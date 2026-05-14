import os
import cv2
import torch
import numpy as np
from collections import deque
from transformers import VideoMAEForVideoClassification, AutoImageProcessor
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

# Resolve model paths relative to the PROJECT ROOT (one level up from src/)
_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# HuggingFace model ID — pre-trained violence vs non-violence classifier (98% accuracy)
# Loaded safely from 100% offline local directory
_MODEL_ID = os.path.join(_DIR, "models", "videomae-violence-local")


class ViolencePreFilter:
    """
    Lightweight MobileNetV2 CNN deployed as a pre-filter.
    Evaluates single frames/crops to determine if the scene looks suspicious.
    If it evaluates < threshold, the expensive VideoMAE inference is skipped.
    """
    def __init__(self, model_path, device="cuda"):
        self.device = device
        print(f"[CNNPreFilter] Loading MobileNetV2 from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.num_classes = checkpoint.get("num_classes", 2)
        
        # Rebuild architecture
        self.model = models.mobilenet_v2()
        original_classifier_in = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(original_classifier_in, self.num_classes),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=checkpoint.get("imagenet_mean", [0.485, 0.456, 0.406]), 
                std=checkpoint.get("imagenet_std", [0.229, 0.224, 0.225])
            ),
        ])
        print(f"[CNNPreFilter] Loaded successfully.")
        
    def predict(self, frame_bgr):
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb_frame).unsqueeze(0).to(self.device)
        
        if self.device == "cuda":
            input_tensor = input_tensor.half()
            
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            
        return probs[1].item()  # index 1 = violence


class AltercationTracker:
    """
    Violence/altercation detector using VideoMAE Video Transformer + ROI Motion Gate.

    Architecture:
        YOLO (CNN) detects humans → VideoMAE (Transformer) classifies violence.
        YOLO is the core CNN-based model; VideoMAE is the helper for action recognition.

    Two-layer detection:
        1. VideoMAE analyzes 16-frame clips for violence probability
        2. ROI-focused optical flow measures MOTION INTENSITY only where people are

        Both must agree for an alert:
        - High VideoMAE violence + High ROI motion → FIGHT (alert)
        - High VideoMAE violence + Low ROI motion  → HUG/safe interaction (suppressed)
        - Low VideoMAE violence + Any motion        → Safe (no alert)

    Key improvement over full-frame optical flow:
        - Full-frame flow is diluted by static background (80%+ of CCTV frame)
        - ROI-focused flow measures ONLY the bounding box where people are
        - This makes hugging register as low motion (gentle) and fighting as high (chaotic)
    """

    def __init__(self, num_frames=16, stride=8, vote_window=4,
                 alert_threshold=0.55, monitor_threshold=0.35,
                 motion_threshold=1.5, min_alert_votes=1,
                 use_prefilter=True, prefilter_threshold=0.30):
        """
        Args:
            num_frames: Number of frames per clip for VideoMAE (default 16)
            stride: Run inference every N frames (8 = every half clip, overlapping)
            vote_window: Number of recent clip predictions to keep for voting
            alert_threshold: Violence probability >= this → candidate for ALERT
            monitor_threshold: Violence probability >= this → MONITORING
            motion_threshold: Minimum avg ROI optical flow magnitude for "violent motion"
                              In CCTV footage (distant camera, low res), motion values are LOW:
                              Hugging ROI scores 0.1-0.8, fighting ROI scores 1.5-4.0
                              Default 1.5 is tuned for typical CCTV conditions.
            min_alert_votes: Minimum number of recent clips that must BOTH have
                             high violence AND high motion to trigger an alert.
                             Prevents single-clip false positives.
        """
        print(f"[AltercationTracker] Loading VideoMAE model from offline directory: {_MODEL_ID}")

        # Load the pre-trained VideoMAE model and image processor completely offline
        self.processor = AutoImageProcessor.from_pretrained(_MODEL_ID, local_files_only=True)
        self.model = VideoMAEForVideoClassification.from_pretrained(_MODEL_ID, local_files_only=True)
        self.model.to(_DEVICE)
        self.model.eval()

        # Use FP16 on CUDA for memory efficiency (RTX 3050 = 4GB VRAM)
        if _DEVICE == "cuda":
            self.model.half()

        self.num_frames = num_frames
        self.stride = stride
        self.alert_threshold = alert_threshold
        self.monitor_threshold = monitor_threshold
        self.motion_threshold = motion_threshold
        self.min_alert_votes = min_alert_votes

        # === CNN Pre-filter initialization ===
        self.use_prefilter = use_prefilter
        self.prefilter_threshold = prefilter_threshold
        if self.use_prefilter:
            model_path = os.path.join(_DIR, "models", "mobilenet_violence_prefilter.pt")
            if os.path.exists(model_path):
                self.cnn_prefilter = ViolencePreFilter(model_path, _DEVICE)
                if _DEVICE == "cuda":
                    self.cnn_prefilter.model.half()
            else:
                print(f"[AltercationTracker] WARNING: Pre-filter model not found at {model_path}. Disabling pre-filter.")
                self.use_prefilter = False

        # Frame buffer — collects raw RGB frames for VideoMAE
        self.frame_buffer = deque(maxlen=num_frames)

        # Gray frame buffer — collects grayscale frames for optical flow
        self.gray_buffer = deque(maxlen=num_frames)

        # ROI buffer — stores the merged person bounding box per frame
        # Each entry is (x1, y1, x2, y2) or None if no persons
        self.roi_buffer = deque(maxlen=num_frames)

        # Temporal voting — stores recent (violence_prob, motion_intensity) tuples
        self.vote_history = deque(maxlen=vote_window)

        # Frame counter for stride-based inference
        self._frame_count = 0

        # Cache the latest results for display between inferences
        self._latest_violence_prob = 0.0
        self._latest_motion = 0.0

        # Track alerting state
        self._currently_alerting = False

        # Determine the violence class index from model config
        self._violence_idx = self._find_violence_class()

        print(f"[AltercationTracker] Model loaded successfully on {_DEVICE}.")
        print(f"[AltercationTracker] Violence class index: {self._violence_idx}")
        print(f"[AltercationTracker] Labels: {self.model.config.id2label}")
        print(f"[AltercationTracker] Motion threshold (ROI): {self.motion_threshold}")
        print(f"[AltercationTracker] Min alert votes: {self.min_alert_votes}/{vote_window}")

    def _find_violence_class(self):
        """
        Find which class index corresponds to violence/fight.
        
        IMPORTANT: Must exclude labels that contain negation prefixes like 'non'.
        For example, 'NonViolence' contains 'violen' but is NOT the violence class.
        """
        id2label = self.model.config.id2label
        violence_keywords = ["violen", "fight", "crime", "assault"]
        negation_prefixes = ["non", "no_", "not_", "safe", "normal"]
        
        for idx, label in id2label.items():
            label_lower = label.lower()
            
            # Check if label matches a violence keyword
            has_violence_keyword = any(kw in label_lower for kw in violence_keywords)
            if not has_violence_keyword:
                continue
            
            # Check if label is NEGATED (e.g., NonViolence, NoFight)
            is_negated = any(label_lower.startswith(neg) for neg in negation_prefixes)
            if is_negated:
                print(f"[AltercationTracker] Skipping negated label: '{label}' (index {idx})")
                continue
            
            return int(idx)

        # If no explicit violence label found, assume class 1
        # (common convention: 0=non-violence, 1=violence)
        print(f"[AltercationTracker] WARNING: No explicit violence label found in {id2label}")
        print(f"[AltercationTracker] Defaulting to class index 1")
        return 1

    def analyze(self, frame, persons=None):
        """
        Analyze a frame for physical altercation.

        Uses TWO signals:
            1. VideoMAE violence probability (temporal visual patterns)
            2. ROI-focused optical flow (how chaotic/fast the movement is WHERE people are)

        Both must be high AND sustained across multiple clips to trigger an alert.

        Args:
            frame: BGR video frame from OpenCV
            persons: List of person dicts with 'bbox' key from HumanGate.
                     Used to focus optical flow on the people region only.
                     If None, falls back to full-frame optical flow.

        Returns:
            frame: The video frame (unmodified by this module)
            detections: List of detection strings (empty if no sustained altercation)
            status: Dict with keys 'type', 'confidence', 'motion' for UI overlay.
                    type is one of: 'alert', 'safe_contact', 'monitoring', 'none'
        """
        detections = []
        status = {"type": "none", "confidence": 0.0, "motion": 0.0}

        # Convert BGR -> RGB for VideoMAE, and BGR -> Gray for optical flow
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.frame_buffer.append(rgb_frame)
        self.gray_buffer.append(gray_frame)

        # Store merged ROI for this frame
        roi = self._merge_person_rois(persons, frame.shape) if persons else None
        self.roi_buffer.append(roi)

        self._frame_count += 1

        # Check if buffer is full enough for inference
        if len(self.frame_buffer) < self.num_frames:
            # Still buffering - skip silently
            return frame, detections, status

        # Run inference every `stride` frames (overlapping clips)
        if self._frame_count % self.stride == 0:
            run_videomae = True

            if self.use_prefilter:
                cnn_frame = frame
                if roi is not None:
                    x1, y1, x2, y2 = roi
                    cnn_frame = frame[y1:y2, x1:x2]

                if cnn_frame.size > 0:
                    cnn_score = self.cnn_prefilter.predict(cnn_frame)
                    if cnn_score < self.prefilter_threshold:
                        run_videomae = False
                        print(f"  [PRE-FILTER] Score {cnn_score:.3f} < {self.prefilter_threshold:.2f} (Safe) -> Skipping VideoMAE")
                    else:
                        print(f"  [PRE-FILTER] Score {cnn_score:.3f} >= {self.prefilter_threshold:.2f} (Suspicious) -> Running VideoMAE")

            if run_videomae:
                violence_prob = self._run_inference()
                motion_intensity = self._compute_motion_intensity()

                self._latest_violence_prob = violence_prob
                self._latest_motion = motion_intensity
                self.vote_history.append((violence_prob, motion_intensity))

                # Console debug output
                print(f"  [CLIP] violence={violence_prob:.4f}  motion(ROI)={motion_intensity:.2f}  "
                      f"buf={len(self.frame_buffer)}/{self.num_frames}")
            else:
                # If we skip VideoMAE because it's safe, feed safe values into history
                # so that any past violence votes get flushed out.
                self.vote_history.append((0.0, 0.0))
                self._latest_violence_prob = 0.0
                self._latest_motion = 0.0

        # Use consensus-based voting across recent predictions exactly like the old system
        if len(self.vote_history) > 0:
            avg_violence = sum(v for v, m in self.vote_history) / len(self.vote_history)
            avg_motion = sum(m for v, m in self.vote_history) / len(self.vote_history)

            # Collect only the votes that meet BOTH alert thresholds
            # These are the clips that actually caused the system to alert
            alert_qualifying_votes = [
                (v, m) for v, m in self.vote_history
                if v >= self.alert_threshold and m >= self.motion_threshold
            ]
            alert_votes = len(alert_qualifying_votes)

            # CONSENSUS requirement: need min_alert_votes clips agreeing
            if alert_votes >= self.min_alert_votes:
                # Sustained high violence + high motion -> REAL FIGHT
                # Show the REAL confidence: average of only the clips that triggered the alert,
                # not the diluted average that includes safe 0.0 entries from CNN pre-filter skips
                real_confidence = sum(v for v, m in alert_qualifying_votes) / alert_votes
                real_motion = sum(m for v, m in alert_qualifying_votes) / alert_votes
                status = {"type": "alert", "confidence": real_confidence, "motion": real_motion}
                if "Physical Altercation" not in detections:
                    detections.append("Physical Altercation")
                self._currently_alerting = True
                
            elif avg_violence >= self.alert_threshold and avg_motion < self.motion_threshold:
                # High violence + Low motion -> likely HUG or calm interaction
                status = {"type": "safe_contact", "confidence": avg_violence, "motion": avg_motion}
                self._currently_alerting = False
                
            elif avg_violence >= self.monitor_threshold:
                # Ambiguous - monitoring
                status = {"type": "monitoring", "confidence": avg_violence, "motion": avg_motion}
                self._currently_alerting = False
                
            else:
                self._currently_alerting = False

        return frame, detections, status

    def _merge_person_rois(self, persons, frame_shape):
        """
        Merge all person bounding boxes into one combined ROI.
        This gives us the region of the frame where ALL people are.
        Adds a small margin around the merged box.
        """
        if not persons:
            return None

        h, w = frame_shape[:2]
        margin = 60  # Increased padding to capture arms/weapons swinging OUTSIDE the torso area

        x1 = min(p["bbox"][0] for p in persons)
        y1 = min(p["bbox"][1] for p in persons)
        x2 = max(p["bbox"][2] for p in persons)
        y2 = max(p["bbox"][3] for p in persons)

        # Add margin, clamp to frame bounds
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        return (x1, y1, x2, y2)

    def _run_inference(self):
        """
        Run VideoMAE inference on the current frame buffer.

        Returns:
            float: Violence probability (0.0 to 1.0)
        """
        # Get the 16 frames from the buffer
        clip_frames = list(self.frame_buffer)

        # Process frames using the HuggingFace image processor
        # This handles resizing to 224x224, normalization, etc.
        inputs = self.processor(clip_frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(_DEVICE)

        # Use FP16 on CUDA
        if _DEVICE == "cuda":
            pixel_values = pixel_values.half()

        # Run the model
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        violence_prob = probs[self._violence_idx].item()
        return violence_prob

    def _compute_motion_intensity(self):
        """
        Compute average optical flow magnitude ONLY within the person ROI.

        By focusing on the bounding box where people are:
        - Hugging: gentle embrace motion in ROI → LOW magnitude (1-3)
        - Fighting: chaotic strikes in ROI → HIGH magnitude (4-10+)
        - Background motion (trees, camera shake) is EXCLUDED

        Falls back to full-frame if no ROI data is available.

        Returns:
            float: Average optical flow magnitude within the people region.
        """
        gray_frames = list(self.gray_buffer)
        rois = list(self.roi_buffer)

        if len(gray_frames) < 2:
            return 0.0

        total_flow = 0.0
        num_pairs = 0

        # Sample every other frame pair to save computation
        for i in range(0, len(gray_frames) - 1, 2):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i], gray_frames[i + 1],
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            # flow shape: (H, W, 2) — dx, dy per pixel
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

            # Use ROI if available for this frame pair
            roi = rois[i + 1] if (i + 1) < len(rois) and rois[i + 1] is not None else None

            if roi is not None:
                x1, y1, x2, y2 = roi
                roi_mag = magnitude[y1:y2, x1:x2]
                if roi_mag.size > 0:
                    # Use the 90th percentile instead of mean to capture peak motion
                    # Mean dilutes chaotic motion with static body parts
                    # 90th percentile captures the MOST moving part (fists, arms)
                    total_flow += np.percentile(roi_mag, 90)
                else:
                    total_flow += np.mean(magnitude)
            else:
                # Fallback: full-frame (less accurate but works without person data)
                total_flow += np.mean(magnitude)

            num_pairs += 1

        avg_flow = total_flow / max(num_pairs, 1)
        return avg_flow

    def _draw_alert(self, frame, confidence, motion):
        """Draw a red PHYSICAL ALTERCATION alert banner on the frame."""
        h, w = frame.shape[:2]
        scale_factor = w / 640.0

        # Red banner at top
        banner_h = int(40 * scale_factor)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        text = f"PHYSICAL ALTERCATION ({confidence:.0%})"
        font_scale = 0.6 * scale_factor
        thickness = max(1, int(1.5 * scale_factor))
        cv2.putText(frame, text, (int(8 * scale_factor), int(banner_h * 0.72)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Motion info (smaller, bottom-right)
        motion_text = f"Motion(ROI): {motion:.1f}"
        small_scale = 0.35 * scale_factor
        small_thick = max(1, int(1.0 * scale_factor))
        (tw, _), _ = cv2.getTextSize(motion_text, cv2.FONT_HERSHEY_SIMPLEX, small_scale, small_thick)
        cv2.putText(frame, motion_text, (w - tw - int(10 * scale_factor), h - int(10 * scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, small_scale, (0, 0, 255), small_thick)

        # Red border around entire frame
        border = max(2, int(3 * scale_factor))
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), border)

    def _draw_safe_contact(self, frame, confidence, motion):
        """Draw a green 'safe interaction' indicator when VideoMAE flags violence but motion is low."""
        h, w = frame.shape[:2]
        scale_factor = w / 640.0

        text = f"SAFE INTERACTION (motion: {motion:.1f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5 * scale_factor
        thickness = max(1, int(1.5 * scale_factor))
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        padding = int(8 * scale_factor)
        x1 = w - text_w - padding * 2
        y1 = 5
        x2 = w - 5
        y2 = y1 + text_h + padding * 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, text, (x1 + padding, y2 - padding),
                    font, font_scale, (0, 200, 0), thickness)

    def _draw_monitoring(self, frame, confidence, motion):
        """Draw a yellow MONITORING banner on the frame."""
        h, w = frame.shape[:2]
        scale_factor = w / 640.0

        text = f"MONITORING ({confidence:.0%} | motion: {motion:.1f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5 * scale_factor
        thickness = max(1, int(1.5 * scale_factor))
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        padding = int(8 * scale_factor)
        x1 = w - text_w - padding * 2
        y1 = 5
        x2 = w - 5
        y2 = y1 + text_h + padding * 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, text, (x1 + padding, y2 - padding),
                    font, font_scale, (0, 255, 255), thickness)

    def _draw_buffering(self, frame, progress):
        """Draw a buffering progress indicator on the frame."""
        h, w = frame.shape[:2]
        scale_factor = w / 640.0

        text = f"BUFFERING... ({progress:.0%})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5 * scale_factor
        thickness = max(1, int(1.5 * scale_factor))
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        padding = int(8 * scale_factor)
        x1 = w - text_w - padding * 2
        y1 = 5
        x2 = w - 5
        y2 = y1 + text_h + padding * 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, text, (x1 + padding, y2 - padding),
                    font, font_scale, (200, 200, 200), thickness)

    def buffer_frame(self, frame, persons=None):
        """
        Add a frame to the buffer WITHOUT running inference or drawing overlays.

        Call this when humans are detected but people aren't close enough
        to trigger altercation analysis. This keeps the frame buffer filled
        with contiguous frames so that when people DO become close, the
        first inference has a valid, recent clip to analyze.

        Args:
            frame: BGR video frame from OpenCV
            persons: Optional list of person dicts from HumanGate
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_buffer.append(rgb_frame)
        self.gray_buffer.append(gray_frame)

        roi = self._merge_person_rois(persons, frame.shape) if persons else None
        self.roi_buffer.append(roi)

        self._frame_count += 1

    def soft_reset(self):
        """
        Clear vote history but KEEP the frame buffer.
        
        Call this when transitioning from 'not close' to 'close'.
        We keep the frame buffer because:
        1. In CCTV footage, proximity detection flaps frequently (people
           toggle between close/not-close every few frames due to detection noise)
        2. If we clear the buffer every time, we NEVER accumulate 16 frames
           and NEVER run inference — the system becomes completely blind
        3. The frame buffer contains recent visual context that IS relevant
        
        We only clear votes to prevent stale fight/safe verdicts from carrying over.
        """
        self.vote_history.clear()
        self._latest_violence_prob = 0.0
        self._latest_motion = 0.0
        self._currently_alerting = False

    def reset(self):
        """Full reset — call when switching videos or when no humans detected."""
        self.frame_buffer.clear()
        self.gray_buffer.clear()
        self.roi_buffer.clear()
        self.vote_history.clear()
        self._frame_count = 0
        self._latest_violence_prob = 0.0
        self._latest_motion = 0.0
        self._currently_alerting = False

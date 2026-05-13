import os
import cv2
import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Resolve model paths relative to THIS file's directory
_DIR = os.path.dirname(os.path.abspath(__file__))
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class WeaponTracker:
    """
    Weapon (knife + gun) detector.
    
    OPTIMIZED: Accepts pre-detected human bounding boxes from the HumanGate
    and runs all crops in a single GPU batch for maximum throughput.
    
    Knife scan: Uses custom 'knife_detector_adamw.pt' (class 0) — fine-tuned
    on hand-held knife images for surveillance accuracy.
    
    Gun scan: Uses 'Subh775/Firearm_Detection_Yolov8n' (class 0) from Hugging Face
    with a temporal vote filter to suppress false positives.
    """

    def __init__(self):
        from collections import deque

        # --- Knife model (custom hand-held knife detector) ---
        self.knife_model = YOLO(os.path.join(_DIR, "models", "knife_detector_adamw.pt"))

        # --- Gun model (CCTV Weapon Medium) ---
        print("[WeaponTracker] Loading custom CCTV gun detection model...")
        _gun_model_path = os.path.join(_DIR, "models", "cctv_gun_detector.pt")
        self.gun_model = YOLO(_gun_model_path)
        print("[WeaponTracker] Gun model loaded.")

        # --- Temporal voting (knife): alert only if seen in >= 6 of last 12 frames ---
        # Increased from 4/8 to 6/12 to reduce annotation clutter from brief false positives.
        # The weapon must be consistently visible for a longer period before boxes are drawn.
        self._vote_window = 12
        self._vote_threshold = 6
        self._vote_history = deque(maxlen=self._vote_window)

        # --- Temporal voting (gun): same window/threshold as knife ---
        self._gun_vote_history = deque(maxlen=self._vote_window)
    
    def scan_frame(self, frame, persons=None):
        """
        Scan for weapons in the frame.
        
        Args:
            frame: The video frame to analyze
            persons: List of person dicts from HumanGate (each has 'bbox' key).
        
        Returns:
            frame: Annotated frame with weapon bounding boxes
            detections: List of detected weapon labels
        """
        detections = []
        
        if persons is None:
            return frame, detections
        
        raw_gun_detections = []

        # ----------------- BATCH PREPARATION -----------------
        # Collect all human crops first to run them through YOLO in a single batch.
        # This preserves the "optical zoom" accuracy while restoring speed!
        crops = []
        crop_offsets = [] # Keep track of coordinates to translate boxes back to full-frame
        
        for person in persons:
            x1, y1, x2, y2 = person["bbox"]
            margin = 30
            h, w = frame.shape[:2]
            pad_x1 = max(0, x1 - margin)
            pad_y1 = max(0, y1 - margin)
            pad_x2 = min(w, x2 + margin)
            pad_y2 = min(h, y2 + margin)
            
            human_crop = frame[pad_y1:pad_y2, pad_x1:pad_x2]
            if human_crop.size > 0:
                crops.append(human_crop)
                crop_offsets.append((pad_x1, pad_y1, x1, y1, x2, y2))
                
        if not crops:
            return frame, detections
            
        # ----------------- BATCH INFERENCE -----------------
        # Run YOLO on the list of crops. PyTorch processes them in parallel!
        all_weapon_results = self.knife_model(crops, classes=[0], conf=0.35, device=_DEVICE, verbose=False)
        all_gun_results = self.gun_model(crops, classes=[1], conf=0.15, device=_DEVICE, verbose=False)
        
        h, w = frame.shape[:2]
        scale_factor = w / 640.0
        box_thick = max(1, int(2 * scale_factor))
        font_scale = 0.35 * scale_factor
        txt_thick = max(1, int(1.5 * scale_factor))

        # ----------------- PROCESS KNIFE RESULTS -----------------
        raw_knife_boxes = []
        for i, weapon_result in enumerate(all_weapon_results):
            pad_x1, pad_y1, hx1, hy1, hx2, hy2 = crop_offsets[i]
            for w_box in weapon_result.boxes:
                wx1, wy1, wx2, wy2 = map(int, w_box.xyxy[0])
                
                # Convert crop coordinates back to full-frame coordinates
                main_wx1 = pad_x1 + wx1
                main_wy1 = pad_y1 + wy1
                main_wx2 = pad_x1 + wx2
                main_wy2 = pad_y1 + wy2
                
                # --- IoW Filter: Knife must overlap with the specific human bbox ---
                ix1 = max(main_wx1, hx1)
                iy1 = max(main_wy1, hy1)
                ix2 = min(main_wx2, hx2)
                iy2 = min(main_wy2, hy2)
                
                if ix1 < ix2 and iy1 < iy2:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                else:
                    intersection = 0
                    
                weapon_area = max(1, (main_wx2 - main_wx1) * (main_wy2 - main_wy1))
                iow = intersection / weapon_area
                
                # Require at least 20% overlap with human box
                if iow >= 0.20:
                    conf = float(w_box.conf[0])
                    raw_knife_boxes.append((main_wx1, main_wy1, main_wx2, main_wy2, conf))

        # --- Temporal voting (knife) ---
        raw_knife_found = len(raw_knife_boxes) > 0
        self._vote_history.append(raw_knife_found)
        knife_votes = sum(self._vote_history)

        if knife_votes >= self._vote_threshold:
            # Vote passed! Draw the boxes and register detection
            for box in raw_knife_boxes:
                main_wx1, main_wy1, main_wx2, main_wy2, conf = box
                cv2.rectangle(frame, (main_wx1, main_wy1), (main_wx2, main_wy2), (0, 0, 255), box_thick)
                cv2.putText(frame, f"Knife ({conf:.0%})", (main_wx1, main_wy1 + int(15 * scale_factor)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), txt_thick)
            if raw_knife_found:
                detections.append("Knife")
        else:
            if raw_knife_found:
                print(f"  [WEAPON] Knife spike suppressed ({knife_votes}/{self._vote_threshold} votes)")

        # ----------------- PROCESS GUN RESULTS -----------------
        raw_gun_boxes = []
        for i, gun_result in enumerate(all_gun_results):
            pad_x1, pad_y1, hx1, hy1, hx2, hy2 = crop_offsets[i]
            for g_box in gun_result.boxes:
                gx1, gy1, gx2, gy2 = map(int, g_box.xyxy[0])
                
                # Convert crop coordinates back to full-frame coordinates
                main_gx1 = pad_x1 + gx1
                main_gy1 = pad_y1 + gy1
                main_gx2 = pad_x1 + gx2
                main_gy2 = pad_y1 + gy2
                
                # --- IoW Filter: Gun must overlap with the specific human bbox ---
                ix1 = max(main_gx1, hx1)
                iy1 = max(main_gy1, hy1)
                ix2 = min(main_gx2, hx2)
                iy2 = min(main_gy2, hy2)
                
                if ix1 < ix2 and iy1 < iy2:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                else:
                    intersection = 0
                    
                gun_area = max(1, (main_gx2 - main_gx1) * (main_gy2 - main_gy1))
                iow = intersection / gun_area
                
                # Require at least 20% overlap with human box
                if iow >= 0.20:
                    conf = float(g_box.conf[0])
                    raw_gun_boxes.append((main_gx1, main_gy1, main_gx2, main_gy2, conf))
        
        # --- Temporal voting (gun) ---
        raw_gun_found = len(raw_gun_boxes) > 0
        self._gun_vote_history.append(raw_gun_found)
        gun_votes = sum(self._gun_vote_history)

        if gun_votes >= self._vote_threshold:
            # Vote passed! Draw the boxes and register detection
            for box in raw_gun_boxes:
                main_gx1, main_gy1, main_gx2, main_gy2, conf = box
                cv2.rectangle(frame, (main_gx1, main_gy1), (main_gx2, main_gy2), (0, 165, 255), box_thick)
                cv2.putText(frame, f"Gun ({conf:.0%})", (main_gx1, main_gy1 + int(15 * scale_factor)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), txt_thick)
            if raw_gun_found:
                detections.append("Gun")
        else:
            if raw_gun_found:
                print(f"  [WEAPON] Gun spike suppressed ({gun_votes}/{self._vote_threshold} votes)")

        return frame, detections

import os
import cv2
import torch
from ultralytics import YOLO

# Resolve model paths relative to THIS file's directory
_DIR = os.path.dirname(os.path.abspath(__file__))
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class HumanGate:
    """
    Human detector using YOLOv8s for reliable person detection.
    Runs ONCE per frame and shares results with other modules.
    
    Uses YOLOv8s instead of YOLOv8n because CCTV/surveillance footage
    often has small, distant people that the nano model misses.
    """
    
    def __init__(self):
        # YOLOv8s — more accurate for CCTV footage with small/distant people
        self.model = YOLO(os.path.join(_DIR, "models", "yolov8s.pt"))
    
    def detect_humans(self, frame):
        """
        Detect all humans in the frame.
        Returns a list of person dicts with bounding box and confidence.
        """
        # Raised confidence (0.40) to reduce false positives
        results = self.model(frame, classes=[0], conf=0.10, iou=0.4, device=_DEVICE, verbose=False)
        
        persons = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                persons.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf
                })
        
        return persons
    
    def are_people_close(self, persons, frame_shape=None, proximity_ratio=0.15):
        """
        Check if any two detected people are within proximity of each other
        (measured center-to-center). Only then should altercation detection run.
        
        The proximity threshold is calculated as a percentage of the frame diagonal,
        so it adapts to different video resolutions automatically.
        
        Args:
            persons: List of person dicts with 'bbox' key
            frame_shape: (height, width) of the frame — used to compute threshold
            proximity_ratio: Fraction of frame diagonal to use as threshold (default 15%)
        """
        if len(persons) < 2:
            return False
        
        # Compute proximity threshold relative to frame size
        if frame_shape is not None:
            h, w = frame_shape[:2]
            diagonal = (h ** 2 + w ** 2) ** 0.5
            proximity_threshold = diagonal * proximity_ratio
        else:
            # Fallback to a generous fixed value if frame_shape not provided
            proximity_threshold = 300
        
        for i, p1 in enumerate(persons):
            for p2 in persons[i + 1:]:
                cx1 = (p1["bbox"][0] + p1["bbox"][2]) / 2
                cy1 = (p1["bbox"][1] + p1["bbox"][3]) / 2
                cx2 = (p2["bbox"][0] + p2["bbox"][2]) / 2
                cy2 = (p2["bbox"][1] + p2["bbox"][3]) / 2
                dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                if dist < proximity_threshold:
                    return True
        
        return False
    
    def draw_human_boxes(self, frame, persons):
        """Draw bounding boxes for all detected humans on the frame."""
        h, w = frame.shape[:2]
        scale_factor = w / 640.0
        font_scale = 0.35 * scale_factor
        thickness = max(1, int(1.5 * scale_factor))
        box_thick = max(1, int(2 * scale_factor))
        for person in persons:
            x1, y1, x2, y2 = person["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), box_thick)
            cv2.putText(frame, "Human", (x1, y1 + int(15 * scale_factor)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
        return frame

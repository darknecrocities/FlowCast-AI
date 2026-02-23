import os
import cv2
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Generator
from ultralytics import YOLO

import torch
import functools

# Monkeypatch torch.load to bypass the strict weights_only=True default in PyTorch 2.6+
# This is necessary because YOLOv8 models contain many custom internal classes 
# that are not on the default allowlist.
_original_torch_load = torch.load
@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

class DetectionEngine:
    """
    YOLOv8-based Object Detection Engine for FlowCast AI.
    Handles loading the model, moving it to the appropriate device (GPU/CPU),
    and performing batch/stream inference.
    """
    def __init__(self, model_id: str = "yolov8n.pt", confidence: float = 0.4, classes: List[int] = None, device: str = "auto"):
        self.confidence = confidence
        self.classes = classes if classes else [0, 2, 3, 5, 7] # def: person, car, motorcycle, bus, truck
        
        # Determine optimal device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"[Engine] Loading {model_id} on {self.device}...")
        self.model = YOLO(model_id)
        self.model.to(self.device)
        print(f"[Engine] Detection Engine initialized successfully.")

    def detect_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on a single bounding box frame.
        
        Args:
            frame: A numpy array representing an image/video frame (BGR format from cv2).
            
        Returns:
            List of dictionaries containing bounding box coordinates (xyxy), confidence, and class ID.
        """
        # Run inference
        results = self.model.predict(
            source=frame, 
            conf=self.confidence, 
            classes=self.classes, 
            device=self.device,
            verbose=False
        )[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": conf,
                "class_id": cls
            })
            
        return detections

    def detect_stream(self, source: str) -> Generator[Tuple[np.ndarray, List[Dict[str, Any]]], None, None]:
        """
        Yields frames and their detections from a video stream or file.
        
        Args:
            source: Path to video file or camera stream index (e.g., 0).
            
        Yields:
            Tuple of the original frame and the list of detections.
        """
        # Note: We use OpenCV VideoCapture instead of ultralytics stream to have 
        # granular control over frame emission and downstream tracking/rendering.
        try:
            source_idx = int(source)
        except ValueError:
            source_idx = source
            
        cap = cv2.VideoCapture(source_idx)
        
        if not cap.isOpened():
            raise ValueError(f"Unable to open video source: {source}")
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            detections = self.detect_frame(frame)
            yield frame, detections
            
        cap.release()

if __name__ == "__main__":
    # Sanity Check
    print("Running Detection Engine Sanity Check...")
    engine = DetectionEngine()
    # Create a dummy image
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = engine.detect_frame(dummy_img)
    print(f"Sanity Check Passed. Detected {len(dets)} objects in blank array.")

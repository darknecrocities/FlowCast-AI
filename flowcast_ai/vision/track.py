import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Any, Tuple

def box_iou(boxes1, boxes2):
    """
    Computes Intersection over Union (IoU) between two sets of bounding boxes.
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
        
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, 0, None)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    return inter / np.clip(union, 1e-6, None)

class Track:
    def __init__(self, obj_id, bbox, class_id=0, confidence=0.0):
        self.obj_id = obj_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.confidence = confidence
        self.history = [] # Past positions: (timestamp, cx, cy)
        self.velocity = (0.0, 0.0) # (vx, vy)
        self.time_since_update = 0
        self.hits = 1

    def update(self, bbox, confidence, timestamp):
        # 1. Store previous center for velocity calculation
        prev_cx, prev_cy = self.get_center()
        prev_ts = self.history[-1][0] if len(self.history) > 0 else None
        
        # 2. Update current state
        self.bbox = bbox
        self.confidence = confidence
        curr_cx, curr_cy = self.get_center()
        
        # 3. Append to history
        self.history.append((timestamp, curr_cx, curr_cy))
        if len(self.history) > 30:
            self.history.pop(0)
            
        # 4. Calculate velocity: (pixels change) / (time change)
        if prev_ts is not None:
            dt = timestamp - prev_ts
            if dt > 0:
                self.velocity = ((curr_cx - prev_cx) / dt, (curr_cy - prev_cy) / dt)
            
        self.time_since_update = 0
        self.hits += 1

    def get_center(self):
        return (self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0

class ByteTracker:
    """
    Simplified ByteTrack implementation.
    1. Match high-score detections with existing tracks.
    2. Match remaining low-score detections with unmatched tracks.
    """
    def __init__(self, track_buffer=30, high_thresh=0.6, low_thresh=0.1, match_thresh=0.8):
        self.track_buffer = track_buffer
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.tracks: List[Track] = []
        self.next_id = 1

    def update(self, detections: List[Dict[str, Any]], timestamp: float) -> List[Dict[str, Any]]:
        """
        Takes raw detections and assigns tracking IDs.
        Yields list of dicts: {ID, x_center, y_center, velocity, timestamp}
        """
        if len(detections) == 0:
            for t in self.tracks:
                t.time_since_update += 1
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.track_buffer]
            return []

        # Split detections based on confidence
        det_boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        det_classes = np.array([d['class_id'] for d in detections])
        
        high_idx = np.where(scores >= self.high_thresh)[0]
        low_idx = np.where((scores >= self.low_thresh) & (scores < self.high_thresh))[0]
        
        high_dets = det_boxes[high_idx]
        high_classes = det_classes[high_idx]
        low_dets = det_boxes[low_idx]
        low_classes = det_classes[low_idx]
        
        # Step 1: Match high score detections with existing tracks
        track_boxes = np.array([t.bbox for t in self.tracks])
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_high_dets = list(range(len(high_dets)))
        
        if len(track_boxes) > 0 and len(high_dets) > 0:
            ious = box_iou(track_boxes, high_dets)
            cost_matrix = 1.0 - ious
            
            # Using Hungarian Algorithm to solve the assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] > self.match_thresh:
                    continue
                self.tracks[r].update(high_dets[c], scores[high_idx][c], timestamp)
                unmatched_tracks.remove(r)
                unmatched_high_dets.remove(c)

        # Step 2: Match low score detections with remaining unmatched tracks
        unmatched_track_boxes = np.array([self.tracks[i].bbox for i in unmatched_tracks])
        
        if len(unmatched_track_boxes) > 0 and len(low_dets) > 0:
            ious_low = box_iou(unmatched_track_boxes, low_dets)
            cost_matrix_low = 1.0 - ious_low
            
            row_ind_low, col_ind_low = linear_sum_assignment(cost_matrix_low)
            
            matches_low = []
            for r, c in zip(row_ind_low, col_ind_low):
                if cost_matrix_low[r, c] <= self.match_thresh:
                    matches_low.append((unmatched_tracks[r], c))

            for track_idx, c in matches_low:
                self.tracks[track_idx].update(low_dets[c], scores[low_idx][c], timestamp)
                unmatched_tracks.remove(track_idx)

        # Step 3: Instantiate new tracks for unmatched high-score detections
        for c in unmatched_high_dets:
            new_track = Track(self.next_id, high_dets[c], high_classes[c], scores[high_idx][c])
            new_track.history.append((timestamp, new_track.get_center()[0], new_track.get_center()[1]))
            self.tracks.append(new_track)
            self.next_id += 1

        # Remove tracks that haven't been updated
        for i in unmatched_tracks:
            self.tracks[i].time_since_update += 1
            
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.track_buffer]

        # Formatting Output
        output = []
        for t in self.tracks:
            if t.time_since_update == 0:  # Only output active tracks in current frame
                cx, cy = t.get_center()
                output.append({
                    "ID": t.obj_id,
                    "class_id": t.class_id,
                    "x_center": cx,
                    "y_center": cy,
                    "velocity": t.velocity,
                    "timestamp": timestamp,
                    "bbox": t.bbox, # Kept for drawing
                    "confidence": t.confidence
                })
                
        return output

if __name__ == "__main__":
    # Sanity Check
    print("Running ByteTrack Sanity Check...")
    tracker = ByteTracker()
    dets = [{"bbox": [10, 10, 50, 50], "confidence": 0.9, "class_id": 0}]
    ts = time.time()
    res = tracker.update(dets, ts)
    print(f"Frame 1: {res}")
    
    time.sleep(0.1)
    dets2 = [{"bbox": [15, 15, 55, 55], "confidence": 0.85, "class_id": 0}]
    res2 = tracker.update(dets2, time.time())
    print(f"Frame 2: {res2}")

import numpy as np
import math
from typing import Dict, List, Tuple

class TrajectoryEngine:
    """
    Computes and stores trajectory histories, motion vectors, and acceleration.
    Operates on top of the generic tracking outputs.
    """
    def __init__(self, max_history: int = 600):
        # Maps ID to a list of (x, y, timestamp, vx, vy)
        self.trajectories: Dict[int, List[Tuple[float, float, float, float, float]]] = {}
        self.max_history = max_history

    def update(self, tracked_objects: List[Dict]) -> Dict[int, Dict]:
        """
        Updates the trajectory history with new tracked objects.
        
        Args:
            tracked_objects: Output from ByteTracker update.
            
        Returns:
            Dictionary of active trajectory intelligence features.
        """
        active_features = {}
        
        for obj in tracked_objects:
            obj_id = obj['ID']
            x = obj['x_center']
            y = obj['y_center']
            vx, vy = obj['velocity']
            ts = obj['timestamp']
            
            if obj_id not in self.trajectories:
                self.trajectories[obj_id] = []
                
            history = self.trajectories[obj_id]
            history.append((x, y, ts, vx, vy))
            
            if len(history) > self.max_history:
                history.pop(0)
                
            # Compute advanced features
            motion_vector, acceleration, direction_angle = self._compute_kinematics(obj_id)
            
            active_features[obj_id] = {
                "history": history,
                "motion_vector": motion_vector,     # Current [vx, vy] smoothed
                "acceleration": acceleration,       # [ax, ay]
                "direction_angle": direction_angle, # in degrees
                "speed": math.hypot(*motion_vector) # magnitude
            }
            
        # Optional: cleanup inactive trajectories to prevent memory leaks
        # We can pass an active_ids set and remove missing ones, 
        # but realistically, tracks might reappear, so cleanup should happen sparingly.
        
        return active_features

    def _compute_kinematics(self, obj_id: int) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """
        Computes motion vector, acceleration, and angle for a given object ID 
        using its recent history.
        """
        history = self.trajectories.get(obj_id, [])
        if len(history) < 2:
            return (0.0, 0.0), (0.0, 0.0), 0.0
            
        # Recent state
        curr = history[-1]
        prev = history[-2]
        
        # Velocity from tracker
        vx_curr, vy_curr = curr[3], curr[4]
        vx_prev, vy_prev = prev[3], prev[4]
        
        dt = curr[2] - prev[2]
        if dt > 0:
            ax = (vx_curr - vx_prev) / dt
            ay = (vy_curr - vy_prev) / dt
        else:
            ax, ay = (0.0, 0.0)
            
        # Smooth motion vector
        motion_vector = (vx_curr, vy_curr)
        # 0 = Right, 90 = Down, 180 = Left, -90 = Up (Image coordinates)
        direction_angle = math.degrees(math.atan2(vy_curr, vx_curr))
        
        return motion_vector, (ax, ay), direction_angle
        
    def cleanup(self, active_ids: set):
        """
        Removes dead trajectories to free up memory.
        Should be called periodically rather than every frame.
        """
        stale_ids = [tid for tid in self.trajectories if tid not in active_ids]
        for tid in stale_ids:
            del self.trajectories[tid]

if __name__ == "__main__":
    eng = TrajectoryEngine()
    
    # Simulate a linearly moving object
    res = eng.update([{"ID": 1, "x_center": 10, "y_center": 10, "velocity": (0,0), "timestamp": 0}])
    res = eng.update([{"ID": 1, "x_center": 20, "y_center": 10, "velocity": (10,0), "timestamp": 1}])
    res = eng.update([{"ID": 1, "x_center": 30, "y_center": 10, "velocity": (10,0), "timestamp": 2}])
    
    print(f"Features: {res[1]}")

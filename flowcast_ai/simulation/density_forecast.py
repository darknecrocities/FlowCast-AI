import numpy as np
import cv2
from typing import Dict, Any, List

class FlowSimulationEngine:
    """
    Transforms the abstract ST-GNN predictions (on the grid graph)
    into visual overlays: heatmaps, flow vectors, risk zones.
    """
    def __init__(self, frame_width: int, frame_height: int, grid_size: int = 25):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.grid_size = grid_size
        
        self.cell_w = self.frame_width // self.grid_size
        self.cell_h = self.frame_height // self.grid_size

    def generate_heatmap(self, node_densities: np.ndarray, max_density: float = 10.0) -> np.ndarray:
        """
        Converts a 1D array of node densities into a 2D color heatmap image
        that matches the original frame resolution.
        """
        assert len(node_densities) == self.grid_size * self.grid_size
        
        # Reshape to grid
        grid_density = node_densities.reshape((self.grid_size, self.grid_size))
        
        # Normalize to 0-255 based on assumed max density per cell
        normalized = np.clip(grid_density / max_density, 0, 1.0)
        heatmap_gray = (normalized * 255).astype(np.uint8)
        
        # Resize to frame dimensions using interpolation
        heatmap_resized = cv2.resize(heatmap_gray, (self.frame_width, self.frame_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply colormap (Color: Blue -> Free, Yellow -> Pressure, Red -> Congestion)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # Mask out zero-density regions completely
        mask = heatmap_resized > 10
        result = np.zeros_like(heatmap_color)
        result[mask] = heatmap_color[mask]
        
        return result

    def get_risk_zones(self, node_pressures: np.ndarray, threshold: float = 15.0) -> List[Dict[str, Any]]:
        """
        Identifies high-pressure areas as bounding boxes for UI rendering.
        """
        zones = []
        for i, pressure in enumerate(node_pressures):
            if pressure > threshold:
                gy = i // self.grid_size
                gx = i % self.grid_size
                
                # Bounding box of the grid cell
                x1 = gx * self.cell_w
                y1 = gy * self.cell_h
                x2 = x1 + self.cell_w
                y2 = y1 + self.cell_h
                
                zones.append({
                    "bbox": [x1, y1, x2, y2],
                    "pressure": float(pressure),
                    "node_id": i
                })
        return zones

    def render_overlay(self, frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Blends the target heatmap over the original video frame.
        """
        return cv2.addWeighted(heatmap, alpha, frame, 1 - alpha, 0)
        
if __name__ == "__main__":
    # Sanity Check
    print("Running FlowSimulationEngine Sanity Check...")
    engine = FlowSimulationEngine(1920, 1080, 5) # 5x5 grid
    
    # Mock density (highest in center)
    densities = np.zeros(25)
    densities[12] = 8.0 # Center node
    densities[11] = 4.0
    densities[13] = 4.0
    
    # Generate Heatmap
    heatmap = engine.generate_heatmap(densities)
    print(f"Heatmap Shape: {heatmap.shape}")
    
    # Mock pressures
    pressures = np.zeros(25)
    pressures[12] = 20.0 # High pressure in center
    
    zones = engine.get_risk_zones(pressures)
    print(f"Risk Zones Identified: {len(zones)}")
    print(f"Sample Zone: {zones[0]}")

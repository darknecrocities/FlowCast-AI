import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, Tuple, List

class SpatialGraphBuilder:
    """
    Partitions the video frame into a grid of zones. 
    Each zone is a Node in the Spatial Graph.
    Objects in zones contribute to feature vectors (density, velocity).
    """
    def __init__(self, frame_width: int = 1920, frame_height: int = 1080, grid_size: int = 25, decay: float = 0.9):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.grid_size = grid_size
        self.decay = decay
        
        self.num_nodes = grid_size * grid_size
        self.cell_w = frame_width / grid_size
        self.cell_h = frame_height / grid_size
        
        # Build adjacency matrix (edges) for the grid (8-way connectivity)
        self.edge_index = self._build_grid_edges()
        
        # Historical state features: shape [num_nodes, feature_dim]
        # Features: [density, avg_vx, avg_vy, pressure]
        self.node_features = np.zeros((self.num_nodes, 4), dtype=np.float32)

    def _build_grid_edges(self) -> torch.Tensor:
        """
        Creates edge pairs for an N x N grid with 8-way connectivity.
        """
        edges = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                u = y * self.grid_size + x
                
                # Connect to 8 neighbors
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ny = y + dy
                        nx = x + dx
                        if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                            v = ny * self.grid_size + nx
                            edges.append([u, v])
                            
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def get_node_index(self, x: float, y: float) -> int:
        """ Maps (x,y) pixel coordinate to a 1D node index. """
        gx = min(int(x / self.cell_w), self.grid_size - 1)
        gy = min(int(y / self.cell_h), self.grid_size - 1)
        return gy * self.grid_size + gx

    def update_graph(self, trajectories: Dict[int, Dict]) -> Data:
        """
        Aggregates frame-level object features into node-level graph features.
        
        Args:
            trajectories: Output from TrajectoryEngine.
                          Format: Dict[obj_id, features]
                          
        Returns:
            PyTorch Geometric Data object containing edge_index and x (features)
        """
        # Current frame observations
        curr_density = np.zeros(self.num_nodes, dtype=np.float32)
        curr_vx = np.zeros(self.num_nodes, dtype=np.float32)
        curr_vy = np.zeros(self.num_nodes, dtype=np.float32)
        
        # Map objects to nodes
        for obj_id, features in trajectories.items():
            # Get latest position
            x, y, _, _, _ = features['history'][-1]
            node_idx = self.get_node_index(x, y)
            
            curr_density[node_idx] += 1.0
            vx, vy = features['motion_vector']
            curr_vx[node_idx] += vx
            curr_vy[node_idx] += vy
            
        # Average velocity
        mask = curr_density > 0
        curr_vx[mask] /= curr_density[mask]
        curr_vy[mask] /= curr_density[mask]
        
        # Calculate Pressure (density * speed)
        curr_speed = np.sqrt(curr_vx**2 + curr_vy**2)
        curr_pressure = curr_density * curr_speed
        
        # Combine into current feature set
        curr_features = np.stack([curr_density, curr_vx, curr_vy, curr_pressure], axis=1)
        
        # Exponential Moving Average for smoothing
        self.node_features = self.decay * self.node_features + (1 - self.decay) * curr_features
        
        # Convert to PyTorch Geometric Data object
        x = torch.tensor(self.node_features, dtype=torch.float32)
        return Data(x=x, edge_index=self.edge_index)

if __name__ == "__main__":
    builder = SpatialGraphBuilder(1920, 1080, 5, 0.9)
    print(f"Num Nodes: {builder.num_nodes}")
    print(f"Edge Index Shape: {builder.edge_index.shape}")
    
    # Simulate a single car pointing right
    trajs = {
        1: {
            "history": [(100, 100, 0, 0, 0)],
            "motion_vector": (5.0, 0.0)
        }
    }
    
    graph_data = builder.update_graph(trajs)
    print(f"Graph Features Shape: {graph_data.x.shape}")

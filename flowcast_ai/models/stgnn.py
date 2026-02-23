import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from typing import Tuple

class STGNNBlock(nn.Module):
    """
    Spatio-Temporal GNN Block.
    Combines Graph Convolution (spatial) with 1D Convolution/LSTM (temporal).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(STGNNBlock, self).__init__()
        # Spatial Graph Convolution
        self.gcn = GCNConv(in_channels, out_channels)
        # Temporal Convolution across the time dimension
        self.tconv = nn.Conv1d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            padding=1
        )
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, num_nodes, seq_len, in_channels]
        Because PyG GCN expects [num_nodes, in_channels], we reshape/process.
        For simplicity in this research-grade prototype, we'll assume batch_size=1
        and process the spatial features per timestep, then apply temporal conv.
        
        Args:
            x: [num_nodes, seq_len, in_channels]
            edge_index: [2, num_edges]
        """
        num_nodes, seq_len, in_channels = x.shape
        
        # 1. Spatial Processing (apply GCN to each time step independently)
        spatial_out = []
        for t in range(seq_len):
            x_t = x[:, t, :] # [num_nodes, in_channels]
            out_t = self.relu(self.gcn(x_t, edge_index)) # [num_nodes, out_channels]
            spatial_out.append(out_t)
            
        # Stack back to [num_nodes, seq_len, out_channels]
        x_spatial = torch.stack(spatial_out, dim=1) 
        
        # 2. Temporal Processing (1D Conv over sequences)
        # Conv1d expects [batch, channels, length], so we treat num_nodes as batch
        x_temp = x_spatial.permute(0, 2, 1) # [num_nodes, out_channels, seq_len]
        x_temp = self.relu(self.tconv(x_temp))
        
        # Return to [num_nodes, seq_len, out_channels]
        out = x_temp.permute(0, 2, 1)
        return out

class FlowForecastModel(nn.Module):
    """
    Predicts future Flow states (density, pressure, risk) 
    given historical spatial graphs.
    """
    def __init__(self, node_features: int = 4, hidden_dim: int = 32, horizon: int = 3):
        super(FlowForecastModel, self).__init__()
        self.horizon = horizon
        
        self.st_block1 = STGNNBlock(node_features, hidden_dim)
        self.st_block2 = STGNNBlock(hidden_dim, hidden_dim * 2)
        
        # Fully connected to predict future states
        # Input to FC is flattened seq_len * hidden_dim*2 per node
        # Output is horizon * node_features
        # We will use Global average pooling over time for simplicity, 
        # or just take the last time step representation.
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, horizon * node_features) # Predicts 4 features per horizon step
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, seq_len, node_features]
            edge_index: [2, num_edges]
            
        Returns:
            [num_nodes, horizon, node_features]
        """
        # Spatio-Temporal feature extraction
        out = self.st_block1(x, edge_index)
        out = self.st_block2(out, edge_index)
        
        # Take the feature representation of the last time step
        # out shape: [num_nodes, seq_len, hidden_dim * 2]
        last_step_features = out[:, -1, :] # [num_nodes, hidden_dim * 2]
        
        # Predict future steps
        predictions = self.fc(last_step_features) # [num_nodes, horizon * 4]
        
        # Reshape to [num_nodes, horizon, 4]
        num_nodes = x.shape[0]
        predictions = predictions.view(num_nodes, self.horizon, -1)
        
        # Apply softplus to ensure non-negative density and pressure predictions
        return torch.nn.functional.softplus(predictions)

if __name__ == "__main__":
    # Sanity Check
    print("Running ST-GNN Sanity Check...")
    
    num_nodes = 25
    seq_len = 10
    node_features = 4
    horizon = 3 # Forecast 3 steps ahead
    
    # Mock node history [num_nodes, seq_len, features]
    x = torch.rand((num_nodes, seq_len, node_features))
    
    # Mock edges (dummy grid)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    
    model = FlowForecastModel(node_features=node_features, hidden_dim=16, horizon=horizon)
    preds = model(x, edge_index)
    
    print(f"Input Shape: {x.shape}")
    print(f"Output Predictions Shape: {preds.shape}")
    print("Sanity Check Passed.")

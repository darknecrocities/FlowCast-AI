import argparse
import sys
import yaml
import cv2
import time
import torch
import numpy as np

# FlowCast AI Modules
from flowcast_ai.vision.detect import DetectionEngine
from flowcast_ai.vision.track import ByteTracker
from flowcast_ai.vision.trajectories import TrajectoryEngine
from flowcast_ai.graph.build_graph import SpatialGraphBuilder
from flowcast_ai.models.stgnn import FlowForecastModel
from flowcast_ai.simulation.density_forecast import FlowSimulationEngine

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        sys.exit(1)

def run_vision_pipeline(config, source):
    """
    Orchestrates the entire vision, tracking, graph, and rendering pipeline.
    """
    # 1. Initialization
    print("Initializing Vision Pipeline components...")
    
    # Get frame dimensions for simulator
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    has_frame, first_frame = cap.read()
    if not has_frame:
        print("Failed to read video source.")
        return
    
    h, w = first_frame.shape[:2]
    cap.release()
    
    device = config['system']['device']
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
    engine = DetectionEngine(
        model_id=config['vision']['model_id'],
        confidence=config['vision']['confidence'],
        classes=config['vision']['classes'],
        device=device
    )
    
    tracker = ByteTracker(track_buffer=config['tracking']['buffer_frames'])
    trajectories = TrajectoryEngine()
    graph_builder = SpatialGraphBuilder(w, h, grid_size=config['graph']['zone_grid_size'], decay=config['graph']['heatmap_decay'])
    
    # Mocking ST-GNN weights for prototype
    print("Initializing Forecasting Model (Prototype weights)...")
    stgnn = FlowForecastModel(horizon=3).to(device)
    stgnn.eval()
    
    simulator = FlowSimulationEngine(w, h, grid_size=config['graph']['zone_grid_size'])
    
    # Define class colors for boxes (B, G, R)
    COLOR_PALETTE = {
        0: (255, 128, 0),   # Person: Blue
        2: (0, 255, 255),   # Car: Yellow
        3: (0, 128, 255),   # Motorcycle: Orange
        5: (255, 0, 127),   # Bus: Purple
        6: (255, 255, 0),   # Train: Cyan
        7: (0, 0, 255)      # Truck: Red
    }
    CLASS_NAMES = {0: "Person", 2: "Car", 3: "Motorcycle", 5: "Bus", 6: "Train", 7: "Truck"}
    
    print(f"Starting inference on {source}...")
    
    # Global analytics for HUD
    start_time = time.time()
    
    for frame, detections in engine.detect_stream(source):
        loop_start = time.time()
        
        # --- PHASE 3: Tracking (Class Persistent) ---
        timestamp = time.time()
        tracked = tracker.update(detections, timestamp)
        
        # --- PHASE 4: Trajectories (Lightweight) ---
        features = trajectories.update(tracked)
        
        # --- DRAWING ENGINE (Flow & Group Augmented) ---
        group_centers = {"Person": [], "Vehicle": []}
        active_counts = {"Person": 0, "Vehicle": 0}
        group_bounds = {"Person": [float('inf'), float('inf'), 0, 0], 
                        "Vehicle": [float('inf'), float('inf'), 0, 0]}
        
        for t_obj in tracked:
            obj_id = t_obj['ID']
            det_class = t_obj['class_id']
            
            # Map Classes to Groups
            group_name = "Person" if det_class == 0 else "Vehicle"
            color = (255, 128, 0) if group_name == "Person" else (0, 255, 255)
            
            active_counts[group_name] += 1
            cx, cy = int(t_obj['x_center']), int(t_obj['y_center'])
            group_centers[group_name].append((cx, cy))
            
            # Update Group Boundaries
            x1, y1, x2, y2 = map(int, t_obj['bbox'])
            group_bounds[group_name][0] = min(group_bounds[group_name][0], x1)
            group_bounds[group_name][1] = min(group_bounds[group_name][1], y1)
            group_bounds[group_name][2] = max(group_bounds[group_name][2], x2)
            group_bounds[group_name][3] = max(group_bounds[group_name][3], y2)
            
            # Individual Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{group_name} {obj_id}", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw "Collective" Group Boxes & Labels
        for g_name, bounds in group_bounds.items():
            if active_counts[g_name] > 1:
                gx1, gy1, gx2, gy2 = map(int, bounds)
                g_color = (255, 128, 0) if g_name == "Person" else (0, 255, 255)
                
                # Draw the Master Box (Dashed style simulation)
                cv2.rectangle(frame, (gx1-10, gy1-10), (gx2+10, gy2+10), (255, 255, 255), 1)
                
                # Stylized "Only Text Box" Label
                label = f" {g_name.upper()}S : {active_counts[g_name]} "
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (gx1-10, gy1-th-25), (gx1-10+tw, gy1-10), g_color, -1)
                cv2.putText(frame, label, (gx1-10, gy1-17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Draw "Social Flow" Connections (Proximity-Based)
        for g_name, centers in group_centers.items():
            g_color = (255, 128, 0) if g_name == "Person" else (0, 255, 255)
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
                    if dist < 300:
                        cv2.line(frame, centers[i], centers[j], g_color, 1, cv2.LINE_AA)
                        mid_x, mid_y = (centers[i][0] + centers[j][0]) // 2, (centers[i][1] + centers[j][1]) // 2
                        cv2.putText(frame, f"{int(dist)}px", (mid_x, mid_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # --- REFINED HUD (Compiled Summary) ---
        inf_fps = 1.0 / (time.time() - loop_start)
        
        # Transparent Overlay for HUD
        cv2.rectangle(frame, (5, 5), (280, 100), (0, 0, 0), -1)
        
        cv2.putText(frame, f"FLOWCAST AI | {inf_fps:.1f} FPS", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"PEOPLE: {active_counts['Person']}", (15, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 1)
        cv2.putText(frame, f"VEHICLES: {active_counts['Vehicle']}", (15, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
        # Display Result
        cv2.imshow('FlowCast AI', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        print(f"Inference Running | FPS: {inf_fps:.1f} | Objects: {len(tracked)}")

def main():
    parser = argparse.ArgumentParser(description="FlowCast AI: Predictive Mobility Intelligence")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--source", type=str, help="Video source (e.g., path/to/video.mp4 or 0 for webcam)")
    parser.add_argument("--server", action="store_true", help="Start the FastAPI server")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    if args.server:
        print("Starting API Server...")
        import uvicorn
        uvicorn.run("flowcast_ai.api.server:app", host=config['api']['host'], port=config['api']['port'])
    elif args.source is not None:
        run_vision_pipeline(config, args.source)
    else:
        print("Error: Please specify a video source with --source or start the server with --server")
        parser.print_help()

if __name__ == "__main__":
    main()

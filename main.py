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
        0: (255, 128, 0),    # Person: Blue-ish
        1: (255, 178, 102),  # Bicycle
        2: (0, 255, 255),    # Car: Yellow
        3: (0, 128, 255),    # Motorcycle: Orange
        5: (255, 0, 127),    # Bus: Purple
        6: (255, 255, 0),    # Train: Cyan
        7: (0, 0, 255),      # Truck: Red
        9: (0, 255, 0),      # Traffic light (Stoplight): Green
        10: (127, 255, 0),   # Fire hydrant
        11: (255, 255, 255)  # Stop sign
    }
    CLASS_NAMES = {
        0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 
        5: "Bus", 6: "Train", 7: "Truck", 9: "Traffic Light",
        10: "Fire Hydrant", 11: "Stop Sign"
    }
    DEFAULT_COLOR = (200, 200, 200) # For others
    
    print(f"Starting inference on {source}...")
    
    # Global analytics for HUD
    start_time = time.time()
    
    # --- PHASE PRE-20: Predictive AI Buffers ---
    node_history = []
    max_history = 10
    forecast_results = None
    forecast_interval = 10 # Predict every 10 frames
    frame_count = 0
    
    for frame, detections in engine.detect_stream(source):
        loop_start = time.time()
        
        # --- PHASE 3: Tracking (Class Persistent) ---
        timestamp = time.time()
        tracked = tracker.update(detections, timestamp)
        
        # --- PHASE 4: Trajectories (Lightweight) ---
        features = trajectories.update(tracked)
        
        # Update Spatial Graph for Heatmap
        graph_builder.update_graph(features)
        
        # --- PHASE 5: Predictive AI (ST-GNN Deployment) ---
        frame_count += 1
        node_history.append(graph_builder.get_features())
        if len(node_history) > max_history:
            node_history.pop(0)
            
        if len(node_history) == max_history and frame_count % forecast_interval == 0:
            # Prepare input sequence: [num_nodes, seq_len, node_features]
            x_seq = torch.tensor(np.stack(node_history, axis=1), dtype=torch.float32).to(device)
            edge_index = graph_builder.edge_index.to(device)
            
            with torch.no_grad():
                forecast_results = stgnn(x_seq, edge_index) # [num_nodes, horizon, 4]
        
        # --- DRAWING ENGINE (Flow & Group Augmented) ---
        active_counts = {"Person": 0, "Vehicle": 0, "Infrastructure": 0}
        all_objects = []
        
        # 1. Draw Heatmap (Subtle Grid Overlay - Optimized)
        heatmap_overlay = frame.copy()
        draw_heatmap = False
        for node_idx in range(graph_builder.num_nodes):
            density = graph_builder.node_features[node_idx, 0]
            if density > 0.1:
                draw_heatmap = True
                gx = int((node_idx % graph_builder.grid_size) * graph_builder.cell_w)
                gy = int((node_idx // graph_builder.grid_size) * graph_builder.cell_h)
                cv2.rectangle(heatmap_overlay, (gx, gy), 
                             (int(gx + graph_builder.cell_w), int(gy + graph_builder.cell_h)), 
                             (0, 255, 0), -1)
        
        if draw_heatmap:
            cv2.addWeighted(heatmap_overlay, 0.15, frame, 0.85, 0, frame)
        
        for t_obj in tracked:
            obj_id = t_obj['ID']
            det_class = t_obj['class_id']
            conf = t_obj.get('confidence', 0.0)
            
            # Get trajectory intelligence
            feat = features.get(obj_id, {})
            history = feat.get("history", [])
            mv = feat.get("motion_vector", (0, 0))
            speed = feat.get("speed", 0.0)
            
            # Map Classes to Groups/Labels
            class_name = CLASS_NAMES.get(det_class, f"Obj_{det_class}")
            color = COLOR_PALETTE.get(det_class, DEFAULT_COLOR)
            
            # --- 1. Draw Historical Trail (Performance Optimized) ---
            if len(history) > 2:
                # Use polyline for single-call drawing
                trail_points = np.array([pt[:2] for pt in history[-20:]], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [trail_points], False, color, 1, cv2.LINE_AA)

            # Update Category Counts for HUD
            if det_class == 0:
                active_counts["Person"] += 1
            elif det_class in [1, 2, 3, 5, 6, 7]:
                active_counts["Vehicle"] += 1
            else:
                active_counts["Infrastructure"] = active_counts.get("Infrastructure", 0) + 1
            
            cx, cy = int(t_obj['x_center']), int(t_obj['y_center'])
            all_objects.append({'center': (cx, cy), 'color': color})
            
            # Individual Bounding Box
            x1, y1, x2, y2 = map(int, t_obj['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # --- 2. Advanced Label (Class | ID | Conf | Speed) ---
            label = f"{class_name.upper()} {obj_id} | {conf:.0%} | {int(speed)}px/s"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
                        
            # --- 3. Object Base Circle ---
            cv2.ellipse(frame, (cx, y2), (20, 10), 0, 0, 360, color, 1)

        # Draw Global Proximity Lines (Connect Everything)
        for i in range(len(all_objects)):
            for j in range(i + 1, len(all_objects)):
                p1 = all_objects[i]['center']
                p2 = all_objects[j]['center']
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                
                if dist < 400: # Threshold for distance lines
                    # Determine Status (Danger zone)
                    is_danger = dist < 120
                    line_color = (0, 0, 255) if is_danger else (150, 150, 150)
                    thickness = 2 if is_danger else 1
                    
                    cv2.line(frame, p1, p2, line_color, thickness, cv2.LINE_AA)
                    
                    mid_x, mid_y = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                    
                    if is_danger:
                        cv2.putText(frame, "COLLISION ALERT", (mid_x - 40, mid_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                    
                    # Display distance
                    cv2.putText(frame, f"{int(dist)}px", (mid_x, mid_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # --- PHASE 6: Predictive UI (Risk Zones) ---
        if forecast_results is not None:
            # 1. Draw Future Risk Zones in Main View (ST-GNN Derived)
            future_pressures = forecast_results[:, -1, 3].cpu().numpy()
            risk_zones = simulator.get_risk_zones(future_pressures, threshold=10.0)
            
            for zone in risk_zones:
                zx1, zy1, zx2, zy2 = map(int, zone['bbox'])
                # Pulse effect or subtle glow for risk zones
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 1)
                cv2.putText(frame, "!", (zx1 + 5, zy1 + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # --- REFINED HUD (Compiled Summary) ---
        inf_fps = 1.0 / (time.time() - loop_start)
        
        # Transparent Overlay for HUD
        cv2.rectangle(frame, (5, 5), (320, 130), (0, 0, 0), -1)
        
        cv2.putText(frame, f"FLOWCAST AI | {inf_fps:.1f} FPS", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        status_text = "AI FORECAST: ACTIVE" if forecast_results is not None else "AI FORECAST: BUFFERING..."
        cv2.putText(frame, status_text, (15, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(frame, f"PEOPLE: {active_counts['Person']}", (15, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 1)
        cv2.putText(frame, f"VEHICLES: {active_counts['Vehicle']}", (15, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"INFRASTRUCTURE: {active_counts.get('Infrastructure', 0)}", (15, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (127, 255, 0), 1)
            
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

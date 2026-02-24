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

def vision_pipeline_stream(config, source):
    """
    Generator that processes video and yields (frame, analytics).
    """
    # 1. Initialization
    print("Initializing Vision Pipeline components...")
    
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    has_frame, first_frame = cap.read()
    if not has_frame:
        print("Failed to read video source.")
        return
    
    h, w = first_frame.shape[:2]
    
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
    
    print("Initializing Forecasting Model (Prototype weights)...")
    stgnn = FlowForecastModel(horizon=3).to(device)
    stgnn.eval()
    
    simulator = FlowSimulationEngine(w, h, grid_size=config['graph']['zone_grid_size'])
    
    COLOR_PALETTE = {
        0: (255, 128, 0), 1: (255, 178, 102), 2: (0, 255, 255), 3: (0, 128, 255),
        5: (255, 0, 127), 6: (255, 255, 0), 7: (0, 0, 255), 9: (0, 255, 0),
        10: (127, 255, 0), 11: (255, 255, 255)
    }
    CLASS_NAMES = {
        0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 
        5: "Bus", 6: "Train", 7: "Truck", 9: "Traffic Light",
        10: "Fire Hydrant", 11: "Stop Sign"
    }
    DEFAULT_COLOR = (200, 200, 200)
    
    node_history = []
    max_history = 10
    forecast_results = None
    forecast_interval = 10
    frame_count = 0
    
    for frame, detections in engine.detect_stream(source):
        loop_start = time.time()
        timestamp = time.time()
        tracked = tracker.update(detections, timestamp)
        features = trajectories.update(tracked)
        graph_builder.update_graph(features)
        
        frame_count += 1
        node_history.append(graph_builder.get_features())
        if len(node_history) > max_history:
            node_history.pop(0)
            
        if len(node_history) == max_history and frame_count % forecast_interval == 0:
            x_seq = torch.tensor(np.stack(node_history, axis=1), dtype=torch.float32).to(device)
            edge_index = graph_builder.edge_index.to(device)
            with torch.no_grad():
                forecast_results = stgnn(x_seq, edge_index)
        
        active_counts = {"Person": 0, "Vehicle": 0, "Infrastructure": 0}
        all_objects = []
        
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
            obj_id, det_class = t_obj['ID'], t_obj['class_id']
            conf, feat = t_obj.get('confidence', 0.0), features.get(obj_id, {})
            history, speed = feat.get("history", []), feat.get("speed", 0.0)
            class_name = CLASS_NAMES.get(det_class, f"Obj_{det_class}")
            color = COLOR_PALETTE.get(det_class, DEFAULT_COLOR)
            
            if len(history) > 2:
                trail_points = np.array([pt[:2] for pt in history[-20:]], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [trail_points], False, color, 1, cv2.LINE_AA)

            if det_class == 0: active_counts["Person"] += 1
            elif det_class in [1, 2, 3, 5, 6, 7]: active_counts["Vehicle"] += 1
            else: active_counts["Infrastructure"] = active_counts.get("Infrastructure", 0) + 1
            
            cx, cy = int(t_obj['x_center']), int(t_obj['y_center'])
            all_objects.append({'center': (cx, cy), 'color': color})
            x1, y1, x2, y2 = map(int, t_obj['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name.upper()} {obj_id} | {conf:.0%} | {int(speed)}px/s"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            cv2.ellipse(frame, (cx, y2), (20, 10), 0, 0, 360, color, 1)

        for i in range(len(all_objects)):
            for j in range(i + 1, len(all_objects)):
                p1, p2 = all_objects[i]['center'], all_objects[j]['center']
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if dist < 400:
                    is_danger = dist < 120
                    cv2.line(frame, p1, p2, (0, 0, 255) if is_danger else (150, 150, 150), 2 if is_danger else 1, cv2.LINE_AA)
                    mid_x, mid_y = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                    if is_danger:
                        cv2.putText(frame, "COLLISION ALERT", (mid_x - 40, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                    cv2.putText(frame, f"{int(dist)}px", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        max_pressure = 0.0
        if forecast_results is not None:
            future_pressures = forecast_results[:, -1, 3].cpu().numpy()
            max_pressure = float(np.max(future_pressures))
            risk_zones = simulator.get_risk_zones(future_pressures, threshold=10.0)
            for zone in risk_zones:
                zx1, zy1, zx2, zy2 = map(int, zone['bbox'])
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 1)
                cv2.putText(frame, "!", (zx1 + 5, zy1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        inf_fps = 1.0 / (time.time() - loop_start)
        cv2.rectangle(frame, (5, 5), (320, 130), (0, 0, 0), -1)
        cv2.putText(frame, f"FLOWCAST AI | {inf_fps:.1f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        status_text = "AI FORECAST: ACTIVE" if forecast_results is not None else "AI FORECAST: BUFFERING..."
        cv2.putText(frame, status_text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"PEOPLE: {active_counts['Person']}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 1)
        cv2.putText(frame, f"VEHICLES: {active_counts['Vehicle']}", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"INFRASTRUCTURE: {active_counts.get('Infrastructure', 0)}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (127, 255, 0), 1)
        
        analytics = {
            "fps": inf_fps,
            "active_objects": len(tracked),
            "max_pressure": max_pressure,
            "counts": active_counts
        }
        
        yield frame, analytics

    cap.release()

def run_vision_pipeline(config, source):
    """
    Local runner that consumes the generator and displays outcomes in a window.
    """
    for frame, analytics in vision_pipeline_stream(config, source):
        cv2.imshow('FlowCast AI', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(f"Inference Running | FPS: {analytics['fps']:.1f} | Objects: {analytics['active_objects']}")

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

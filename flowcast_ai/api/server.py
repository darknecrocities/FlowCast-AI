import cv2
import asyncio
import json
import base64
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# In a real app, this would be imported from the main orchestration loop
# For this prototype, we'll demonstrate the streaming mechanics
import time
import numpy as np

app = FastAPI(title="FlowCast AI API")

# Mount UI static files
app.mount("/ui", StaticFiles(directory="flowcast_ai/ui"), name="ui")

# Global state for current analytics
current_analytics = {
    "fps": 0.0,
    "active_objects": 0,
    "max_pressure": 0.0,
    "predictions": []
}

class SystemConfig(BaseModel):
    horizon_minutes: int
    confidence: float

@app.get("/")
async def root():
    return HTMLResponse(content="<h1>FlowCast AI Backend Running.</h1><p>Visit /ui/dashboard.html</p>")

@app.post("/config")
async def update_config(config: SystemConfig):
    # In a real app, update the global/YAML config
    return {"status": "success", "new_config": config.dict()}

@app.get("/analytics")
async def get_analytics():
    return current_analytics

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Simulates the vision pipeline output for the dashboard via WebSockets.
    Sends base64 encoded JPEG frames + JSON analytics.
    """
    await websocket.accept()
    
    # We would normally connect this to the actual `run_vision_pipeline` yielding frames
    # Here, we generate a synthetic feed to demonstrate the WebSocket architecture
    
    cap = cv2.VideoCapture(0) # Fallback to local webcam for demo
    if not cap.isOpened():
        print("Camera not found, using blank frames.")
        
    try:
        while True:
            ret, frame = cap.read() if cap.isOpened() else (True, np.zeros((480, 640, 3), dtype=np.uint8))
            
            # Simulate processing delay
            await asyncio.sleep(0.05)
            
            # --- Synthetic Overlay for Demo Server ---
            cv2.putText(frame, "FlowCast AI Live Demo", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64_img = base64.b64encode(buffer).decode('utf-8')
            
            # Update mocking analytics
            global current_analytics
            current_analytics["fps"] = 20.0 + np.random.randn()
            current_analytics["active_objects"] = int(5 + np.random.rand() * 10)
            current_analytics["max_pressure"] = float(15.0 + np.random.randn() * 2)
            
            # Payload
            payload = {
                "image": f"data:image/jpeg;base64,{b64_img}",
                "analytics": current_analytics
            }
            
            await websocket.send_text(json.dumps(payload))
            
    except WebSocketDisconnect:
        print("Client disconnected.")
    finally:
        if cap.isOpened():
            cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

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

from main import vision_pipeline_stream, load_config
import threading
from queue import Queue

# Initialize pipeline resources
config = load_config()
source = config.get("vision", {}).get("source", "sample.mp4")

# Thread-safe queue for frame delivery
frame_queue = Queue(maxsize=10)

def run_pipeline_thread():
    """Background thread to run the vision pipeline and populate the queue."""
    for frame, analytics in vision_pipeline_stream(config, source):
        if frame_queue.full():
            frame_queue.get() # Drop old frames if client is slow
        frame_queue.put((frame, analytics))

# Start pipeline thread
threading.Thread(target=run_pipeline_thread, daemon=True).start()

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Broadcasts real-time frames and analytics from the vision pipeline.
    """
    await websocket.accept()
    print("Client connected to stream.")
    
    try:
        while True:
            # Check for new data from the pipeline thread
            if not frame_queue.empty():
                frame, analytics = frame_queue.get()
                
                # Update global analytics state
                global current_analytics
                current_analytics.update(analytics)
                
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64_img = base64.b64encode(buffer).decode('utf-8')
                
                payload = {
                    "image": f"data:image/jpeg;base64,{b64_img}",
                    "analytics": current_analytics
                }
                
                await websocket.send_text(json.dumps(payload))
            
            await asyncio.sleep(0.01) # High frequency polling
            
    except WebSocketDisconnect:
        print("Client disconnected.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

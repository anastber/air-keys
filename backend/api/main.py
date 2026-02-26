"""
AirKeys FastAPI application main module.

Provides WebSocket endpoints for real-time hand landmark streaming.
"""

import asyncio
import json
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.gesture.landmark_extractor import (
    LandmarkExtractor,
    WebcamCapture,
    format_landmarks_for_json,
)

# Create FastAPI app
app = FastAPI(
    title="AirKeys API",
    description="Camera-based gesture instrument API",
    version="0.1.0",
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    """Manages WebSocket connections for landmark streaming."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and store a WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_to_connection(self, data: dict[str, Any], websocket: WebSocket):
        """Send data to a specific WebSocket connection."""
        try:
            await websocket.send_text(json.dumps(data))
        except Exception:
            # Connection might be closed, remove it
            self.disconnect(websocket)


# Connection manager instance
manager = ConnectionManager()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "AirKeys API is running", "status": "healthy"}


@app.websocket("/ws/landmarks")
async def websocket_landmarks(websocket: WebSocket):
    """
    WebSocket endpoint for real-time hand landmark streaming.
    
    Streams normalized hand landmark coordinates as JSON at ~30fps.
    Each message contains:
    - timestamp: Unix timestamp of when landmarks were captured
    - hands: List of detected hands, each with:
      - handedness: 'Left' or 'Right'
      - landmarks: List of 21 landmarks with x, y, z coordinates
    """
    await manager.connect(websocket)

    # Initialize landmark extractor and webcam capture with lower thresholds
    extractor = LandmarkExtractor(
        static_image_mode=False,
        max_num_hands=2,  # Allow both hands
        min_detection_confidence=0.3,  # Lower threshold for easier detection
        min_tracking_confidence=0.3,   # Lower threshold for easier tracking
    )

    webcam = WebcamCapture(camera_index=0)

    try:
        # Start webcam capture
        if not webcam.start_capture():
            await websocket.send_text(json.dumps({
                'error': 'Failed to initialize webcam',
                'timestamp': time.time()
            }))
            return

        # Target 30 FPS
        target_fps = 30
        frame_duration = 1.0 / target_fps

        while True:
            start_time = time.time()

            # Capture frame from webcam
            frame = webcam.get_frame()
            if frame is None:
                await websocket.send_text(json.dumps({
                    'error': 'Failed to capture frame',
                    'timestamp': time.time()
                }))
                await asyncio.sleep(0.1)
                continue

            # Extract landmarks
            landmarks_data = extractor.extract_landmarks(frame)

            # Format for JSON transmission
            json_data = format_landmarks_for_json(landmarks_data or [])

            # Send to client
            await manager.send_to_connection(json_data, websocket)

            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_duration - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        # Send error to client before closing
        try:
            await websocket.send_text(json.dumps({
                'error': f'Server error: {str(e)}',
                'timestamp': time.time()
            }))
        except:
            pass
        manager.disconnect(websocket)
    finally:
        # Clean up resources
        webcam.stop_capture()
        extractor.close()


@app.get("/api/health")
async def health_check():
    """Detailed health check for monitoring."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "0.1.0",
        "services": {
            "mediapipe": "ready",
            "webcam": "ready"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

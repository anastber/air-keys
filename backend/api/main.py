"""
AirKeys FastAPI application main module.

Provides WebSocket endpoints for real-time hand landmark streaming.
"""

import asyncio
import json
import time
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.gesture.landmark_extractor import (
    LandmarkExtractor,
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
    WebSocket endpoint for real-time hand landmark extraction.

    The client owns the camera (browser getUserMedia) and pushes JPEG-encoded
    frames as binary messages. For each frame received, responds with the
    extracted landmarks as JSON:
    - timestamp: Unix timestamp of when landmarks were extracted
    - hands: List of detected hands, each with:
      - handedness: 'Left' or 'Right'
      - landmarks: List of 21 landmarks with x, y, z coordinates
    """
    await manager.connect(websocket)

    # Initialize landmark extractor with lower thresholds
    extractor = LandmarkExtractor(
        static_image_mode=False,
        max_num_hands=2,  # Allow both hands
        min_detection_confidence=0.3,  # Lower threshold for easier detection
        min_tracking_confidence=0.3,  # Lower threshold for easier tracking
    )

    try:
        while True:
            # Receive a JPEG frame captured client-side
            frame_bytes = await websocket.receive_bytes()

            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Extract landmarks off the event loop (MediaPipe call is blocking)
            landmarks_data = await asyncio.to_thread(extractor.extract_landmarks, frame)

            # Format for JSON transmission
            json_data = format_landmarks_for_json(landmarks_data or [])

            # Send to client
            await manager.send_to_connection(json_data, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        # Send error to client before closing
        try:
            await websocket.send_text(
                json.dumps(
                    {"error": f"Server error: {str(e)}", "timestamp": time.time()}
                )
            )
        except Exception:
            pass
        manager.disconnect(websocket)
    finally:
        # Clean up resources
        extractor.close()


@app.get("/api/health")
async def health_check():
    """Detailed health check for monitoring."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "0.1.0",
        "services": {"mediapipe": "ready"},
    }


if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )

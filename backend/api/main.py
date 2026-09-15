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
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.gesture import dataset
from backend.gesture.classifier import get_classifier
from backend.gesture.labels import GESTURE_LABELS
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


class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float


class GestureSample(BaseModel):
    label: str
    landmarks: list[LandmarkPoint]
    handedness: str = "Right"


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "AirKeys API is running", "status": "healthy"}


@app.get("/api/gestures/status")
async def gesture_status():
    """Recorded sample counts per gesture and whether a model is trained."""
    classifier = get_classifier()
    return {
        "labels": GESTURE_LABELS,
        "sample_counts": dataset.sample_counts(),
        "model_trained": classifier.is_trained,
        "trained_at": classifier.trained_at,
        "accuracy": classifier.accuracy,
    }


@app.post("/api/gestures/samples")
async def add_gesture_sample(sample: GestureSample):
    """Record one labeled gesture sample for later training."""
    if sample.label not in GESTURE_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown gesture label '{sample.label}'. Must be one of: "
            f"{', '.join(GESTURE_LABELS)}.",
        )
    if len(sample.landmarks) != 21:
        raise HTTPException(
            status_code=400, detail="Expected exactly 21 landmarks per hand."
        )

    dataset.append_sample(
        sample.label,
        [p.model_dump() for p in sample.landmarks],
        sample.handedness,
    )
    return {"sample_counts": dataset.sample_counts()}


@app.post("/api/gestures/train")
async def train_gesture_classifier():
    """Retrain the classifier on every sample recorded so far."""
    samples = dataset.load_samples()
    classifier = get_classifier()

    try:
        # Runs on a thread since sklearn's fit() is blocking (it's fast for
        # this data size, but this keeps the event loop responsive regardless).
        result = await asyncio.to_thread(classifier.train, samples)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    classifier.save()
    return result


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
      - gesture, confidence: classified pose (see backend.gesture.classifier),
        present only once a model has been trained via /api/gestures/train
    """
    await manager.connect(websocket)

    # Initialize landmark extractor with lower thresholds
    extractor = LandmarkExtractor(
        static_image_mode=False,
        max_num_hands=2,  # Allow both hands
        min_detection_confidence=0.3,  # Lower threshold for easier detection
        min_tracking_confidence=0.3,  # Lower threshold for easier tracking
    )
    classifier = get_classifier()

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

            # Classify each detected hand's pose. Cheap (a few hundred
            # microseconds per hand on a random forest) so it stays inline
            # rather than another to_thread hop.
            if landmarks_data and classifier.is_trained:
                for hand_data in landmarks_data:
                    prediction = classifier.predict(
                        hand_data["landmarks"], hand_data["handedness"]
                    )
                    if prediction:
                        hand_data["gesture"] = prediction["gesture"]
                        hand_data["confidence"] = prediction["confidence"]

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

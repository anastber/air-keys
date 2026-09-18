"""
AirKeys FastAPI application main module.

Hand tracking, gesture recognition (both the pretrained base set and a
visitor's own taught gestures), and audio all run client-side now — see
frontend/src/lib/gestureRecognition.ts and customGestures.ts, and the git
history for the earlier server-side classifier this replaced (and why: a
single shared model/dataset doesn't work once more than one visitor can hit
the app at once — see README's case study section).

This backend is intentionally a stub for now — no gesture or audio logic
runs here. What it's used for next hasn't been decided yet.
"""

import time

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AirKeys API",
    description="Camera-based gesture instrument API",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "AirKeys API is running", "status": "healthy"}


@app.get("/api/health")
async def health_check():
    """Detailed health check for monitoring."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "0.2.0",
    }


if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )

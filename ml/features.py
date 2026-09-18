"""Landmark normalization — must stay numerically identical to the
TypeScript port at frontend/src/lib/customGestures.ts::normalizeLandmarks.

Both the offline-trained classifier here and the client-side kNN classifier
derive from the same original technique (see git history: the deleted
backend/gesture/features.py). Keeping this Python copy and the TS copy in
lockstep is what lets a model trained here run correctly against landmarks
produced by the browser's MediaPipe pipeline at inference time.
"""

import numpy as np

WRIST = 0
MIDDLE_FINGER_MCP = 9


def normalize_landmarks(
    landmarks: list[dict[str, float]], handedness: str
) -> np.ndarray:
    """Translation/scale/mirror-invariant feature vector for one hand.

    landmarks: 21 dicts of {"x", "y", "z"}, in MediaPipe's normalized
    image-relative coordinates (matches frontend/src/lib/types.ts::Landmark).
    Returns a flat (63,) float32 vector: 21 landmarks x (x, y, z).
    """
    points = np.array([[p["x"], p["y"], p["z"]] for p in landmarks], dtype=np.float32)
    origin = points[WRIST].copy()
    centered = points - origin

    scale = max(float(np.hypot(*centered[MIDDLE_FINGER_MCP][:2])), 1e-6)
    mirror = -1.0 if handedness == "Left" else 1.0

    features = centered.copy()
    features[:, 0] = (mirror * features[:, 0]) / scale
    features[:, 1] = features[:, 1] / scale
    features[:, 2] = features[:, 2] / scale

    return features.flatten()

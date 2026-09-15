"""Turn raw MediaPipe hand landmarks into a pose classifier's input features.

Raw landmarks are normalized coordinates in [0, 1] relative to the camera
frame, which means the same physical gesture produces completely different
numbers depending on where the hand is in frame and how far it is from the
camera. A classifier trained on raw landmarks would just be learning "where
was the hand", not "what pose is the hand making".

This module removes that: it re-centers on the wrist, scales by hand size,
and mirrors left hands so the feature space is the same regardless of which
hand or where it is.
"""

import numpy as np

# MediaPipe hand landmark indices used as normalization anchors.
WRIST = 0
MIDDLE_FINGER_MCP = 9


def normalize_landmarks(
    landmarks: list[dict[str, float]], handedness: str = "Right"
) -> np.ndarray:
    """
    Convert 21 raw (x, y, z) landmarks into a translation/scale/mirror
    invariant feature vector.

    Args:
        landmarks: 21 dicts with 'x', 'y', 'z' keys, as produced by
            LandmarkExtractor (x, y normalized to [0, 1] by MediaPipe,
            z is relative depth).
        handedness: 'Left' or 'Right'. Left hands are mirrored on x so a
            gesture looks the same to the classifier regardless of hand.

    Returns:
        A flat (63,) float32 array: 21 landmarks x (x, y, z).
    """
    points = np.array([[p["x"], p["y"], p["z"]] for p in landmarks], dtype=np.float32)

    # Center on the wrist so position in frame doesn't matter.
    origin = points[WRIST].copy()
    points -= origin

    # Scale by wrist -> middle-MCP distance so distance from camera / hand
    # size doesn't matter. Guard against a degenerate (zero-length) case.
    scale = float(np.linalg.norm(points[MIDDLE_FINGER_MCP][:2]))
    if scale < 1e-6:
        scale = 1e-6
    points /= scale

    # Mirror left hands on x so "fist" looks the same regardless of hand.
    if handedness == "Left":
        points[:, 0] *= -1

    return points.flatten()

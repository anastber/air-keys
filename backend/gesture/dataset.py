"""On-disk storage for labeled gesture samples.

Samples are appended as they're recorded (one JSON object per line) so a
training session never loses data to a crash, and so the raw landmarks stay
around if the normalization or feature set ever changes later.
"""

import json
import os
import time
from typing import Any

from backend.gesture.labels import GESTURE_LABELS

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(_REPO_ROOT, "ml", "data", "gestures.jsonl")


def append_sample(
    label: str, landmarks: list[dict[str, float]], handedness: str = "Right"
) -> None:
    """Append one labeled gesture sample to the dataset file."""
    if label not in GESTURE_LABELS:
        raise ValueError(f"Unknown gesture label: {label!r}")

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    record = {
        "label": label,
        "landmarks": landmarks,
        "handedness": handedness,
        "timestamp": time.time(),
    }
    with open(DATA_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_samples() -> list[dict[str, Any]]:
    """Load every recorded sample. Returns [] if nothing has been recorded yet."""
    if not os.path.exists(DATA_PATH):
        return []

    samples = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def sample_counts() -> dict[str, int]:
    """Number of recorded samples per gesture label (0 for labels with none)."""
    counts = dict.fromkeys(GESTURE_LABELS, 0)
    for sample in load_samples():
        if sample["label"] in counts:
            counts[sample["label"]] += 1
    return counts


def clear_samples() -> None:
    """Delete all recorded samples, e.g. to start a fresh training session."""
    if os.path.exists(DATA_PATH):
        os.remove(DATA_PATH)

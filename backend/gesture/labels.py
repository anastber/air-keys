"""Canonical set of gesture classes the classifier recognizes.

This is the single source of truth for gesture labels: the training UI
offers exactly these, samples are validated against this list, and the
classifier's output classes always come from it.
"""

GESTURE_LABELS: list[str] = [
    "fist",
    "open_palm",
    "pinch",
    "point",
    "peace",
    "thumbs_up",
]

# Minimum number of samples per class required before training is allowed.
# Below this, accuracy numbers are too noisy to mean anything.
MIN_SAMPLES_PER_CLASS = 8

"""Gesture pose classifier.

A random forest over the normalized landmark features. Random forests were
picked deliberately over a neural net here: training sets are tiny (tens of
samples per class, collected live), need to train in well under a second on
CPU with no tuning, and don't need feature scaling — all of which a forest
gives for free where an MLP would need careful handling to not be flaky live.
"""

import os
import time
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from backend.gesture.features import normalize_landmarks
from backend.gesture.labels import GESTURE_LABELS, MIN_SAMPLES_PER_CLASS

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(_REPO_ROOT, "ml", "experiments", "gesture_classifier.joblib")

# Below this confidence, treat the prediction as "no gesture recognized"
# rather than acting on a guess.
CONFIDENCE_THRESHOLD = 0.6


class GestureClassifier:
    """Wraps a trained sklearn model with normalization, save/load, and a
    confidence-gated predict() suitable for calling once per video frame."""

    def __init__(self) -> None:
        self.model: RandomForestClassifier | None = None
        self.trained_at: float | None = None
        self.accuracy: float | None = None

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def train(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Train (or retrain) on a full set of raw samples (as returned by
        dataset.load_samples()).

        Returns a summary dict with accuracy, a confusion matrix, and
        per-class sample counts, meant to be shown live right after training.
        """
        counts: dict[str, int] = dict.fromkeys(GESTURE_LABELS, 0)
        for s in samples:
            if s["label"] in counts:
                counts[s["label"]] += 1

        present_classes = [label for label in GESTURE_LABELS if counts[label] > 0]
        if len(present_classes) < 2:
            raise ValueError("Need samples for at least 2 different gestures to train.")
        under_min = [
            label for label in present_classes if counts[label] < MIN_SAMPLES_PER_CLASS
        ]
        if under_min:
            raise ValueError(
                f"Need at least {MIN_SAMPLES_PER_CLASS} samples for: "
                f"{', '.join(under_min)} (have "
                f"{', '.join(f'{c}={counts[c]}' for c in under_min)})."
            )

        X = np.stack(
            [
                normalize_landmarks(s["landmarks"], s.get("handedness", "Right"))
                for s in samples
                if s["label"] in counts
            ]
        )
        y = np.array([s["label"] for s in samples if s["label"] in counts])

        # Held-out split for an honest accuracy number, stratified so every
        # class is represented on both sides.
        min_class_count = min(counts[label] for label in present_classes)
        can_split = min_class_count >= 2
        if can_split:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=42
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = X, y  # falls back to train accuracy, not held-out

        model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = float((y_pred == y_test).mean())
        labels_in_data = sorted(present_classes)
        cm = confusion_matrix(y_test, y_pred, labels=labels_in_data)

        # Refit on everything for the model we actually deploy, so no
        # collected sample goes to waste.
        model.fit(X, y)

        self.model = model
        self.trained_at = time.time()
        self.accuracy = accuracy

        return {
            "accuracy": accuracy,
            "held_out_evaluation": can_split,
            "sample_counts": counts,
            "confusion_matrix": {
                "labels": labels_in_data,
                "matrix": cm.tolist(),
            },
            "trained_at": self.trained_at,
        }

    def predict(
        self, landmarks: list[dict[str, float]], handedness: str = "Right"
    ) -> dict[str, Any] | None:
        """
        Classify one hand's landmarks.

        Returns {"gesture": str, "confidence": float} or None if no model
        is trained yet, or the top prediction is below CONFIDENCE_THRESHOLD.
        """
        if self.model is None:
            return None

        features = normalize_landmarks(landmarks, handedness).reshape(1, -1)
        probs = self.model.predict_proba(features)[0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])

        if confidence < CONFIDENCE_THRESHOLD:
            return None

        return {
            "gesture": str(self.model.classes_[best_idx]),
            "confidence": confidence,
        }

    def save(self, path: str = MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "trained_at": self.trained_at,
                "accuracy": self.accuracy,
            },
            path,
        )

    def load(self, path: str = MODEL_PATH) -> bool:
        """Load a previously saved model. Returns False if none exists yet."""
        if not os.path.exists(path):
            return False
        state = joblib.load(path)
        self.model = state["model"]
        self.trained_at = state["trained_at"]
        self.accuracy = state["accuracy"]
        return True


_classifier: GestureClassifier | None = None


def get_classifier() -> GestureClassifier:
    """Process-wide classifier singleton, lazily loading a saved model on
    first use so a trained gesture set survives a backend restart."""
    global _classifier
    if _classifier is None:
        _classifier = GestureClassifier()
        _classifier.load()
    return _classifier

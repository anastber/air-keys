"""Candidate classifiers, and the deployed MLP as the app runs it.

Every candidate is a scikit-learn estimator built by a factory in MODELS,
so evaluation protocols can train any of them the same way. The deployed
model is different: it is never retrained here, but reloaded from its
exported weights and run exactly as frontend/src/lib/trainedGestures.ts
does (same forward pass, same confidence threshold, same background class).
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOYED_WEIGHTS_PATH = REPO_ROOT / "ml" / "experiments" / "gesture_mlp_weights.json"
BACKGROUND_LABEL = "no_gesture"
RANDOM_STATE = 42

MODELS: dict[str, Callable[[], BaseEstimator]] = {
    # Linear baseline.
    "logreg": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    ),
    # Same family as the app's personal-gesture classifier
    # (lib/customGestures.ts): Euclidean distance on normalized landmarks.
    "knn": lambda: KNeighborsClassifier(n_neighbors=5),
    # Same settings as the original baseline in ml/train.py.
    "rf": lambda: RandomForestClassifier(
        n_estimators=150, max_depth=10, random_state=RANDOM_STATE
    ),
    # Same architecture as the deployed model, so it can ship the same way.
    "mlp": lambda: MLPClassifier(
        hidden_layer_sizes=(32, 16), max_iter=2000, random_state=RANDOM_STATE
    ),
}


class DeployedMLP(ClassifierMixin, BaseEstimator):
    """The shipped MLP, reloaded from its exported weights JSON.

    Predicts BACKGROUND_LABEL whenever the app would show no gesture: the
    winning class is the background class, or its probability is below the
    exported confidence threshold.
    """

    def __init__(self, weights_path: Path = DEPLOYED_WEIGHTS_PATH):
        self.weights_path = weights_path
        payload = json.loads(Path(weights_path).read_text())
        self.classes_ = np.array(payload["classLabels"])
        self.confidence_threshold = payload["confidenceThreshold"]
        self._layers = [
            (np.array(layer["weights"]), np.array(layer["biases"]))
            for layer in payload["layers"]
        ]

    def fit(self, x, y):  # noqa: ARG002 — the deployed model is never refit
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        activations = np.asarray(x, dtype=np.float64)
        for i, (weights, biases) in enumerate(self._layers):
            activations = activations @ weights + biases
            if i < len(self._layers) - 1:
                activations = np.maximum(activations, 0.0)
        exps = np.exp(activations - activations.max(axis=1, keepdims=True))
        return exps / exps.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)
        best = probs.argmax(axis=1)
        labels = self.classes_[best].astype(object)
        below = probs[np.arange(len(best)), best] < self.confidence_threshold
        labels[below] = BACKGROUND_LABEL
        return labels.astype(str)


def build_model(name: str) -> BaseEstimator:
    if name == "deployed":
        return DeployedMLP()
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {available_models()}")
    return MODELS[name]()


def available_models() -> list[str]:
    return [*MODELS, "deployed"]

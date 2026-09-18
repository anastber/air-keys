"""Offline training + evaluation for the self-trained gesture classifier.

Trains two models on the same normalized features for comparison:
  - RandomForestClassifier: evaluation baseline / sanity check.
  - MLPClassifier: the deployed model — small enough that its weight
    matrices export directly to a hand-rolled JS forward pass (no ONNX/
    TF.js runtime needed client-side).

Input:  ml/data/gestures_v1.json — a JSON array of
        {"label": str, "landmarks": [{"x","y","z"} x 21],
         "handedness": str, "timestamp": float}
        exported by frontend/src/components/DatasetCollector.tsx.

Output (ml/experiments/):
  - confusion_matrix.png     confusion matrix plot for the deployed MLP
  - gesture_classifier.joblib  full sklearn MLP, for reproducibility
  - gesture_mlp_weights.json   weights/biases/labels/threshold for the
                                frontend's hand-rolled forward pass
                                (also copied to frontend/public/models/)

Usage (from repo root): uv run python -m ml.train
"""

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.neural_network import MLPClassifier

from ml.features import normalize_landmarks

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "ml" / "data" / "gestures_v1.json"
EXPERIMENTS_DIR = REPO_ROOT / "ml" / "experiments"
FRONTEND_MODELS_DIR = REPO_ROOT / "frontend" / "public" / "models"

CONFIDENCE_THRESHOLD = 0.6
RANDOM_STATE = 42

# DatasetCollector.tsx captures ~20 near-identical frames per Record click
# (one "recording burst"). A plain random train/test split would scatter
# frames from the same burst across both sides, so the model would partly
# be tested on near-duplicates of its own training data — inflating
# accuracy without proving generalization. Instead we group samples by
# burst (same label, no >2s gap from the previous sample) and split whole
# bursts between train/test, never splitting one burst across both.
BURST_GAP_SECONDS = 2.0


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def featurize(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = np.array(
        [normalize_landmarks(r["landmarks"], r["handedness"]) for r in records],
        dtype=np.float32,
    )
    labels = np.array([r["label"] for r in records])
    return features, labels


def assign_recording_bursts(records: list[dict]) -> np.ndarray:
    groups = np.empty(len(records), dtype=np.int64)
    group_id = -1
    prev_label = None
    prev_ts = None
    for i, r in enumerate(records):
        if (
            prev_ts is None
            or r["label"] != prev_label
            or r["timestamp"] - prev_ts > BURST_GAP_SECONDS
        ):
            group_id += 1
        groups[i] = group_id
        prev_label = r["label"]
        prev_ts = r["timestamp"]
    return groups


def group_stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for label in sorted(set(y)):
        label_indices = np.where(y == label)[0]
        label_groups = groups[label_indices]
        unique_groups = np.unique(label_groups)
        rng.shuffle(unique_groups)
        n_test_groups = max(1, round(len(unique_groups) * test_size))
        test_groups = set(unique_groups[:n_test_groups])
        for idx, g in zip(label_indices, label_groups, strict=True):
            (test_idx if g in test_groups else train_idx).append(idx)
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def evaluate(name: str, model, x_test: np.ndarray, y_test: np.ndarray) -> None:
    preds = model.predict(x_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"\n=== {name} ===")
    print(f"Held-out accuracy: {accuracy:.3f}")
    print(classification_report(y_test, preds, zero_division=0))


def export_mlp_weights(
    mlp: MLPClassifier, class_labels: list[str], out_path: Path
) -> None:
    payload = {
        "classLabels": class_labels,
        "confidenceThreshold": CONFIDENCE_THRESHOLD,
        "layers": [
            {"weights": w.tolist(), "biases": b.tolist()}
            for w, b in zip(mlp.coefs_, mlp.intercepts_, strict=True)
        ],
    }
    out_path.write_text(json.dumps(payload))


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(
            f"No dataset at {DATA_PATH}. Export one from DatasetCollector.tsx first."
        )

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records(DATA_PATH)
    x, y = featurize(records)
    groups = assign_recording_bursts(records)
    print(f"Loaded {len(y)} samples across {len(set(y))} classes: {sorted(set(y))}")
    print(
        f"Grouped into {len(set(groups))} recording bursts (held out by burst, not by frame)"
    )

    x_train, x_test, y_train, y_test = group_stratified_split(
        x, y, groups, test_size=0.25, random_state=RANDOM_STATE
    )

    rf = RandomForestClassifier(
        n_estimators=150, max_depth=10, random_state=RANDOM_STATE
    )
    rf.fit(x_train, y_train)
    evaluate("RandomForest (baseline)", rf, x_test, y_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    mlp.fit(x_train, y_train)
    evaluate("MLP (deployed model)", mlp, x_test, y_test)

    ConfusionMatrixDisplay.from_estimator(mlp, x_test, y_test, xticks_rotation=45)
    cm_path = EXPERIMENTS_DIR / "confusion_matrix.png"
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(cm_path)
    print(f"\nSaved confusion matrix to {cm_path}")

    # Refit the deployed model on all data before export/deployment.
    mlp.fit(x, y)
    joblib.dump(mlp, EXPERIMENTS_DIR / "gesture_classifier.joblib")

    weights_path = EXPERIMENTS_DIR / "gesture_mlp_weights.json"
    export_mlp_weights(mlp, list(mlp.classes_), weights_path)
    print(f"Saved MLP weights to {weights_path}")

    FRONTEND_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(weights_path, FRONTEND_MODELS_DIR / "gesture_mlp_weights.json")
    print(f"Copied weights to {FRONTEND_MODELS_DIR / 'gesture_mlp_weights.json'}")


if __name__ == "__main__":
    main()

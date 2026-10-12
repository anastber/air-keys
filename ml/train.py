"""Offline training + evaluation for the self-trained gesture classifier.

Trains two models on the same normalized features for comparison:
  - RandomForestClassifier: evaluation baseline / sanity check.
  - MLPClassifier: the deployed model — small enough that its weight
    matrices export directly to a hand-rolled JS forward pass (no ONNX/
    TF.js runtime needed client-side).

Input:  ml/data/gestures_v1.json, exported by
        frontend/src/components/DatasetCollector.tsx (record format and
        burst-grouped splitting: see ml/data.py).

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from ml.data import (
    assign_recording_bursts,
    featurize,
    group_stratified_split,
    load_records,
)
from ml.evaluate import evaluate, save_confusion_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "ml" / "data" / "gestures_v1.json"
EXPERIMENTS_DIR = REPO_ROOT / "ml" / "experiments"
FRONTEND_MODELS_DIR = REPO_ROOT / "frontend" / "public" / "models"

CONFIDENCE_THRESHOLD = 0.6
RANDOM_STATE = 42


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

    cm_path = EXPERIMENTS_DIR / "confusion_matrix.png"
    save_confusion_matrix(mlp, x_test, y_test, cm_path)
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

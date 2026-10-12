"""Evaluation metrics and plots for gesture classifiers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)


def evaluate(name: str, model, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Print a held-out report for `model` and return its headline metrics."""
    preds = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro", zero_division=0),
    }
    print(f"\n=== {name} ===")
    print(f"Held-out accuracy: {metrics['accuracy']:.3f}")
    print(f"Held-out macro-F1: {metrics['macro_f1']:.3f}")
    print(classification_report(y_test, preds, zero_division=0))
    return metrics


def save_confusion_matrix(
    model, x_test: np.ndarray, y_test: np.ndarray, out_path: Path
) -> None:
    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

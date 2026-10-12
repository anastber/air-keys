"""Evaluation metrics and plots for gesture classifiers."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no GUI backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

# Silence matplotlib's font-fallback chatter in CLI output.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


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


# ---------------------------------------------------------------------------
# Evaluation-harness metrics (ml/eval.py)
# ---------------------------------------------------------------------------


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> np.ndarray:
    """Confusion counts over `labels`, plus a last column for predictions outside them."""
    index = {label: i for i, label in enumerate(labels)}
    k = len(labels)
    true_idx = np.array([index[y] for y in y_true])
    pred_idx = np.array([index.get(y, k) for y in y_pred])
    return np.bincount(true_idx * (k + 1) + pred_idx, minlength=k * (k + 1)).reshape(
        k, k + 1
    )


def _scores(cm: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Accuracy, macro-F1, and per-class precision/recall/F1 from a confusion matrix."""
    k = cm.shape[0]
    tp = np.diag(cm[:, :k]).astype(float)
    predicted = cm[:, :k].sum(axis=0)
    actual = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actual > 0, tp / actual, 0.0)
        denom = 2 * tp + (predicted - tp) + (actual - tp)
        f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
    present = actual > 0
    accuracy = tp.sum() / max(actual.sum(), 1)
    return accuracy, float(f1[present].mean()), precision, recall, f1


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    """Headline and per-class metrics; macro-F1 averages the classes present in y_true."""
    cm = _confusion(y_true, y_pred, labels)
    accuracy, macro_f1, precision, recall, f1 = _scores(cm)
    return {
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(cm[i].sum()),
            }
            for i, label in enumerate(labels)
        },
        "labels": labels,
        # Rows: true label; columns: predicted label, then "other" (a
        # prediction outside `labels`).
        "confusion_matrix": cm.tolist(),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    labels: list[str],
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, list[float]]:
    """Percentile bootstrap CIs for accuracy and macro-F1, resampling *groups*.

    Samples from one subject (or recording burst) are correlated, so
    resampling individual samples would understate the uncertainty; whole
    groups are resampled with replacement instead.
    """
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(unique))]
    accuracies, macro_f1s = [], []
    for _ in range(n_boot):
        picked = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([members[g] for g in picked])
        accuracy, macro_f1, *_ = _scores(_confusion(y_true[idx], y_pred[idx], labels))
        accuracies.append(accuracy)
        macro_f1s.append(macro_f1)
    tail = (1 - level) / 2 * 100
    return {
        "accuracy": np.percentile(accuracies, [tail, 100 - tail]).tolist(),
        "macro_f1": np.percentile(macro_f1s, [tail, 100 - tail]).tolist(),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str,
) -> None:
    """Row-normalized confusion matrix (each row: where that true class went)."""
    y_pred = np.where(np.isin(y_pred, labels), y_pred, "other")
    display_labels = labels + (["other"] if "other" in y_pred else [])
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=display_labels,
        normalize="true",
        values_format=".2f",
        xticks_rotation=45,
        colorbar=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

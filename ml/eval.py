"""Evaluation harness: train/evaluate one model under one protocol, write a report.

Every run validates its datasets first (and refuses invalid ones), uses
them exactly as ml.data.load_dataset would (deduplicated), and writes to
ml/experiments/<date>-<protocol>-<model>/:
  - metrics.json   all numbers, plus provenance: dataset SHA-256s, git
                   commit, model parameters, the command that reproduces it
  - report.md      the same, human-readable
  - confusion_matrix.png

Usage (from repo root):
    uv run python -m ml.eval --protocol p1 --model mlp
    make eval PROTOCOL=p1 MODEL=mlp
"""

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from ml.data import dataset_sha256, featurize, load_records
from ml.evaluate import bootstrap_ci, compute_metrics, plot_confusion_matrix
from ml.models import available_models, build_model
from ml.protocols import HAGRID_PATH, OWN_PATH, PROTOCOLS, build_protocol
from ml.validation import Report, hagrid_attributes, validate

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "ml" / "experiments"
CV_FOLDS = 5
N_BOOT = 1000
SEED = 0
# Groups smaller than this are reported but flagged: their accuracy is noisy.
MIN_GROUP_SIZE = 100


def load_validated(path: Path) -> tuple[list[dict], dict[str, Any]]:
    """Validated, deduplicated records plus the dataset's identity."""
    records = load_records(path)
    report = Report(path=str(path), sha256=dataset_sha256(path), n_records=len(records))
    checked = validate(records, report)
    if not report.ok:
        raise SystemExit(
            f"{path} failed validation — run `make validate DATA={path}`:\n  "
            + "\n  ".join(report.errors)
        )
    return checked, {
        "path": str(
            path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        ),
        "sha256": report.sha256,
        "records_raw": report.n_records,
        "records_used": len(checked),
        "duplicates_removed": report.stats["duplicates_removed"],
    }


def git_commit() -> str:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()

    commit = run("rev-parse", "--short", "HEAD") or "unknown"
    return commit + ("-dirty" if run("status", "--porcelain") else "")


def model_params(model) -> dict[str, str]:
    try:
        return {k: repr(v) for k, v in sorted(model.get_params(deep=False).items())}
    except Exception:  # noqa: BLE001 — provenance only, never fatal
        return {}


def run_cv(train: list[dict], model_name: str) -> dict[str, Any]:
    x, y = featurize(train)
    groups = np.array([r["user_id"] for r in train])
    labels = sorted(set(y))
    folds = []
    for fold, (fit_idx, val_idx) in enumerate(
        GroupKFold(n_splits=CV_FOLDS).split(x, y, groups)
    ):
        model = build_model(model_name).fit(x[fit_idx], y[fit_idx])
        metrics = compute_metrics(y[val_idx], model.predict(x[val_idx]), labels)
        folds.append(
            {
                "fold": fold,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )
        print(f"  fold {fold}: accuracy {metrics['accuracy']:.3f}")
    summary = {
        metric: {
            "mean": float(np.mean([f[metric] for f in folds])),
            "std": float(np.std([f[metric] for f in folds])),
        }
        for metric in ("accuracy", "macro_f1")
    }
    return {
        "folds": folds,
        "summary": summary,
        "n_train": len(train),
        "train_subjects": len(set(groups)),
    }


def run_holdout(data, model_name: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model = build_model(model_name)
    fit_seconds = 0.0
    if data.train is not None:
        x_train, y_train = featurize(data.train)
        start = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - start

    x_test, y_test = featurize(data.test)
    start = time.perf_counter()
    y_pred = model.predict(x_test)
    predict_seconds = time.perf_counter() - start

    metrics = compute_metrics(y_test, y_pred, data.labels)
    metrics["ci95"] = bootstrap_ci(
        y_test, y_pred, data.test_groups, data.labels, n_boot=N_BOOT, seed=SEED
    )
    handedness = np.array([r["handedness"] for r in data.test])
    metrics["accuracy_by_handedness"] = {
        hand: float((y_pred[handedness == hand] == y_test[handedness == hand]).mean())
        for hand in sorted(set(handedness))
    }
    if all(r.get("source") == "hagrid" for r in data.test):
        attributes = hagrid_attributes(data.test)
        if attributes:
            metrics["accuracy_by_group"] = accuracy_by_group(
                data.test, y_test, y_pred, attributes
            )
    result = {
        "metrics": metrics,
        "n_train": len(data.train) if data.train is not None else 0,
        "train_subjects": (
            len({r["user_id"] for r in data.train}) if data.train is not None else 0
        ),
        "n_test": len(data.test),
        "test_groups": len(set(data.test_groups.tolist())),
        "fit_seconds": round(fit_seconds, 2),
        # Python/NumPy timing on this machine — not the in-browser latency.
        "predict_ms_per_sample": round(1000 * predict_seconds / len(data.test), 4),
        "model_params": model_params(model),
    }
    return result, y_test, y_pred


def accuracy_by_group(
    test: list[dict], y_test: np.ndarray, y_pred: np.ndarray, attributes: dict
) -> dict[str, dict[str, dict[str, float]]]:
    """Accuracy per demographic group (HaGRID meta: gender, race, age)."""
    correct = y_test == y_pred
    out: dict[str, dict[str, dict[str, float]]] = {}
    for key in ("gender", "race", "age"):
        values = np.array([attributes[r["image_id"]][key] for r in test])
        out[key] = {
            value: {
                "accuracy": float(correct[values == value].mean()),
                "n": int((values == value).sum()),
            }
            for value in sorted(set(values))
        }
    return out


def write_report(out_dir: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['protocol'].upper()} — `{result['model']}`",
        "",
        f"**Question:** {result['question']}",
        "",
        f"- Train: {result['train_description']}",
        f"- Test: {result['test_description']}",
        f"- Date: {result['date']} · commit `{result['git_commit']}`",
        f"- Reproduce: `{result['command']}`",
        "",
        "## Datasets",
        "",
        "| Dataset | SHA-256 | Records used (raw) | Duplicates removed |",
        "|---|---|---:|---:|",
    ]
    for name, d in result["datasets"].items():
        lines.append(
            f"| `{name}` (`{d['path']}`) | `{d['sha256'][:12]}…` | "
            f"{d['records_used']} ({d['records_raw']}) | {d['duplicates_removed']} |"
        )
    lines.append("")

    if result["protocol"] == "cv":
        cv = result["cv"]
        lines += [
            f"## {CV_FOLDS}-fold grouped cross-validation",
            "",
            f"{cv['n_train']} samples from {cv['train_subjects']} subjects; "
            "folds never share a subject.",
            "",
            "| Metric | Mean ± std |",
            "|---|---|",
        ]
        for metric, s in cv["summary"].items():
            lines.append(f"| {metric} | {s['mean']:.3f} ± {s['std']:.3f} |")
    else:
        m = result["metrics"]
        lines += [
            "## Results",
            "",
            f"{result['n_test']} test samples, {result['test_groups']} independent "
            "groups (bootstrap unit). "
            + (
                f"Trained on {result['n_train']} samples from "
                f"{result['train_subjects']} subjects."
                if result["n_train"]
                else "Pre-trained model: nothing fitted in this run."
            ),
            "",
            "| Metric | Value | 95 % CI |",
            "|---|---:|---|",
        ]
        for metric in ("accuracy", "macro_f1"):
            low, high = m["ci95"][metric]
            lines.append(f"| {metric} | {m[metric]:.3f} | [{low:.3f}, {high:.3f}] |")
        lines += [
            "",
            "### Per class",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
        for label, c in m["per_class"].items():
            lines.append(
                f"| `{label}` | {c['precision']:.3f} | {c['recall']:.3f} | "
                f"{c['f1']:.3f} | {c['support']} |"
            )
        by_hand = ", ".join(
            f"{hand}: {acc:.3f}" for hand, acc in m["accuracy_by_handedness"].items()
        )
        lines += ["", f"Accuracy by handedness — {by_hand}."]
        if "accuracy_by_group" in m:
            lines += [
                "",
                "### By demographic group",
                "",
                "HaGRID meta, automatically estimated (FairFace / MiVOLO), so "
                f"approximate. Groups under {MIN_GROUP_SIZE} samples are marked *.",
                "",
                "| Attribute | Group | Accuracy | n |",
                "|---|---|---:|---:|",
            ]
            for key, groups in m["accuracy_by_group"].items():
                for value, g in groups.items():
                    star = "*" if g["n"] < MIN_GROUP_SIZE else ""
                    lines.append(
                        f"| {key} | {value}{star} | {g['accuracy']:.3f} | {g['n']} |"
                    )
        lines += [
            "",
            "![Confusion matrix](confusion_matrix.png)",
            "",
            f"Inference: {result['predict_ms_per_sample']} ms/sample (Python, "
            f"not in-browser). Fit time: {result['fit_seconds']} s.",
        ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--protocol", required=True, choices=list(PROTOCOLS))
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--hagrid", type=Path, default=HAGRID_PATH)
    parser.add_argument("--own", type=Path, default=OWN_PATH)
    parser.add_argument("--out-root", type=Path, default=EXPERIMENTS_DIR)
    args = parser.parse_args()

    hagrid, hagrid_info = load_validated(args.hagrid)
    datasets = {"hagrid_landmarks_v1": hagrid_info}
    own = None
    if args.protocol == "p3":
        if not args.own.exists():
            raise SystemExit(f"P3 needs {args.own} — record it with /collect first.")
        own, datasets["own_v2"] = load_validated(args.own)

    try:
        data = build_protocol(args.protocol, args.model, hagrid, own)
    except ValueError as error:
        raise SystemExit(str(error)) from None

    date = datetime.date.today().isoformat()
    result: dict[str, Any] = {
        "protocol": args.protocol,
        "model": args.model,
        "question": data.question,
        "train_description": (
            "HaGRID train subjects"
            if data.train is not None
            else "none — deployed weights (author's own recordings, gestures_v1)"
        ),
        "test_description": data.test_description,
        "date": date,
        "git_commit": git_commit(),
        "command": "uv run python -m ml.eval " + " ".join(sys.argv[1:]),
        "datasets": datasets,
        "seed": SEED,
    }

    out_dir = args.out_root / f"{date}-{args.protocol}-{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.protocol.upper()} / {args.model}: {data.question}")

    if args.protocol == "cv":
        result["cv"] = run_cv(data.train, args.model)
        s = result["cv"]["summary"]
        print(
            f"CV accuracy {s['accuracy']['mean']:.3f} ± {s['accuracy']['std']:.3f}, "
            f"macro-F1 {s['macro_f1']['mean']:.3f} ± {s['macro_f1']['std']:.3f}"
        )
    else:
        holdout, y_test, y_pred = run_holdout(data, args.model)
        result |= holdout
        plot_confusion_matrix(
            y_test,
            y_pred,
            data.labels,
            out_dir / "confusion_matrix.png",
            title=f"{args.protocol.upper()} — {args.model}",
        )
        m = result["metrics"]
        print(
            f"accuracy {m['accuracy']:.3f} {np.round(m['ci95']['accuracy'], 3)}, "
            f"macro-F1 {m['macro_f1']:.3f} {np.round(m['ci95']['macro_f1'], 3)}"
        )

    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    write_report(out_dir, result)
    print(f"Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()

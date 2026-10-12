"""Dataset loading, featurization and leakage-safe train/test splitting.

Every record carries at least:
    {"label": str, "landmarks": [{"x","y","z"} x 21], "handedness": str}

Own recordings (frontend/src/components/DatasetCollector.tsx) add
    "timestamp": float  — used to group frames into recording bursts
and, from schema_version 2 on (own_v2):
    "schema_version": 2, "source": "own", "session_id": str,
    "camera": str, "resolution": {"width": int, "height": int},
    "lighting": "daylight" | "artificial" | "dim" | "backlit"

HaGRID records (ml/datasets/hagrid.py) add "source": "hagrid", "user_id"
and HaGRID's official "split" instead.
"""

import json
from pathlib import Path

import numpy as np

from ml.features import normalize_landmarks

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

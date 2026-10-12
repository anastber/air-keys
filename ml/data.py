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

import hashlib
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
    """A JSON array (own recordings) or JSON Lines (extracted datasets)."""
    if path.suffix == ".jsonl":
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
    return json.loads(path.read_text())


def dataset_sha256(path: Path) -> str:
    """Content hash identifying a dataset version in validation and experiment reports."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Two samples closer than this in normalized feature space are treated as
# the same photo. On hagrid_landmarks_v1, nearest-neighbor distances have a
# median of ~0.52 and a 1st percentile of ~0.25; the handful of pairs below
# 0.05 are exact copies or the same scene filed under two HaGRID user_ids.
NEAR_DUPLICATE_DISTANCE = 0.05
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def duplicates_to_drop(
    records: list[dict], max_distance: float = NEAR_DUPLICATE_DISTANCE
) -> set[int]:
    """Indices of duplicated samples to remove before any experiment.

    - Exact landmark copies: every copy after the first is dropped.
    - Records with a "split" field (HaGRID): near-duplicates (feature
      distance < max_distance) in *different* splits are leakage, since
      HaGRID files a few identical scenes under different user_ids. The
      copy in the later split (train < val < test) is dropped, so the
      evaluation sets never contain a near-copy of a training sample.
    """
    from scipy.spatial import cKDTree

    drop: set[int] = set()
    seen: dict[tuple, int] = {}
    for i, record in enumerate(records):
        key = tuple((p["x"], p["y"], p["z"]) for p in record["landmarks"])
        if key in seen:
            drop.add(i)
        else:
            seen[key] = i

    if all("split" in r for r in records):
        x, _ = featurize(records)
        for i, j in cKDTree(x).query_pairs(r=max_distance):
            if records[i]["split"] != records[j]["split"]:
                later = max(i, j, key=lambda k: SPLIT_ORDER[records[k]["split"]])
                drop.add(later)
    return drop


def load_dataset(path: Path) -> list[dict]:
    """Records as every experiment should consume them: loaded, then deduplicated."""
    records = load_records(path)
    drop = duplicates_to_drop(records)
    return [r for i, r in enumerate(records) if i not in drop]


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

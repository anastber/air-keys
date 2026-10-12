"""Dataset validation: schema, leakage, balance and quality checks.

Run before any experiment uses a dataset. Problems are split in two:
  - errors break the data or the evaluation methodology (malformed
    records, unknown labels, duplicated images, a subject on both sides of
    a split) — the command exits non-zero;
  - warnings are worth knowing but don't invalidate the data (class
    imbalance, fewer recording sessions than planned, noisy landmarks).

Checks run on the records exactly as experiments consume them
(ml.data.load_dataset: loaded, then deduplicated); what deduplication
removed is reported. The report also carries the dataset file's SHA-256,
which identifies the exact dataset version in experiment reports (see
ROADMAP.md, methodology rule 7).

Usage (from repo root):
    uv run python -m ml.validation ml/data/hagrid_landmarks_v1/records.jsonl
    uv run python -m ml.validation ml/data/own_v2.json --report report.json
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ml.data import dataset_sha256, duplicates_to_drop, load_records
from ml.datasets.hagrid import CACHE_DIR, load_label_config

KNOWN_LABELS = {spec["label"] for spec in load_label_config()["classes"].values()}
HANDEDNESS = {"Left", "Right"}
# Keep in sync with LIGHTING_OPTIONS in frontend/src/components/DatasetCollector.tsx.
LIGHTING = {"daylight", "artificial", "dim", "backlit"}
HAGRID_SPLITS = {"train", "val", "test"}
NUM_LANDMARKS = 21

# Planned number of own recording sessions (ROADMAP.md, Phase 0.2): own_v2
# is a test set only, recorded in the real playing conditions.
MIN_OWN_SESSIONS = 2
# Within a split (HaGRID) or the whole dataset (own), a class with fewer
# than this fraction of the largest class's samples is flagged.
MIN_CLASS_RATIO = 0.5
# Mean distance between our landmarks and HaGRID's own 2D annotations,
# as a fraction of the hand box size, above which a sample is flagged as a
# likely landmark error (see the Phase 0.1 extraction results).
MAX_HAGRID_GAP = 0.15
# How many offending record indices to quote per error message.
MAX_EXAMPLES = 5


@dataclass
class Report:
    path: str
    sha256: str
    n_records: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def source_of(record: dict[str, Any]) -> str:
    # Own recordings exported before schema v2 have no "source" field.
    return record.get("source", "own")


def check_record(record: dict[str, Any]) -> list[str]:
    """Schema problems of a single record (empty if valid)."""
    problems = []
    landmarks = record.get("landmarks")
    if not isinstance(landmarks, list) or len(landmarks) != NUM_LANDMARKS:
        problems.append(f"landmarks: expected {NUM_LANDMARKS} points")
    elif not all(
        isinstance(p, dict)
        and all(
            isinstance(p.get(k), int | float) and math.isfinite(p[k]) for k in "xyz"
        )
        for p in landmarks
    ):
        problems.append("landmarks: missing or non-finite x/y/z")
    if record.get("label") not in KNOWN_LABELS:
        problems.append(f"unknown label {record.get('label')!r}")
    if record.get("handedness") not in HANDEDNESS:
        problems.append(f"invalid handedness {record.get('handedness')!r}")

    source = source_of(record)
    if source == "hagrid":
        if not record.get("user_id"):
            problems.append("hagrid: missing user_id")
        if record.get("split") not in HAGRID_SPLITS:
            problems.append(f"hagrid: invalid split {record.get('split')!r}")
        if not record.get("image_id"):
            problems.append("hagrid: missing image_id")
    elif source == "own":
        if not isinstance(record.get("timestamp"), int | float):
            problems.append("own: missing timestamp")
        if record.get("schema_version", 1) >= 2:
            if not record.get("session_id"):
                problems.append("own: missing session_id")
            if record.get("lighting") not in LIGHTING:
                problems.append(f"own: invalid lighting {record.get('lighting')!r}")
            if not record.get("camera"):
                problems.append("own: missing camera")
    else:
        problems.append(f"unknown source {source!r}")
    return problems


def hagrid_landmark_gap(record: dict[str, Any]) -> float | None:
    """Mean gap to HaGRID's own 2D landmarks, as a fraction of the hand box size."""
    theirs = record.get("hagrid_landmarks")
    if not theirs or len(theirs) != NUM_LANDMARKS:
        return None
    ours = np.array([[p["x"], p["y"]] for p in record["landmarks"]])
    box_size = max(record["bbox"][2], record["bbox"][3])
    return float(np.linalg.norm(ours - np.array(theirs), axis=1).mean() / box_size)


def _examples(indices: list[int]) -> str:
    shown = ", ".join(map(str, indices[:MAX_EXAMPLES]))
    more = (
        f", +{len(indices) - MAX_EXAMPLES} more" if len(indices) > MAX_EXAMPLES else ""
    )
    return f"records {shown}{more}"


def _check_balance(
    counts: dict[str, Counter], where: str, report: Report, labels: set[str]
) -> None:
    for group, label_counts in sorted(counts.items()):
        missing = sorted(labels - set(label_counts))
        if missing:
            report.warnings.append(f"{where} {group}: no samples for {missing}")
        largest = max(label_counts.values())
        small = {
            label: n
            for label, n in sorted(label_counts.items())
            if n < MIN_CLASS_RATIO * largest
        }
        if small:
            report.warnings.append(
                f"{where} {group}: under-represented classes {small} "
                f"(largest class has {largest})"
            )


def validate(records: list[dict[str, Any]], report: Report) -> list[dict[str, Any]]:
    """Fill `report`; return the valid, deduplicated records the checks ran on."""
    # 1. Schema, per record. Later checks only look at valid records.
    problems: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        for problem in check_record(record):
            problems[problem].append(i)
    for problem, indices in sorted(problems.items()):
        report.errors.append(f"{problem} ({_examples(indices)})")
    invalid = {i for indices in problems.values() for i in indices}
    valid = [r for i, r in enumerate(records) if i not in invalid]

    # 2. Deduplicate, as ml.data.load_dataset does, and report what went.
    drop = duplicates_to_drop(valid) if valid else set()
    if drop:
        dropped = Counter(valid[i].get("split", "n/a") for i in drop)
        report.warnings.append(
            f"{len(drop)} duplicated records removed by deduplication "
            f"(by split: {dict(sorted(dropped.items()))}); "
            "ml.data.load_dataset applies the same removal"
        )
    report.stats["duplicates_removed"] = len(drop)
    valid = [r for i, r in enumerate(valid) if i not in drop]

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in valid:
        by_source[source_of(record)].append(record)
    report.stats["by_source"] = {s: len(rs) for s, rs in by_source.items()}
    report.stats["by_label"] = dict(sorted(Counter(r["label"] for r in valid).items()))
    report.stats["handedness"] = dict(Counter(r["handedness"] for r in valid))

    if by_source.get("hagrid"):
        _validate_hagrid(by_source["hagrid"], report)
    if by_source.get("own"):
        _validate_own(by_source["own"], report)
    return valid


def _validate_hagrid(records: list[dict[str, Any]], report: Report) -> None:
    # The same image id twice means a broken extraction, not a HaGRID quirk.
    image_counts = Counter(r["image_id"] for r in records)
    duplicated = [image for image, n in image_counts.items() if n > 1]
    if duplicated:
        report.errors.append(f"hagrid: {len(duplicated)} image_ids appear twice")

    # Leakage: a subject must never appear in more than one split.
    splits_of_user: dict[str, set[str]] = defaultdict(set)
    for r in records:
        splits_of_user[r["user_id"]].add(r["split"])
    leaking = [u for u, splits in splits_of_user.items() if len(splits) > 1]
    if leaking:
        report.errors.append(
            f"hagrid: {len(leaking)} subjects appear in more than one split"
        )

    counts: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        counts[r["split"]][r["label"]] += 1
    _check_balance(counts, "hagrid split", report, KNOWN_LABELS)

    gaps = [g for r in records if (g := hagrid_landmark_gap(r)) is not None]
    noisy = sum(g > MAX_HAGRID_GAP for g in gaps)
    if noisy:
        report.warnings.append(
            f"hagrid: {noisy} records ({noisy / len(records):.1%}) disagree with "
            f"HaGRID's landmarks by > {MAX_HAGRID_GAP:.0%} of hand size "
            "(likely landmark errors)"
        )

    report.stats["hagrid"] = {
        "subjects": len(splits_of_user),
        "by_split": {s: dict(sorted(c.items())) for s, c in sorted(counts.items())},
        "landmark_gap_median": round(float(np.median(gaps)), 4) if gaps else None,
        "landmark_gap_p95": round(float(np.percentile(gaps, 95)), 4) if gaps else None,
        "records_without_hagrid_landmarks": len(records) - len(gaps),
    }


def _validate_own(records: list[dict[str, Any]], report: Report) -> None:
    legacy = [r for r in records if r.get("schema_version", 1) < 2]
    if legacy:
        report.warnings.append(
            f"own: {len(legacy)} pre-v2 records have no session; they can't be "
            "used in session-disjoint evaluation"
        )

    sessioned = [r for r in records if r.get("session_id")]
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in sessioned:
        sessions[r["session_id"]].append(r)
    if len(sessions) < MIN_OWN_SESSIONS:
        report.warnings.append(
            f"own: {len(sessions)} session(s), fewer than the {MIN_OWN_SESSIONS} "
            "planned"
        )

    # Balance over the whole dataset, then per-session class coverage.
    _check_balance(
        {"dataset": Counter(r["label"] for r in records)}, "own", report, KNOWN_LABELS
    )
    for session_id, rs in sorted(sessions.items()):
        missing = sorted(KNOWN_LABELS - {r["label"] for r in rs})
        if missing:
            report.warnings.append(
                f"own session {session_id}: no samples for {missing}"
            )

    report.stats["own"] = {
        "sessions": {
            session_id: {
                "records": len(rs),
                "lighting": sorted({r["lighting"] for r in rs}),
                "camera": sorted({r["camera"] for r in rs}),
            }
            for session_id, rs in sorted(sessions.items())
        },
        "pre_v2_records": len(legacy),
    }


def hagrid_demographics(
    records: list[dict[str, Any]], annotations_dir: Path = CACHE_DIR / "annotations"
) -> dict[str, dict[str, int]] | None:
    """Gender / race / age distribution of the sampled HaGRID subjects.

    Uses HaGRID's own meta annotations (automatically estimated with FairFace
    and MiVOLO, so approximate), read from the annotation cache written by
    ml/datasets/hagrid.py. Counted once per subject. Returns None if the
    cache isn't available.
    """
    needed = {(r["split"], r["hagrid_label"]) for r in records}
    meta_by_image: dict[str, dict[str, Any]] = {}
    for split, hagrid_label in needed:
        path = annotations_dir / split / f"{hagrid_label}.json"
        if not path.exists():
            return None
        for image_id, annotation in json.loads(path.read_text()).items():
            meta_by_image[image_id] = annotation.get("meta") or {}

    meta_by_user: dict[str, dict[str, Any]] = {}
    for r in records:
        meta_by_user.setdefault(r["user_id"], meta_by_image.get(r["image_id"], {}))

    def first(meta: dict[str, Any], key: str) -> Any:
        values = meta.get(key) or [None]
        return values[0]

    def age_bucket(age: float | None) -> str:
        if age is None:
            return "unknown"
        low = min(int(age // 10) * 10, 60)
        return "60+" if low == 60 else f"{low}-{low + 9}"

    users = list(meta_by_user.values())
    return {
        "gender": dict(Counter(first(m, "gender") or "unknown" for m in users)),
        "race": dict(Counter(first(m, "race") or "unknown" for m in users)),
        "age": dict(
            sorted(Counter(age_bucket(first(m, "age")) for m in users).items())
        ),
    }


def validate_file(path: Path) -> Report:
    records = load_records(path)
    report = Report(path=str(path), sha256=dataset_sha256(path), n_records=len(records))
    checked = validate(records, report)
    hagrid = [r for r in checked if source_of(r) == "hagrid"]
    if hagrid:
        demographics = hagrid_demographics(hagrid)
        if demographics:
            report.stats["hagrid"]["demographics"] = demographics
    return report


def print_report(report: Report) -> None:
    print(f"Dataset: {report.path}")
    print(f"SHA-256: {report.sha256}")
    print(f"Records: {report.n_records}")
    print(json.dumps(report.stats, indent=2))
    for title, items in (("Errors", report.errors), ("Warnings", report.warnings)):
        print(f"\n{title} ({len(items)}):")
        for item in items:
            print(f"  - {item}")
    print("\nVALID" if report.ok else "\nINVALID")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path, help="also write the report as JSON")
    args = parser.parse_args()

    report = validate_file(args.dataset)
    print_report(report)
    if args.report:
        args.report.write_text(json.dumps(asdict(report), indent=2))
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()

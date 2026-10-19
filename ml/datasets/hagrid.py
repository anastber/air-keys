"""Extract MediaPipe hand landmarks from a subject-stratified sample of HaGRID.

HaGRID (https://github.com/hukenovs/hagrid) provides multi-subject data
without recruiting anyone: tens of thousands of distinct people, each image
tagged with a `user_id`, and an official train/val/test split that is
already subject-disjoint. Its archives are ~40 GB per class, so instead of
downloading them this script reads only the sampled images out of the remote
zips (HTTP range requests via remotezip) and never writes images to disk —
only the extracted landmarks.

Landmarks are re-extracted with MediaPipe's GestureRecognizer, using the
same gesture_recognizer.task model as the frontend
(frontend/scripts/setup-mediapipe.mjs), rather than reusing HaGRID's own
landmark annotations (2D only, different extraction settings). That keeps
train-time and inference-time preprocessing identical. HaGRID's 2D landmarks
are kept alongside, for later train/serve skew analysis.

Known differences from the frontend, documented rather than hidden: still
photos run in IMAGE mode (no tracking across frames), and on the CPU rather
than the browser's GPU delegate.

Each image's target hand is the annotated bbox carrying the class label;
the detected hand whose landmark bbox overlaps it best (IoU) is kept. Every
image that yields no usable hand is logged to failures.jsonl with a reason,
never silently dropped — per-class failure rates are themselves a finding.

License: HaGRID is distributed under a variant of CC BY-SA 4.0
(attribution required, share-alike for adapted material); see the
repository's license/en_us.pdf.

Usage (from repo root):
    # the app's 9 gestures (hagrid_landmarks_v1)
    uv run python -m ml.datasets.hagrid --n-train 1000 --n-val 200 --n-test 300
    # rejection negatives, and gesture variants (test only)
    uv run python -m ml.datasets.hagrid --roles negative negative_heldout \
        --n-train 500 --n-val 100 --n-test 150 --out-dir ml/data/hagrid_negatives_v1
    uv run python -m ml.datasets.hagrid --roles variant \
        --n-train 0 --n-val 0 --n-test 150 --out-dir ml/data/hagrid_variants_v1
"""

import argparse
import hashlib
import json
import random
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_CONFIG_PATH = Path(__file__).with_name("hagrid_labels.json")
CACHE_DIR = REPO_ROOT / "ml" / "data" / "hagrid_cache"
DEFAULT_OUT_DIR = REPO_ROOT / "ml" / "data" / "hagrid_landmarks_v1"

ANNOTATIONS_URL = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru"
    "/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"
)
# Same model the frontend downloads (frontend/scripts/setup-mediapipe.mjs).
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer"
    "/gesture_recognizer/float16/1/gesture_recognizer.task"
)

SPLITS = ("train", "val", "test")
# Same as the frontend (lib/gestureRecognition.ts): both hands are detected,
# then the one matching the annotated bbox is kept.
NUM_HANDS = 2
# Minimum IoU between the annotated bbox and a detected hand's landmark
# bbox for that detection to count as the annotated hand. Landmark bboxes
# are tighter than HaGRID's hand boxes, so this is deliberately loose; the
# IoU of every kept hand is recorded, so the threshold can be revisited.
MIN_MATCH_IOU = 0.3
# Network retries per image before giving up (the run stays resumable).
READ_ATTEMPTS = 5

BBox = tuple[float, float, float, float]  # normalized (x, y, width, height)


@dataclass(frozen=True)
class Sample:
    image_id: str
    user_id: str
    split: str
    hagrid_label: str
    bbox: BBox
    hagrid_landmarks: list[list[float]] | None


@dataclass(frozen=True)
class DetectedHand:
    landmarks: list[dict[str, float]]
    handedness: str
    handedness_score: float


# ---------------------------------------------------------------------------
# Pure logic (unit-tested)
# ---------------------------------------------------------------------------


def load_label_config(path: Path = LABEL_CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def target_hands(
    annotation: dict[str, Any], hagrid_label: str
) -> list[tuple[BBox, list[list[float]] | None]]:
    """The (bbox, HaGRID landmarks) of every hand labeled `hagrid_label`."""
    landmarks = annotation.get("hand_landmarks") or []
    hands = []
    for i, (bbox, label) in enumerate(
        zip(annotation["bboxes"], annotation["labels"], strict=True)
    ):
        if label == hagrid_label:
            hand_landmarks = landmarks[i] if i < len(landmarks) else None
            hands.append((tuple(bbox), hand_landmarks or None))
    return hands


def sample_images(
    annotations: dict[str, dict[str, Any]],
    hagrid_label: str,
    split: str,
    n: int,
    max_per_user: int,
    seed: int,
) -> list[Sample]:
    """Pick up to `n` images, spreading them over as many subjects as possible.

    Subjects are visited round-robin in a seeded random order, taking at
    most `max_per_user` images each, so the sample favors subject diversity
    over many shots of the same few people. Images with zero or several
    hands carrying the target label are skipped: which hand is "the" sample
    would be ambiguous. So are images without a user_id (some HaGRID classes
    have them): their subject could be on the other side of the split.
    """
    by_user: dict[str, list[str]] = defaultdict(list)
    for image_id in sorted(annotations):
        if not annotations[image_id].get("user_id"):
            continue
        if len(target_hands(annotations[image_id], hagrid_label)) == 1:
            by_user[annotations[image_id]["user_id"]].append(image_id)

    rng = random.Random(seed)
    users = sorted(by_user)
    rng.shuffle(users)
    for user in users:
        rng.shuffle(by_user[user])

    chosen: list[str] = []
    for round_index in range(max_per_user):
        for user in users:
            if len(chosen) == n:
                break
            if round_index < len(by_user[user]):
                chosen.append(by_user[user][round_index])

    samples = []
    for image_id in chosen:
        annotation = annotations[image_id]
        (bbox, hagrid_landmarks) = target_hands(annotation, hagrid_label)[0]
        samples.append(
            Sample(
                image_id=image_id,
                user_id=annotation["user_id"],
                split=split,
                hagrid_label=hagrid_label,
                bbox=bbox,
                hagrid_landmarks=hagrid_landmarks,
            )
        )
    return samples


def landmarks_bbox(landmarks: list[dict[str, float]]) -> BBox:
    xs = [p["x"] for p in landmarks]
    ys = [p["y"] for p in landmarks]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def iou(a: BBox, b: BBox) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    inter_w = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    inter_h = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match_hand(
    hands: list[DetectedHand], target_bbox: BBox, min_iou: float = MIN_MATCH_IOU
) -> tuple[int, float] | None:
    """Index and IoU of the detected hand best matching the annotated bbox."""
    if not hands:
        return None
    scores = [iou(landmarks_bbox(h.landmarks), target_bbox) for h in hands]
    best = max(range(len(hands)), key=scores.__getitem__)
    return (best, scores[best]) if scores[best] >= min_iou else None


def build_record(
    sample: Sample, hand: DetectedHand, match_iou: float, label: str
) -> dict[str, Any]:
    """One extracted sample, in the record format ml/data.py consumes."""
    return {
        "source": "hagrid",
        "image_id": sample.image_id,
        "user_id": sample.user_id,
        "split": sample.split,
        "hagrid_label": sample.hagrid_label,
        "label": label,
        "landmarks": hand.landmarks,
        "handedness": hand.handedness,
        "handedness_score": hand.handedness_score,
        "match_iou": match_iou,
        "bbox": list(sample.bbox),
        "hagrid_landmarks": sample.hagrid_landmarks,
    }


def build_failure(sample: Sample, reason: str) -> dict[str, Any]:
    return {
        "source": "hagrid",
        "image_id": sample.image_id,
        "user_id": sample.user_id,
        "split": sample.split,
        "hagrid_label": sample.hagrid_label,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# I/O: remote archives, MediaPipe
# ---------------------------------------------------------------------------


def fetch_annotations(split: str, hagrid_label: str) -> dict[str, dict[str, Any]]:
    """One split/class annotation file, read from the remote zip once, then cached."""
    from remotezip import RemoteZip

    cached = CACHE_DIR / "annotations" / split / f"{hagrid_label}.json"
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        with RemoteZip(ANNOTATIONS_URL) as archive:
            cached.write_bytes(archive.read(f"annotations/{split}/{hagrid_label}.json"))
    return json.loads(cached.read_text())


def ensure_model() -> Path:
    path = CACHE_DIR / "gesture_recognizer.task"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class ArchiveReader:
    """Reads single images out of a remote zip, one connection per thread."""

    def __init__(self, url: str):
        self.url = url
        self._local = threading.local()
        self._index: dict[str, str] | None = None

    def _archive(self):
        from remotezip import RemoteZip

        if not hasattr(self._local, "archive"):
            self._local.archive = RemoteZip(self.url)
        return self._local.archive

    def build_index(self) -> None:
        # v1 archives nest files under "<class>/", v2 archives are flat:
        # index by file stem (= HaGRID image id) to handle both.
        self._index = {
            Path(info.filename).stem: info.filename
            for info in self._archive().infolist()
            if not info.is_dir()
        }

    def read(self, image_id: str) -> bytes | None:
        """One image's bytes, retrying with a fresh connection on network errors.

        The HaGRID server occasionally drops connections mid-transfer; a
        persistent failure still raises, and the run can be resumed.
        """
        from remotezip import RemoteIOError

        assert self._index is not None, "call build_index() first"
        name = self._index.get(image_id)
        if name is None:
            return None
        for attempt in range(1, READ_ATTEMPTS + 1):
            try:
                return self._archive().read(name)
            except RemoteIOError:
                if attempt == READ_ATTEMPTS:
                    raise
                # Reconnect on the next attempt (the failure may have come
                # from opening the connection itself, so it may not exist).
                self._local.__dict__.pop("archive", None)
                time.sleep(2**attempt)


class HandDetector:
    """MediaPipe GestureRecognizer on still images (not thread-safe)."""

    def __init__(self, model_path: Path):
        from mediapipe.tasks.python import BaseOptions, vision

        self._recognizer = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=NUM_HANDS,
            )
        )

    def detect(self, jpeg: bytes) -> list[DetectedHand] | None:
        """Detected hands, or None if the image can't be decoded."""
        import cv2
        import mediapipe as mp
        import numpy as np

        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self._recognizer.recognize(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )
        return [
            DetectedHand(
                landmarks=[{"x": p.x, "y": p.y, "z": p.z} for p in landmarks],
                handedness=handedness[0].category_name,
                handedness_score=handedness[0].score,
            )
            for landmarks, handedness in zip(
                result.hand_landmarks, result.handedness, strict=True
            )
        ]

    def close(self) -> None:
        self._recognizer.close()


def extract(
    samples: list[Sample],
    reader: ArchiveReader,
    detector: HandDetector,
    label: str,
    workers: int,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ("record" | "failure", row) per sample; downloads run in parallel."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        images = pool.map(lambda s: reader.read(s.image_id), samples)
        for sample, jpeg in zip(samples, images, strict=True):
            if jpeg is None:
                yield "failure", build_failure(sample, "image_not_in_archive")
                continue
            hands = detector.detect(jpeg)
            if hands is None:
                yield "failure", build_failure(sample, "image_decode_error")
            elif not hands:
                yield "failure", build_failure(sample, "no_hand_detected")
            elif (match := match_hand(hands, sample.bbox)) is None:
                yield "failure", build_failure(sample, "no_matching_hand")
            else:
                index, match_iou = match
                yield "record", build_record(sample, hands[index], match_iou, label)


def already_processed(*paths: Path) -> set[str]:
    done = set()
    for path in paths:
        if path.exists():
            with path.open() as f:
                done.update(json.loads(line)["image_id"] for line in f)
    return done


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--classes", nargs="+", help="HaGRID classes to extract")
    parser.add_argument(
        "--roles",
        nargs="+",
        default=["gesture"],
        help="extract every class with these roles in hagrid_labels.json "
        "(default: gesture); ignored if --classes is given",
    )
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--max-per-user", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    import mediapipe as mp

    config = load_label_config()
    classes = args.classes or [
        name for name, spec in config["classes"].items() if spec["role"] in args.roles
    ]
    unknown = set(classes) - set(config["classes"])
    if unknown:
        raise SystemExit(f"Not in {LABEL_CONFIG_PATH.name}: {sorted(unknown)}")
    n_per_split = {"train": args.n_train, "val": args.n_val, "test": args.n_test}

    model_path = ensure_model()
    manifest = {
        "dataset": "hagrid",
        "label_config_version": config["version"],
        "classes": classes,
        "n_per_split": n_per_split,
        "max_per_user": args.max_per_user,
        "seed": args.seed,
        "min_match_iou": MIN_MATCH_IOU,
        "num_hands": NUM_HANDS,
        "mediapipe_version": mp.__version__,
        "model_url": MODEL_URL,
        "model_sha256": sha256(model_path),
        "annotations_url": ANNOTATIONS_URL,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    records_path = args.out_dir / "records.jsonl"
    failures_path = args.out_dir / "failures.jsonl"
    # Resuming is only safe with the exact same settings: otherwise the
    # output would silently mix two different samplings or pipelines.
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise SystemExit(
            f"{manifest_path} was produced with different settings; "
            "use a new --out-dir (datasets are immutable once written)."
        )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    done = already_processed(records_path, failures_path)
    if done:
        print(f"Resuming: {len(done)} images already processed")

    detector = HandDetector(model_path)
    outcomes: Counter[tuple[str, str, str]] = Counter()

    with records_path.open("a") as records, failures_path.open("a") as failures:
        for hagrid_label in classes:
            label = config["classes"][hagrid_label]["label"]
            reader = ArchiveReader(config["classes"][hagrid_label]["archive"])
            reader.build_index()
            for split in SPLITS:
                if n_per_split[split] == 0:
                    continue
                samples = sample_images(
                    fetch_annotations(split, hagrid_label),
                    hagrid_label,
                    split,
                    n=n_per_split[split],
                    max_per_user=args.max_per_user,
                    seed=args.seed,
                )
                todo = [s for s in samples if s.image_id not in done]
                print(
                    f"{hagrid_label}/{split}: {len(samples)} sampled "
                    f"from {len({s.user_id for s in samples})} subjects, "
                    f"{len(todo)} to extract"
                )
                for kind, row in extract(todo, reader, detector, label, args.workers):
                    out = records if kind == "record" else failures
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    outcomes[(hagrid_label, split, row.get("reason", "ok"))] += 1
    detector.close()

    print("\nOutcomes this run (class/split: reason -> count):")
    for (hagrid_label, split, reason), count in sorted(outcomes.items()):
        print(f"  {hagrid_label}/{split}: {reason} -> {count}")
    print(f"\nWrote {records_path} and {failures_path}")


if __name__ == "__main__":
    main()

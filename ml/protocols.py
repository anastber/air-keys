"""Evaluation protocols (ROADMAP.md, Phase 0): which data trains, which data tests.

  P1 in-domain        HaGRID train subjects -> HaGRID test subjects
                      Does the model generalize to people it never saw?
  P2 cross-dataset    deployed MLP (trained on the author's hand only)
                      -> HaGRID test subjects
                      The honest version of the README's single-subject score.
  P3 deployment shift HaGRID train subjects -> own_v2 (live webcam)
                      Does a model learned from photos work in the app?
  cv                  grouped k-fold CV on HaGRID train subjects only —
                      for model selection, never touches a test set.

HaGRID's val split is not used here: it is reserved for calibration and
threshold tuning (ROADMAP.md, Phase 1).

The deployed MLP only knows 4 classes (ok_sign, rock_on, call_me,
no_gesture). It is scored on a 4-class view of the test set where every
other pose becomes no_gesture — which is how the app treats them: the
deployed model should stay silent on a fist or a peace sign.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.data import assign_recording_bursts
from ml.models import BACKGROUND_LABEL, DeployedMLP

REPO_ROOT = Path(__file__).resolve().parent.parent
HAGRID_PATH = REPO_ROOT / "ml" / "data" / "hagrid_landmarks_v1" / "records.jsonl"
OWN_PATH = REPO_ROOT / "ml" / "data" / "own_v2.json"

PROTOCOLS = {
    "p1": "Does the model generalize to people it never saw?",
    "p2": "Does the deployed model, trained on one person's hand, work on thousands of others?",
    "p3": "Does a model learned from HaGRID photos work on live webcam video in the app?",
    "cv": "Model selection: grouped cross-validation on HaGRID training subjects.",
}


@dataclass
class ProtocolData:
    name: str
    question: str
    train: list[dict] | None  # None: pre-trained model, nothing to fit
    test: list[dict]
    test_groups: np.ndarray  # resampling unit for confidence intervals
    labels: list[str]  # classes the test is scored on
    test_description: str


def deployed_view(records: list[dict]) -> list[dict]:
    """Relabel every class the deployed model doesn't know as background."""
    known = set(DeployedMLP().classes_)
    return [
        r if r["label"] in known else {**r, "label": BACKGROUND_LABEL} for r in records
    ]


def own_groups(records: list[dict]) -> np.ndarray:
    """Recording bursts within sessions: the correlated unit in own data.

    Bursts are numbered per session, so a burst never spans two sessions
    even if their timestamps happen to be close.
    """
    by_session: dict[str, list[int]] = {}
    for i, record in enumerate(records):
        by_session.setdefault(record.get("session_id", ""), []).append(i)

    groups = np.empty(len(records), dtype=np.int64)
    offset = 0
    for session in sorted(by_session):
        order = sorted(by_session[session], key=lambda i: records[i]["timestamp"])
        bursts = assign_recording_bursts([records[i] for i in order])
        groups[order] = bursts + offset
        offset += int(bursts.max()) + 1
    return groups


def build_protocol(
    name: str, model: str, hagrid: list[dict], own: list[dict] | None
) -> ProtocolData:
    if name not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {name!r}; choose from {list(PROTOCOLS)}")
    deployed = model == "deployed"
    if name == "p2" and not deployed:
        raise ValueError("P2 evaluates the deployed model only (--model deployed)")
    if name in ("p1", "cv") and deployed:
        raise ValueError(f"{name} trains a model; the deployed one is evaluated by P2")

    hagrid_train = [r for r in hagrid if r["split"] == "train"]
    hagrid_test = [r for r in hagrid if r["split"] == "test"]

    if name in ("p1", "cv", "p2"):
        test, description = hagrid_test, "HaGRID test subjects"
        groups = np.array([r["user_id"] for r in test])
        if name == "cv":
            test, groups, description = [], np.array([]), "none (cross-validation)"
    else:  # p3
        if not own:
            raise ValueError(
                "P3 needs own recordings (ml/data/own_v2.json, see /collect)"
            )
        test, description = own, "own_v2 (live webcam)"
        groups = own_groups(own)

    if deployed:
        test = deployed_view(test)
        labels = sorted(DeployedMLP().classes_)
        train = None
    else:
        train = hagrid_train
        labels = sorted({r["label"] for r in train})

    return ProtocolData(
        name=name,
        question=PROTOCOLS[name],
        train=train,
        test=test,
        test_groups=groups,
        labels=labels,
        test_description=description,
    )

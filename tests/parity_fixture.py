"""Shared fixture for the Python <-> TypeScript parity tests.

Models are trained and evaluated in Python but run in the browser in
TypeScript, so both sides must compute the same numbers:
  - landmark normalization: ml/features.py vs lib/customGestures.ts
  - the deployed MLP's forward pass: ml/models.py::DeployedMLP vs
    lib/trainedGestures.ts::forward

This module builds tests/fixtures/parity.json — inputs plus the outputs the
Python side produces. tests/test_parity.py checks the file still matches
Python; frontend/src/lib/parity.test.ts checks TypeScript against it.
Inputs are synthetic hands (no dataset content is committed).

Regenerate after changing either implementation or the deployed weights:
    uv run python -m tests.parity_fixture
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from ml.features import normalize_landmarks
from ml.models import DEPLOYED_WEIGHTS_PATH, DeployedMLP

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "parity.json"
N_RANDOM_HANDS = 12
SEED = 7


def _hand(rng: np.random.Generator) -> list[dict[str, float]]:
    """A plausible hand: a wrist plus 20 points spread above it, MediaPipe-like ranges."""
    wrist = rng.uniform([0.2, 0.5, 0.0], [0.8, 0.9, 0.0])
    offsets = rng.normal(
        loc=[0.0, -0.15, -0.03], scale=[0.08, 0.08, 0.03], size=(20, 3)
    )
    points = np.vstack([wrist, wrist + offsets])
    return [{"x": float(x), "y": float(y), "z": float(z)} for x, y, z in points]


def build_cases() -> list[dict]:
    rng = np.random.default_rng(SEED)
    hands = [(_hand(rng), "Left" if i % 2 else "Right") for i in range(N_RANDOM_HANDS)]
    # Edge case: middle-finger MCP on top of the wrist, so the hand-span
    # scale hits its 1e-6 floor in both implementations.
    degenerate = _hand(rng)
    degenerate[9] = dict(degenerate[0])
    hands.append((degenerate, "Right"))

    model = DeployedMLP()
    cases = []
    for landmarks, handedness in hands:
        features = normalize_landmarks(landmarks, handedness)
        probabilities = model.predict_proba(features[None, :])[0]
        cases.append(
            {
                "handedness": handedness,
                "landmarks": landmarks,
                "features": features.astype(float).tolist(),
                "probabilities": probabilities.tolist(),
            }
        )
    return cases


def build_fixture() -> dict:
    return {
        "deployed_weights_sha256": hashlib.sha256(
            DEPLOYED_WEIGHTS_PATH.read_bytes()
        ).hexdigest(),
        "cases": build_cases(),
    }


if __name__ == "__main__":
    FIXTURE_PATH.parent.mkdir(exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(build_fixture(), indent=1) + "\n")
    print(f"Wrote {FIXTURE_PATH}")

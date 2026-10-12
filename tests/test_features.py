import numpy as np
import pytest

from ml.features import normalize_landmarks


@pytest.fixture
def hand() -> list[dict[str, float]]:
    rng = np.random.default_rng(0)
    points = rng.uniform(0.2, 0.8, size=(21, 3))
    return [{"x": x, "y": y, "z": z} for x, y, z in points]


def transform(hand, dx=0.0, dy=0.0, scale=1.0):
    return [
        {"x": p["x"] * scale + dx, "y": p["y"] * scale + dy, "z": p["z"] * scale}
        for p in hand
    ]


def test_output_shape_and_dtype(hand):
    features = normalize_landmarks(hand, "Right")
    assert features.shape == (63,)
    assert features.dtype == np.float32


def test_wrist_is_origin(hand):
    features = normalize_landmarks(hand, "Right").reshape(21, 3)
    np.testing.assert_allclose(features[0], 0.0)


def test_translation_invariant(hand):
    np.testing.assert_allclose(
        normalize_landmarks(hand, "Right"),
        normalize_landmarks(transform(hand, dx=0.1, dy=-0.05), "Right"),
        atol=1e-5,
    )


def test_scale_invariant(hand):
    np.testing.assert_allclose(
        normalize_landmarks(hand, "Right"),
        normalize_landmarks(transform(hand, scale=0.5), "Right"),
        atol=1e-5,
    )


def test_left_hand_is_mirrored_on_x_only(hand):
    right = normalize_landmarks(hand, "Right").reshape(21, 3)
    left = normalize_landmarks(hand, "Left").reshape(21, 3)
    np.testing.assert_allclose(left[:, 0], -right[:, 0])
    np.testing.assert_allclose(left[:, 1:], right[:, 1:])

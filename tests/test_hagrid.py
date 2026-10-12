import pytest

from ml.data import featurize
from ml.datasets.hagrid import (
    DetectedHand,
    Sample,
    build_record,
    iou,
    load_label_config,
    match_hand,
    sample_images,
    target_hands,
)


def annotation(user_id: str, labels: list[str], bboxes=None) -> dict:
    bboxes = bboxes or [[0.1 * i, 0.1, 0.2, 0.2] for i in range(len(labels))]
    return {"user_id": user_id, "labels": labels, "bboxes": bboxes}


def hand_at(x: float, y: float, size: float = 0.2) -> DetectedHand:
    # 21 points spread over a size x size square starting at (x, y).
    points = [
        {"x": x + size * (i % 5) / 4, "y": y + size * (i // 5) / 4, "z": 0.0}
        for i in range(21)
    ]
    return DetectedHand(landmarks=points, handedness="Right", handedness_score=0.9)


def test_label_config_maps_every_class_to_a_label_and_archive():
    config = load_label_config()
    assert config["version"] >= 1
    for spec in config["classes"].values():
        assert spec["label"]
        assert spec["archive"].startswith("https://")


def test_target_hands_ignores_other_labels():
    ann = annotation("u", ["no_gesture", "ok"])
    ann["hand_landmarks"] = [[[0.0, 0.0]], [[0.5, 0.5]]]
    assert target_hands(ann, "ok") == [((0.1, 0.1, 0.2, 0.2), [[0.5, 0.5]])]


def test_sampling_spreads_over_subjects_before_repeating_one():
    annotations = {f"a{i}": annotation("alice", ["ok"]) for i in range(10)}
    annotations |= {f"b{i}": annotation("bob", ["ok"]) for i in range(10)}

    samples = sample_images(annotations, "ok", "train", n=4, max_per_user=5, seed=0)

    users = [s.user_id for s in samples]
    assert len(samples) == 4
    assert users.count("alice") == users.count("bob") == 2


def test_sampling_respects_max_per_user_even_if_n_is_not_reached():
    annotations = {f"a{i}": annotation("alice", ["ok"]) for i in range(10)}
    samples = sample_images(annotations, "ok", "train", n=5, max_per_user=2, seed=0)
    assert len(samples) == 2


def test_sampling_skips_images_without_exactly_one_target_hand():
    annotations = {
        "single": annotation("u1", ["ok", "no_gesture"]),
        "double": annotation("u2", ["ok", "ok"]),
        "none": annotation("u3", ["no_gesture"]),
    }
    samples = sample_images(annotations, "ok", "test", n=10, max_per_user=1, seed=0)
    assert [s.image_id for s in samples] == ["single"]


def test_sampling_is_deterministic_for_a_seed():
    annotations = {f"img{i}": annotation(f"user{i % 7}", ["ok"]) for i in range(50)}
    first = sample_images(annotations, "ok", "train", n=10, max_per_user=2, seed=1)
    second = sample_images(annotations, "ok", "train", n=10, max_per_user=2, seed=1)
    assert first == second


def test_iou():
    box = (0.0, 0.0, 1.0, 1.0)
    assert iou(box, box) == pytest.approx(1.0)
    assert iou(box, (2.0, 2.0, 1.0, 1.0)) == 0.0
    assert iou(box, (0.5, 0.0, 1.0, 1.0)) == pytest.approx(1 / 3)


def test_match_hand_picks_the_hand_overlapping_the_annotated_box():
    hands = [hand_at(0.7, 0.7), hand_at(0.1, 0.1)]
    match = match_hand(hands, target_bbox=(0.1, 0.1, 0.2, 0.2))
    assert match is not None
    assert match[0] == 1
    assert match[1] == pytest.approx(1.0)


def test_match_hand_rejects_a_hand_elsewhere_in_the_image():
    assert match_hand([hand_at(0.7, 0.7)], target_bbox=(0.1, 0.1, 0.2, 0.2)) is None
    assert match_hand([], target_bbox=(0.1, 0.1, 0.2, 0.2)) is None


def test_records_are_consumable_by_the_training_pipeline():
    sample = Sample(
        image_id="img",
        user_id="u",
        split="train",
        hagrid_label="ok",
        bbox=(0.1, 0.1, 0.2, 0.2),
        hagrid_landmarks=None,
    )
    record = build_record(sample, hand_at(0.1, 0.1), match_iou=0.8, label="ok_sign")

    x, y = featurize([record])
    assert x.shape == (1, 63)
    assert y.tolist() == ["ok_sign"]

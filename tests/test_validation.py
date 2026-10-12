import math

import pytest

from ml.data import duplicates_to_drop
from ml.validation import MIN_OWN_SESSIONS, Report, check_record, validate

_counter = iter(range(10**6))


def landmarks(offset: float = 0.0) -> list[dict[str, float]]:
    # A distinct hand shape per offset, so records aren't duplicates.
    return [
        {"x": 0.3 + 0.01 * i + offset * (i % 3), "y": 0.4 + 0.02 * i, "z": 0.0}
        for i in range(21)
    ]


def hagrid_record(label="ok_sign", split="train", user="u1", offset=None) -> dict:
    n = next(_counter)
    return {
        "source": "hagrid",
        "image_id": f"img{n}",
        "user_id": user,
        "split": split,
        "hagrid_label": "ok",
        "label": label,
        "landmarks": landmarks(n * 0.01 if offset is None else offset),
        "handedness": "Right",
        "bbox": [0.3, 0.4, 0.2, 0.4],
        "hagrid_landmarks": None,
    }


def own_record(label="ok_sign", session="s1", lighting="daylight") -> dict:
    n = next(_counter)
    return {
        "schema_version": 2,
        "source": "own",
        "label": label,
        "landmarks": landmarks(n * 0.01),
        "handedness": "Left",
        "timestamp": float(n),
        "session_id": session,
        "camera": "test cam",
        "resolution": {"width": 640, "height": 480},
        "lighting": lighting,
    }


def run(records) -> Report:
    report = Report(path="test", sha256="", n_records=len(records))
    validate(records, report)
    return report


def test_valid_records_have_no_schema_problems():
    assert check_record(hagrid_record()) == []
    assert check_record(own_record()) == []


@pytest.mark.parametrize(
    "change, expected",
    [
        (lambda r: r["landmarks"].pop(), "landmarks: expected 21 points"),
        (lambda r: r["landmarks"][0].update(x=math.nan), "non-finite"),
        (lambda r: r.update(label="wave"), "unknown label"),
        (lambda r: r.update(handedness="Both"), "invalid handedness"),
        (lambda r: r.update(split="dev"), "invalid split"),
        (lambda r: r.pop("user_id"), "missing user_id"),
    ],
)
def test_malformed_hagrid_record_is_an_error(change, expected):
    record = hagrid_record()
    change(record)
    report = run([record])
    assert not report.ok
    assert any(expected in error for error in report.errors)


def test_own_v2_record_requires_session_and_lighting():
    record = own_record()
    record.pop("session_id")
    record["lighting"] = "sunny"
    problems = check_record(record)
    assert "own: missing session_id" in problems
    assert any("invalid lighting" in p for p in problems)


def test_pre_v2_own_record_is_accepted_with_a_warning():
    record = own_record()
    for key in ("schema_version", "source", "session_id", "camera", "lighting"):
        record.pop(key)
    report = run([record])
    assert report.ok
    assert any("pre-v2" in w for w in report.warnings)


def test_subject_in_two_splits_is_leakage():
    report = run(
        [
            hagrid_record(split="train", user="alice"),
            hagrid_record(split="test", user="alice"),
        ]
    )
    assert any("more than one split" in e for e in report.errors)


def test_fewer_sessions_than_planned_is_a_warning():
    report = run([own_record(session=f"s{i}") for i in range(MIN_OWN_SESSIONS - 1)])
    assert report.ok
    assert any("fewer than" in w for w in report.warnings)


def test_class_imbalance_is_a_warning():
    records = [hagrid_record(label="ok_sign", user=f"a{i}") for i in range(10)]
    records += [hagrid_record(label="fist", user=f"b{i}") for i in range(2)]
    report = run(records)
    assert any("under-represented" in w and "fist" in w for w in report.warnings)


def test_exact_copies_are_dropped_after_the_first():
    original = own_record()
    copy = dict(original)
    assert duplicates_to_drop([original, own_record(), copy]) == {2}


def test_near_duplicate_across_splits_drops_the_evaluation_copy():
    train = hagrid_record(split="train", user="a", offset=0.5)
    test = hagrid_record(split="test", user="b", offset=0.5001)
    far = hagrid_record(split="test", user="c", offset=3.0)
    assert duplicates_to_drop([test, train, far]) == {0}


def test_near_duplicates_within_a_split_are_kept():
    a = hagrid_record(split="train", user="a", offset=0.5)
    b = hagrid_record(split="train", user="b", offset=0.5001)
    assert duplicates_to_drop([a, b]) == set()

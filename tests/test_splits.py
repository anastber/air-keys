import numpy as np

from ml.train import assign_recording_bursts, group_stratified_split


def record(label: str, timestamp: float) -> dict:
    return {"label": label, "timestamp": timestamp}


def test_new_burst_on_label_change_or_time_gap():
    records = [
        record("ok_sign", 0.0),
        record("ok_sign", 0.1),
        record("rock_on", 0.2),  # label change
        record("rock_on", 5.0),  # > 2s gap
        record("rock_on", 5.1),
    ]
    assert assign_recording_bursts(records).tolist() == [0, 0, 1, 2, 2]


def test_split_never_puts_a_burst_on_both_sides():
    labels = np.repeat(["a", "b"], 50)
    groups = np.repeat(np.arange(10), 10)  # 10 bursts of 10 frames, 5 per label
    x = np.arange(len(labels)).reshape(-1, 1)

    x_train, x_test, y_train, y_test = group_stratified_split(
        x, labels, groups, test_size=0.2, random_state=0
    )

    train_groups = set(groups[x_train.ravel()])
    test_groups = set(groups[x_test.ravel()])
    assert train_groups.isdisjoint(test_groups)
    assert len(x_train) + len(x_test) == len(x)
    # Every class is represented in the test set.
    assert set(y_test) == {"a", "b"}

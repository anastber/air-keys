import json

import numpy as np
import pytest
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

from ml.evaluate import bootstrap_ci, compute_metrics
from ml.models import BACKGROUND_LABEL, DeployedMLP, available_models, build_model
from ml.protocols import build_protocol, deployed_view, own_groups
from ml.train import export_mlp_weights


def test_every_registered_model_builds():
    for name in available_models():
        assert hasattr(build_model(name), "predict")


@pytest.fixture
def trained_mlp_weights(tmp_path):
    rng = np.random.default_rng(0)
    labels = np.array(["call_me", "no_gesture", "ok_sign", "rock_on"])
    centers = rng.normal(size=(4, 63))
    y = rng.integers(0, 4, size=400)
    x = centers[y] + rng.normal(scale=0.3, size=(400, 63))
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=0)
    mlp.fit(x, labels[y])
    path = tmp_path / "weights.json"
    export_mlp_weights(mlp, list(mlp.classes_), path)
    return mlp, x, path


def test_deployed_forward_pass_matches_sklearn(trained_mlp_weights):
    mlp, x, path = trained_mlp_weights
    deployed = DeployedMLP(weights_path=path)
    np.testing.assert_allclose(
        deployed.predict_proba(x), mlp.predict_proba(x), atol=1e-6
    )


def test_deployed_model_stays_silent_below_its_threshold(trained_mlp_weights):
    mlp, x, path = trained_mlp_weights
    payload = json.loads(path.read_text())
    payload["confidenceThreshold"] = 1.01  # nothing is ever confident enough
    path.write_text(json.dumps(payload))
    assert set(DeployedMLP(weights_path=path).predict(x)) == {BACKGROUND_LABEL}


def test_metrics_match_sklearn():
    labels = ["a", "b", "c"]
    y_true = np.array(["a", "a", "b", "b", "c", "c"])
    y_pred = np.array(["a", "b", "b", "b", "c", "a"])
    metrics = compute_metrics(y_true, y_pred, labels)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["macro_f1"] == pytest.approx(
        f1_score(y_true, y_pred, average="macro")
    )
    assert metrics["per_class"]["b"]["recall"] == 1.0


def test_prediction_outside_scored_labels_counts_as_an_error():
    metrics = compute_metrics(np.array(["a", "b"]), np.array(["a", "zzz"]), ["a", "b"])
    assert metrics["accuracy"] == 0.5
    assert metrics["confusion_matrix"][1] == [0, 0, 1]  # "b" went to "other"


def test_bootstrap_ci_brackets_the_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y_true = rng.choice(["a", "b"], size=300)
    y_pred = np.where(rng.random(300) < 0.8, y_true, rng.choice(["a", "b"], size=300))
    groups = np.repeat(np.arange(100), 3)
    ci = bootstrap_ci(y_true, y_pred, groups, ["a", "b"], n_boot=200, seed=3)
    accuracy = compute_metrics(y_true, y_pred, ["a", "b"])["accuracy"]
    assert ci["accuracy"][0] <= accuracy <= ci["accuracy"][1]
    assert ci == bootstrap_ci(y_true, y_pred, groups, ["a", "b"], n_boot=200, seed=3)


def _hagrid(label: str, split: str, user: str) -> dict:
    return {"label": label, "split": split, "user_id": user, "landmarks": []}


HAGRID = [
    _hagrid("ok_sign", "train", "u1"),
    _hagrid("fist", "train", "u2"),
    _hagrid("ok_sign", "val", "u3"),
    _hagrid("fist", "test", "u4"),
    _hagrid("ok_sign", "test", "u5"),
]


def test_p1_trains_on_train_subjects_and_tests_on_test_subjects():
    data = build_protocol("p1", "mlp", HAGRID, own=None)
    train_users = {r["user_id"] for r in data.train}
    test_users = {r["user_id"] for r in data.test}
    assert train_users == {"u1", "u2"}
    assert test_users == {"u4", "u5"}
    assert data.labels == ["fist", "ok_sign"]


def test_protocol_and_model_combinations_are_enforced():
    with pytest.raises(ValueError, match="deployed model only"):
        build_protocol("p2", "mlp", HAGRID, own=None)
    with pytest.raises(ValueError, match="evaluated by P2"):
        build_protocol("p1", "deployed", HAGRID, own=None)
    with pytest.raises(ValueError, match="own recordings"):
        build_protocol("p3", "mlp", HAGRID, own=None)


def test_p2_scores_the_deployed_model_on_a_four_class_view():
    data = build_protocol("p2", "deployed", HAGRID, own=None)
    assert data.train is None
    assert [r["label"] for r in data.test] == [BACKGROUND_LABEL, "ok_sign"]
    assert data.labels == sorted(DeployedMLP().classes_)


def test_deployed_view_keeps_known_labels():
    relabeled = deployed_view([{"label": "peace"}, {"label": "rock_on"}])
    assert [r["label"] for r in relabeled] == [BACKGROUND_LABEL, "rock_on"]


def test_own_groups_are_recording_bursts_within_sessions():
    records = [
        {"label": "fist", "session_id": "s2", "timestamp": 0.0},
        {"label": "fist", "session_id": "s1", "timestamp": 0.0},
        {"label": "fist", "session_id": "s1", "timestamp": 0.1},
        {"label": "fist", "session_id": "s1", "timestamp": 9.0},  # > 2 s gap
    ]
    groups = own_groups(records)
    assert groups[1] == groups[2]
    assert len({groups[0], groups[1], groups[3]}) == 3

import json

import numpy as np

from tests.parity_fixture import FIXTURE_PATH, build_fixture

REGENERATE = "regenerate with `uv run python -m tests.parity_fixture`"


def test_fixture_matches_current_python_implementation():
    stored = json.loads(FIXTURE_PATH.read_text())
    current = build_fixture()
    assert stored["deployed_weights_sha256"] == current["deployed_weights_sha256"], (
        f"deployed weights changed — {REGENERATE}"
    )
    assert len(stored["cases"]) == len(current["cases"]), REGENERATE
    for stored_case, current_case in zip(
        stored["cases"], current["cases"], strict=True
    ):
        np.testing.assert_array_equal(
            stored_case["features"], current_case["features"], err_msg=REGENERATE
        )
        np.testing.assert_allclose(
            stored_case["probabilities"],
            current_case["probabilities"],
            rtol=0,
            atol=1e-12,
            err_msg=REGENERATE,
        )

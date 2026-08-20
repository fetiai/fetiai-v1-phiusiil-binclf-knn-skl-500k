"""The golden row: does this repo reproduce the training run's recorded numbers?"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phiusiil import schema
from server import load, predict
from server.selftest import run_golden

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


@pytest.fixture(scope="module")
def loaded():
    return load.load_all(MODEL_DIR)


@pytest.fixture(scope="module")
def golden():
    return json.loads((MODEL_DIR / "golden_row.json").read_text(encoding="utf-8"))


def test_feature_order_matches_the_frozen_schema(loaded):
    _, stats, _ = loaded
    assert len(stats.feature_order) == 49
    assert tuple(stats.feature_order) == schema.FEATURE_ORDER


def test_scaler_covers_exactly_the_numeric_columns(loaded):
    _, stats, _ = loaded
    assert tuple(stats.scaler.columns) == tuple(stats.numerical_columns)


def test_transformed_vector_is_bitwise_identical(loaded, golden):
    """Exact equality, not a tolerance.

    A tolerance-based comparison would pass while a fitted statistic quietly differed,
    which is the one thing this test exists to catch.
    """
    model, stats, _ = loaded
    frame = predict.build_frame([golden["raw"]])
    _, _, matrix = predict.score(model, stats, frame)
    assert np.array_equal(matrix[0], np.asarray(golden["vector"], dtype=np.float32))


def test_prediction_and_score_match_the_recorded_run(loaded, golden):
    model, stats, _ = loaded
    frame = predict.build_frame([golden["raw"]])
    labels, scores, _ = predict.score(model, stats, frame)
    assert int(labels[0]) == int(golden["prediction"])
    assert float(scores[0]) == float(golden["score"])


def test_manifest_hashes_are_intact(loaded):
    _, _, manifest = loaded
    load.verify(MODEL_DIR, manifest)


def test_tampering_is_detected(tmp_path, loaded):
    """Hash verification has to actually fail on a changed file, or it proves nothing."""
    _, _, manifest = loaded
    import shutil

    copy = tmp_path / "model"
    shutil.copytree(MODEL_DIR, copy)
    target = copy / "fitted_stats.json"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(load.ArtifactError, match="does not match its recorded hash"):
        load.verify(copy, manifest)


def test_selftest_entrypoint_reports_no_failures():
    assert run_golden(MODEL_DIR) == []

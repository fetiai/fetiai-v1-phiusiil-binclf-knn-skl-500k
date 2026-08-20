"""The HTTP contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.app import app

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


@pytest.fixture(scope="module")
def client():
    # As a context manager, so that the lifespan hook -- and therefore the start-up
    # self-test and the model warm-up -- actually runs. A bare TestClient skips it.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def golden():
    return json.loads((MODEL_DIR / "golden_row.json").read_text(encoding="utf-8"))


def _features(golden):
    from phiusiil import schema

    return {k: golden["raw"].get(k) for k in schema.FEATURE_ORDER}


def _request(golden):
    """The golden row as the API takes it: the 49 features plus the intermediates.

    The intermediates are not decoration. Several of the golden row's features are null,
    and the URL-derived fill chain recomputes those from the URL and Title when they are
    supplied. Send only the 49 and those features are imputed to the training mean
    instead, which is a different -- and legitimately different -- answer.
    """
    return {
        "features": _features(golden),
        "url": golden["raw"].get("URL"),
        "domain": golden["raw"].get("Domain"),
        "tld": golden["raw"].get("TLD"),
        "title": golden["raw"].get("Title"),
    }


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metadata_lists_the_feature_contract(client):
    body = client.get("/metadata").json()
    assert len(body["feature_order"]) == 49
    assert body["positive_class"] == {"label": 0, "meaning": "phishing"}
    assert body["model"]["parameter_count"] > 0


def test_predict_reproduces_the_golden_row(client, golden):
    response = client.post("/predict", json=_request(golden))
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == int(golden["prediction"])
    assert body["phishing_score"] == float(golden["score"])
    assert body["verdict"] in {"phishing", "legitimate"}


def test_debug_returns_the_transformed_vector(client, golden):
    body = client.post("/predict?debug=true", json=_request(golden)).json()
    assert body["vector"] == golden["vector"]


def test_supplying_the_url_recomputes_rather_than_imputes(client, golden):
    """The optional intermediates change the answer, and that is the point.

    21 of the 49 features are derived from the URL string. When one of them arrives null
    and the URL does not, it is imputed from the training distribution; when the URL is
    present it is recomputed from it. The golden row exercises exactly this, so dropping
    the URL must move the vector.
    """
    with_url = client.post("/predict?debug=true", json=_request(golden)).json()
    without = client.post(
        "/predict?debug=true", json={"features": _features(golden)}
    ).json()
    assert with_url["vector"] == golden["vector"]
    assert without["vector"] != golden["vector"]


def test_batch_matches_single(client, golden):
    """Row-by-row and batched must agree.

    The transformed vectors are compared exactly -- that invariance is the correctness
    property the preprocessing exists to guarantee, and a tolerance there would hide a
    statistic leaking from the batch. The scores get a relative tolerance instead: the
    Gaussian Naive Bayes score is a product of 49 densities evaluated in log space, and
    summation order genuinely varies with batch shape, moving the last few ulps of a
    value that can sit near 1e-77. The label is unaffected and is compared exactly.
    """
    payload = _request(golden)
    single = client.post("/predict?debug=true", json=payload).json()
    batch = client.post(
        "/predict/batch?debug=true", json={"rows": [payload] * 3}
    ).json()

    assert len(batch) == 3
    assert all(row["vector"] == single["vector"] for row in batch)
    assert all(row["label"] == single["label"] for row in batch)
    assert all(
        row["phishing_score"] == pytest.approx(single["phishing_score"], rel=1e-9)
        for row in batch
    )


def test_missing_feature_is_named_not_silently_imputed(client, golden):
    features = _features(golden)
    features.pop("URLLength")
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 422
    assert "URLLength" in response.json()["detail"]


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"rows": []}).status_code == 422


def test_readyz_reports_ready_after_startup(client):
    assert client.get("/readyz").json()["status"] == "ready"


def test_evidence_counts_reflect_what_was_supplied(client, golden):
    """A thin row must be marked thin.

    Imputation fills from a 92.5%-legitimate training distribution, so an empty row
    scores as confidently "legitimate" as a complete one. The counts are the only thing
    distinguishing them.
    """
    from phiusiil import schema

    full = client.post("/predict", json={"features": _features(golden)}).json()
    empty = client.post(
        "/predict", json={"features": {k: None for k in schema.FEATURE_ORDER}}
    ).json()

    assert full["n_provided"] > empty["n_provided"]
    assert empty["n_provided"] == 0
    assert empty["n_imputed"] == 49
    assert empty["coverage_ratio"] == 0.0
    assert empty["low_evidence"] is True


def test_an_entirely_null_row_still_scores(client):
    """Every one of the 49 null, and it must return 200 rather than raise.

    This is a regression guard on a subtle dependency. The integer coercion applied to the
    binary and discrete columns runs after imputation, so it only survives a fully null
    row because every one of those columns is guaranteed non-null by then -- including
    HasObfuscation, which is non-null only because its fill function collapses everything
    to 0 or 1. If that function is ever "corrected", this test fails first.
    """
    from phiusiil import schema

    response = client.post(
        "/predict", json={"features": {k: None for k in schema.FEATURE_ORDER}}
    )
    assert response.status_code == 200
    assert response.json()["verdict"] in {"phishing", "legitimate"}


def test_unknown_feature_name_is_rejected(client, golden):
    features = _features(golden) | {"NotAFeature": 1.0}
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 422
    assert "NotAFeature" in json.dumps(response.json())


def test_fractional_value_in_a_binary_column_is_rejected(client, golden):
    """It would otherwise miss the cascade's mode table and then be truncated to 0."""
    features = _features(golden) | {"HasTitle": 0.7}
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 422
    assert "HasTitle" in json.dumps(response.json())


def test_batch_over_the_cap_is_rejected(client, golden):
    from server.app import MAX_BATCH

    rows = [{"features": _features(golden)}] * (MAX_BATCH + 1)
    assert client.post("/predict/batch", json={"rows": rows}).status_code == 422

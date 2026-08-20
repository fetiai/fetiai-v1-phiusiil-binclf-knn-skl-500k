"""HTTP surface for a single model.

Scope, stated plainly because it is the difference between this and the full application:
this server scores a feature row that someone else extracted. It does not fetch pages, it
has no URL feature extractor, and it therefore has no abstention rule -- the caller who
produced the row owns the question of whether enough of it is real.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from phiusiil import __version__, schema
from server import load, predict

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

#: A batch large enough to amortise the fixed per-call preprocessing cost, small enough
#: that one request cannot monopolise the process. Preprocessing costs roughly the same
#: for 1 row as for 100; beyond about a thousand the distance kernel dominates instead.
MAX_BATCH = 1000

MODEL, STATS, MANIFEST = load.load_all(MODEL_DIR)

_BINARY_COLUMNS = frozenset(schema.CATEGORICAL_COLUMNS_FILTERED)
_FEATURE_NAMES = frozenset(schema.FEATURE_ORDER)

READY = {"value": False}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Score the golden row once before accepting traffic.

    Two jobs in one pass. It refuses to start if the artifact no longer reproduces the
    prediction its training run recorded, and it pays the first-call cost up front --
    scikit-learn's k-NN builds its neighbour index lazily, so the first real request
    would otherwise absorb about a second of setup that has nothing to do with it.
    """
    from server.selftest import run_golden

    failures = run_golden(MODEL_DIR)
    if failures:
        raise RuntimeError("artifact self-test failed: " + "; ".join(failures))
    READY["value"] = True
    yield


app = FastAPI(
    title=MANIFEST["model_name"],
    version=__version__,
    lifespan=lifespan,
    description=(
        "Phishing URL classification from a pre-extracted 49-feature row. "
        "Coursework reimplementation, not a security product."
    ),
)


class Row(BaseModel):
    features: dict[str, float | None] = Field(
        ..., description="All 49 feature columns. null is allowed and will be imputed."
    )
    url: str | None = None
    domain: str | None = None
    tld: str | None = None
    title: str | None = None

    @field_validator("features")
    @classmethod
    def _known_and_well_formed(
        cls, value: dict[str, float | None]
    ) -> dict[str, float | None]:
        unknown = sorted(set(value) - _FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown feature columns: {unknown}")

        # A fractional value in a binary column is accepted by every layer below and
        # corrupts two of them silently: it misses the cascade's mode table, so the fill
        # falls back to the global mode, and it is then truncated toward zero by the
        # integer cast. Neither leaves a trace in the response.
        bad = sorted(
            name
            for name, v in value.items()
            if name in _BINARY_COLUMNS and v is not None and v not in (0, 1)
        )
        if bad:
            raise ValueError(f"binary columns must be 0, 1 or null: {bad}")
        return value

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = dict(self.features)
        for name, value in (
            ("URL", self.url),
            ("Domain", self.domain),
            ("TLD", self.tld),
            ("Title", self.title),
        ):
            if value is not None:
                record[name] = value
        return record


class Batch(BaseModel):
    rows: list[Row]


class Prediction(BaseModel):
    model: str
    label: int
    verdict: str
    phishing_score: float
    n_provided: int
    n_imputed: int
    coverage_ratio: float
    low_evidence: bool
    vector: list[float] | None = None


def _predict(records: list[dict[str, Any]], debug: bool) -> list[Prediction]:
    try:
        frame = predict.build_frame(records)
        labels, scores, matrix = predict.score(MODEL, STATS, frame)
    except predict.InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return [
        Prediction(
            model=MANIFEST["model_key"],
            label=int(label),
            verdict=predict.verdict(label),
            phishing_score=float(score_),
            vector=[float(v) for v in matrix[i]] if debug else None,
            **predict.evidence(records[i]),
        )
        for i, (label, score_) in enumerate(zip(labels, scores, strict=True))
    ]


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness only. Deliberately does no scoring -- see /readyz for that."""
    return {"status": "ok", "model": MANIFEST["model_key"]}


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    if not READY["value"]:
        raise HTTPException(status_code=503, detail="self-test has not completed")
    return {"status": "ready", "model": MANIFEST["model_key"]}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    """Everything a caller needs to build a valid request, plus what this model scored."""
    import json

    return {
        "model": {
            "key": MANIFEST["model_key"],
            "name": MANIFEST["model_name"],
            "family": MANIFEST["family"],
            "is_scratch": MANIFEST["is_scratch"],
            "parameter_count": MANIFEST["parameter_count"],
        },
        "feature_order": list(schema.FEATURE_ORDER),
        "demoted_features": MANIFEST["demoted_features"],
        "positive_class": {"label": schema.PHISHING_LABEL, "meaning": "phishing"},
        "metrics": json.loads((MODEL_DIR / "metrics.json").read_text("utf-8")),
        "manifest": MANIFEST,
    }


@app.post("/predict", response_model=Prediction)
def predict_one(row: Row, debug: bool = Query(False)) -> Prediction:
    return _predict([row.as_record()], debug)[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(batch: Batch, debug: bool = Query(False)) -> list[Prediction]:
    if not batch.rows:
        raise HTTPException(status_code=422, detail="rows must not be empty")
    if len(batch.rows) > MAX_BATCH:
        raise HTTPException(
            status_code=422,
            detail=f"batch of {len(batch.rows)} exceeds the limit of {MAX_BATCH}",
        )
    return _predict([r.as_record() for r in batch.rows], debug)

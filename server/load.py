"""Load one model and its fitted preprocessing state, and prove they belong together.

This deliberately does not reuse the parent project's bundle loader. That loader requires
all four trained models to be present and verifies a manifest describing all of them, so
it cannot express "one model, on its own" at all. What it does have -- hash verification
before anything is trusted -- is what matters, and that is kept here.

A server that boots with an artifact it cannot account for is worse than one that refuses
to boot: it serves predictions nobody can trace back to a training run, and nothing about
its behaviour looks wrong from outside.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from phiusiil import ARTIFACT_SCHEMA_VERSION
from phiusiil.preprocess.stats import (
    CategoricalFillStep,
    ClipBound,
    FittedStats,
    HtmlFallback,
    NumericFill,
    ScalerParams,
)
import joblib

from phiusiil.models.sk import SklearnKNN, wrap_knn

#: JSON has no tuple keys, so a cascade key is joined with a separator that cannot occur
#: in a numeric literal.
KEY_SEPARATOR = "|"

MODEL_FILE = "knn_sklearn.joblib"


class ArtifactError(RuntimeError):
    """The artifact directory is absent, incomplete, or does not match its manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_key(encoded: str) -> tuple[float, ...]:
    return tuple(float(part) for part in encoded.split(KEY_SEPARATOR))


def verify(model_dir: Path, manifest: dict[str, Any]) -> None:
    """Fail loudly when what is on disk is not what was trained."""
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError(
            f"artifact schema version mismatch: artifact "
            f"{manifest.get('artifact_schema_version')!r}, build "
            f"{ARTIFACT_SCHEMA_VERSION!r}"
        )
    for relative, record in manifest.get("files", {}).items():
        path = model_dir / relative
        if not path.exists():
            raise ArtifactError(f"artifact file missing: {relative}")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise ArtifactError(
                f"artifact file {relative} does not match its recorded hash "
                f"(expected {record['sha256'][:12]}, found {actual[:12]})"
            )


def load_manifest(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "manifest.json"
    if not path.exists():
        raise ArtifactError(f"no manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_stats(model_dir: Path) -> FittedStats:
    """Rebuild the fitted statistics from JSON.

    Plain JSON rather than a pickled transformer, on purpose. A pickled estimator arrives
    with a fit method attached, and the defect this whole pipeline exists to avoid is
    someone calling it at serving time. Numbers cannot be re-fitted.
    """
    payload = json.loads((model_dir / "fitted_stats.json").read_text(encoding="utf-8"))
    version = payload.get("artifact_schema_version")
    if version != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError(
            f"artifact schema version mismatch: fitted stats are {version!r}, "
            f"this build expects {ARTIFACT_SCHEMA_VERSION!r}"
        )

    columns = payload["columns"]
    return FittedStats(
        char_prob={str(k): float(v) for k, v in payload["char_prob"].items()},
        tld_prob_mean={
            str(k): float(v) for k, v in payload["tld"]["prob_mean"].items()
        },
        tld_prob_global_fill=float(payload["tld"]["global_fill"]),
        tld_skew=float(payload["tld"]["skew"]),
        tld_fill_method=payload["tld"]["fill_method"],
        numeric_fill=tuple(NumericFill(**f) for f in payload["numeric_fill"]),
        categorical_cascade=tuple(
            CategoricalFillStep(
                column=s["column"],
                groupby_cols=tuple(s["groupby_cols"]),
                modes={_decode_key(k): float(v) for k, v in s["modes"].items()},
                global_mode=float(s["global_mode"]),
            )
            for s in payload["categorical_cascade"]
        ),
        clip_bounds=tuple(ClipBound(**b) for b in payload["clip_bounds"]),
        scaler=ScalerParams(
            columns=tuple(payload["scaler"]["columns"]),
            mean_=tuple(float(v) for v in payload["scaler"]["mean_"]),
            scale_=tuple(float(v) for v in payload["scaler"]["scale_"]),
        ),
        nb_drop=tuple(payload["nb_drop"]),
        nb_drop_reasons=payload["nb_drop_reasons"],
        html_fallbacks=tuple(HtmlFallback(**f) for f in payload["html_fallbacks"]),
        feature_order=tuple(columns["feature_order"]),
        numerical_columns=tuple(columns["numerical_columns"]),
        continuous_columns=tuple(columns["continuous_columns"]),
        discrete_columns=tuple(columns["discrete_columns"]),
        categorical_columns_filtered=tuple(columns["categorical_columns_filtered"]),
        n_train_rows=int(payload.get("n_train_rows", 0)),
        demoted_features=tuple(payload.get("demoted_features", ())),
    )


def load_model(model_dir: Path, manifest: dict[str, Any]) -> SklearnKNN:
    """Unpickle the bare estimator, then put the wrapper interface back around it."""
    estimator = joblib.load(model_dir / MODEL_FILE)
    return wrap_knn(estimator, k=int(manifest["knn_k"]))



def load_all(model_dir: Path, *, check: bool = True):
    """Return (model, stats, manifest) for the artifact directory."""
    manifest = load_manifest(model_dir)
    if check:
        verify(model_dir, manifest)
    return load_model(model_dir, manifest), load_stats(model_dir), manifest

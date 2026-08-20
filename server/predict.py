"""Turn a submitted feature row into a scored prediction.

Shared by the HTTP layer and the self-test so that both take exactly the same path. A
self-test that exercised a different code path from the server would verify the wrong
thing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phiusiil import schema
from phiusiil.preprocess.stats import FittedStats
from phiusiil.preprocess.transformer import FeatureContractError, Preprocessor


#: Coverage below which the answer is flagged as thin evidence. Matches the parent
#: application's abstention threshold.
COVERAGE_MIN_RATIO = 0.60


class InputError(ValueError):
    """The submitted row does not satisfy the feature contract."""


def build_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Assemble the raw-schema frame the preprocessor expects.

    Absent intermediates (URL, Domain, TLD, Title) become NaN rather than an error: they
    are optional context that lets the URL-derived fill chain recompute a missing feature
    instead of imputing it. Absent *feature* columns are a broken contract and are
    reported by name, because an extractor that dropped a column and an extractor that
    genuinely could not determine a value are different faults with different fixes.

    Nothing here coerces a dtype or reorders a column. transform() reindexes and casts
    from its own recorded state, and every coercion added on this side would be another
    chance to diverge from what training did.
    """
    for index, row in enumerate(rows):
        missing = sorted(set(schema.FEATURE_ORDER) - set(row))
        if missing:
            raise InputError(f"row {index}: feature columns absent: {missing}")

    frame = pd.DataFrame(rows)
    for column in schema.WORKING_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[list(schema.WORKING_COLUMNS)]


def evidence(record: dict[str, Any]) -> dict[str, Any]:
    """How much of the row was actually supplied, rather than imputed.

    This has to travel with the verdict, and it is the one thing a bare
    label-and-score response would throw away.

    Imputation fills a missing feature from the training distribution, and that
    distribution is 92.5% legitimate. So a mostly-empty row does not produce a *neutral*
    prediction -- it produces one biased toward "legitimate", which is precisely the wrong
    direction for a phishing detector. Without these counts a caller cannot tell a verdict
    drawn from 49 real values from one drawn from six, and the second looks exactly as
    confident as the first.
    """
    total = len(schema.FEATURE_ORDER)
    provided = sum(1 for name in schema.FEATURE_ORDER if record.get(name) is not None)
    ratio = provided / total
    return {
        "n_provided": provided,
        "n_imputed": total - provided,
        "coverage_ratio": round(ratio, 4),
        "low_evidence": ratio < COVERAGE_MIN_RATIO,
    }


def score(
    model: Any, stats: FittedStats, frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (labels, phishing scores, the transformed matrix).

    A fresh Preprocessor per call, never a shared one. transform() tallies cascade
    fallbacks on the instance, so a module-global would both accumulate counts across
    unrelated requests and race between concurrent ones. Constructing one only stores a
    reference to the fitted statistics, so this costs nothing.
    """
    try:
        matrix = Preprocessor(stats).transform_matrix(frame)
    except FeatureContractError as exc:
        raise InputError(str(exc)) from exc
    except (ValueError, TypeError) as exc:
        # The integer coercion of the binary and discrete columns is the realistic source
        # here: a null in a column with no recorded fill value reaches the cast still NaN.
        raise InputError(
            f"a submitted value could not be coerced to the trained column types: {exc}"
        ) from exc

    return model.predict(matrix), model.score_phishing(matrix), matrix


def verdict(label: int) -> str:
    return "phishing" if int(label) == schema.PHISHING_LABEL else "legitimate"

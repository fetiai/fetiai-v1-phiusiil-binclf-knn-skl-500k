"""Start-up self-test.

Run at container build time and on demand. It answers one question: does this repo's copy
of the scoring path still produce the numbers the training run recorded?

The check is the golden row -- one real record, its expected 49-vector, and its expected
prediction -- pushed through the whole path and compared exactly. The vector and the
prediction are asserted separately on purpose: a wrong vector means the preprocessing
drifted, a right vector with a wrong label means the model artifact did.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from phiusiil import schema
from server import load, predict

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def run_golden(model_dir: Path) -> list[str]:
    """Return a list of failures; empty means the artifact is trustworthy."""
    failures: list[str] = []

    model, stats, manifest = load.load_all(model_dir)

    if len(stats.feature_order) != 49:
        failures.append(
            f"feature_order has {len(stats.feature_order)} entries, expected 49"
        )
    if tuple(stats.feature_order) != schema.FEATURE_ORDER:
        failures.append("fitted feature order does not match this build's frozen order")
    if tuple(stats.scaler.columns) != tuple(stats.numerical_columns):
        failures.append("scaler column set does not match the recorded numeric columns")

    golden = json.loads((model_dir / "golden_row.json").read_text(encoding="utf-8"))

    frame = predict.build_frame([golden["raw"]])
    labels, scores, matrix = predict.score(model, stats, frame)

    expected_vector = np.asarray(golden["vector"], dtype=np.float32)
    if not np.array_equal(matrix[0], expected_vector):
        differing = np.nonzero(matrix[0] != expected_vector)[0]
        detail = ", ".join(
            f"{schema.FEATURE_ORDER[i]}: expected {expected_vector[i]!r}, "
            f"got {matrix[0][i]!r}"
            for i in differing[:5]
        )
        failures.append(
            f"golden row vector differs in {len(differing)} feature(s): {detail}"
        )

    if int(labels[0]) != int(golden["prediction"]):
        failures.append(
            f"golden row predicted {int(labels[0])}, "
            f"artifact recorded {int(golden['prediction'])}"
        )
    if float(scores[0]) != float(golden["score"]):
        failures.append(
            f"golden row scored {float(scores[0])!r}, "
            f"artifact recorded {float(golden['score'])!r}"
        )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", action="store_true", help="run the golden-row check")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    failures = run_golden(args.model_dir)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("self-test passed: artifact reproduces its recorded golden row")
    return 0


if __name__ == "__main__":
    sys.exit(main())

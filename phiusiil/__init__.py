"""phiusiil -- the serving subset of the PhiUSIIL phishing URL classifier.

One trained model, the fitted preprocessing state it needs, and nothing else. The module
boundary rules that matter here:

1. ``features/`` is pure: no network, no fitted state. It turns raw values into raw
   values, emitting ``NaN`` for anything it cannot determine.
2. ``preprocess/`` owns all fitted state. ``features/`` produces ``NaN``s; ``preprocess/``
   decides what they become, reading every number from persisted state and computing none
   of them at serving time.

There is no fetch layer and no feature extractor for live URLs. This package scores a
feature row that someone else produced.
"""

__version__ = "1.0.0"
ARTIFACT_SCHEMA_VERSION = "v1"

__all__ = ["ARTIFACT_SCHEMA_VERSION", "__version__"]

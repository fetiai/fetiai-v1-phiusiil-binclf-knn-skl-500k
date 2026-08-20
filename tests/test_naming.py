"""The parent project's internal library name must not appear anywhere in this repo.

The needle is stored encoded rather than as a literal, because writing it out would put
the very string this test forbids into the repository it is checking.

Binary files are included in the sweep on purpose. A serialised model is the realistic
place for a stale module path to survive: a pickle records the import path of every class
it contains, so re-serialising the wrong object reintroduces the name invisibly.
"""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NEEDLE = base64.b64decode("cGhpc2hndWFyZA==")


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z"],
        capture_output=True, check=True,
    )
    names = [n for n in result.stdout.split(b"\0") if n]
    return [REPO / n.decode() for n in names]


def test_no_tracked_file_contains_the_name():
    offenders = []
    for path in _tracked_files():
        if not path.is_file():
            continue
        if NEEDLE.lower() in path.read_bytes().lower():
            offenders.append(str(path.relative_to(REPO)))
    assert offenders == [], f"forbidden name present in: {offenders}"


def test_no_path_contains_the_name():
    offenders = [
        str(p.relative_to(REPO))
        for p in _tracked_files()
        if NEEDLE.lower() in str(p).encode().lower()
    ]
    assert offenders == [], f"forbidden name present in path: {offenders}"

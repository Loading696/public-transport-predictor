"""Deprecated entry point. Kept so existing muscle memory and CI keep working.

This used to be the only way to produce a submission. It is now a thin shim over
``ml/make_submission.py`` -- the command that actually validates the result and
exits non-zero when the file is bad. Two entry points that each build a CSV
their own way is precisely the failure mode this task removed, so there is only
one implementation now.

    py ml/generate_submission.py [output.csv]   ->  same as below
    py ml/make_submission.py --out [output.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from ml.make_submission import main as _main


def main() -> int:
    sys.stderr.write(
        "note: ml/generate_submission.py is deprecated, "
        "use py ml/make_submission.py (same behaviour, plus validation)\n"
    )
    if len(sys.argv) > 1:
        # Insert the flag, do not overwrite the destination argument.
        sys.argv[1:1] = ["--out"]
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())

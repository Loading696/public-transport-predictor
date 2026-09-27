"""Direct (no-HTTP) access to the ML pipeline, for tests and offline scripts.

The deployed path is Backend -> HTTP -> ML service.  But the *pipeline* itself
(feature building, CatBoost, detectors) is also worth testing on its own, and
several offline scripts want to run it without a second process.  This module
loads the same :class:`service.MlEngine` the container serves, in-process.

Keeping the ``sys.path`` juggling for the dash-named ``ml-service/`` directory in
one place matters: every test that needs the engine would otherwise repeat the
same three lines, and a stale copy would be easy to leave behind.

What this is **not**: a backend fallback.  The backend never calls this -- it
only ever speaks HTTP to the service.  If you find yourself reaching for it from
``src/`` or ``backend/``, you are about to re-couple the two services.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ML_SERVICE_DIR = ROOT / "ml-service"

if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_engine(root: str | Path | None = None, *, wait: bool = True):
    """Return a fully loaded :class:`service.MlEngine` for ``root``."""
    from service import MlEngine

    engine = MlEngine(Path(root) if root is not None else ROOT)
    if wait:
        engine.thread.join()
    return engine


def require_ready(engine) -> None:
    """Raise if the engine did not finish loading, with the reason."""
    health = engine.health()
    if health["status"] != "ok":
        raise RuntimeError(f"ML pipeline not ready: {health.get('error') or health['status']}")


def predict(points: Any, telemetry: Any = None, *, root: str | Path | None = None) -> list[dict[str, Any]]:
    """Run the batch pipeline in-process and return prediction records."""
    engine = load_engine(root)
    require_ready(engine)
    return engine.predict(points, telemetry)


def write_submission(
    root: str | Path | None = None, output_path: str | Path | None = None
) -> dict[str, Any]:
    """Recompute ``submission.csv`` for the validate period, in-process.

    Used by the offline tools in this directory.  It runs the pipeline directly
    rather than over HTTP so ``sanity_check.py`` still works from a fresh clone
    with nothing else running; the deployed Backend -> ML service path is
    covered by ``ml/test_ml_integration.py``.

    The file itself is written by :mod:`ml.submission`, which owns the format
    contract and validates the result -- this function only produces the
    numbers.
    """
    import pandas as pd

    from ml.submission import write_submission as write

    project_root = Path(root) if root is not None else ROOT
    engine = load_engine(project_root)
    require_ready(engine)
    dataset = project_root / "dataset" / "validate"
    points = pd.read_csv(dataset / "points.csv", low_memory=False)
    telemetry = pd.read_csv(dataset / "traffic.csv", low_memory=False)
    records = engine.predict(points.to_dict("records"), telemetry)

    destination = Path(output_path) if output_path is not None else project_root / "submission.csv"
    report = write(points, [record["prediction"] for record in records], destination)
    if not report.ok:
        raise ValueError("submission failed validation: " + "; ".join(report.errors))
    return report.as_dict()

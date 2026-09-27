"""Build the final submission from the prepared validate/test data -- one command.

    py ml/make_submission.py                       # -> submission.csv, then validate
    py ml/make_submission.py --out out/sub.csv
    py ml/make_submission.py --check-only          # validate an existing file
    py ml/make_submission.py --via http            # go through the running stack

The pipeline
------------
    dataset/validate/points.csv   what to predict: id, T, target stop, target time
    dataset/validate/traffic.csv  telemetry, cut causally at each point's T
    dataset/validate/schedule_plan.csv  the plan
                 |
                 v  preprocessing (ml-service/service.py: normalise, parse WKT,
                 |                   cut event_time <= T per point)
                 v  inference  (CatBoost + risk / class / calibrated p_late
                 |              + pattern detectors)
                 v  submission (ml/submission.py: one format, validated on write)
                 |
    submission.csv   sample_id;prediction, one row per point, points.csv order

Exits non-zero if validation fails, so it is usable as a CI or pre-submit gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml-service"))

DEFAULT_OUT = ROOT / "submission.csv"


def _points_and_traffic() -> tuple[object, object]:
    import pandas as pd

    ds = ROOT / "dataset" / "validate"
    for name in ("points.csv", "traffic.csv", "schedule_plan.csv"):
        if not (ds / name).exists():
            raise SystemExit(
                f"[make] missing {ds / name}\n"
                f"       the validate slice ships with the repository; if it is gone, "
                f"restore it from your dataset copy."
            )
    return (
        pd.read_csv(ds / "points.csv", low_memory=False),
        pd.read_csv(ds / "traffic.csv", low_memory=False),
    )


def predict_inprocess(points, traffic) -> list[float]:
    """Run the ML pipeline locally -- no services required."""
    from ml.ml_pipeline import load_engine, require_ready

    engine = load_engine(ROOT)
    require_ready(engine)
    return [record["prediction"] for record in engine.predict(points.to_dict("records"), traffic)]


def predict_via_stack(points, traffic, url: str) -> list[float]:
    """Go through the running deployment: Backend -> ML service, the real path."""
    import pandas as pd
    from src.ml_client import MlClient

    async def run() -> list[float]:
        client = MlClient(url.rstrip("/"), timeout_s=120, connect_timeout_s=10)
        try:
            records = await client.predict_batch(points.to_dict("records"), traffic)
        finally:
            await client.aclose()
        return [float(record["prediction"]) for record in records]

    return asyncio.run(run())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="destination CSV")
    parser.add_argument("--check-only", action="store_true", help="validate an existing file and exit")
    parser.add_argument(
        "--via", choices=("local", "http"), default="local",
        help="local: run the pipeline in-process; http: call a running Backend",
    )
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000",
        help="Backend base URL when --via http",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args()

    from ml.submission import validate_submission, write_submission

    points, traffic = _points_and_traffic()

    if args.check_only:
        report = validate_submission(args.out, reference_points=points)
    else:
        source = "local pipeline" if args.via == "local" else f"Backend at {args.url}"
        print(f"[make] {len(points)} points, {len(traffic)} telemetry rows -> {source}")
        predictions = (
            predict_inprocess(points, traffic)
            if args.via == "local"
            else predict_via_stack(points, traffic, args.url)
        )
        report = write_submission(points, predictions, args.out)

    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print_rendered(report)

    if not report.ok:
        print("\n[make] FAILED")
        for problem in report.errors:
            print(f"  - {problem}")
        return 1
    print("\n[make] OK")
    return 0


def print_rendered(report) -> None:
    """The human-facing report: counts first, then numbers, then problems."""
    rows = report.rows
    print()
    print("  submission")
    print(f"    file                 {report.path}")
    print(f"    rows                 {rows}" + (f" / {report.expected_rows} expected" if report.expected_rows else ""))
    print(f"    unique sample_id     {report.unique_ids}")
    print(f"    duplicates           {report.duplicate_ids}")
    print(f"    empty ids            {report.empty_ids}")
    print(f"    non-numeric          {report.non_numeric}")
    print(f"    non-finite (NaN/Inf) {report.non_finite}")
    print(f"    out of order         {report.out_of_order}")
    print(f"    missing ids          {report.missing_ids}")
    print(f"    unexpected ids       {report.unexpected_ids}")
    print(f"    outside horizon      {report.horizon_violations}  (target - T must be in (600, 900] s)")
    if report.min_prediction is not None:
        print("  predictions")
        print(f"    min / max            {report.min_prediction:.3f} / {report.max_prediction:.3f} s")
        print(f"    mean                 {report.mean_prediction:.3f} s")
    for warning in report.warnings:
        print(f"    warning: {warning}")
    for problem in report.errors:
        print(f"    error:   {problem}")


if __name__ == "__main__":
    raise SystemExit(main())

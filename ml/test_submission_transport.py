"""Regression tests for the submission telemetry transport.

The bug this guards against
---------------------------
The backend hands telemetry to the ML service as JSON. Timestamps were serialised
with ``Timestamp.isoformat()``, which emits a fractional part *only when the value
has one*: a column ends up mixing ``2026-01-06T03:35:00`` with
``2026-01-06T03:35:00.462764``.

On the receiving side ``pd.to_datetime(series, errors="coerce")`` infers a single
format from the first element and applies it to the whole column, so the
fractional rows became ``NaT`` and were then dropped by
``dropna(subset=["event_time"])``. In the validate telemetry **6 547 of 105 945
rows** carry sub-second precision, so the loss was not marginal: it changed
``telemetry_points`` and every ``speed_*`` window, and a prediction computed over
HTTP differed from the same prediction computed locally on 142 of 151 rows.

Two independent guards now exist and both are asserted here:
  * the client emits one canonical timestamp shape, so the service never has to
    guess (fixed in ``src/ml_client.py::_iso``);
  * the service parses ISO8601 and falls back per element, instead of trusting a
    format inferred from the first row (fixed in ``service._ensure_datetime``).

Run: py ml\\test_submission_transport.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml-service"))

import service as ml_service  # noqa: E402
from src.ml_client import _iso, _telemetry_rows  # noqa: E402

DS = ROOT / "dataset" / "validate"


def test_client_emits_one_canonical_shape() -> None:
    """Every timestamp the client sends parses the same way, fraction or not."""
    values = [
        pd.Timestamp("2026-01-06 03:35:00"),
        pd.Timestamp("2026-01-06 03:35:00.462764"),
        "2026-01-06 03:35:01",
        "2026-01-06T03:35:02",
    ]
    sent = [_iso(value) for value in values]
    assert len(set(sent)) == len(sent), sent
    assert all(" " in item for item in sent), sent
    # Every one of them must survive the receiver's parser.
    parsed = pd.to_datetime(pd.Series(sent), errors="coerce", format="ISO8601")
    assert int(parsed.isna().sum()) == 0, list(parsed)
    # And the format is fixed-width, which is the actual invariant.
    assert all(len(item) == 26 for item in sent), sent
    print(f"   client:     canonical shape, e.g. {sent[1]!r}")


def test_client_rejects_unusable_timestamps() -> None:
    for bad in (None, "", "   ", "not-a-date"):
        try:
            _iso(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should have been rejected")
    print("   client:     rejects None / empty / unparseable")


def test_service_parses_mixed_shapes() -> None:
    """The receiver must not infer one format for a heterogeneous column."""
    column = pd.Series(
        [
            "2026-01-06T03:35:00",              # no fraction, T separator
            "2026-01-06T03:35:00.462764",       # fraction, T separator
            "2026-01-06 03:35:01",              # space separator
            "2026-01-06 03:35:02.500000",        # space separator, canonical
        ]
    )
    parsed = ml_service._ensure_datetime(column)
    assert int(parsed.isna().sum()) == 0, list(parsed)
    assert pd.Timestamp(parsed.iloc[1]).microsecond == 462764, list(parsed)
    print("   service:    parses T/space x with/without fraction, keeps microseconds")


def test_service_drops_only_genuinely_broken_rows() -> None:
    column = pd.Series(["2026-01-06 03:35:00", "not-a-date", None, "2026-01-06 03:35:01"])
    parsed = ml_service._ensure_datetime(column)
    assert int(parsed.isna().sum()) == 2, list(parsed)
    assert not pd.isna(parsed.iloc[0]) and not pd.isna(parsed.iloc[3])
    print("   service:    only truly unparseable values become NaT")


def test_full_telemetry_survives_the_json_round_trip() -> None:
    """The end-to-end property: no telemetry row is lost on the way to the model.

    This is the assertion that failed before the fix, and it is the one that
    matters -- every windowed feature is computed from these rows, so a silent
    drop changes predictions without any visible error.
    """
    if not (DS / "traffic.csv").exists():
        print("   skipped:   dataset/validate/traffic.csv отсутствует")
        return
    raw = pd.read_csv(DS / "traffic.csv")
    direct = ml_service.normalise_telemetry(raw)
    routed = ml_service.normalise_telemetry(
        pd.DataFrame(_telemetry_rows(ml_service.normalise_telemetry(raw)))
    )
    assert len(direct) == len(routed), (
        f"telemetry lost in transport: {len(direct)} -> {len(routed)} rows"
    )
    assert list(direct.columns) == list(routed.columns)
    for column in direct.columns:
        left = direct[column].reset_index(drop=True)
        right = routed[column].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(left):
            assert np.array_equal(
                left.to_numpy(dtype=float), right.to_numpy(dtype=float), equal_nan=True
            ), column
        else:
            assert left.astype(str).equals(right.astype(str)), column
    sub_second = int((pd.to_datetime(raw["event_time"]).dt.microsecond > 0).sum())
    print(
        f"   transport:  {len(direct)} rows preserved end-to-end "
        f"({sub_second} of them sub-second)"
    )


def test_predictions_match_across_the_transport() -> None:
    """The user-visible consequence: local and routed predictions must agree."""
    if not (DS / "points.csv").exists():
        print("   skipped:   dataset/validate/points.csv отсутствует")
        return
    from ml.ml_pipeline import load_engine

    engine = load_engine(ROOT)
    points = pd.read_csv(DS / "points.csv")
    traffic = pd.read_csv(DS / "traffic.csv")
    direct = [r["prediction"] for r in engine.predict(points.to_dict("records"), traffic)]
    routed = [
        r["prediction"]
        for r in engine.predict(
            points.to_dict("records"),
            pd.DataFrame(_telemetry_rows(ml_service.normalise_telemetry(traffic))),
        )
    ]
    assert len(direct) == len(routed) == len(points), (len(direct), len(routed))
    worst = max(abs(a - b) for a, b in zip(direct, routed))
    assert worst == 0.0, f"transport changed a prediction by {worst}"
    print(f"   predictions: local == routed for all {len(direct)} rows (max diff {worst:.1e})")


def main() -> None:
    tests = [
        test_client_emits_one_canonical_shape,
        test_client_rejects_unusable_timestamps,
        test_service_parses_mixed_shapes,
        test_service_drops_only_genuinely_broken_rows,
        test_full_telemetry_survives_the_json_round_trip,
        test_predictions_match_across_the_transport,
    ]
    for test in tests:
        print(f"[run ] {test.__name__}")
        test()
    print("ALL SUBMISSION-TRANSPORT TESTS PASSED")


if __name__ == "__main__":
    main()

"""The submission pipeline: one format, one place to write it, one validator.

Before this module, three call sites each wrote ``submission.csv`` with their own
copy of the format rules -- ``ml/baseline_v2.py`` (training), ``ml/ml_pipeline``
(offline) and ``backend/main.py`` (HTTP). They happened to agree, which is exactly
the condition under which a silent divergence goes unnoticed. Format now lives
here; the other call sites delegate.

Format contract
----------------
* header exactly ``sample_id;prediction``
* ``;`` separator, UTF-8, no index column
* one row per point in ``dataset/validate/points.csv``, **in that file's order**
  -- the platform matches rows by position as well as by id
* ``sample_id`` copied verbatim; ``prediction`` finite, ``%.6f``

Usage
-----
    from ml.submission import write_submission
    report = write_submission(validate_points, predictions, destination)
    assert report.ok
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

#: Column names, in order. Anything else is a format error.
COLUMNS = ("sample_id", "prediction")
SEPARATOR = ";"
FLOAT_FORMAT = "%.6f"

#: Prediction sanity band. The model is a delay regressor in seconds; a value
#: outside this range is not a "rare delay", it is a broken input. 86 400 s is a
#: day, so the band can only be crossed by a bug.
PLAUSIBLE_MIN_S = -86_400.0
PLAUSIBLE_MAX_S = 86_400.0

#: The competition's forecast horizon, in seconds: (T + 10 min, T + 15 min].
HORIZON_MIN_S = 600
HORIZON_MAX_S = 900


class SubmissionFormatError(ValueError):
    """Raised when the input cannot produce a valid submission."""


@dataclass
class SubmissionReport:
    """Result of building and validating a submission file."""

    path: str
    rows: int = 0
    expected_rows: int = 0
    unique_ids: int = 0
    duplicate_ids: int = 0
    missing_ids: int = 0
    unexpected_ids: int = 0
    out_of_order: int = 0
    non_finite: int = 0
    non_numeric: int = 0
    empty_ids: int = 0
    out_of_range: int = 0
    horizon_violations: int = 0
    min_prediction: float | None = None
    max_prediction: float | None = None
    mean_prediction: float | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the file is submittable. Warnings do not block submission."""
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "rows": self.rows,
            "expected_rows": self.expected_rows,
            "unique_ids": self.unique_ids,
            "duplicate_ids": self.duplicate_ids,
            "missing_ids": self.missing_ids,
            "unexpected_ids": self.unexpected_ids,
            "out_of_order": self.out_of_order,
            "non_finite": self.non_finite,
            "non_numeric": self.non_numeric,
            "empty_ids": self.empty_ids,
            "out_of_range": self.out_of_range,
            "horizon_violations": self.horizon_violations,
            "min_prediction": self.min_prediction,
            "max_prediction": self.max_prediction,
            "mean_prediction": self.mean_prediction,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def validate_submission(
    path: str | Path,
    *,
    reference_points: Any = None,
) -> SubmissionReport:
    """Inspect a submission file and report every problem found.

    ``reference_points`` is the ``points.csv`` DataFrame. When given, the ids, the
    row count, the row order and the 10-15 minute horizon are all checked
    against it, which is the only way to catch a submission that is well-formed
    but attached to the wrong points.
    """
    path = Path(path)
    report = SubmissionReport(path=str(path))
    if not path.is_file():
        report.errors.append(f"file not found: {path}")
        return report

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        report.errors.append("file is empty")
        return report

    header = lines[0].split(SEPARATOR)
    if tuple(header) != COLUMNS:
        report.errors.append(
            f"header is {header!r}, expected {list(COLUMNS)!r} joined by {SEPARATOR!r}"
        )
        return report

    ids: list[str] = []
    values: list[float] = []
    for line_no, line in enumerate(lines[1:], start=2):
        if not line.strip():
            report.warnings.append(f"line {line_no}: blank line skipped")
            continue
        parts = line.split(SEPARATOR)
        if len(parts) != 2:
            report.non_numeric += 1
            report.errors.append(f"line {line_no}: expected 2 fields, got {len(parts)}")
            continue
        sample_id, raw = parts[0].strip(), parts[1].strip()
        if not sample_id:
            report.empty_ids += 1
            report.errors.append(f"line {line_no}: empty sample_id")
        ids.append(sample_id)
        try:
            value = float(raw)
        except ValueError:
            report.non_numeric += 1
            report.errors.append(f"line {line_no}: prediction {raw!r} is not a number")
            values.append(math.nan)
            continue
        if not math.isfinite(value):
            report.non_finite += 1
            report.errors.append(f"line {line_no}: prediction {raw!r} is not finite")
        values.append(value)

    report.rows = len(ids)
    report.unique_ids = len(set(ids))
    report.duplicate_ids = report.rows - report.unique_ids
    if report.duplicate_ids:
        report.errors.append(f"{report.duplicate_ids} duplicate sample_id")

    finite = [v for v in values if math.isfinite(v)]
    if finite:
        report.min_prediction = min(finite)
        report.max_prediction = max(finite)
        report.mean_prediction = sum(finite) / len(finite)
        out = [v for v in finite if not PLAUSIBLE_MIN_S <= v <= PLAUSIBLE_MAX_S]
        report.out_of_range = len(out)
        if out:
            report.errors.append(
                f"{len(out)} prediction(s) outside the plausible band "
                f"[{PLAUSIBLE_MIN_S:.0f}, {PLAUSIBLE_MAX_S:.0f}] s"
            )
    elif report.rows:
        report.errors.append("no finite predictions")

    if reference_points is None:
        return report

    expected = [str(value) for value in reference_points["sample_id"]]
    report.expected_rows = len(expected)
    if report.rows != report.expected_rows:
        report.errors.append(
            f"row count {report.rows} != {report.expected_rows} in points.csv"
        )
    expected_set = set(expected)
    actual_set = set(ids)
    report.missing_ids = len(expected_set - actual_set)
    report.unexpected_ids = len(actual_set - expected_set)
    if report.missing_ids:
        report.errors.append(f"{report.missing_ids} sample_id from points.csv are missing")
    if report.unexpected_ids:
        report.errors.append(f"{report.unexpected_ids} sample_id are not in points.csv")
    shared = min(len(ids), len(expected))
    report.out_of_order = sum(1 for a, b in zip(ids[:shared], expected[:shared]) if a != b)
    if report.out_of_order:
        report.errors.append(
            f"{report.out_of_order} row(s) are out of points.csv order "
            "(rows are matched by position as well as by id)"
        )
    report.horizon_violations = _count_horizon_violations(reference_points)
    if report.horizon_violations:
        report.errors.append(
            f"{report.horizon_violations} point(s) outside the "
            f"({HORIZON_MIN_S}, {HORIZON_MAX_S}] s horizon"
        )
    return report


def _count_horizon_violations(reference_points: Any) -> int:
    """How many reference points fall outside the 10-15 minute target window."""
    import pandas as pd

    try:
        lead = (
            pd.to_datetime(reference_points["target_time_begin"])
            - pd.to_datetime(reference_points["T"])
        ).dt.total_seconds()
    except (KeyError, TypeError, ValueError):
        return 0
    return int(((lead <= HORIZON_MIN_S) | (lead > HORIZON_MAX_S)).sum())


def build_submission_frame(
    points: Any,
    predictions: Sequence[float],
) -> Any:
    """Assemble the two-column frame, refusing anything that cannot be written.

    The row count and the id order are checked *here* rather than after writing,
    so a mismatch is a loud error next to its cause instead of a subtly wrong
    file discovered by the platform.
    """
    import pandas as pd

    ids = [str(value) for value in points["sample_id"]]
    if len(predictions) != len(ids):
        raise SubmissionFormatError(
            f"got {len(predictions)} predictions for {len(ids)} points; "
            "the submission must have exactly one row per point"
        )
    values = []
    for index, raw in enumerate(predictions):
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise SubmissionFormatError(
                f"prediction {index} is not a number: {raw!r}"
            ) from exc
        if not math.isfinite(value):
            raise SubmissionFormatError(
                f"prediction {index} is not finite: {raw!r}"
            )
        values.append(value)
    return pd.DataFrame({"sample_id": ids, "prediction": values}, columns=list(COLUMNS))


def write_submission(
    points: Any,
    predictions: Sequence[float],
    destination: str | Path,
) -> SubmissionReport:
    """Write the submission and immediately validate what landed on disk."""
    frame = build_submission_frame(points, predictions)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        destination, index=False, sep=SEPARATOR, float_format=FLOAT_FORMAT,
        encoding="utf-8", lineterminator="\n",
    )
    return validate_submission(destination, reference_points=points)

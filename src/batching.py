"""Request coalescing for the prediction path.

Why micro-batching
------------------
Profiling showed that a single-row ``predict_frame`` costs ~40 ms while a
50-row batch costs ~25 ms per row, because feature building carries a large
fixed cost (schedule merge, telemetry slicing, dataframe concatenation) and
almost none per row.  The CatBoost model itself is 0.13 ms per row, so the
model was never the problem.

That makes concurrent single-row requests pathological: N simultaneous clients
each pay the full fixed cost and the GIL serialises their pandas work anyway.
:class:`MicroBatcher` lets them share one call instead.

Why the merge is causally sound
--------------------------------
Batching is only legal if merging two requests cannot change either answer.
It cannot, and the argument is short:

* the per-point causal cut lives in
  :func:`src.predictor._vehicle_telemetry_features` and is
  ``end = searchsorted(times, T, side='right')`` -- it depends only on that
  point's own ``T``;
* ``predict_frame`` pre-trims telemetry to ``max(T)`` over the batch, which can
  only *add* rows relative to any individual request;
* a row added by the union is visible to a point only if
  ``event_time <= that point's T``, in which case the other request legitimately
  had that observation at its own ``T`` anyway.

Appending genuinely future rows is therefore a no-op, and unioning the telemetry
of concurrent requests is exact.  Both claims are asserted in
``ml/test_batching.py`` (max abs difference 0.0).

A batch therefore returns byte-identical predictions to running the same
requests one by one -- the win is purely in fixed-cost amortisation, never in
approximation.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import pandas as pd

from src.runtime import InferenceService, _normalise_telemetry


@dataclass
class _Job:
    """One pending caller: its points, its telemetry, and where to put results."""

    points: list[dict[str, Any]]
    telemetry: pd.DataFrame | None
    future: asyncio.Future
    offset: int = 0
    count: int = 0


@dataclass
class BatcherStats:
    """Counters exposed through ``GET /metrics``."""

    batches: int = 0
    jobs: int = 0
    rows: int = 0
    empty_batches: int = 0
    closed: bool = False
    histogram: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "batches": self.batches,
            "jobs": self.jobs,
            "rows": self.rows,
            "rows_per_batch": round(self.rows / self.batches, 2) if self.batches else 0.0,
            "jobs_per_batch": round(self.jobs / self.batches, 2) if self.batches else 0.0,
            "empty_batches": self.empty_batches,
            "closed": self.closed,
        }


class MicroBatcher:
    """Coalesce concurrent prediction requests into a single model call.

    Parameters
    ----------
    service : InferenceService
        Provides the model and the pre-normalised default telemetry.
    max_batch_rows : int
        Upper bound on points per batch.  Beyond this the batch is flushed, which
        caps peak memory: the feature frame is roughly 150 float64 columns per
        row, so 256 rows is a few MB, not a leak.
    window_ms : float
        Collection window.  Small enough that a single uncached request pays
        almost nothing extra, large enough to catch a burst from a dashboard
        that fires several calls at once.
    max_concurrency : int
        How many inference calls may run at once.  Feature building is pure
        Python and holds the GIL, so >1 buys nothing and multiplies peak RAM;
        the default of 1 is deliberate.
    """

    def __init__(
        self,
        service: InferenceService,
        *,
        max_batch_rows: int = 256,
        window_ms: float = 4.0,
        max_concurrency: int = 1,
    ) -> None:
        self._service = service
        self._max_batch_rows = max(1, int(max_batch_rows))
        self._window = max(0.0, float(window_ms)) / 1000.0
        self._queue: list[_Job] = []
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        self.stats = BatcherStats()

    # -- lifecycle --------------------------------------------------------- #

    async def start(self) -> None:
        """Start the drain task.  Idempotent."""
        if self._task is None or self._task.done():
            self.stats.closed = False
            self._task = asyncio.create_task(self._drain())

    async def stop(self) -> None:
        """Stop the drain task and fail anything still queued."""
        self.stats.closed = True
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        pending, self._queue = self._queue, []
        for job in pending:
            if not job.future.done():
                job.future.set_exception(RuntimeError("batcher stopped"))

    # -- public API -------------------------------------------------------- #

    async def predict_records(
        self,
        points: Iterable[dict[str, Any]],
        shared_telemetry: Iterable[dict[str, Any]] | pd.DataFrame | None = None,
        *,
        bypass: bool = False,
    ) -> list[dict[str, Any]]:
        """Predict for one caller, joining a shared batch when possible.

        ``bypass=True`` runs the call inline, which the replay simulator uses: it
        already batches internally, and making it queue behind the drain loop
        would add a full window of latency per tick for no coalescing benefit.
        """
        point_rows = [dict(point) for point in points]
        if not point_rows:
            return []
        if bypass or self._task is None or self._window <= 0.0:
            return await self._run_now(point_rows, shared_telemetry)

        loop = asyncio.get_running_loop()
        job = _Job(points=point_rows, telemetry=_as_frame(shared_telemetry), future=loop.create_future())
        self._queue.append(job)
        self.stats.jobs += 1
        self._wake.set()
        return await job.future

    # -- internals --------------------------------------------------------- #

    async def _run_now(self, points, telemetry):
        async with self._semaphore:
            return await asyncio.to_thread(
                self._service.predict_records, points, telemetry
            )

    async def _drain(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            if self._queue:
                # Let a burst arrive instead of flushing on the first setter.
                await asyncio.sleep(self._window)
            jobs, self._queue = self._queue, []
            if not jobs:
                continue
            try:
                await self._execute(jobs)
            except Exception as exc:  # noqa: BLE001 - never kill the worker
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(exc)

    async def _execute(self, jobs: list[_Job]) -> None:
        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        offset = 0
        for job in jobs:
            job.offset = offset
            job.count = len(job.points)
            rows.extend(job.points)
            offset += job.count

        if len(rows) > self._max_batch_rows:
            overflow = rows[self._max_batch_rows :]
            rows = rows[: self._max_batch_rows]
            overflow_jobs = [job for job in jobs if job.offset >= self._max_batch_rows]
            jobs = [job for job in jobs if job.offset < self._max_batch_rows]
        else:
            overflow = []
            overflow_jobs = []

        telemetry = _merge_telemetry([job.telemetry for job in jobs])

        async with self._semaphore:
            try:
                records = await asyncio.to_thread(
                    self._service.predict_records, rows, telemetry
                )
            except Exception as exc:  # noqa: BLE001
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(exc)
                for job in overflow_jobs:
                    if not job.future.done():
                        job.future.set_result(
                            await asyncio.to_thread(
                                self._service.predict_records,
                                [row for row in overflow],
                                _merge_telemetry([job.telemetry]),
                            )
                        )
                return

        for job in jobs:
            if job.future.done():
                continue
            chunk = records[job.offset : job.offset + job.count]
            job.future.set_result(chunk)

        if overflow_jobs:
            for job in overflow_jobs:
                if job.future.done():
                    continue
                job.future.set_result(
                    await asyncio.to_thread(
                        self._service.predict_records,
                        overflow,
                        _merge_telemetry([job.telemetry]),
                    )
                )

        self.stats.batches += 1
        self.stats.rows += len(rows)
        if not rows:
            self.stats.empty_batches += 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        bucket = _bucket(elapsed_ms)
        self.stats.histogram[bucket] = self.stats.histogram.get(bucket, 0) + 1


def _as_frame(telemetry) -> pd.DataFrame | None:
    if telemetry is None:
        return None
    if isinstance(telemetry, pd.DataFrame):
        return telemetry
    rows = list(telemetry)
    return pd.DataFrame(rows) if rows else None


def _merge_telemetry(frames: Sequence[pd.DataFrame | None]) -> pd.DataFrame | None:
    """Union of the callers' telemetry frames.

    Returns ``None`` when every caller relies on the service default, which is
    the fast path: no concatenation at all.  Distinct frames are unioned because
    that cannot change any result (see the module docstring); repeated copies of
    the *same* frame object are taken once.
    """
    unique: list[pd.DataFrame] = []
    seen: set[int] = set()
    uses_default = False
    for frame in frames:
        if frame is None:
            uses_default = True
            continue
        key = id(frame)
        if key in seen:
            continue
        seen.add(key)
        unique.append(frame)
    if uses_default:
        return None
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    return pd.concat(unique, ignore_index=True)


def _bucket(elapsed_ms: float) -> str:
    for edge in (5, 10, 25, 50, 100, 250, 500, 1000):
        if elapsed_ms < edge:
            return f"<{edge}ms"
    return ">=1000ms"

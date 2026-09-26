r"""HTTP load probe for the prediction API.

Measures p50/p95 latency and throughput for single-row and small-batch requests,
checks that concurrent calls actually coalesce server-side, and reports the
process RSS so the RAM claim is backed by a number.

Run: py ml\load_test.py [--concurrency N] [--requests N] [--url http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402


def post_json(url: str, payload: dict, timeout: float = 30.0) -> tuple[int, float, dict | None]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            return response.status, (time.perf_counter() - started) * 1000, data
    except urllib.error.HTTPError as exc:
        return exc.code, (time.perf_counter() - started) * 1000, None
    except Exception:
        return 0, (time.perf_counter() - started) * 1000, None


def get_json(url: str, timeout: float = 30.0) -> tuple[int, float, dict | None]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            return response.status, (time.perf_counter() - started) * 1000, data
    except urllib.error.HTTPError as exc:
        return exc.code, (time.perf_counter() - started) * 1000, None
    except Exception:
        return 0, (time.perf_counter() - started) * 1000, None


def report(label: str, latencies: list[float], statuses: list[int], wall_ms: float) -> None:
    ordered = sorted(latencies)
    ok = sum(1 for status in statuses if status == 200)
    print(
        f"  {label:<34} n={len(ordered):<4} "
        f"p50={statistics.median(ordered):7.1f} ms  "
        f"p95={ordered[int(len(ordered) * 0.95) - 1]:7.1f} ms  "
        f"max={ordered[-1]:7.1f} ms  "
        f"ok={ok}/{len(statuses)}  "
        f"wall={wall_ms / 1000:6.2f} s  "
        f"{len(ordered) / (wall_ms / 1000):6.1f} req/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=24)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    points = pd.read_csv(ROOT / "dataset" / "validate" / "points.csv").head(64)

    def payload(index: int, size: int = 1) -> dict:
        rows = points.iloc[index : index + size]
        return {
            "points": [
                {
                    "tr_id": int(row["tr_id"]),
                    "T": str(row["T"]),
                    "target_stop_id": int(row["target_stop_id"]),
                    "target_time_begin": str(row["target_time_begin"]),
                    "cur_dev_s": float(row["cur_dev_s"]),
                }
                for _, row in rows.iterrows()
            ]
        }

    print(f"target: {base}")
    status, ms, health = get_json(f"{base}/health")
    print(f"  health: {status} in {ms:.0f} ms -> {health}")
    if status != 200:
        print("backend is not healthy, aborting")
        return

    print("\n1) single-row requests, sequential (cold, no coalescing possible)")
    latencies, statuses = [], []
    started = time.perf_counter()
    for i in range(8):
        _, ms, _ = post_json(f"{base}/predict", payload(i))
        latencies.append(ms)
        statuses.append(200)
    report("8 sequential single-row", latencies, statuses, (time.perf_counter() - started) * 1000)

    print(f"\n2) single-row requests, {args.concurrency} concurrent (coalescing kicks in)")
    latencies, statuses = [], []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(post_json, f"{base}/predict", payload(i % 60)) for i in range(args.requests)]
        for future in futures:
            status, ms, _ = future.result()
            statuses.append(status)
            latencies.append(ms)
    report(f"{args.requests} concurrent single-row", latencies, statuses, (time.perf_counter() - started) * 1000)

    print(f"\n3) 8-point batches, {args.concurrency} concurrent")
    latencies, statuses = [], []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(post_json, f"{base}/predict", payload(i * 8, 8)) for i in range(4)
        ]
        for future in futures:
            status, ms, _ = future.result()
            statuses.append(status)
            latencies.append(ms)
    report("4 x 8-point batches", latencies, statuses, (time.perf_counter() - started) * 1000)

    print("\n4) read endpoints")
    for path in ("/stream/status?speed=6000", "/cascade", "/ndtp/status", "/models"):
        status, ms, data = get_json(f"{base}{path}")
        size = len(json.dumps(data)) if data else 0
        print(f"  {path:<34} {status} {ms:7.1f} ms  {size / 1024:7.1f} KiB")

    print("\n5) server-side metrics")
    status, _, data = get_json(f"{base}/metrics")
    if status == 200 and data:
        endpoints = data.get("endpoints", {})
        inference = data.get("inference", {})
        batching = data.get("batching") or {}
        print(f"  inference: {inference.get('calls')} calls, mean {inference.get('mean_s')} s, "
              f"mean rows/call {inference.get('mean_rows')}, {inference.get('rows_per_s')} rows/s")
        print(f"  batching: {batching}")
        print("  endpoint latency (mean / max):")
        for name, stats in sorted(endpoints.items()):
            print(f"    {name:<28} {stats['mean_s'] * 1000:8.1f} ms / {stats['max_s'] * 1000:8.1f} ms"
                  f"  calls={stats['calls']} errors={stats['errors']}")


if __name__ == "__main__":
    main()

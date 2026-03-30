"""
Benchmark the MLflow AI Gateway.

Uses aiohttp with connection pooling to measure throughput and latency
under concurrent load against a running MLflow server with a gateway endpoint.

Usage:
    python benchmark_gateway.py --url http://127.0.0.1:5000/gateway/benchmark-chat/mlflow/invocations
    python benchmark_gateway.py --url http://... --requests 5000 --max-concurrent 100 --runs 3
"""

import argparse
import asyncio
import statistics
import time

import aiohttp

BODY = {
    "messages": [{"role": "user", "content": "benchmark request"}],
    "temperature": 0.0,
    "max_tokens": 50,
}

HEADERS = {
    "Content-Type": "application/json",
}


async def send_request(session, url, semaphore, failure_counts):
    async with semaphore:
        start = time.perf_counter()
        try:
            async with session.post(url, json=BODY, headers=HEADERS) as resp:
                await resp.read()
                elapsed = time.perf_counter() - start
                if resp.status != 200:
                    key = f"HTTP {resp.status}"
                    failure_counts[key] = failure_counts.get(key, 0) + 1
                    return None
                return elapsed
        except Exception as e:
            key = type(e).__name__
            failure_counts[key] = failure_counts.get(key, 0) + 1
            return None


async def run_benchmark(url, n_requests, max_concurrent):
    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(
        limit=max(max_concurrent * 2, 200),
        limit_per_host=max(max_concurrent, 200),
        force_close=False,
        enable_cleanup_closed=True,
    )
    failure_counts = {}
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        warmup_n = min(max(50, max_concurrent), n_requests)
        warmup_failures = {}
        await asyncio.gather(*[
            send_request(session, url, semaphore, warmup_failures) for _ in range(warmup_n)
        ])

        # Timed run
        wall_start = time.perf_counter()
        results = await asyncio.gather(*[
            send_request(session, url, semaphore, failure_counts) for _ in range(n_requests)
        ])
        wall_elapsed = time.perf_counter() - wall_start

    latencies = sorted(r for r in results if r is not None)
    failures = sum(1 for r in results if r is None)
    n = len(latencies)

    if not latencies:
        return {
            "mean": 0,
            "p50": 0,
            "p95": 0,
            "p99": 0,
            "throughput": 0,
            "failures": n_requests,
            "failure_breakdown": dict(failure_counts),
            "wall_time": wall_elapsed,
            "n_requests": n_requests,
        }

    return {
        "mean": statistics.mean(latencies) * 1000,
        "p50": latencies[n // 2] * 1000,
        "p95": latencies[int(n * 0.95)] * 1000,
        "p99": latencies[int(n * 0.99)] * 1000,
        "throughput": n_requests / wall_elapsed,
        "failures": failures,
        "failure_breakdown": dict(failure_counts),
        "wall_time": wall_elapsed,
        "n_requests": n_requests,
    }


async def main():
    parser = argparse.ArgumentParser(description="MLflow AI Gateway benchmark")
    parser.add_argument("--url", required=True, help="MLflow gateway invocation URL")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    print(f"\nBenchmarking MLflow Gateway at {args.url}")
    print(f"  {args.requests} requests, {args.max_concurrent} concurrency, {args.runs} run(s)")

    all_results = []
    for run_num in range(1, args.runs + 1):
        result = await run_benchmark(args.url, args.requests, args.max_concurrent)
        all_results.append(result)
        failure_detail = ""
        if result["failures"] > 0 and result.get("failure_breakdown"):
            parts = [f"{k}={v}" for k, v in result["failure_breakdown"].items()]
            failure_detail = f" ({', '.join(parts)})"
        print(
            f"  Run {run_num}/{args.runs}: {result['throughput']:.0f} rps, "
            f"p50={result['p50']:.1f}ms, p99={result['p99']:.1f}ms, "
            f"failures={result['failures']}{failure_detail}"
        )

    # Aggregate across runs
    total_failures = sum(r["failures"] for r in all_results)
    total_requests = sum(r["n_requests"] for r in all_results)
    all_wall_times = [r["wall_time"] for r in all_results]
    avg_tp = total_requests / sum(all_wall_times) if sum(all_wall_times) > 0 else 0

    # Compute percentiles from per-run stats (weighted by request count)
    avg_p50 = statistics.mean(r["p50"] for r in all_results)
    avg_p95 = statistics.mean(r["p95"] for r in all_results)
    avg_p99 = statistics.mean(r["p99"] for r in all_results)
    avg_mean = statistics.mean(r["mean"] for r in all_results)

    print(f"\n{'=' * 60}")
    print(f"  MLflow Gateway ({args.runs} runs, {total_requests} total requests)")
    print(f"{'=' * 60}")
    print(f"  Failures:    {total_failures}")
    if total_failures > 0:
        combined = {}
        for r in all_results:
            for k, v in r.get("failure_breakdown", {}).items():
                combined[k] = combined.get(k, 0) + v
        for reason, count in sorted(combined.items(), key=lambda x: -x[1]):
            print(f"    - {reason}: {count}")
    print(f"  Throughput:  {avg_tp:.0f} req/s")
    print(f"  Mean:        {avg_mean:.2f} ms")
    print(f"  P50:         {avg_p50:.2f} ms")
    print(f"  P95:         {avg_p95:.2f} ms")
    print(f"  P99:         {avg_p99:.2f} ms")


if __name__ == "__main__":
    asyncio.run(main())

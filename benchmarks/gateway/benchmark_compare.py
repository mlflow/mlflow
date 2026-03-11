"""
Head-to-head benchmark: MLflow Gateway vs LiteLLM proxy.

Uses the same aiohttp-based methodology as LiteLLM's benchmark_mock.py
for a fair comparison. Both proxies hit the same fake OpenAI server.

Usage:
    # Start fake server + both proxies first, then:
    python benchmark_compare.py --target mlflow --requests 2000 --max-concurrent 50 --runs 3
    python benchmark_compare.py --target litellm --requests 2000 --max-concurrent 50 --runs 3
    python benchmark_compare.py --target both --requests 2000 --max-concurrent 50 --runs 3
"""

import argparse
import asyncio
import statistics
import time

import aiohttp

MLFLOW_URL = "http://127.0.0.1:5000/gateway/benchmark-chat/mlflow/invocations"
LITELLM_URL = "http://127.0.0.1:4000/chat/completions"

MLFLOW_BODY = {
    "messages": [{"role": "user", "content": "benchmark request"}],
    "temperature": 0.0,
    "max_tokens": 50,
}

LITELLM_BODY = {
    "model": "benchmark-chat",
    "messages": [{"role": "user", "content": "benchmark request"}],
    "max_tokens": 50,
}

LITELLM_HEADERS = {
    "Authorization": "Bearer sk-1234",
    "Content-Type": "application/json",
}

MLFLOW_HEADERS = {
    "Content-Type": "application/json",
}


async def send_request(session, url, body, headers, semaphore):
    async with semaphore:
        start = time.perf_counter()
        try:
            async with session.post(url, json=body, headers=headers) as resp:
                await resp.read()
                elapsed = time.perf_counter() - start
                return elapsed if resp.status == 200 else None
        except Exception:
            return None


async def run_benchmark(url, body, headers, n_requests, max_concurrent):
    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(
        limit=min(max_concurrent * 2, 200),
        limit_per_host=max_concurrent,
        force_close=False,
        enable_cleanup_closed=True,
    )
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        warmup_n = min(50, n_requests)
        await asyncio.gather(*[
            send_request(session, url, body, headers, semaphore) for _ in range(warmup_n)
        ])

        # Timed run
        wall_start = time.perf_counter()
        results = await asyncio.gather(*[
            send_request(session, url, body, headers, semaphore) for _ in range(n_requests)
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
            "wall_time": wall_elapsed,
            "n_requests": n_requests,
            "latencies": [],
        }

    return {
        "mean": statistics.mean(latencies) * 1000,
        "p50": latencies[n // 2] * 1000,
        "p95": latencies[int(n * 0.95)] * 1000,
        "p99": latencies[int(n * 0.99)] * 1000,
        "throughput": n_requests / wall_elapsed,
        "failures": failures,
        "wall_time": wall_elapsed,
        "n_requests": n_requests,
        "latencies": latencies,
    }


def print_results(name, results):
    all_latencies = sorted(lat for r in results for lat in r["latencies"])
    total_failures = sum(r["failures"] for r in results)
    total_requests = sum(r["n_requests"] for r in results)
    n = len(all_latencies)

    if not all_latencies:
        print(f"\n  {name}: all {total_requests} requests failed")
        return

    mean = statistics.mean(all_latencies) * 1000
    p50 = all_latencies[n // 2] * 1000
    p95 = all_latencies[int(n * 0.95)] * 1000
    p99 = all_latencies[int(n * 0.99)] * 1000
    avg_tp = statistics.mean(r["throughput"] for r in results)

    print(f"\n{'=' * 60}")
    print(f"  {name} ({len(results)} runs, {total_requests} total requests)")
    print(f"{'=' * 60}")
    print(f"  Failures:    {total_failures}")
    print(f"  Throughput:  {avg_tp:.0f} req/s")
    print(f"  Mean:        {mean:.2f} ms")
    print(f"  P50:         {p50:.2f} ms")
    print(f"  P95:         {p95:.2f} ms")
    print(f"  P99:         {p99:.2f} ms")

    for i, r in enumerate(results, 1):
        print(
            f"    Run {i}: {r['throughput']:.0f} rps, p50={r['p50']:.1f}ms, "
            f"p99={r['p99']:.1f}ms, failures={r['failures']}"
        )


def print_comparison(mlflow_results, litellm_results):
    def agg(results):
        lats = sorted(lat for r in results for lat in r["latencies"])
        n = len(lats)
        if not lats:
            return {}
        return {
            "p50": lats[n // 2] * 1000,
            "p95": lats[int(n * 0.95)] * 1000,
            "p99": lats[int(n * 0.99)] * 1000,
            "throughput": statistics.mean(r["throughput"] for r in results),
            "failures": sum(r["failures"] for r in results),
        }

    m = agg(mlflow_results)
    l = agg(litellm_results)

    if not m or not l:
        return

    print(f"\n{'=' * 60}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<20} {'MLflow Gateway':>15} {'LiteLLM':>15}")
    print(f"  {'-' * 50}")
    print(f"  {'P50 (ms)':<20} {m['p50']:>15.1f} {l['p50']:>15.1f}")
    print(f"  {'P95 (ms)':<20} {m['p95']:>15.1f} {l['p95']:>15.1f}")
    print(f"  {'P99 (ms)':<20} {m['p99']:>15.1f} {l['p99']:>15.1f}")
    print(f"  {'RPS':<20} {m['throughput']:>15.0f} {l['throughput']:>15.0f}")
    print(f"  {'Failures':<20} {m['failures']:>15} {l['failures']:>15}")


async def bench_target(name, url, body, headers, n_requests, max_concurrent, runs):
    print(f"\nBenchmarking {name} at {url}")
    print(f"  {n_requests} requests, {max_concurrent} concurrency, {runs} run(s)")

    results = []
    for run_num in range(1, runs + 1):
        result = await run_benchmark(url, body, headers, n_requests, max_concurrent)
        results.append(result)
        print(
            f"  Run {run_num}/{runs}: {result['throughput']:.0f} rps, "
            f"p50={result['p50']:.1f}ms, p99={result['p99']:.1f}ms, "
            f"failures={result['failures']}"
        )

    print_results(name, results)
    return results


async def main():
    parser = argparse.ArgumentParser(description="MLflow Gateway vs LiteLLM benchmark")
    parser.add_argument(
        "--target",
        choices=["mlflow", "litellm", "both"],
        default="both",
    )
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--mlflow-url", default=MLFLOW_URL)
    parser.add_argument("--litellm-url", default=LITELLM_URL)
    args = parser.parse_args()

    mlflow_results = None
    litellm_results = None

    if args.target in ("mlflow", "both"):
        mlflow_results = await bench_target(
            "MLflow Gateway",
            args.mlflow_url,
            MLFLOW_BODY,
            MLFLOW_HEADERS,
            args.requests,
            args.max_concurrent,
            args.runs,
        )

    if args.target in ("litellm", "both"):
        litellm_results = await bench_target(
            "LiteLLM",
            args.litellm_url,
            LITELLM_BODY,
            LITELLM_HEADERS,
            args.requests,
            args.max_concurrent,
            args.runs,
        )

    if mlflow_results and litellm_results:
        print_comparison(mlflow_results, litellm_results)


if __name__ == "__main__":
    asyncio.run(main())

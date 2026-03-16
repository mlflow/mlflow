"""
Head-to-head benchmark: MLflow Gateway vs LiteLLM proxy vs Portkey AI Gateway.

Uses the same aiohttp-based methodology as LiteLLM's benchmark_mock.py
for a fair comparison. All proxies hit the same fake OpenAI server.

Usage:
    # Start fake server + proxies first, then:
    python benchmark_compare.py --target mlflow --requests 2000 --max-concurrent 50 --runs 3
    python benchmark_compare.py --target litellm --requests 2000 --max-concurrent 50 --runs 3
    python benchmark_compare.py --target portkey --requests 2000 --max-concurrent 50 --runs 3
    python benchmark_compare.py --target both --requests 2000 --max-concurrent 50 --runs 3
    python benchmark_compare.py --target all --requests 2000 --max-concurrent 50 --runs 3
"""

import argparse
import asyncio
import statistics
import time

import aiohttp

MLFLOW_URL = "http://127.0.0.1:5000/gateway/benchmark-chat/mlflow/invocations"
LITELLM_URL = "http://127.0.0.1:4000/chat/completions"
PORTKEY_URL = "http://127.0.0.1:8787/v1/chat/completions"

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

PORTKEY_BODY = {
    "model": "gpt-4o-mini",
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

PORTKEY_HEADERS = {
    "Content-Type": "application/json",
    "x-portkey-provider": "openai",
    "x-portkey-custom-host": "http://127.0.0.1:9000/v1",
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


def print_comparison(targets):
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

    aggregated = [(name, agg(results)) for name, results in targets]
    aggregated = [(name, a) for name, a in aggregated if a]

    if len(aggregated) < 2:
        return

    col_width = max(len(name) for name, _ in aggregated)
    col_width = max(col_width, 15)
    total_width = 20 + col_width * len(aggregated) + 2 * len(aggregated)
    header_width = max(total_width, 60)

    print(f"\n{'=' * header_width}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * header_width}")

    header = f"  {'Metric':<20}"
    for name, _ in aggregated:
        header += f" {name:>{col_width}}"
    print(header)
    print(f"  {'-' * (total_width - 2)}")

    for metric, key, fmt in [
        ("P50 (ms)", "p50", ".1f"),
        ("P95 (ms)", "p95", ".1f"),
        ("P99 (ms)", "p99", ".1f"),
        ("RPS", "throughput", ".0f"),
    ]:
        row = f"  {metric:<20}"
        for _, a in aggregated:
            row += f" {a[key]:{f'>{col_width}{fmt}'}}"
        print(row)

    row = f"  {'Failures':<20}"
    for _, a in aggregated:
        row += f" {a['failures']:>{col_width}}"
    print(row)


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
    parser = argparse.ArgumentParser(description="MLflow Gateway vs LiteLLM vs Portkey benchmark")
    parser.add_argument(
        "--target",
        choices=["mlflow", "litellm", "portkey", "both", "all"],
        default="both",
    )
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--mlflow-url", default=MLFLOW_URL)
    parser.add_argument("--litellm-url", default=LITELLM_URL)
    parser.add_argument("--portkey-url", default=PORTKEY_URL)
    args = parser.parse_args()

    comparison_targets = []

    if args.target in ("mlflow", "both", "all"):
        mlflow_results = await bench_target(
            "MLflow Gateway",
            args.mlflow_url,
            MLFLOW_BODY,
            MLFLOW_HEADERS,
            args.requests,
            args.max_concurrent,
            args.runs,
        )
        comparison_targets.append(("MLflow Gateway", mlflow_results))

    if args.target in ("litellm", "both", "all"):
        litellm_results = await bench_target(
            "LiteLLM",
            args.litellm_url,
            LITELLM_BODY,
            LITELLM_HEADERS,
            args.requests,
            args.max_concurrent,
            args.runs,
        )
        comparison_targets.append(("LiteLLM", litellm_results))

    if args.target in ("portkey", "all"):
        portkey_headers = {**PORTKEY_HEADERS}
        # Allow overriding custom-host via URL to match fake server port
        if args.portkey_url != PORTKEY_URL:
            portkey_headers["x-portkey-custom-host"] = "http://127.0.0.1:9000/v1"
        portkey_results = await bench_target(
            "Portkey",
            args.portkey_url,
            PORTKEY_BODY,
            portkey_headers,
            args.requests,
            args.max_concurrent,
            args.runs,
        )
        comparison_targets.append(("Portkey", portkey_results))

    if len(comparison_targets) >= 2:
        print_comparison(comparison_targets)


if __name__ == "__main__":
    asyncio.run(main())

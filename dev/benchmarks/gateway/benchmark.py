# /// script
# requires-python = ">=3.10"
# dependencies = ["aiohttp>=3.13.3,<4", "rich>=14.3.3,<15"]
# ///
"""Async HTTP benchmark client for the MLflow AI Gateway.

Can be imported by run.py or used standalone:
    uv run benchmark.py --url http://127.0.0.1:5731/gateway/benchmark-chat/mlflow/invocations
    uv run benchmark.py --url http://... --requests 5000 --max-concurrent 100 --runs 3
"""

import argparse
import asyncio
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from rich.console import Console  # type: ignore[import-not-found]
from rich.progress import (  # type: ignore[import-not-found]
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table  # type: ignore[import-not-found]

console = Console()

_BODY = {
    "messages": [{"role": "user", "content": "benchmark request"}],
    "temperature": 0.0,
    "max_tokens": 50,
}


@dataclass
class RunResult:
    latencies_ms: list[float] = field(default_factory=list)
    failures: dict[str, int] = field(default_factory=dict)
    wall_time: float = 0.0

    @property
    def n_success(self) -> int:
        return len(self.latencies_ms)

    @property
    def n_failures(self) -> int:
        return sum(self.failures.values())

    @property
    def throughput(self) -> float:
        return self.n_success / self.wall_time if self.wall_time > 0 else 0.0

    def percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        idx = max(0, math.ceil(p / 100 * len(s)) - 1)
        return s[idx]


async def _send(
    session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore
) -> tuple[float, str | None]:
    async with sem:
        t0 = time.perf_counter()
        try:
            async with session.post(url, json=_BODY) as resp:
                await resp.read()
                ms = (time.perf_counter() - t0) * 1000
                if resp.status == 200:
                    return ms, None
                return ms, f"HTTP {resp.status}"
        except Exception as e:
            return (time.perf_counter() - t0) * 1000, type(e).__name__


async def _run_once(
    url: str, n: int, max_concurrent: int, progress: Progress, task_id: int
) -> RunResult:
    sem = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(
        limit=max(max_concurrent * 2, 200),
        limit_per_host=max(max_concurrent, 200),
        force_close=False,
        enable_cleanup_closed=True,
    )
    result = RunResult()
    total_time = 0.0
    max_time = 0.0

    async with aiohttp.ClientSession(connector=connector) as session:
        t0 = time.perf_counter()
        for coro in asyncio.as_completed([_send(session, url, sem) for _ in range(n)]):
            ms, error = await coro
            if error:
                result.failures[error] = result.failures.get(error, 0) + 1
            else:
                result.latencies_ms.append(ms)
                total_time += ms
                if ms > max_time:
                    max_time = ms

            n_ok = result.n_success
            n_fail = result.n_failures
            mean = total_time / n_ok if n_ok else 0.0
            fail_part = f"[red]✗{n_fail}[/red]  " if n_fail else ""
            live = f"{fail_part}✓{n_ok}  mean={mean:.0f}ms  max={max_time:.0f}ms"
            progress.update(task_id, advance=1, live=live)

        result.wall_time = time.perf_counter() - t0
    return result


async def _warmup(url: str, n: int, max_concurrent: int) -> None:
    sem = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(limit=max(max_concurrent * 2, 200))
    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(*[_send(session, url, sem) for _ in range(n)])


def run_benchmark(
    url: str, n_requests: int = 2000, max_concurrent: int = 50, runs: int = 3
) -> list[RunResult]:
    warmup_n = min(max(50, max_concurrent), n_requests)
    console.print(f"  [dim]Warming up ({warmup_n} requests)...[/dim]")
    asyncio.run(_warmup(url, warmup_n, max_concurrent))

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("  {task.fields[live]}"),
        console=console,
    ) as progress:
        for i in range(runs):
            task_id = progress.add_task(f"  Run {i + 1}/{runs}", total=n_requests, live="")
            results.append(
                asyncio.run(_run_once(url, n_requests, max_concurrent, progress, task_id))
            )
    return results


def results_to_dict(results: list[RunResult]) -> dict[str, Any]:
    runs = [
        {
            "n_success": r.n_success,
            "n_failures": r.n_failures,
            "failures": r.failures,
            "wall_time_s": r.wall_time,
            "mean_ms": statistics.mean(r.latencies_ms) if r.latencies_ms else 0.0,
            "p50_ms": r.percentile(50),
            "p95_ms": r.percentile(95),
            "p99_ms": r.percentile(99),
            "max_ms": max(r.latencies_ms) if r.latencies_ms else 0.0,
            "rps": r.throughput,
        }
        for r in results
    ]
    summary: dict[str, Any] = (
        {
            "avg_mean_ms": statistics.mean(
                statistics.mean(r.latencies_ms) if r.latencies_ms else 0.0 for r in results
            ),
            "avg_p50_ms": statistics.mean(r.percentile(50) for r in results),
            "avg_p99_ms": statistics.mean(r.percentile(99) for r in results),
            "avg_rps": statistics.mean(r.throughput for r in results),
        }
        if results
        else {}
    )
    return {"runs": runs, "summary": summary}


def print_results(results: list[RunResult]) -> None:
    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    table.add_column("Run", style="dim", width=5)
    table.add_column("Mean ms", justify="right")
    table.add_column("P50 ms", justify="right")
    table.add_column("P95 ms", justify="right")
    table.add_column("P99 ms", justify="right")
    table.add_column("Max ms", justify="right")
    table.add_column("Req/s", justify="right")
    table.add_column("Failures", justify="right")

    means = []
    p50s = []
    p95s = []
    p99s = []
    maxes = []
    throughputs = []
    for i, r in enumerate(results):
        mean = statistics.mean(r.latencies_ms) if r.latencies_ms else 0.0
        p50 = r.percentile(50)
        p95 = r.percentile(95)
        p99 = r.percentile(99)
        mx = max(r.latencies_ms) if r.latencies_ms else 0.0
        means.append(mean)
        p50s.append(p50)
        p95s.append(p95)
        p99s.append(p99)
        maxes.append(mx)
        throughputs.append(r.throughput)
        fail_str = f"[red]{r.n_failures}[/red]" if r.n_failures else "0"
        table.add_row(
            str(i + 1),
            f"{mean:.1f}",
            f"{p50:.1f}",
            f"{p95:.1f}",
            f"{p99:.1f}",
            f"{mx:.1f}",
            f"{r.throughput:.0f}",
            fail_str,
        )

    if len(results) > 1:
        table.add_section()
        table.add_row(
            "[bold]avg[/bold]",
            f"[bold]{statistics.mean(means):.1f}[/bold]",
            f"[bold]{statistics.mean(p50s):.1f}[/bold]",
            f"[bold]{statistics.mean(p95s):.1f}[/bold]",
            f"[bold]{statistics.mean(p99s):.1f}[/bold]",
            f"[bold]{statistics.mean(maxes):.1f}[/bold]",
            f"[bold]{statistics.mean(throughputs):.0f}[/bold]",
            "",
        )

    console.print()
    console.print(table)

    combined: dict[str, int] = {}
    for r in results:
        for k, v in r.failures.items():
            combined[k] = combined.get(k, 0) + v
    if combined:
        console.print()
        console.print("[red]Failure breakdown:[/red]")
        for reason, count in sorted(combined.items(), key=lambda x: -x[1]):
            console.print(f"  {reason}: {count}")


def check_thresholds(
    results: list[RunResult],
    min_rps: float | None = None,
    max_p50_ms: float | None = None,
    max_p99_ms: float | None = None,
) -> bool:
    """Check results against performance thresholds. Returns True if all pass."""
    avg_rps = statistics.mean(r.throughput for r in results)
    avg_p50 = statistics.mean(r.percentile(50) for r in results)
    avg_p99 = statistics.mean(r.percentile(99) for r in results)
    passed = True

    if min_rps is not None and avg_rps < min_rps:
        console.print(
            f"\n[red]THRESHOLD FAILED:[/red] avg throughput {avg_rps:.0f} req/s"
            f" < minimum {min_rps:.0f} req/s"
        )
        passed = False

    if max_p50_ms is not None and avg_p50 > max_p50_ms:
        console.print(
            f"\n[red]THRESHOLD FAILED:[/red] avg P50 {avg_p50:.1f} ms > maximum {max_p50_ms:.1f} ms"
        )
        passed = False

    if max_p99_ms is not None and avg_p99 > max_p99_ms:
        console.print(
            f"\n[red]THRESHOLD FAILED:[/red] avg P99 {avg_p99:.1f} ms > maximum {max_p99_ms:.1f} ms"
        )
        passed = False

    if passed and (min_rps is not None or max_p50_ms is not None or max_p99_ms is not None):
        console.print("\n[green]All thresholds passed.[/green]")

    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Async HTTP benchmark client for MLflow Gateway")
    parser.add_argument("--url", required=True, help="Gateway invocation URL")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--min-rps",
        type=float,
        default=None,
        metavar="N",
        help="Fail (exit 1) if average throughput falls below N req/s",
    )
    parser.add_argument(
        "--max-p50-ms",
        type=float,
        default=None,
        metavar="N",
        help="Fail (exit 1) if average P50 latency exceeds N ms",
    )
    parser.add_argument(
        "--max-p99-ms",
        type=float,
        default=None,
        metavar="N",
        help="Fail (exit 1) if average P99 latency exceeds N ms",
    )
    args = parser.parse_args()

    console.print(f"\n[bold]Benchmarking[/bold] {args.url}")
    console.print(
        f"  {args.requests} requests · {args.max_concurrent} concurrent · {args.runs} runs\n"
    )
    results = run_benchmark(args.url, args.requests, args.max_concurrent, args.runs)
    print_results(results)

    if not check_thresholds(
        results, min_rps=args.min_rps, max_p50_ms=args.max_p50_ms, max_p99_ms=args.max_p99_ms
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

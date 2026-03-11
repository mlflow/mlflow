"""
Post-processing script for MLflow AI Gateway benchmark results.

Reads Locust CSV stats and resource CSV, prints a summary table with
median/p95/p99 latency, RPS, error rate, and resource usage.

Usage:
    python analyze_results.py <results_directory>
"""

import csv
import sys
from pathlib import Path


def read_config(results_dir):
    config = {}
    config_path = results_dir / "config.txt"
    if config_path.exists():
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    key, value = line.split("=", 1)
                    config[key] = value
    return config


def read_locust_stats(results_dir):
    stats_path = results_dir / "locust_stats.csv"
    if not stats_path.exists():
        return None

    with open(stats_path) as f:
        return list(csv.DictReader(f))


def read_resources(results_dir):
    resources_path = results_dir / "resources.csv"
    if not resources_path.exists():
        return None

    with open(resources_path) as f:
        return list(csv.DictReader(f))


def format_table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = [
            max(len(h), max((len(str(r[i])) for r in rows), default=0)) + 2
            for i, h in enumerate(headers)
        ]

    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "".join("-" * w for w in col_widths)
    lines = [header_line, separator]
    lines.extend("".join(str(v).ljust(w) for v, w in zip(row, col_widths)) for row in rows)
    return "\n".join(lines)


def _stats_row_to_table_entry(row, label):
    return [
        label,
        row.get("Request Count", "0"),
        row.get("Failure Count", "0"),
        row.get("50%", "N/A"),
        row.get("95%", "N/A"),
        row.get("99%", "N/A"),
        row.get("Average Response Time", "N/A"),
        row.get("Requests/s", "N/A"),
    ]


def analyze(results_dir):
    results_dir = Path(results_dir)
    config = read_config(results_dir)
    stats_rows = read_locust_stats(results_dir)
    resource_rows = read_resources(results_dir)

    print("=" * 70)
    print("MLflow AI Gateway Benchmark Results")
    print("=" * 70)

    if config:
        print("\nConfiguration:")
        print(f"  Workers:        {config.get('gateway_workers', 'N/A')}")
        print(f"  Fake delay:     {config.get('fake_response_delay_ms', 'N/A')}ms")
        print(f"  Locust users:   {config.get('locust_users', 'N/A')}")
        print(f"  Run time:       {config.get('locust_run_time', 'N/A')}")

    if stats_rows:
        print("\n--- Request Statistics ---\n")
        headers = [
            "Endpoint",
            "Requests",
            "Failures",
            "Median (ms)",
            "P95 (ms)",
            "P99 (ms)",
            "Avg (ms)",
            "RPS",
        ]
        table_rows = []

        for row in stats_rows:
            name = row.get("Name", "")
            req_type = row.get("Type", "")
            if not name or name == "Aggregated":
                continue
            table_rows.append(_stats_row_to_table_entry(row, f"{req_type} {name}"))

        for row in stats_rows:
            if row.get("Name") == "Aggregated":
                table_rows.append(_stats_row_to_table_entry(row, "--- TOTAL ---"))
                total_reqs = int(row.get("Request Count", 0))
                total_fails = int(row.get("Failure Count", 0))
                if total_reqs > 0:
                    error_rate = (total_fails / total_reqs) * 100
                    print(f"  Error rate: {error_rate:.2f}%")
                break

        print(format_table(headers, table_rows))

        if overhead_rows := [r for r in stats_rows if r.get("Type") == "OVERHEAD"]:
            print("\n--- Gateway Overhead ---\n")
            oh_headers = [
                "Metric",
                "Median (ms)",
                "P95 (ms)",
                "P99 (ms)",
                "Avg (ms)",
            ]
            oh_table = [
                [
                    row.get("Name", ""),
                    row.get("50%", "N/A"),
                    row.get("95%", "N/A"),
                    row.get("99%", "N/A"),
                    row.get("Average Response Time", "N/A"),
                ]
                for row in overhead_rows
            ]
            print(format_table(oh_headers, oh_table))

    if resource_rows:
        print("\n--- Resource Usage ---\n")
        cpu_vals = [float(r["cpu_percent"]) for r in resource_rows if r["cpu_percent"]]
        rss_vals = [float(r["rss_mb"]) for r in resource_rows if r["rss_mb"]]

        if cpu_vals:
            avg_cpu = sum(cpu_vals) / len(cpu_vals)
            print(f"  CPU:    avg={avg_cpu:.1f}%  peak={max(cpu_vals):.1f}%")
        if rss_vals:
            avg_rss = sum(rss_vals) / len(rss_vals)
            print(f"  Memory: avg={avg_rss:.1f} MB  peak={max(rss_vals):.1f} MB")
        print(f"  Samples: {len(resource_rows)}")

    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results_directory>", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]
    if not Path(results_dir).is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    analyze(results_dir)

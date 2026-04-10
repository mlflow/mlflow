# /// script
# requires-python = ">=3.10"
# dependencies = ["aiohttp>=3.13.3,<4", "psycopg2-binary>=2.9,<3", "rich>=14.3.3,<15"]
# ///
"""MLflow AI Gateway benchmark runner.

Orchestrates fake OpenAI server, MLflow server(s), optional PostgreSQL and
nginx (via Docker), then runs the async benchmark client.

Usage:
    uv run run.py                              # 4 instances, PostgreSQL, nginx (Docker)
    uv run run.py --instances 1               # single instance, SQLite, no Docker
    uv run run.py --instances 1 --database postgres
    uv run run.py --instances 8 --workers 8
    uv run run.py --url http://...            # benchmark an existing endpoint directly
"""

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import benchmark as bm  # local module; path inserted above
from rich.console import Console  # type: ignore[import-not-found]
from rich.panel import Panel  # type: ignore[import-not-found]
from rich.progress import (  # type: ignore[import-not-found]
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

SCRIPT_DIR = Path(__file__).parent

FAKE_SERVER_PORT = 9137
FAKE_SERVER_WORKERS = 8
MLFLOW_PORT = 5731
INSTANCE_BASE_PORT = 5800
POSTGRES_PORT = int(os.environ.get("GATEWAY_BENCH_POSTGRES_PORT", "5432"))
POSTGRES_PASSWORD = "benchmarkpass"
ENDPOINT_NAME = "benchmark-chat"

_API_SECRET_CREATE = "gateway/secrets/create"
_API_MODEL_DEF_CREATE = "gateway/model-definitions/create"
_API_ENDPOINT_CREATE = "gateway/endpoints/create"

console = Console()


def _uv_prefix() -> list[str]:
    """Return uv run prefix when inside the mlflow repo, else empty list."""
    in_repo = (
        shutil.which("uv")
        and subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR, capture_output=True
        ).returncode
        == 0
    )
    return ["uv", "run", "--no-build-isolation", "--extra", "gateway"] if in_repo else []


def _subprocess_env() -> dict[str, str]:
    return os.environ | {"OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES"}


def _wait_for_port(port: int, label: str, log_file: Path | None = None, timeout: int = 30) -> None:
    url = f"http://127.0.0.1:{port}/health"
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"  Waiting for {label}...", total=None)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1):
                    break
            except Exception:
                time.sleep(0.5)
        else:
            console.print(f"  [red]✗ {label} failed to start within {timeout}s[/red]")
            if log_file and log_file.exists():
                console.print("  [yellow]Last 20 lines of log:[/yellow]")
                for line in log_file.read_text().splitlines()[-20:]:
                    console.print(f"    [dim]{line}[/dim]")
            sys.exit(1)
    console.print(f"  [green]✓[/green] {label} ready")


@contextlib.contextmanager
def _start_fake_server(
    work_dir: str, port: int = FAKE_SERVER_PORT, workers: int = FAKE_SERVER_WORKERS
) -> Generator[None, None, None]:
    prefix = _uv_prefix()
    log_file = Path(work_dir) / "fake_server.log"
    with (
        log_file.open("w") as f,
        subprocess.Popen(
            [
                *prefix,
                "uvicorn",
                "fake_server:app",
                "--workers",
                str(workers),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=SCRIPT_DIR,
            stdout=f,
            stderr=f,
            env=_subprocess_env(),
        ) as proc,
    ):
        _wait_for_port(port, "fake OpenAI server", log_file)
        try:
            yield
        finally:
            proc.terminate()


@contextlib.contextmanager
def _start_mlflow(
    work_dir: str,
    port: int,
    workers: int,
    backend_uri: str,
    label: str = "MLflow server",
    host: str = "127.0.0.1",
) -> Generator[None, None, None]:
    prefix = _uv_prefix()
    log_file = Path(work_dir) / f"mlflow-{port}.log"
    with (
        log_file.open("w") as f,
        subprocess.Popen(
            [
                *prefix,
                "mlflow",
                "server",
                "--backend-store-uri",
                backend_uri,
                "--host",
                host,
                "--port",
                str(port),
                "--workers",
                str(workers),
                "--disable-security-middleware",
            ],
            stdout=f,
            stderr=f,
            env=_subprocess_env(),
        ) as proc,
    ):
        _wait_for_port(port, label, log_file)
        try:
            yield
        finally:
            proc.terminate()


def _check_docker() -> None:
    try:
        result = subprocess.run(["docker", "info"], capture_output=True)
    except FileNotFoundError:
        console.print(
            "[red]Docker is not installed. Install it at https://docs.docker.com/get-docker/[/red]"
        )
        sys.exit(1)
    if result.returncode != 0:
        console.print("[red]Docker daemon is not running. Please start Docker and try again.[/red]")
        sys.exit(1)


@contextlib.contextmanager
def _start_postgres(container_name: str = "benchmark-postgres") -> Generator[str, None, None]:
    """Start a PostgreSQL Docker container. Yields the connection URI."""
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    with subprocess.Popen(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "-e",
            f"POSTGRES_PASSWORD={POSTGRES_PASSWORD}",
            "-e",
            "POSTGRES_DB=mlflow",
            "-p",
            f"127.0.0.1:{POSTGRES_PORT}:5432",
            "postgres:16-alpine",
            "-c",
            "max_connections=500",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("  Starting PostgreSQL...", total=None)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if (
                    subprocess.run(
                        ["docker", "exec", container_name, "pg_isready", "-U", "postgres"],
                        capture_output=True,
                    ).returncode
                    == 0
                ):
                    break
                time.sleep(0.5)
            else:
                console.print("  [red]✗ PostgreSQL failed to start within 30s[/red]")
                sys.exit(1)

        console.print("  [green]✓[/green] PostgreSQL ready")
        try:
            yield f"postgresql://postgres:{POSTGRES_PASSWORD}@127.0.0.1:{POSTGRES_PORT}/mlflow"
        finally:
            subprocess.run(["docker", "kill", container_name], capture_output=True)


def _api_post(tracking_uri: str, path: str, body: dict[str, Any]) -> Any:
    url = f"{tracking_uri.rstrip('/')}/api/3.0/mlflow/{path}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        console.print(f"  [red]API error {e.code} at {url}: {e.read().decode()}[/red]")
        sys.exit(1)
    except urllib.error.URLError as e:
        console.print(f"  [red]API error at {url}: {e.reason}[/red]")
        sys.exit(1)


def _setup_endpoint(
    tracking_uri: str, fake_server_url: str, endpoint_name: str, usage_tracking: bool
) -> str:
    """Create secret → model definition → endpoint. Returns the invocation URL."""
    console.print("  Creating secret...")
    secret_id = _api_post(
        tracking_uri,
        _API_SECRET_CREATE,
        {
            "secret_name": "benchmark-secret",
            "secret_value": {"api_key": "fake-benchmark-key"},
            "provider": "openai",
            "auth_config": {"api_base": fake_server_url},
        },
    )["secret"]["secret_id"]

    console.print("  Creating model definition...")
    model_def_id = _api_post(
        tracking_uri,
        _API_MODEL_DEF_CREATE,
        {
            "name": "benchmark-model",
            "secret_id": secret_id,
            "provider": "openai",
            "model_name": "gpt-4o-mini",
        },
    )["model_definition"]["model_definition_id"]

    console.print(f"  Creating endpoint '{endpoint_name}' (usage_tracking={usage_tracking})...")
    _api_post(
        tracking_uri,
        _API_ENDPOINT_CREATE,
        {
            "name": endpoint_name,
            "model_configs": [
                {"model_definition_id": model_def_id, "linkage_type": "PRIMARY", "weight": 1.0}
            ],
            "usage_tracking": usage_tracking,
        },
    )

    invoke_url = f"{tracking_uri.rstrip('/')}/gateway/{endpoint_name}/mlflow/invocations"
    console.print(f"  [green]✓[/green] Endpoint ready: [cyan]{invoke_url}[/cyan]")
    return invoke_url


def _sanity_check(url: str) -> None:
    console.print("  Sending sanity-check request...")
    body = json.dumps({"messages": [{"role": "user", "content": "test"}]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                console.print(f"  [red]✗ Sanity check failed: HTTP {resp.status}[/red]")
                sys.exit(1)
    except Exception as e:
        console.print(f"  [red]✗ Sanity check failed: {e}[/red]")
        sys.exit(1)
    console.print("  [green]✓[/green] Sanity check passed")


def _run_benchmark(
    url: str,
    n_requests: int,
    max_concurrent: int,
    runs: int,
    min_rps: float | None = None,
    max_p50_ms: float | None = None,
    max_p99_ms: float | None = None,
    output: Path | None = None,
) -> None:
    results = bm.run_benchmark(url, n_requests, max_concurrent, runs)
    bm.print_results(results)
    if output is not None:
        output.write_text(json.dumps(bm.results_to_dict(results), indent=2))
        console.print(f"  Results saved to [cyan]{output}[/cyan]")
    if not bm.check_thresholds(
        results, min_rps=min_rps, max_p50_ms=max_p50_ms, max_p99_ms=max_p99_ms
    ):
        raise SystemExit(1)


@contextlib.contextmanager
def _start_nginx(
    work_dir: str, instance_ports: list[int], port: int, container_name: str = "benchmark-nginx"
) -> Generator[None, None, None]:
    nginx_dir = Path(work_dir) / "nginx"
    conf_d = nginx_dir / "conf.d"
    conf_d.mkdir(parents=True)

    upstream_lines = "\n".join(f"    server host.docker.internal:{p};" for p in instance_ports)
    (conf_d / "mlflow.conf").write_text(
        f"upstream mlflow_backends {{\n"
        f"{upstream_lines}\n"
        f"    keepalive 512;\n"
        f"    keepalive_requests 100000;\n"
        f"    keepalive_timeout 60s;\n"
        f"}}\n"
        f"server {{\n"
        f"    listen {port} reuseport backlog=65535;\n"
        f"    location / {{\n"
        f"        proxy_pass http://mlflow_backends;\n"
        f"        proxy_http_version 1.1;\n"
        f'        proxy_set_header Connection "";\n'
        f"        proxy_set_header Host $host;\n"
        f"        proxy_set_header X-Real-IP $remote_addr;\n"
        f"        proxy_connect_timeout 5s;\n"
        f"        proxy_send_timeout 60s;\n"
        f"        proxy_read_timeout 60s;\n"
        f"    }}\n"
        f"}}\n"
    )
    (nginx_dir / "nginx.conf").write_text(
        "worker_processes auto;\n"
        "worker_rlimit_nofile 65535;\n"
        "events {\n"
        "    worker_connections 16384;\n"
        "    use epoll;\n"
        "    multi_accept on;\n"
        "}\n"
        "http {\n"
        "    access_log off;\n"
        "    tcp_nodelay on;\n"
        "    keepalive_timeout 65;\n"
        "    keepalive_requests 100000;\n"
        "    reset_timedout_connection on;\n"
        "    include /etc/nginx/conf.d/*.conf;\n"
        "}\n"
    )

    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("  Starting nginx...", total=None)
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-d",
                "--name",
                container_name,
                "--add-host=host.docker.internal:host-gateway",
                "--ulimit",
                "nofile=65535:65535",
                "-v",
                f"{nginx_dir / 'nginx.conf'}:/etc/nginx/nginx.conf:ro",
                "-v",
                f"{conf_d}:/etc/nginx/conf.d:ro",
                "-p",
                f"127.0.0.1:{port}:{port}",
                "nginx:alpine",
            ],
            check=True,
            capture_output=True,
        )

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if (
                subprocess.run(
                    ["docker", "exec", container_name, "nginx", "-t"], capture_output=True
                ).returncode
                == 0
            ):
                break
            time.sleep(0.5)
        else:
            console.print("  [red]✗ nginx failed to start[/red]")
            sys.exit(1)

    console.print("  [green]✓[/green] nginx ready")
    try:
        yield
    finally:
        subprocess.run(["docker", "kill", container_name], capture_output=True)


def cmd_bench(args: argparse.Namespace) -> None:
    instances = args.instances
    mode = "1 instance" if instances == 1 else f"{instances} instances, nginx LB"

    if args.url:
        console.print(
            Panel.fit(
                f"[bold]Gateway Benchmark[/bold] ({mode})\n"
                f"URL: [cyan]{args.url}[/cyan]\n"
                f"Requests: {args.requests}  ·  Concurrency: {args.max_concurrent}"
                f"  ·  Runs: {args.runs}",
                border_style="cyan",
            )
        )
        console.print("\n[bold]Running benchmark[/bold]")
        _run_benchmark(
            args.url,
            args.requests,
            args.max_concurrent,
            args.runs,
            args.min_rps,
            args.max_p50_ms,
            args.max_p99_ms,
            args.output,
        )
        return

    needs_docker = instances > 1 or args.database == "postgres"
    if needs_docker:
        _check_docker()

    with tempfile.TemporaryDirectory(prefix="mlflow-bench-") as work_dir:
        port = args.port
        fake_port = args.fake_server_port
        instance_ports = [args.base_port + i for i in range(instances)]

        if instances == 1:
            panel = (
                f"[bold]Gateway Benchmark[/bold] ({mode})\n"
                f"Workers: {args.workers}  ·  DB: {args.database.upper()}  ·  "
                f"Usage tracking: {args.usage_tracking}\n"
                f"Requests: {args.requests}  ·  Concurrency: {args.max_concurrent}  ·  "
                f"Runs: {args.runs}  ·  Fake delay: {args.fake_delay_ms}ms\n"
                f"Ports: MLflow :{port}  ·  Fake server :{fake_port}"
            )
        else:
            panel = (
                f"[bold]Gateway Benchmark[/bold] ({mode})\n"
                f"Workers/instance: {args.workers}  ·  "
                f"Total workers: {instances * args.workers}  ·  "
                f"Usage tracking: {args.usage_tracking}\n"
                f"Requests: {args.requests}  ·  Concurrency: {args.max_concurrent}  ·  "
                f"Runs: {args.runs}  ·  Fake delay: {args.fake_delay_ms}ms\n"
                f"Ports: instances {instance_ports[0]}–{instance_ports[-1]}"
                f"  ·  LB :{port}  ·  Fake server :{fake_port}"
            )
        console.print(Panel.fit(panel, border_style="cyan"))

        with contextlib.ExitStack() as stack:
            stack.callback(lambda: console.print("\n[dim]Cleaning up...[/dim]"))

            # Backend
            if instances > 1 or args.database == "postgres":
                console.print("\n[bold]PostgreSQL[/bold]")
                backend_uri = stack.enter_context(_start_postgres())
            else:
                db_path = Path(work_dir) / "mlflow.db"
                backend_uri = f"sqlite:///{db_path}"
                console.print(f"\n[dim]Using SQLite: {db_path}[/dim]")

            # Servers
            console.print("\n[bold]Starting servers[/bold]")
            stack.enter_context(
                _start_fake_server(work_dir, port=fake_port, workers=FAKE_SERVER_WORKERS)
            )

            if instances == 1:
                stack.enter_context(_start_mlflow(work_dir, port, args.workers, backend_uri))

                console.print("\n[bold]Setting up gateway endpoint[/bold]")
                invoke_url = _setup_endpoint(
                    f"http://127.0.0.1:{port}",
                    f"http://127.0.0.1:{fake_port}/v1",
                    ENDPOINT_NAME,
                    usage_tracking=args.usage_tracking,
                )
                _sanity_check(invoke_url)
            else:
                # Start instance 0 first — it initializes the DB schema.
                # All instances share the same PostgreSQL DB, so starting concurrently
                # can cause CREATE TABLE race conditions.
                stack.enter_context(
                    _start_mlflow(
                        work_dir,
                        instance_ports[0],
                        args.workers,
                        backend_uri,
                        "MLflow instance 0",
                        host="0.0.0.0",
                    )
                )
                for i, p in enumerate(instance_ports[1:], start=1):
                    stack.enter_context(
                        _start_mlflow(
                            work_dir,
                            p,
                            args.workers,
                            backend_uri,
                            f"MLflow instance {i}",
                            host="0.0.0.0",
                        )
                    )

                console.print("\n[bold]Setting up gateway endpoint[/bold]")
                _setup_endpoint(
                    f"http://127.0.0.1:{instance_ports[0]}",
                    f"http://127.0.0.1:{fake_port}/v1",
                    ENDPOINT_NAME,
                    usage_tracking=args.usage_tracking,
                )

                console.print("\n[bold]Starting nginx load balancer[/bold]")
                nginx_container = "benchmark-nginx"
                stack.enter_context(
                    _start_nginx(
                        work_dir, instance_ports, port=port, container_name=nginx_container
                    )
                )
                subprocess.run(
                    ["docker", "exec", nginx_container, "nginx", "-s", "reload"],
                    capture_output=True,
                )
                time.sleep(1)

                invoke_url = f"http://127.0.0.1:{port}/gateway/{ENDPOINT_NAME}/mlflow/invocations"
                _sanity_check(invoke_url)

            console.print("\n[bold]Running benchmark[/bold]")
            _run_benchmark(
                invoke_url,
                args.requests,
                args.max_concurrent,
                args.runs,
                args.min_rps,
                args.max_p50_ms,
                args.max_p99_ms,
                args.output,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLflow AI Gateway benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url",
        metavar="URL",
        help="Benchmark this endpoint URL directly, skipping server setup entirely",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=int(os.environ.get("INSTANCES", "4")),
        metavar="N",
        help=(
            "Number of MLflow instances to run (default: 4). "
            "Values >1 require Docker (postgres + nginx). "
            "Use --instances 1 for a single instance with optional SQLite."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("WORKERS_PER_INSTANCE", "4")),
        metavar="N",
        help="Gunicorn/uvicorn worker processes per MLflow instance (default: 4)",
    )
    parser.add_argument(
        "--database",
        choices=["sqlite", "postgres"],
        default="sqlite",
        help=(
            "Database to use — only applies when --instances 1. "
            "'postgres' auto-starts a Docker container. (default: sqlite)"
        ),
    )
    parser.add_argument(
        "--no-usage-tracking",
        dest="usage_tracking",
        action="store_false",
        default=True,
        help="Disable usage tracking (tracing) on the benchmark endpoint",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MLFLOW_PORT", str(MLFLOW_PORT))),
        metavar="N",
        help=(
            "Port the benchmark client sends requests to. "
            "For --instances 1 this is the MLflow port; "
            "for --instances >1 this is the nginx load balancer port. (default: 5731)"
        ),
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=int(os.environ.get("BASE_PORT", str(INSTANCE_BASE_PORT))),
        metavar="N",
        help=(
            "Starting port for MLflow instances in multi mode. "
            "Instances listen on base-port, base-port+1, … (default: 5800)"
        ),
    )
    parser.add_argument(
        "--fake-server-port",
        type=int,
        metavar="N",
        default=int(os.environ.get("FAKE_SERVER_PORT", str(FAKE_SERVER_PORT))),
        help="Port for the fake OpenAI server that simulates provider latency (default: 9137)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=int(os.environ.get("REQUESTS", "2000")),
        metavar="N",
        help="Total requests to send per benchmark run (default: 2000)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=int(os.environ.get("MAX_CONCURRENT", "50")),
        metavar="N",
        help="Maximum number of in-flight requests at any time (default: 50)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=int(os.environ.get("RUNS", "3")),
        metavar="N",
        help="Number of timed runs; results are reported per-run and averaged (default: 3)",
    )
    parser.add_argument(
        "--fake-delay-ms",
        type=int,
        default=int(os.environ.get("FAKE_RESPONSE_DELAY_MS", "50")),
        metavar="N",
        help=(
            "Simulated provider latency in ms. Set to 0 to measure pure MLflow overhead "
            "with no provider delay. (default: 50)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write benchmark results as JSON to FILE (useful for CI artifact upload)",
    )
    parser.add_argument(
        "--min-rps",
        type=float,
        default=None,
        metavar="N",
        help="Exit 1 if average throughput across runs falls below N req/s (CI threshold)",
    )
    parser.add_argument(
        "--max-p50-ms",
        type=float,
        default=None,
        metavar="N",
        help="Exit 1 if average P50 latency across runs exceeds N ms (CI threshold)",
    )
    parser.add_argument(
        "--max-p99-ms",
        type=float,
        default=None,
        metavar="N",
        help="Exit 1 if average P99 latency across runs exceeds N ms (CI threshold)",
    )

    args = parser.parse_args()
    os.environ["FAKE_RESPONSE_DELAY_MS"] = str(args.fake_delay_ms)
    cmd_bench(args)


if __name__ == "__main__":
    main()

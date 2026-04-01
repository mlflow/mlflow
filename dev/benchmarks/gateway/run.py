# /// script
# requires-python = ">=3.10"
# dependencies = ["aiohttp", "rich"]
# ///
"""MLflow AI Gateway benchmark runner.

Orchestrates fake OpenAI server, MLflow server(s), optional PostgreSQL and
nginx (via Docker), then runs the async benchmark client.

Usage:
    uv run run.py single                     # Single instance, SQLite
    uv run run.py single --backend postgres  # Single instance, PostgreSQL (Docker)
    uv run run.py multi                      # 4 instances behind nginx (Docker)
    uv run run.py multi --instances 8        # 8 instances
"""

import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

SCRIPT_DIR = Path(__file__).parent
FAKE_SERVER_PORT = 9137
MLFLOW_LB_PORT = 5731
INSTANCE_BASE_PORT = 5800
ENDPOINT_NAME = "benchmark-chat"

console = Console()

_procs: list[subprocess.Popen] = []
_docker_containers: list[str] = []
_tmpdir: str | None = None


def _cleanup():
    console.print("\n[dim]Cleaning up...[/dim]")
    for proc in reversed(_procs):
        try:
            proc.terminate()
        except Exception:
            pass
    time.sleep(0.5)
    for proc in reversed(_procs):
        try:
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    for name in _docker_containers:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    if _tmpdir:
        shutil.rmtree(_tmpdir, ignore_errors=True)


atexit.register(_cleanup)


def _uv_prefix() -> list[str]:
    """Return uv run prefix when inside the mlflow repo, else empty list."""
    pyproject = (SCRIPT_DIR / "../../../pyproject.toml").resolve()
    if shutil.which("uv") and pyproject.exists():
        return ["uv", "run", "--no-build-isolation", "--extra", "gateway"]
    return []


def _subprocess_env() -> dict[str, str]:
    return os.environ | {"OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES"}


def _wait_for_port(port: int, label: str, log_file: Path | None = None, timeout: int = 30):
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


def _start_fake_server(port: int = FAKE_SERVER_PORT, workers: int = 8):
    prefix = _uv_prefix()
    log_file = Path(_tmpdir) / "fake_server.log"
    with log_file.open("w") as f:
        proc = subprocess.Popen(
            [
                *prefix,
                "uvicorn",
                "fake_server:app",
                "--workers",
                str(workers),
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=SCRIPT_DIR,
            stdout=f,
            stderr=f,
            env=_subprocess_env(),
        )
    _procs.append(proc)
    _wait_for_port(port, "fake OpenAI server", log_file)


def _start_mlflow(port: int, workers: int, backend_uri: str) -> Path:
    """Start an MLflow server and return the log file path."""
    prefix = _uv_prefix()
    log_file = Path(_tmpdir) / f"mlflow-{port}.log"
    with log_file.open("w") as f:
        proc = subprocess.Popen(
            [
                *prefix,
                "mlflow",
                "server",
                "--backend-store-uri",
                backend_uri,
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
                "--workers",
                str(workers),
                "--disable-security-middleware",
            ],
            stdout=f,
            stderr=f,
            env=_subprocess_env(),
        )
    _procs.append(proc)
    return log_file


def _check_docker():
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


def _start_postgres(container_name: str = "benchmark-postgres") -> str:
    """Start a PostgreSQL Docker container. Returns the connection URI."""
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("  Starting PostgreSQL...", total=None)
        proc = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
                "-e",
                "POSTGRES_PASSWORD=benchmarkpass",
                "-e",
                "POSTGRES_DB=mlflow",
                "-p",
                "5432:5432",
                "postgres:16-alpine",
                "-c",
                "max_connections=500",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _procs.append(proc)
        _docker_containers.append(container_name)

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
    return "postgresql://postgres:benchmarkpass@127.0.0.1:5432/mlflow"


def _install_psycopg2():
    prefix = _uv_prefix()
    cmd = (
        [*prefix, "pip", "install", "psycopg2-binary", "-q"]
        if prefix
        else [sys.executable, "-m", "pip", "install", "psycopg2-binary", "-q"]
    )
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("  Installing psycopg2-binary...", total=None)
        subprocess.run(cmd, capture_output=True)
    console.print("  [green]✓[/green] psycopg2-binary ready")


def _setup_endpoint(
    tracking_uri: str, fake_server_url: str, endpoint_name: str, usage_tracking: bool
) -> str:
    """Create secret → model definition → endpoint. Returns the invocation URL."""

    def api_post(path: str, body: dict) -> dict:
        url = f"{tracking_uri.rstrip('/')}/api/3.0/mlflow/{path}"
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            console.print(f"  [red]API error {e.code} at {url}: {e.read().decode()}[/red]")
            sys.exit(1)

    console.print("  Creating secret...")
    secret_id = api_post(
        "gateway/secrets/create",
        {
            "secret_name": "benchmark-secret",
            "secret_value": {"api_key": "fake-benchmark-key"},
            "provider": "openai",
            "auth_config": {"api_base": fake_server_url},
        },
    )["secret"]["secret_id"]

    console.print("  Creating model definition...")
    model_def_id = api_post(
        "gateway/model-definitions/create",
        {
            "name": "benchmark-model",
            "secret_id": secret_id,
            "provider": "openai",
            "model_name": "gpt-4o-mini",
        },
    )["model_definition"]["model_definition_id"]

    console.print(f"  Creating endpoint '{endpoint_name}' (usage_tracking={usage_tracking})...")
    api_post(
        "gateway/endpoints/create",
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


def _sanity_check(url: str):
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


def _run_benchmark(url: str, n_requests: int, max_concurrent: int, runs: int):
    sys.path.insert(0, str(SCRIPT_DIR))
    import benchmark as bm

    results = bm.run_benchmark(url, n_requests, max_concurrent, runs)
    bm.print_results(results)


def _start_nginx(
    instance_ports: list[int],
    lb_port: int = MLFLOW_LB_PORT,
    container_name: str = "benchmark-nginx",
):
    nginx_dir = Path(_tmpdir) / "nginx"
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
        f"    listen {lb_port} reuseport backlog=65535;\n"
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
                f"{lb_port}:{lb_port}",
                "nginx:alpine",
            ],
            check=True,
            capture_output=True,
        )
        _docker_containers.append(container_name)

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


def cmd_single(args):
    global _tmpdir

    if args.url:
        console.print(
            Panel.fit(
                f"[bold]Single-Instance Gateway Benchmark[/bold]\n"
                f"URL: [cyan]{args.url}[/cyan]\n"
                f"Requests: {args.requests}  ·  Concurrency: {args.max_concurrent}"
                f"  ·  Runs: {args.runs}",
                border_style="cyan",
            )
        )
        console.print("\n[bold]Running benchmark[/bold]")
        _run_benchmark(args.url, args.requests, args.max_concurrent, args.runs)
        return

    _tmpdir = tempfile.mkdtemp(prefix="mlflow-bench-")
    mlflow_port = args.mlflow_port
    fake_port = args.fake_server_port

    console.print(
        Panel.fit(
            f"[bold]Single-Instance Gateway Benchmark[/bold]\n"
            f"Workers: {args.workers}  ·  DB: {args.backend}  ·  "
            f"Usage tracking: {args.usage_tracking}\n"
            f"Requests: {args.requests}  ·  Concurrency: {args.max_concurrent}  ·  "
            f"Runs: {args.runs}  ·  Fake delay: {args.fake_delay_ms}ms\n"
            f"Ports: MLflow :{mlflow_port}  ·  Fake server :{fake_port}",
            border_style="cyan",
        )
    )

    if args.backend == "postgres":
        _check_docker()
        console.print("\n[bold]PostgreSQL[/bold]")
        backend_uri = _start_postgres()
        _install_psycopg2()
    else:
        db_path = Path(_tmpdir) / "mlflow.db"
        backend_uri = f"sqlite:///{db_path}"
        console.print(f"\n[dim]Using SQLite: {db_path}[/dim]")

    console.print("\n[bold]Starting servers[/bold]")
    _start_fake_server(port=fake_port, workers=8)
    log = _start_mlflow(mlflow_port, args.workers, backend_uri)
    _wait_for_port(mlflow_port, "MLflow server", log)

    console.print("\n[bold]Setting up gateway endpoint[/bold]")
    invoke_url = _setup_endpoint(
        f"http://127.0.0.1:{mlflow_port}",
        f"http://127.0.0.1:{fake_port}/v1",
        ENDPOINT_NAME,
        usage_tracking=args.usage_tracking,
    )
    _sanity_check(invoke_url)

    console.print("\n[bold]Running benchmark[/bold]")
    _run_benchmark(invoke_url, args.requests, args.max_concurrent, args.runs)


def cmd_multi(args):
    global _tmpdir

    if args.url:
        console.print(
            Panel.fit(
                f"[bold]Multi-Instance Gateway Benchmark[/bold]\n"
                f"URL: [cyan]{args.url}[/cyan]\n"
                f"Requests: {args.requests}  ·  Concurrency: {args.max_concurrent}"
                f"  ·  Runs: {args.runs}",
                border_style="cyan",
            )
        )
        console.print("\n[bold]Running benchmark[/bold]")
        _run_benchmark(args.url, args.requests, args.max_concurrent, args.runs)
        return

    _check_docker()
    _tmpdir = tempfile.mkdtemp(prefix="mlflow-bench-")
    lb_port = args.lb_port
    fake_port = args.fake_server_port
    instance_ports = [args.base_port + i for i in range(args.instances)]

    console.print(
        Panel.fit(
            f"[bold]Multi-Instance Gateway Benchmark[/bold]\n"
            f"Instances: {args.instances}  ·  Workers/instance: {args.workers}  ·  "
            f"Total workers: {args.instances * args.workers}"
            f"  ·  Usage tracking: {args.usage_tracking}\n"
            f"Requests: {args.requests}  ·  Concurrency: {args.max_concurrent}  ·  "
            f"Runs: {args.runs}  ·  Fake delay: {args.fake_delay_ms}ms\n"
            f"Ports: instances {instance_ports[0]}–{instance_ports[-1]}  ·  "
            f"LB :{lb_port}  ·  Fake server :{fake_port}",
            border_style="cyan",
        )
    )

    console.print("\n[bold]PostgreSQL[/bold]")
    backend_uri = _start_postgres()
    _install_psycopg2()

    console.print("\n[bold]Starting servers[/bold]")
    _start_fake_server(port=fake_port, workers=16)

    # Start instance 0 first — it initializes the DB schema.
    # All instances share the same PostgreSQL DB, so starting concurrently
    # can cause CREATE TABLE race conditions.
    log0 = _start_mlflow(instance_ports[0], args.workers, backend_uri)
    _wait_for_port(instance_ports[0], "MLflow instance 0", log0)

    remaining_logs = [_start_mlflow(port, args.workers, backend_uri) for port in instance_ports[1:]]
    for i, (port, log) in enumerate(zip(instance_ports[1:], remaining_logs), start=1):
        _wait_for_port(port, f"MLflow instance {i}", log)

    console.print("\n[bold]Setting up gateway endpoint[/bold]")
    _setup_endpoint(
        f"http://127.0.0.1:{instance_ports[0]}",
        f"http://127.0.0.1:{fake_port}/v1",
        ENDPOINT_NAME,
        usage_tracking=args.usage_tracking,
    )

    console.print("\n[bold]Starting nginx load balancer[/bold]")
    _start_nginx(instance_ports, lb_port=lb_port)
    subprocess.run(
        ["docker", "exec", "benchmark-nginx", "nginx", "-s", "reload"],
        capture_output=True,
    )
    time.sleep(1)

    invoke_url = f"http://127.0.0.1:{lb_port}/gateway/{ENDPOINT_NAME}/mlflow/invocations"
    _sanity_check(invoke_url)

    console.print("\n[bold]Running benchmark[/bold]")
    _run_benchmark(invoke_url, args.requests, args.max_concurrent, args.runs)


def _add_benchmark_args(
    p: argparse.ArgumentParser, default_requests: int, default_max_concurrent: int
):
    p.add_argument(
        "--requests", type=int, default=int(os.environ.get("REQUESTS", str(default_requests)))
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=int(os.environ.get("MAX_CONCURRENT", str(default_max_concurrent))),
    )
    p.add_argument("--runs", type=int, default=int(os.environ.get("RUNS", "3")))
    p.add_argument(
        "--fake-delay-ms", type=int, default=int(os.environ.get("FAKE_RESPONSE_DELAY_MS", "50"))
    )


def main():
    parser = argparse.ArgumentParser(
        description="MLflow AI Gateway benchmark orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subs = parser.add_subparsers(dest="command", required=True)

    p_single = subs.add_parser("single", help="Single MLflow instance")
    p_single.add_argument("--url", help="Skip setup and benchmark this endpoint URL directly")
    p_single.add_argument(
        "--workers", type=int, default=int(os.environ.get("TRACKING_SERVER_WORKERS", "4"))
    )
    p_single.add_argument("--backend", choices=["sqlite", "postgres"], default="sqlite")
    p_single.add_argument(
        "--no-usage-tracking", dest="usage_tracking", action="store_false", default=True
    )
    p_single.add_argument(
        "--mlflow-port", type=int, default=int(os.environ.get("MLFLOW_PORT", str(MLFLOW_LB_PORT)))
    )
    p_single.add_argument(
        "--fake-server-port",
        type=int,
        default=int(os.environ.get("FAKE_SERVER_PORT", str(FAKE_SERVER_PORT))),
    )
    _add_benchmark_args(p_single, default_requests=2000, default_max_concurrent=50)

    p_multi = subs.add_parser("multi", help="Multiple instances behind nginx (requires Docker)")
    p_multi.add_argument("--url", help="Skip setup and benchmark this endpoint URL directly")
    p_multi.add_argument("--instances", type=int, default=int(os.environ.get("INSTANCES", "4")))
    p_multi.add_argument(
        "--workers", type=int, default=int(os.environ.get("WORKERS_PER_INSTANCE", "4"))
    )
    p_multi.add_argument(
        "--no-usage-tracking", dest="usage_tracking", action="store_false", default=True
    )
    p_multi.add_argument(
        "--lb-port", type=int, default=int(os.environ.get("LB_PORT", str(MLFLOW_LB_PORT)))
    )
    p_multi.add_argument(
        "--base-port", type=int, default=int(os.environ.get("BASE_PORT", str(INSTANCE_BASE_PORT)))
    )
    p_multi.add_argument(
        "--fake-server-port",
        type=int,
        default=int(os.environ.get("FAKE_SERVER_PORT", str(FAKE_SERVER_PORT))),
    )
    _add_benchmark_args(p_multi, default_requests=10000, default_max_concurrent=200)

    args = parser.parse_args()
    os.environ["FAKE_RESPONSE_DELAY_MS"] = str(args.fake_delay_ms)

    match args.command:
        case "single":
            cmd_single(args)
        case "multi":
            cmd_multi(args)


if __name__ == "__main__":
    main()

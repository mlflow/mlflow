"""Launch the MLflow dev backend and the React dev server for local development.

Cleans up child process groups on exit/SIGINT/SIGTERM so we don't leave zombies.
"""

from __future__ import annotations

import argparse
import atexit
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JS_DIR = REPO_ROOT / "mlflow" / "server" / "js"


def find_free_port(preferred: int, avoid: frozenset[int] = frozenset()) -> int:
    for port in range(preferred, preferred + 100):
        if port in avoid:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit(f"No free port in [{preferred}, {preferred + 100})")


def cleanup(children: list[subprocess.Popen[bytes]], tmp_paths: list[Path]) -> None:
    for proc in children:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    for proc in children:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    for path in tmp_paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def on_signal(signum: int, _frame: object) -> None:
    sys.exit(128 + signum)


def wait_ready(url: str, label: str, timeout: float = 60.0) -> None:
    print(f"Waiting for {label} to be ready...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    print(f"{label} is ready")
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    raise SystemExit(f"Failed to launch {label} (gave up after {timeout:.0f}s)")


def main() -> None:
    # Line-buffer prints so progress shows up live when stdout is redirected to a file.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", help="Path to an env file forwarded to `mlflow`")
    args = parser.parse_args()

    backend_args: list[str] = []
    tmp_paths: list[Path] = []
    if tracking_uri := os.environ.get("MLFLOW_TRACKING_URI"):
        backend_args += ["--backend-store-uri", tracking_uri, "--default-artifact-root", "mlruns"]
    elif backend_uri := os.environ.get("MLFLOW_BACKEND_STORE_URI"):
        backend_args += ["--backend-store-uri", backend_uri, "--default-artifact-root", "mlruns"]
    else:
        db_fd, db_path_str = tempfile.mkstemp(prefix="mlflow-dev-", suffix=".db")
        os.close(db_fd)
        db_path = Path(db_path_str)
        artifacts_path = Path(tempfile.mkdtemp(prefix="mlflow-dev-artifacts-"))
        tmp_paths += [db_path, artifacts_path]
        backend_args += [
            "--backend-store-uri",
            f"sqlite:///{db_path}",
            "--default-artifact-root",
            str(artifacts_path),
        ]
        print(f"Using tmp SQLite store: {db_path} (artifacts: {artifacts_path})")

    if registry_uri := os.environ.get("MLFLOW_REGISTRY_URI"):
        backend_args += ["--registry-store-uri", registry_uri]

    subprocess.check_call(["yarn", "install"], cwd=JS_DIR)

    backend_port = find_free_port(5000)
    frontend_port = find_free_port(3000, avoid=frozenset({backend_port}))
    if backend_port != 5000:
        print(f"Port 5000 is in use; using {backend_port} for the MLflow backend.")
    if frontend_port != 3000:
        print(f"Port 3000 is in use; using {frontend_port} for the React dev server.")

    children: list[subprocess.Popen[bytes]] = []

    atexit.register(cleanup, children, tmp_paths)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)

    mlflow_cmd = [sys.executable, "-m", "mlflow"]
    if args.env_file:
        mlflow_cmd += ["--env-file", args.env_file]
        print(f"Using environment file: {args.env_file}")
    mlflow_cmd += ["server", *backend_args, "--dev", "--port", str(backend_port)]
    print(f"Running tracking server: {shlex.join(mlflow_cmd)}")

    children.append(subprocess.Popen(mlflow_cmd, cwd=REPO_ROOT, start_new_session=True))

    wait_ready(f"http://localhost:{backend_port}/health", "tracking server")

    children.append(
        subprocess.Popen(
            ["yarn", "start"],
            cwd=JS_DIR,
            env={
                **os.environ,
                "PORT": str(frontend_port),
                "MLFLOW_PROXY": f"http://localhost:{backend_port}",
                "MLFLOW_DEV_PROXY_MODE": "1",
            },
            start_new_session=True,
        )
    )

    wait_ready(f"http://localhost:{frontend_port}/", "React dev server", timeout=180)

    # Block until any child exits; atexit reaps the rest.
    while all(proc.poll() is None for proc in children):
        time.sleep(1)
    exited = next(p for p in children if p.poll() is not None)
    print(f"Child process (pid {exited.pid}) exited with code {exited.returncode}")
    sys.exit(exited.returncode or 0)


if __name__ == "__main__":
    main()

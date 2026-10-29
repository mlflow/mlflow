import contextlib
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

SKILL_MD = "---\nname: demo\ndescription: Demo skill\n---\n# Demo\n"
BIG_FILE = b"\0" * 5000


@pytest.fixture
def skill_tree(tmp_path):
    root = tmp_path / "source"
    demo = root / "skills" / "demo"
    demo.mkdir(parents=True)
    (demo / "SKILL.md").write_text(SKILL_MD)
    (demo / "scripts").mkdir()
    (demo / "scripts" / "run.py").write_text("print('hi')\n")
    (root / "README.md").write_text("top-level readme\n")
    (root / "big.bin").write_bytes(BIG_FILE)
    return root


@pytest.fixture
def closed_port():
    # A port that was just bound and released, so connecting to it is refused immediately.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def serve_git_repositories(base_path: Path) -> Iterator[str]:
    """
    Serve every repository under ``base_path`` with ``git daemon`` and yield its base URL.

    Pull refuses ``file://`` Git sources, so tests that pull from Git need a real remote
    transport; ``git://<host>:<port>/<name>`` addresses the repository at ``base_path/<name>``.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with subprocess.Popen([
        "git",
        "daemon",
        "--export-all",
        "--reuseaddr",
        "--listen=127.0.0.1",
        f"--port={port}",
        f"--base-path={base_path}",
    ]) as daemon:
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=1).close()
                    break
                except OSError:
                    if daemon.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("git daemon did not start serving") from None
                    time.sleep(0.1)
            yield f"git://127.0.0.1:{port}"
        finally:
            daemon.terminate()

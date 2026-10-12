import base64
import contextlib
import hashlib
import http.server
import json
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import pytest

import mlflow
from mlflow.entities.skill_source import GitSource, MlflowSource, OCISource, ZipSource
from mlflow.genai.skill_content.archive import package_skill_tree

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


def _git(*args, cwd):
    subprocess.check_call(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=cwd
    )


def _sha256(data):
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


@pytest.fixture
def isolated_credentials(tmp_path, monkeypatch):
    # Keep the developer's own Docker, container, and netrc logins out of every fetch.
    empty = tmp_path / "no-credentials"
    empty.mkdir()
    (empty / "netrc").write_text("")
    monkeypatch.setenv("DOCKER_CONFIG", str(empty / "docker"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(empty / "runtime"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(empty / "config"))
    monkeypatch.setenv("NETRC", str(empty / "netrc"))
    monkeypatch.delenv("REGISTRY_AUTH_FILE", raising=False)


@pytest.fixture
def git_repo(tmp_path, skill_tree):
    repo = tmp_path / "skills.git"
    shutil.copytree(skill_tree, repo)
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "v1", cwd=repo)
    _git("tag", "v1", cwd=repo)
    return repo


@pytest.fixture
def git_server(tmp_path):
    # Every repository a test creates under tmp_path is served at <url>/<directory name>.
    with serve_git_repositories(tmp_path) as url:
        yield url


class _ZipHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/private/"):
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        super().do_GET()


@pytest.fixture
def zip_server(tmp_path, skill_tree):
    serve_dir = tmp_path / "www"
    serve_dir.mkdir()
    shutil.make_archive(str(serve_dir / "skills"), "zip", root_dir=skill_tree)
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(_ZipHandler, directory=str(serve_dir))
    )
    threading.Thread(target=server.serve_forever, name="skill-pull-zip", daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", serve_dir
    finally:
        server.shutdown()
        server.server_close()


class _RegistryHandler(http.server.BaseHTTPRequestHandler):
    blobs = {}
    manifests = {}
    credentials = "Basic " + base64.b64encode(b"user:secret").decode()

    def log_message(self, *args):
        pass

    def _send(self, status, body=b"", content_type="application/json", headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.headers.get("Authorization") != self.credentials:
            self._send(401, headers={"WWW-Authenticate": 'Basic realm="registry"'})
            return
        match self.path.split("/"):
            case ["", "v2", "skills", "demo", "manifests", tag] if tag in self.manifests:
                body, content_type = self.manifests[tag]
                self._send(200, body, content_type=content_type)
            case ["", "v2", "skills", "demo", "blobs", digest] if digest in self.blobs:
                self._send(200, self.blobs[digest], content_type="application/octet-stream")
            case _:
                self._send(404, b"{}")


@pytest.fixture
def oci_registry(tmp_path, skill_tree, monkeypatch):
    layer = package_skill_tree(skill_tree, tmp_path / "layer.tar.gz").read_bytes()
    manifest = json.dumps({
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {"mediaType": "application/vnd.oci.empty.v1+json", "digest": _sha256(b"{}")},
        "layers": [
            {
                "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                "digest": _sha256(layer),
                "size": len(layer),
            }
        ],
    }).encode()
    handler = type(
        "Handler",
        (_RegistryHandler,),
        {
            "blobs": {_sha256(layer): layer},
            "manifests": {"v1": (manifest, "application/vnd.oci.image.manifest.v1+json")},
        },
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, name="skill-pull-oci", daemon=True).start()
    host = f"127.0.0.1:{server.server_address[1]}"
    docker = tmp_path / "docker"
    docker.mkdir()
    auth = base64.b64encode(b"user:secret").decode()
    (docker / "config.json").write_text(json.dumps({"auths": {host: {"auth": auth}}}))
    monkeypatch.setenv("DOCKER_CONFIG", str(docker))
    try:
        yield host
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def mlflow_artifacts(skill_tree):
    with mlflow.start_run() as run:
        mlflow.log_artifacts(str(skill_tree), artifact_path="pkg")
    return f"runs:/{run.info.run_id}/pkg"


@pytest.fixture
def make_source(request):
    def make(kind, subpath=None):
        match kind:
            case "git":
                repo = request.getfixturevalue("git_repo")
                url = request.getfixturevalue("git_server")
                return GitSource(url=f"{url}/{repo.name}", ref="v1", subpath=subpath)
            case "oci":
                host = request.getfixturevalue("oci_registry")
                return OCISource(image=f"oci://{host}/skills/demo:v1", subpath=subpath)
            case "zip":
                base_url, _ = request.getfixturevalue("zip_server")
                return ZipSource(url=f"{base_url}/skills.zip", subpath=subpath)
            case "mlflow":
                uri = request.getfixturevalue("mlflow_artifacts")
                return MlflowSource(artifact_path=uri, subpath=subpath)
            case _:
                raise ValueError(f"Unknown source kind: {kind!r}")

    return make

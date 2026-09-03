import base64
import hashlib
import http.server
import json
import shutil
import subprocess
import threading
import zipfile
from functools import partial
from pathlib import Path

import pytest

import mlflow
from mlflow.entities.skill_source import GitSource, OCISource, SkillSourceType, ZipSource
from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.archive import package_skill_tree
from mlflow.genai.skill_content.digest import compute_tree_digest
from mlflow.genai.skill_content.fetchers import fetch_source
from mlflow.genai.skill_content.fetchers.oci import (
    _load_docker_credentials,
    parse_image_reference,
)

_SKILL_MD = "---\nname: demo\ndescription: Demo skill\n---\n# Demo\n"


@pytest.fixture
def skill_tree(tmp_path):
    root = tmp_path / "source"
    demo = root / "skills" / "demo"
    demo.mkdir(parents=True)
    (demo / "SKILL.md").write_text(_SKILL_MD)
    (demo / "scripts").mkdir()
    (demo / "scripts" / "run.py").write_text("print('hi')\n")
    (root / "README.md").write_text("top-level readme\n")
    return root


def _git(*args, cwd):
    subprocess.check_call(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@pytest.fixture
def git_repo(tmp_path, skill_tree):
    repo = tmp_path / "skills.git"
    shutil.copytree(skill_tree, repo)
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "v1", cwd=repo)
    _git("tag", "v1", cwd=repo)
    (repo / "skills" / "demo" / "SKILL.md").write_text(_SKILL_MD + "\nsecond revision\n")
    _git("commit", "-q", "-am", "v2", cwd=repo)
    return repo


@pytest.fixture
def http_server(tmp_path):
    serve_dir = tmp_path / "www"
    serve_dir.mkdir()
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(serve_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, name="skill-content-http", daemon=True)
    thread.start()
    try:
        yield serve_dir, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_fetch_local_path(skill_tree):
    with fetch_source(str(skill_tree), subpath="skills/demo") as fetched:
        assert fetched.root == skill_tree / "skills" / "demo"
        assert fetched.resolved.is_local is True
        assert fetched.resolved.source_type == SkillSourceType.MLFLOW
    assert (skill_tree / "skills" / "demo" / "SKILL.md").exists()


def test_fetch_local_path_rejects_symlink_and_missing_subpath(skill_tree):
    with pytest.raises(MlflowException, match="does not exist"):
        fetch_source(str(skill_tree), subpath="skills/nope")
    (skill_tree / "skills" / "demo" / "link").symlink_to(skill_tree / "README.md")
    with pytest.raises(MlflowException, match="symbolic links"):
        fetch_source(str(skill_tree), subpath="skills/demo")


def test_fetch_local_path_size_limit(skill_tree):
    with pytest.raises(MlflowException, match="exceeds the size limit"):
        fetch_source(str(skill_tree), max_bytes=5)


def test_fetch_git_by_ref_and_head(git_repo, skill_tree):
    source = GitSource(url=f"file://{git_repo}", ref="v1", subpath="skills/demo")
    with fetch_source(source) as fetched:
        temp_root = fetched.root
        assert (fetched.root / "SKILL.md").read_text() == _SKILL_MD
        assert not (fetched.root.parent.parent / ".git").exists()
        assert compute_tree_digest(fetched.root) == compute_tree_digest(
            skill_tree / "skills" / "demo"
        )
    assert not temp_root.exists()

    with fetch_source(f"file://{git_repo}", subpath="skills/demo") as fetched:
        assert fetched.resolved.source_type == SkillSourceType.GIT
        assert "second revision" in (fetched.root / "SKILL.md").read_text()


def test_fetch_git_missing_ref(git_repo):
    with pytest.raises(MlflowException, match="Failed to fetch skill content.*ref 'nope'"):
        fetch_source(GitSource(url=f"file://{git_repo}", ref="nope"))


def test_fetch_git_size_limit(git_repo):
    with pytest.raises(MlflowException, match="exceeds the skill content size limit"):
        fetch_source(GitSource(url=f"file://{git_repo}"), max_bytes=1)


def test_fetch_git_redacts_credentials():
    url = "https://user:s3cret-token@127.0.0.1:9/skills.git"
    with pytest.raises(MlflowException, match="Failed to fetch skill content") as exc_info:
        fetch_source(url)
    message = str(exc_info.value)
    assert "s3cret-token" not in message
    assert "***@127.0.0.1:9" in message


@pytest.mark.no_mock_requests_get
def test_fetch_zip(http_server, skill_tree):
    serve_dir, base_url = http_server
    shutil.make_archive(str(serve_dir / "skills"), "zip", root_dir=skill_tree)
    with fetch_source(ZipSource(url=f"{base_url}/skills.zip", subpath="skills/demo")) as fetched:
        assert compute_tree_digest(fetched.root) == compute_tree_digest(
            skill_tree / "skills" / "demo"
        )
        assert not (fetched.root.parent.parent.parent / "source.zip").exists()


@pytest.mark.no_mock_requests_get
def test_fetch_zip_errors(http_server, skill_tree):
    serve_dir, base_url = http_server
    with pytest.raises(MlflowException, match="HTTP 404"):
        fetch_source(f"{base_url}/missing.zip")

    shutil.make_archive(str(serve_dir / "skills"), "zip", root_dir=skill_tree)
    with pytest.raises(MlflowException, match="exceeds the skill content size limit"):
        fetch_source(f"{base_url}/skills.zip", max_bytes=10)

    with zipfile.ZipFile(serve_dir / "evil.zip", "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(MlflowException, match="unsafe path"):
        fetch_source(f"{base_url}/evil.zip")


@pytest.mark.no_mock_requests_get
def test_fetch_zip_unreachable():
    with pytest.raises(MlflowException, match="Failed to fetch skill content"):
        fetch_source("http://127.0.0.1:9/skills.zip")


def _sha256(data):
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


class _RegistryHandler(http.server.BaseHTTPRequestHandler):
    blobs = {}
    manifests = {}
    challenge = None
    token = "test-token"

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
        if self.path.startswith("/token"):
            self._send(200, json.dumps({"token": self.token}).encode())
            return
        if self.challenge and self.headers.get("Authorization") != f"Bearer {self.token}":
            realm = f"http://{self.headers['Host']}/token"
            self._send(401, headers={"WWW-Authenticate": self.challenge.format(realm=realm)})
            return
        parts = self.path.split("/")
        if len(parts) >= 5 and parts[-2] == "manifests":
            entry = self.manifests.get(parts[-1])
            if entry is None:
                self._send(404, b"{}")
                return
            body, content_type = entry
            self._send(200, body, content_type=content_type)
            return
        if len(parts) >= 5 and parts[-2] == "blobs":
            blob = self.blobs.get(parts[-1])
            if blob is None:
                self._send(404, b"{}")
                return
            self._send(200, blob, content_type="application/octet-stream")
            return
        self._send(404, b"{}")


@pytest.fixture
def oci_registry(tmp_path, skill_tree):
    layer_path = package_skill_tree(skill_tree, tmp_path / "layer.tar.gz")
    tar_layer = layer_path.read_bytes()
    file_layer = b"extra file from a plain blob\n"
    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {"mediaType": "application/vnd.oci.empty.v1+json", "digest": _sha256(b"{}")},
        "layers": [
            {
                "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                "digest": _sha256(tar_layer),
                "size": len(tar_layer),
            },
            {
                "mediaType": "application/octet-stream",
                "digest": _sha256(file_layer),
                "size": len(file_layer),
                "annotations": {"org.opencontainers.image.title": "docs/README.md"},
            },
        ],
    }
    manifest_body = json.dumps(manifest).encode()
    manifest_type = "application/vnd.oci.image.manifest.v1+json"
    index = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "manifests": [
            {
                "mediaType": manifest_type,
                "digest": "sha256:" + "0" * 64,
                "platform": {"os": "linux", "architecture": "arm64"},
            },
            {
                "mediaType": manifest_type,
                "digest": _sha256(manifest_body),
                "platform": {"os": "linux", "architecture": "amd64"},
            },
        ],
    }
    corrupted = json.loads(manifest_body)
    corrupted["layers"][0]["digest"] = "sha256:" + "1" * 64

    handler = type(
        "Handler",
        (_RegistryHandler,),
        {
            "blobs": {
                _sha256(tar_layer): tar_layer,
                _sha256(file_layer): file_layer,
                "sha256:" + "1" * 64: tar_layer,
            },
            "manifests": {
                "v1": (manifest_body, manifest_type),
                _sha256(manifest_body): (manifest_body, manifest_type),
                "multi": (json.dumps(index).encode(), "application/vnd.oci.image.index.v1+json"),
                "corrupted": (json.dumps(corrupted).encode(), manifest_type),
            },
            "challenge": 'Bearer realm="{realm}",service="test",scope="repository:demo:pull"',
        },
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, name="skill-content-oci", daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{server.server_address[1]}", handler
    finally:
        server.shutdown()
        server.server_close()


def test_fetch_oci_with_bearer_auth(oci_registry, skill_tree):
    host, _ = oci_registry
    with fetch_source(OCISource(image=f"oci://{host}/skills/demo:v1")) as fetched:
        assert fetched.resolved.source == f"{host}/skills/demo:v1"
        assert (fetched.root / "skills" / "demo" / "SKILL.md").read_text() == _SKILL_MD
        assert (fetched.root / "docs" / "README.md").read_bytes().startswith(b"extra file")
        assert compute_tree_digest(fetched.root / "skills" / "demo") == compute_tree_digest(
            skill_tree / "skills" / "demo"
        )


def test_fetch_oci_index_resolves_platform(oci_registry):
    host, _ = oci_registry
    with fetch_source(f"oci://{host}/skills/demo:multi", subpath="skills/demo") as fetched:
        assert (fetched.root / "SKILL.md").exists()


def test_fetch_oci_errors(oci_registry):
    host, handler = oci_registry
    with pytest.raises(MlflowException, match="did not match its digest"):
        fetch_source(f"oci://{host}/skills/demo:corrupted")
    with pytest.raises(MlflowException, match="HTTP 404"):
        fetch_source(f"oci://{host}/skills/demo:missing")
    with pytest.raises(MlflowException, match="exceeds the skill content size limit"):
        fetch_source(f"oci://{host}/skills/demo:v1", max_bytes=10)

    handler.challenge = 'Basic realm="registry"'
    with pytest.raises(MlflowException, match="authentication required"):
        fetch_source(f"oci://{host}/skills/demo:v1")


@pytest.mark.parametrize(
    ("image", "registry", "repository", "reference"),
    [
        ("ghcr.io/acme/skills:v1", "ghcr.io", "acme/skills", "v1"),
        ("ghcr.io/acme/skills", "ghcr.io", "acme/skills", "latest"),
        ("alpine", "registry-1.docker.io", "library/alpine", "latest"),
        ("docker.io/acme/skills:2", "registry-1.docker.io", "acme/skills", "2"),
        ("localhost:5000/skills@sha256:abc", "localhost:5000", "skills", "sha256:abc"),
    ],
)
def test_parse_image_reference(image, registry, repository, reference):
    parsed = parse_image_reference(image)
    assert (parsed.registry, parsed.repository, parsed.reference) == (
        registry,
        repository,
        reference,
    )


@pytest.mark.parametrize("image", ["", "ghcr.io/acme/skills@md5:abc", "ghcr.io/"])
def test_parse_image_reference_invalid(image):
    with pytest.raises(MlflowException, match="OCI image"):
        parse_image_reference(image)


def test_load_docker_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    assert _load_docker_credentials("ghcr.io") is None
    config = {
        "auths": {
            "ghcr.io": {"auth": base64.b64encode(b"user:pass").decode()},
            "https://index.docker.io/v1/": {"username": "hub", "password": "secret"},
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert _load_docker_credentials("ghcr.io") == ("user", "pass")
    assert _load_docker_credentials("registry-1.docker.io") == ("hub", "secret")
    assert _load_docker_credentials("quay.io") is None


def test_fetch_mlflow_artifacts(skill_tree):
    with mlflow.start_run() as run:
        mlflow.log_artifacts(str(skill_tree / "skills" / "demo"), artifact_path="skill")
    uri = f"runs:/{run.info.run_id}/skill"
    with fetch_source(uri) as fetched:
        assert fetched.resolved.source_type == SkillSourceType.MLFLOW
        assert fetched.resolved.is_local is False
        assert compute_tree_digest(fetched.root) == compute_tree_digest(
            skill_tree / "skills" / "demo"
        )
        temp_root = fetched.root
    assert not Path(temp_root).exists()


def test_fetch_mlflow_artifacts_missing():
    with pytest.raises(MlflowException, match="Failed to fetch skill content"):
        fetch_source("runs:/does-not-exist/skill")

import base64
import hashlib
import http.server
import json
import os
import stat
import threading

import pytest

from mlflow.entities.skill_source import OCISource
from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.archive import package_skill_tree
from mlflow.genai.skill_content.digest import compute_tree_digest
from mlflow.genai.skill_content.fetchers import fetch_source
from mlflow.genai.skill_content.fetchers.oci import (
    _load_docker_credentials,
    parse_image_reference,
)

from tests.genai.skill_content.conftest import SKILL_MD


def _sha256(data):
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _basic(user, password):
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


class _RegistryHandler(http.server.BaseHTTPRequestHandler):
    blobs = {}
    manifests = {}
    challenge = None
    basic_credentials = None
    token = "test-token"
    token_requests = []

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

    def _authorized(self):
        auth = self.headers.get("Authorization")
        if self.challenge.startswith("Bearer"):
            return auth == f"Bearer {self.token}"
        return self.basic_credentials is not None and auth == _basic(*self.basic_credentials)

    def do_GET(self):
        if self.path.startswith("/token"):
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            self.token_requests.append({
                "authorization": self.headers.get("Authorization"),
                "query": query,
            })
            self._send(200, json.dumps({"token": self.token}).encode())
            return
        if self.challenge and not self._authorized():
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


_MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"
_INDEX_TYPE = "application/vnd.oci.image.index.v1+json"
_TAR_GZ_TYPE = "application/vnd.oci.image.layer.v1.tar+gzip"


def _manifest(layers):
    return {
        "schemaVersion": 2,
        "mediaType": _MANIFEST_TYPE,
        "config": {"mediaType": "application/vnd.oci.empty.v1+json", "digest": _sha256(b"{}")},
        "layers": layers,
    }


def _tar_layer(data, **extra):
    return {"mediaType": _TAR_GZ_TYPE, "digest": _sha256(data), "size": len(data), **extra}


def _file_layer(data, title):
    return {
        "mediaType": "application/octet-stream",
        "digest": _sha256(data),
        "size": len(data),
        "annotations": {"org.opencontainers.image.title": title},
    }


@pytest.fixture
def oci_registry(tmp_path, skill_tree):
    tar_layer = package_skill_tree(skill_tree, tmp_path / "layer.tar.gz").read_bytes()
    file_layer = b"extra file from a plain blob\n"
    small_dir = tmp_path / "small"
    (small_dir / "docs").mkdir(parents=True)
    (small_dir / "docs" / "a.txt").write_bytes(b"a" * 60)
    small_tar = package_skill_tree(small_dir, tmp_path / "small.tar.gz").read_bytes()
    dir_clash = tmp_path / "clash"
    dir_clash.mkdir()
    (dir_clash / "skills").write_bytes(b"i am a file")
    clash_tar = package_skill_tree(dir_clash, tmp_path / "clash.tar.gz").read_bytes()

    main = json.dumps(
        _manifest([_tar_layer(tar_layer), _file_layer(file_layer, "docs/README.md")])
    ).encode()
    index = {
        "schemaVersion": 2,
        "mediaType": _INDEX_TYPE,
        "manifests": [
            {
                "mediaType": _MANIFEST_TYPE,
                "digest": "sha256:" + "0" * 64,
                "platform": {"os": "linux", "architecture": "arm64"},
            },
            {
                "mediaType": _MANIFEST_TYPE,
                "digest": _sha256(main),
                "platform": {"os": "linux", "architecture": "amd64"},
            },
        ],
    }
    corrupted = json.loads(main)
    corrupted["layers"][0]["digest"] = "sha256:" + "1" * 64
    inner_index_body = json.dumps({
        "schemaVersion": 2,
        "mediaType": _INDEX_TYPE,
        "manifests": [],
    }).encode()
    self_index = {
        "schemaVersion": 2,
        "mediaType": _INDEX_TYPE,
        "manifests": [{"mediaType": _INDEX_TYPE, "digest": _sha256(inner_index_body)}],
    }
    lying = json.loads(main)
    lying["layers"] = [_tar_layer(tar_layer, size=5)]
    nosize = json.loads(main)
    del nosize["layers"][0]["size"]

    manifests = {
        "v1": (main, _MANIFEST_TYPE),
        _sha256(main): (main, _MANIFEST_TYPE),
        "multi": (json.dumps(index).encode(), _INDEX_TYPE),
        "corrupted": (json.dumps(corrupted).encode(), _MANIFEST_TYPE),
        "html": (b"<html>login</html>", "text/html"),
        "self-index": (json.dumps(self_index).encode(), _INDEX_TYPE),
        _sha256(inner_index_body): (inner_index_body, _INDEX_TYPE),
        "sha256:" + "a" * 64: (main, _MANIFEST_TYPE),
        "two-layers": (
            json.dumps(_manifest([_tar_layer(small_tar), _tar_layer(small_tar)])).encode(),
            _MANIFEST_TYPE,
        ),
        "lying-size": (json.dumps(lying).encode(), _MANIFEST_TYPE),
        "no-size": (json.dumps(nosize).encode(), _MANIFEST_TYPE),
        "file-over-dir": (
            json.dumps(_manifest([_tar_layer(tar_layer), _tar_layer(clash_tar)])).encode(),
            _MANIFEST_TYPE,
        ),
    }
    handler = type(
        "Handler",
        (_RegistryHandler,),
        {
            "blobs": {
                _sha256(tar_layer): tar_layer,
                _sha256(file_layer): file_layer,
                _sha256(small_tar): small_tar,
                _sha256(clash_tar): clash_tar,
                "sha256:" + "1" * 64: tar_layer,
            },
            "manifests": manifests,
            "challenge": 'Bearer realm="{realm}",service="test",scope="repository:demo:pull"',
            "token_requests": [],
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


def _docker_config(tmp_path, monkeypatch, config):
    config_dir = tmp_path / "docker"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.json").write_text(json.dumps(config))
    monkeypatch.setenv("DOCKER_CONFIG", str(config_dir))


def test_fetch_oci_with_bearer_auth(oci_registry, skill_tree):
    host, handler = oci_registry
    with fetch_source(OCISource(image=f"oci://{host}/skills/demo:v1")) as fetched:
        assert fetched.resolved.source == f"{host}/skills/demo:v1"
        assert (fetched.root / "skills" / "demo" / "SKILL.md").read_text() == SKILL_MD
        assert (fetched.root / "docs" / "README.md").read_bytes().startswith(b"extra file")
        assert compute_tree_digest(fetched.root / "skills" / "demo") == compute_tree_digest(
            skill_tree / "skills" / "demo"
        )
    assert handler.token_requests[0]["authorization"] is None
    assert "service=test" in handler.token_requests[0]["query"]
    assert "scope=repository" in handler.token_requests[0]["query"]


def test_fetch_oci_bearer_sends_docker_config_credentials(oci_registry, tmp_path, monkeypatch):
    host, handler = oci_registry
    _docker_config(tmp_path, monkeypatch, {"auths": {host: {"username": "u", "password": "p"}}})
    with fetch_source(f"oci://{host}/skills/demo:v1", subpath="skills/demo") as fetched:
        assert (fetched.root / "SKILL.md").exists()
    assert handler.token_requests[0]["authorization"] == _basic("u", "p")


def test_fetch_oci_basic_auth(oci_registry, tmp_path, monkeypatch):
    host, handler = oci_registry
    handler.challenge = 'Basic realm="registry"'
    handler.basic_credentials = ("u", "p")
    with pytest.raises(MlflowException, match="authentication required") as exc:
        fetch_source(f"oci://{host}/skills/demo:v1")
    assert exc.value.error_code == "UNAUTHENTICATED"

    _docker_config(
        tmp_path, monkeypatch, {"auths": {host: {"auth": base64.b64encode(b"u:p").decode()}}}
    )
    with fetch_source(f"oci://{host}/skills/demo:v1", subpath="skills/demo") as fetched:
        assert (fetched.root / "SKILL.md").exists()


@pytest.mark.skipif(os.name == "nt", reason="uses a shell script as a credential helper")
def test_fetch_oci_uses_credential_helper(oci_registry, tmp_path, monkeypatch):
    host, handler = oci_registry
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    helper = bin_dir / "docker-credential-fake"
    helper.write_text(
        '#!/bin/sh\nread server\nprintf \'{"Username":"helper-user","Secret":"helper-secret"}\'\n'
    )
    helper.chmod(helper.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    _docker_config(tmp_path, monkeypatch, {"auths": {host: {}}, "credsStore": "fake"})
    with fetch_source(f"oci://{host}/skills/demo:v1", subpath="skills/demo") as fetched:
        assert (fetched.root / "SKILL.md").exists()
    assert handler.token_requests[0]["authorization"] == _basic("helper-user", "helper-secret")


def test_fetch_oci_index_resolves_platform(oci_registry):
    host, _ = oci_registry
    with fetch_source(f"oci://{host}/skills/demo:multi", subpath="skills/demo") as fetched:
        assert (fetched.root / "SKILL.md").exists()


def test_fetch_oci_subpath_limits_budget(oci_registry):
    host, _ = oci_registry
    with pytest.raises(MlflowException, match="exceeds the skill content size limit"):
        fetch_source(f"oci://{host}/skills/demo:v1", max_bytes=2000)
    with fetch_source(f"oci://{host}/skills/demo:v1", subpath="skills/demo", max_bytes=2000) as f:
        assert (f.root / "SKILL.md").exists()
        assert not (f.root.parent.parent / "docs").exists()


@pytest.mark.parametrize(
    ("reference", "kwargs", "message"),
    [
        ("corrupted", {}, "did not match its digest"),
        ("missing", {}, "HTTP 404"),
        ("html", {}, "not valid JSON"),
        ("self-index", {}, "nests another index"),
        ("@sha256:" + "a" * 64, {}, "manifest .* did not match its digest"),
        ("two-layers", {"max_bytes": 100}, "exceeds the skill content size limit"),
        ("lying-size", {"max_bytes": 100}, "exceeds the skill content size limit"),
        ("no-size", {"max_bytes": 100}, "exceeds the skill content size limit"),
        ("file-over-dir", {}, "OCI layers disagree"),
    ],
)
def test_fetch_oci_errors(oci_registry, reference, kwargs, message):
    host, _ = oci_registry
    separator = "" if reference.startswith("@") else ":"
    with pytest.raises(MlflowException, match=message):
        fetch_source(f"oci://{host}/skills/demo{separator}{reference}", **kwargs)


def test_fetch_oci_two_layers_within_budget(oci_registry):
    host, _ = oci_registry
    with fetch_source(f"oci://{host}/skills/demo:two-layers", max_bytes=200) as fetched:
        assert (fetched.root / "docs" / "a.txt").stat().st_size == 60


def test_fetch_oci_unreachable(closed_port):
    with pytest.raises(MlflowException, match="Failed to fetch skill content") as exc:
        fetch_source(f"oci://127.0.0.1:{closed_port}/skills/demo:v1")
    assert exc.value.error_code == "TEMPORARILY_UNAVAILABLE"


@pytest.mark.parametrize(
    ("image", "registry", "repository", "reference"),
    [
        ("ghcr.io/acme/skills:v1", "ghcr.io", "acme/skills", "v1"),
        ("ghcr.io/acme/skills", "ghcr.io", "acme/skills", "latest"),
        ("alpine", "registry-1.docker.io", "library/alpine", "latest"),
        ("docker.io/acme/skills:2", "registry-1.docker.io", "acme/skills", "2"),
        (
            "localhost:5000/skills@sha256:" + "b" * 64,
            "localhost:5000",
            "skills",
            "sha256:" + "b" * 64,
        ),
    ],
)
def test_parse_image_reference(image, registry, repository, reference):
    parsed = parse_image_reference(image)
    assert (parsed.registry, parsed.repository, parsed.reference) == (
        registry,
        repository,
        reference,
    )


@pytest.mark.parametrize(
    "image",
    ["", "ghcr.io/acme/skills@md5:abc", "ghcr.io/acme/skills@sha256:abc", "ghcr.io/", "ghcr.io//x"],
)
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
            "broken.io": {"auth": 12345},
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert _load_docker_credentials("ghcr.io") == ("user", "pass")
    assert _load_docker_credentials("registry-1.docker.io") == ("hub", "secret")
    assert _load_docker_credentials("broken.io") is None
    assert _load_docker_credentials("quay.io") is None


def test_load_docker_credentials_helper_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"credHelpers": {"quay.io": "does-not-exist"}, "credsStore": "also-missing"})
    )
    assert _load_docker_credentials("quay.io") is None
    assert _load_docker_credentials("ghcr.io") is None

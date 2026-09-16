import base64
import hashlib
import http.server
import io
import json
import os
import stat
import sys
import tarfile
import threading
from unittest import mock

import pytest
import requests
from requests.adapters import BaseAdapter
from urllib3.response import HTTPResponse

from mlflow.entities.skill_source import OCISource
from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.archive import package_skill_tree
from mlflow.genai.skill_content.digest import compute_tree_digest
from mlflow.genai.skill_content.fetchers import fetch_source
from mlflow.genai.skill_content.fetchers import oci as oci_module
from mlflow.genai.skill_content.fetchers.oci import (
    RegistryClient,
    _load_docker_credentials,
    parse_image_reference,
)
from mlflow.genai.skill_content.fetchers.zip import _discard_redirect_body

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
    redirect_blobs = False
    redirect_hits = []

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
        if len(parts) == 3 and parts[1] == "cdn":
            self._send(200, self.blobs[parts[2]], content_type="application/octet-stream")
            return
        if len(parts) < 5 or parts[1] != "v2" or "/".join(parts[2:-2]) != "skills/demo":
            self._send(404, b"{}")
            return
        if parts[-2] == "manifests":
            entry = self.manifests.get(parts[-1])
            if entry is None:
                self._send(404, b"{}")
                return
            body, content_type = entry
            self._send(200, body, content_type=content_type)
            return
        if parts[-2] == "blobs":
            blob = self.blobs.get(parts[-1])
            if blob is None:
                self._send(404, b"{}")
                return
            if self.redirect_blobs:
                # Registries commonly redirect blob pulls to a CDN; the redirect body is large
                # on purpose so buffering it would be observable.
                self.redirect_hits.append(parts[-1])
                self._send(
                    307,
                    b"x" * (64 * 1024),
                    content_type="text/plain",
                    headers={"Location": f"/cdn/{parts[-1]}"},
                )
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
    upper_dir = tmp_path / "upper"
    (upper_dir / "docs").mkdir(parents=True)
    (upper_dir / "docs" / "Notes.txt").write_bytes(b"upper")
    upper_tar = package_skill_tree(upper_dir, tmp_path / "upper.tar.gz").read_bytes()
    lower_dir = tmp_path / "lower"
    (lower_dir / "docs").mkdir(parents=True)
    (lower_dir / "docs" / "notes.txt").write_bytes(b"lower")
    lower_tar = package_skill_tree(lower_dir, tmp_path / "lower.tar.gz").read_bytes()

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
        "too-many-layers": (
            json.dumps(_manifest([_tar_layer(small_tar)] * 257)).encode(),
            _MANIFEST_TYPE,
        ),
        "case-collision": (
            json.dumps(_manifest([_tar_layer(upper_tar), _tar_layer(lower_tar)])).encode(),
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
                _sha256(upper_tar): upper_tar,
                _sha256(lower_tar): lower_tar,
                "sha256:" + "1" * 64: tar_layer,
            },
            "manifests": manifests,
            "challenge": 'Bearer realm="{realm}",service="test",scope="repository:demo:pull"',
            "token_requests": [],
            "redirect_hits": [],
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
        with fetch_source(f"oci://{host}/skills/demo:v1"):
            pass
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
        with fetch_source(f"oci://{host}/skills/demo:v1", max_bytes=2000):
            pass
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
        ("too-many-layers", {}, "has 257 layers; the maximum is 256"),
    ],
)
def test_fetch_oci_errors(oci_registry, reference, kwargs, message):
    host, _ = oci_registry
    separator = "" if reference.startswith("@") else ":"
    with pytest.raises(MlflowException, match=message):
        with fetch_source(f"oci://{host}/skills/demo{separator}{reference}", **kwargs):
            pass


def test_fetch_oci_two_layers_within_budget(oci_registry):
    host, _ = oci_registry
    with fetch_source(f"oci://{host}/skills/demo:two-layers", max_bytes=200) as fetched:
        assert (fetched.root / "docs" / "a.txt").stat().st_size == 60


@pytest.mark.skipif(sys.platform != "linux", reason="needs a case-sensitive filesystem")
def test_fetch_oci_rejects_cross_layer_case_collision(oci_registry):
    # Each layer is validated on its own, so a collision that spans two layers is only
    # visible once they are merged.
    host, _ = oci_registry
    with pytest.raises(MlflowException, match="differ only by letter case"):
        with fetch_source(f"oci://{host}/skills/demo:case-collision"):
            pass


def test_fetch_oci_follows_blob_redirects_without_buffering(oci_registry, skill_tree):
    host, handler = oci_registry
    handler.redirect_blobs = True
    with fetch_source(f"oci://{host}/skills/demo:v1", subpath="skills/demo") as fetched:
        assert compute_tree_digest(fetched.root) == compute_tree_digest(
            skill_tree / "skills" / "demo"
        )
    assert len(handler.redirect_hits) == 2
    assert _discard_redirect_body in RegistryClient(host)._session.hooks["response"]


@pytest.mark.parametrize(
    ("registry", "expected"),
    [
        ("127.0.0.1:5000", "http://127.0.0.1:5000"),
        ("[::1]:5000", "http://[::1]:5000"),
        ("localhost", "http://localhost"),
        ("LOCALHOST:5000", "http://LOCALHOST:5000"),
        ("registry.example.com:5000", "https://registry.example.com:5000"),
        ("localhost.example.com", "https://localhost.example.com"),
    ],
)
def test_registry_client_uses_plain_http_only_for_loopback(registry, expected):
    assert RegistryClient(registry).base_url == expected


@pytest.mark.parametrize(
    "realm",
    ["http://auth.example.com/token", "ftp://auth.example.com/token", "http://[::2]/token"],
)
def test_registry_client_refuses_insecure_token_realm(realm):
    session = mock.Mock(spec=["get", "post", "hooks", "auth"], hooks={"response": []})
    client = RegistryClient("registry.example.com", session=session)
    with pytest.raises(MlflowException, match="token endpoint must use https") as exc:
        client._acquire_token(f'Bearer realm="{realm}",service="test"')
    assert exc.value.error_code == "UNAUTHENTICATED"
    session.get.assert_not_called()
    session.post.assert_not_called()


def _canned_response(status, url, *, headers=None, payload=None):
    response = requests.Response()
    response.status_code = status
    response.url = url
    response.headers.update(headers or {})
    response._content = json.dumps(payload or {}).encode()
    response._content_consumed = True
    return response


def test_registry_client_ignores_challenge_from_redirect_target():
    session = mock.Mock(hooks={"response": []})
    session.get.return_value = _canned_response(
        401,
        "https://cdn.example/blob",
        headers={"WWW-Authenticate": 'Bearer realm="https://cdn.example/token"'},
    )
    with mock.patch(
        "mlflow.genai.skill_content.fetchers.oci._load_docker_credentials",
        return_value=("user", "dummy-password"),
    ):
        client = RegistryClient("registry.example", session=session)
    with mock.patch.object(client, "_request_token", return_value=None) as request_token:
        with pytest.raises(MlflowException, match="redirect target .* requested auth") as exc:
            client.get("/v2/acme/skill/blobs/sha256:" + "a" * 64)
        request_token.assert_not_called()
    assert exc.value.error_code == "UNAUTHENTICATED"
    assert session.get.call_count == 1


def test_registry_client_allows_separate_token_service_named_by_registry():
    registry_url = "https://registry.example/v2/acme/skill/manifests/v1"
    token_url = "https://auth.example/token"
    session = mock.Mock(hooks={"response": []})
    session.get.side_effect = [
        _canned_response(
            401,
            registry_url,
            headers={
                "WWW-Authenticate": (
                    f'Bearer realm="{token_url}",service="registry.example",'
                    'scope="repository:acme/skill:pull"'
                )
            },
        ),
        _canned_response(200, token_url, payload={"token": "dummy-token"}),
        _canned_response(200, registry_url, payload={"layers": []}),
    ]
    with mock.patch(
        "mlflow.genai.skill_content.fetchers.oci._load_docker_credentials",
        return_value=("user", "dummy-password"),
    ):
        client = RegistryClient("registry.example", session=session)
    with client.get("/v2/acme/skill/manifests/v1") as response:
        assert response.status_code == 200
    calls = session.get.call_args_list
    assert len(calls) == 3
    # The registry's own challenge may point at a token service on another host.
    assert calls[1].args[0] == token_url
    assert calls[1].kwargs["auth"] == ("user", "dummy-password")
    assert calls[1].kwargs["params"] == {
        "service": "registry.example",
        "scope": "repository:acme/skill:pull",
    }
    assert calls[2].args[0] == registry_url
    assert calls[2].kwargs["headers"]["Authorization"] == "Bearer dummy-token"


class _RedirectingTokenAdapter(BaseAdapter):
    """Fake transport: the first token URL redirects to plain http, everything else is 200."""

    def __init__(self):
        super().__init__()
        self.sent = []

    def send(self, request, **kwargs):
        self.sent.append((request.url, request.body))
        response = requests.Response()
        response.request = request
        response.url = request.url
        if request.url == "https://auth.example/token":
            response.status_code = 307
            response.headers["Location"] = "http://other.example/token"
        else:
            response.status_code = 200
        # A urllib3 body behaves like a live connection: once closed it yields nothing.
        response.raw = HTTPResponse(
            body=io.BytesIO(b'{"token":"ok"}'), status=response.status_code, preload_content=False
        )
        return response

    def close(self):
        pass


@pytest.mark.parametrize("credentials", [("<token>", "refresh-secret"), ("user", "pw")])
def test_registry_client_does_not_follow_token_redirects(credentials):
    session = requests.Session()
    session.trust_env = False
    adapter = _RedirectingTokenAdapter()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    with mock.patch(
        "mlflow.genai.skill_content.fetchers.oci._load_docker_credentials",
        return_value=credentials,
    ):
        client = RegistryClient("registry.example", session=session)
    with pytest.raises(MlflowException, match="token endpoint redirected") as exc:
        client._request_token("https://auth.example/token", {})
    assert exc.value.error_code == "UNAUTHENTICATED"
    assert [url for url, _ in adapter.sent] == ["https://auth.example/token"]


def _raw_tar_gz(files):
    """Build a gzip tar directly so whiteout markers can be included as plain entries."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tar:
        for name, data in files.items():
            entry = tarfile.TarInfo(name)
            if data is None:
                entry.type = tarfile.DIRTYPE
                entry.mode = 0o755
                tar.addfile(entry)
                continue
            entry.size = len(data)
            entry.mode = 0o644
            tar.addfile(entry, io.BytesIO(data))
    return out.getvalue()


def _fetch_layers(tmp_path, layers, subpath=None):
    """Run ``fetch_oci`` over in-memory tar layers with the registry I/O mocked out."""
    manifest = {"layers": [{"mediaType": _TAR_GZ_TYPE, "payload": blob} for blob in layers]}

    def download(client, ref, layer, target, *, max_bytes):
        target.write_bytes(layer["payload"])

    dest = tmp_path / "content"
    with (
        mock.patch.object(oci_module, "RegistryClient"),
        mock.patch.object(oci_module, "_select_manifest", return_value=manifest),
        mock.patch.object(oci_module, "_download_blob", side_effect=download),
    ):
        oci_module.fetch_oci(
            "example.com/skill:v1", dest, scratch=tmp_path, max_bytes=1024 * 1024, subpath=subpath
        )
    return dest


def _listing(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_fetch_oci_whiteout_removes_lower_file(tmp_path):
    dest = _fetch_layers(
        tmp_path,
        [
            _raw_tar_gz({"SKILL.md": b"current", "old.sh": b"obsolete"}),
            _raw_tar_gz({".wh.old.sh": b""}),
        ],
    )
    assert _listing(dest) == ["SKILL.md"]


def test_fetch_oci_whiteout_removes_lower_directory_tree(tmp_path):
    dest = _fetch_layers(
        tmp_path,
        [
            _raw_tar_gz({"skills/": None, "skills/a/": None, "skills/a/SKILL.md": b"a"}),
            # Deleting the directory and re-adding a file of the same name in one layer.
            _raw_tar_gz({".wh.skills": b"", "skills": b"now a file"}),
        ],
    )
    assert _listing(dest) == ["skills"]
    assert (dest / "skills").read_text() == "now a file"


def test_fetch_oci_opaque_whiteout_clears_lower_directory(tmp_path):
    dest = _fetch_layers(
        tmp_path,
        [
            _raw_tar_gz({"docs/": None, "docs/old.md": b"old", "docs/keep.md": b"lower"}),
            _raw_tar_gz({"docs/": None, "docs/.wh..wh..opq": b"", "docs/new.md": b"new"}),
        ],
    )
    assert _listing(dest) == ["docs", "docs/new.md"]


@pytest.mark.parametrize(
    "deleting_layer",
    [
        {".wh.skills": b""},
        {"skills/": None, "skills/.wh.demo": b""},
        {"skills/": None, "skills/demo/": None, "skills/demo/.wh..wh..opq": b""},
        {".wh..wh..opq": b""},
    ],
)
def test_fetch_oci_whiteout_of_subpath_ancestor_clears_selected_tree(tmp_path, deleting_layer):
    # Entries outside the subpath are never extracted, so their deletions are applied by name.
    base = {
        "skills/": None,
        "skills/demo/": None,
        "skills/demo/SKILL.md": b"lower",
        "README.md": b"outside",
    }
    dest = _fetch_layers(
        tmp_path, [_raw_tar_gz(base), _raw_tar_gz(deleting_layer)], subpath="skills/demo"
    )
    assert _listing(dest) == ["skills", "skills/demo"]


def test_fetch_oci_whiteout_outside_subpath_is_ignored(tmp_path):
    dest = _fetch_layers(
        tmp_path,
        [
            _raw_tar_gz({"skills/": None, "skills/demo/": None, "skills/demo/SKILL.md": b"x"}),
            _raw_tar_gz({".wh.README.md": b"", "skills/": None, "skills/.wh.other": b""}),
        ],
        subpath="skills/demo",
    )
    assert _listing(dest) == ["skills", "skills/demo", "skills/demo/SKILL.md"]


def test_fetch_oci_tag_and_digest_reference(oci_registry, skill_tree):
    host, handler = oci_registry
    digest = next(
        k
        for k, (body, _) in handler.manifests.items()
        if k.startswith("sha256:") and handler.manifests["v1"][0] == body
    )
    with fetch_source(f"oci://{host}/skills/demo:v1@{digest}", subpath="skills/demo") as fetched:
        assert compute_tree_digest(fetched.root) == compute_tree_digest(
            skill_tree / "skills" / "demo"
        )


def test_fetch_oci_decompression_work_is_bounded_across_layers(tmp_path):
    # Each layer is 20 MiB of skipped padding read three times (validate, extract, whiteout
    # scan); with a 1 MiB limit the image-wide work budget of 64 MiB runs out on layer two.
    padded = _raw_tar_gz({"other/padding": b"\0" * (20 * 1024 * 1024), "skill/SKILL.md": b"x"})
    assert len(padded) < 1024 * 1024
    manifest = {"layers": [{"mediaType": _TAR_GZ_TYPE, "payload": padded}] * 2}

    def download(client, ref, layer, target, *, max_bytes):
        target.write_bytes(layer["payload"])

    with (
        mock.patch.object(oci_module, "RegistryClient"),
        mock.patch.object(oci_module, "_select_manifest", return_value=manifest),
        mock.patch.object(oci_module, "_download_blob", side_effect=download),
    ):
        with pytest.raises(MlflowException, match="decompressed more than 67108864 bytes"):
            oci_module.fetch_oci(
                "example.com/skill:v1",
                tmp_path / "content",
                scratch=tmp_path,
                max_bytes=1024 * 1024,
                subpath="skill",
            )


def test_fetch_oci_unreachable(closed_port):
    with pytest.raises(MlflowException, match="Failed to fetch skill content") as exc:
        with fetch_source(f"oci://127.0.0.1:{closed_port}/skills/demo:v1"):
            pass
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
        (
            "localhost:5000/acme/skills:v1@sha256:" + "a" * 64,
            "localhost:5000",
            "acme/skills",
            "sha256:" + "a" * 64,
        ),
        (
            "ghcr.io/acme/skills:v1@sha256:" + "a" * 64,
            "ghcr.io",
            "acme/skills",
            "sha256:" + "a" * 64,
        ),
        (
            "alpine:3@sha256:" + "a" * 64,
            "registry-1.docker.io",
            "library/alpine",
            "sha256:" + "a" * 64,
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


@pytest.mark.parametrize(
    "entry",
    [
        {"auth": base64.b64encode(b"user:").decode(), "identitytoken": "refresh-secret"},
        {"identitytoken": "refresh-secret"},
        {"username": "user", "password": "pw", "identitytoken": "refresh-secret"},
    ],
)
def test_load_docker_credentials_prefers_identity_token(tmp_path, monkeypatch, entry):
    _docker_config(tmp_path, monkeypatch, {"auths": {"registry.example": entry}})
    assert _load_docker_credentials("registry.example") == ("<token>", "refresh-secret")


def test_load_docker_credentials_ignores_empty_identity_token(tmp_path, monkeypatch):
    entry = {"auth": base64.b64encode(b"user:pw").decode(), "identitytoken": ""}
    _docker_config(tmp_path, monkeypatch, {"auths": {"registry.example": entry}})
    assert _load_docker_credentials("registry.example") == ("user", "pw")


def test_load_docker_credentials_helper_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"credHelpers": {"quay.io": "does-not-exist"}, "credsStore": "also-missing"})
    )
    assert _load_docker_credentials("quay.io") is None
    assert _load_docker_credentials("ghcr.io") is None

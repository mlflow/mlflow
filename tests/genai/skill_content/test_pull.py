import base64
import hashlib
import http.server
import json
import os
import shutil
import subprocess
import threading
import traceback
import zipfile
from functools import partial
from pathlib import Path
from unittest import mock

import pytest

import mlflow
from mlflow.entities.skill import SkillStatus
from mlflow.entities.skill_source import (
    GitSource,
    MlflowSource,
    OCISource,
    SkillSourceType,
    ZipSource,
)
from mlflow.entities.skill_version import SkillVersion
from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content import pull as pull_module
from mlflow.genai.skill_content.archive import package_skill_tree
from mlflow.genai.skill_content.digest import compute_tree_digest
from mlflow.genai.skill_content.errors import source_unavailable
from mlflow.genai.skill_content.pull import pull_skill_version, resolve_skill_version
from mlflow.protos.databricks_pb2 import PERMISSION_DENIED, UNAUTHENTICATED, ErrorCode

SUBPATH = "skills/demo"
SOURCE_KINDS = ["git", "oci", "zip", "mlflow"]


def _git(*args, cwd):
    subprocess.check_call(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=cwd
    )


def _sha256(data):
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


@pytest.fixture(autouse=True)
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
                return GitSource(url=f"file://{repo}", ref="v1", subpath=subpath)
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


def _version(source, *, digest=None, status=SkillStatus.ACTIVE, version=1, organization=""):
    source_type = {
        GitSource: SkillSourceType.GIT,
        OCISource: SkillSourceType.OCI,
        ZipSource: SkillSourceType.ZIP,
        MlflowSource: SkillSourceType.MLFLOW,
    }.get(type(source))
    return SkillVersion(
        name="demo",
        version=version,
        organization=organization,
        source=source,
        source_type=source_type,
        digest=digest,
        status=status,
    )


def _listing(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def _assert_no_staging_left(parent):
    assert not [p for p in parent.iterdir() if p.name.startswith(".mlflow-skill-pull-")]


@pytest.mark.parametrize("kind", SOURCE_KINDS)
@pytest.mark.parametrize("subpath", [None, SUBPATH])
def test_pull_every_source_type_honors_persisted_subpath(
    kind, subpath, make_source, skill_tree, tmp_path
):
    expected = skill_tree / subpath if subpath else skill_tree
    version = _version(make_source(kind, subpath), digest=compute_tree_digest(expected))
    destination = tmp_path / "out" / "demo"

    assert pull_skill_version(version, destination) == destination

    assert _listing(destination) == _listing(expected)
    assert compute_tree_digest(destination) == version.digest
    if subpath:
        assert not (destination / "README.md").exists()
        assert (destination / "SKILL.md").exists()
    _assert_no_staging_left(destination.parent)


@pytest.mark.parametrize("kind", SOURCE_KINDS)
def test_pull_into_existing_empty_directory_keeps_the_directory(
    kind, make_source, skill_tree, tmp_path
):
    expected = skill_tree / SUBPATH
    destination = tmp_path / "empty"
    destination.mkdir()
    destination.chmod(0o750)
    inode = destination.stat().st_ino

    pull_skill_version(
        _version(make_source(kind, SUBPATH), digest=compute_tree_digest(expected)), destination
    )

    assert destination.stat().st_ino == inode
    assert destination.stat().st_mode & 0o777 == 0o750
    assert compute_tree_digest(destination) == compute_tree_digest(expected)


@pytest.mark.parametrize("kind", SOURCE_KINDS)
@pytest.mark.parametrize("initially_empty_dir", [False, True])
def test_digest_mismatch_leaves_destination_as_found(
    kind, initially_empty_dir, make_source, tmp_path
):
    destination = tmp_path / "out"
    if initially_empty_dir:
        destination.mkdir()
    version = _version(make_source(kind, SUBPATH), digest="0" * 64)

    with pytest.raises(MlflowException, match="does not match its recorded digest") as exc:
        pull_skill_version(version, destination)

    assert exc.value.error_code == "INVALID_STATE"
    assert "skills:/demo/1" in exc.value.message
    if initially_empty_dir:
        assert destination.is_dir()
        assert list(destination.iterdir()) == []
    else:
        assert not os.path.lexists(destination)
    _assert_no_staging_left(tmp_path)


@pytest.mark.parametrize("kind", SOURCE_KINDS)
def test_missing_digest_skips_verification(kind, make_source, tmp_path):
    with mock.patch(
        "mlflow.genai.skill_content.pull.compute_tree_digest", wraps=compute_tree_digest
    ) as digest:
        pull_skill_version(_version(make_source(kind, SUBPATH)), tmp_path / "out")
    digest.assert_not_called()
    assert (tmp_path / "out" / "SKILL.md").exists()


def test_deprecated_version_is_pullable(make_source, skill_tree, tmp_path):
    expected = skill_tree / SUBPATH
    version = _version(
        make_source("git", SUBPATH),
        digest=compute_tree_digest(expected),
        status=SkillStatus.DEPRECATED,
    )
    pull_skill_version(version, tmp_path / "out")
    assert compute_tree_digest(tmp_path / "out") == version.digest


def test_deleted_version_is_not_pulled(tmp_path):
    version = _version(GitSource(url="https://example.com/s.git"), status=SkillStatus.DELETED)
    with (
        mock.patch("mlflow.genai.skill_content.pull.fetch_source") as fetch,
        pytest.raises(MlflowException, match="has been deleted") as exc,
    ):
        pull_skill_version(version, tmp_path / "out")
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"
    fetch.assert_not_called()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["git", "oci", "zip"])
@pytest.mark.parametrize("initially_empty_dir", [False, True])
def test_authentication_failures_leave_destination_as_found(
    kind, initially_empty_dir, make_source, zip_server, tmp_path, monkeypatch
):
    base_url, _ = zip_server
    match kind:
        case "git":
            # The HTTP server answers 401 under /private/, and no credential helper is set.
            source = GitSource(url=f"{base_url}/private/skills.git")
        case "oci":
            source = make_source("oci")
            monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "no-credentials" / "docker"))
        case "zip":
            source = ZipSource(url=f"{base_url}/private/skills.zip")
    destination = tmp_path / "out"
    if initially_empty_dir:
        destination.mkdir()

    with pytest.raises(MlflowException, match="Failed to fetch skill content") as exc:
        pull_skill_version(_version(source), destination)

    assert exc.value.error_code == "UNAUTHENTICATED"
    assert "secret" not in exc.value.message
    if initially_empty_dir:
        assert list(destination.iterdir()) == []
    else:
        assert not os.path.lexists(destination)
    _assert_no_staging_left(tmp_path)


@pytest.mark.parametrize(
    ("kind", "detail", "error_code"),
    [
        ("git-missing-ref", "no-such-ref", "RESOURCE_DOES_NOT_EXIST"),
        ("oci-missing-tag", "404", "RESOURCE_DOES_NOT_EXIST"),
        ("zip-missing", "404", "RESOURCE_DOES_NOT_EXIST"),
        ("mlflow-missing", "nope", "RESOURCE_DOES_NOT_EXIST"),
        ("zip-unreachable", "127.0.0.1", "TEMPORARILY_UNAVAILABLE"),
    ],
)
def test_unavailable_sources_preserve_the_underlying_error(
    kind, detail, error_code, make_source, tmp_path, closed_port
):
    match kind:
        case "git-missing-ref":
            source = GitSource(url=make_source("git").url, ref="no-such-ref")
        case "oci-missing-tag":
            source = OCISource(image=make_source("oci").image.replace(":v1", ":v9"))
        case "zip-missing":
            source = ZipSource(url=make_source("zip").url.replace("skills.zip", "gone.zip"))
        case "mlflow-missing":
            source = MlflowSource(artifact_path=make_source("mlflow").artifact_path + "/nope")
        case "zip-unreachable":
            source = ZipSource(url=f"http://127.0.0.1:{closed_port}/skills.zip")

    with pytest.raises(MlflowException, match="Failed to fetch skill content") as exc:
        pull_skill_version(_version(source), tmp_path / "out")

    assert detail in exc.value.message
    assert exc.value.error_code == error_code
    assert not os.path.lexists(tmp_path / "out")


def test_source_errors_never_expose_credentials(tmp_path):
    source = ZipSource(url="https://example.com/skills.zip?X-Amz-Signature=presigned-secret")
    error = source_unavailable(
        source.url, "401 for Authorization: Bearer token-secret", error_code=UNAUTHENTICATED
    )
    with (
        mock.patch("mlflow.genai.skill_content.fetchers._fetch_remote", side_effect=error),
        pytest.raises(MlflowException, match="Failed to fetch skill content") as exc,
    ):
        pull_skill_version(_version(source), tmp_path / "out")
    assert exc.value.error_code == "UNAUTHENTICATED"
    assert "secret" not in exc.value.message
    assert "401" in exc.value.message


@pytest.mark.parametrize("kind", SOURCE_KINDS)
@pytest.mark.parametrize("entry", ["symlink", "hardlink", "fifo"])
def test_links_and_special_files_are_rejected_for_every_source_type(kind, entry, tmp_path):
    if entry == "fifo" and not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs are not available on this platform")
    source = {
        "git": GitSource(url="https://example.com/skills.git"),
        "oci": OCISource(image="oci://registry.example.com/skills/demo:v1"),
        "zip": ZipSource(url="https://example.com/skills.zip"),
        "mlflow": MlflowSource(artifact_path="mlflow-artifacts:/skills/demo/token"),
    }[kind]

    def fetch(resolved, dest, scratch, limit):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "SKILL.md").write_text("---\nname: demo\n---\n")
        match entry:
            case "symlink":
                (dest / "link").symlink_to(dest / "SKILL.md")
            case "hardlink":
                os.link(dest / "SKILL.md", dest / "link")
            case "fifo":
                os.mkfifo(dest / "link")
        return dest

    with (
        mock.patch(
            "mlflow.genai.skill_content.fetchers._fetch_remote", side_effect=fetch
        ) as fetch_remote,
        pytest.raises(MlflowException, match="link|regular files") as exc,
    ):
        pull_skill_version(_version(source), tmp_path / "out")
    fetch_remote.assert_called_once()
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    assert not os.path.lexists(tmp_path / "out")


def test_git_symlink_in_repository_is_rejected(tmp_path, skill_tree):
    repo = tmp_path / "linked.git"
    shutil.copytree(skill_tree, repo)
    (repo / SUBPATH / "link").symlink_to("SKILL.md")
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "v1", cwd=repo)

    with pytest.raises(MlflowException, match="symbolic link"):
        pull_skill_version(
            _version(GitSource(url=f"file://{repo}", subpath=SUBPATH)), tmp_path / "out"
        )
    assert not os.path.lexists(tmp_path / "out")


def test_zip_symlink_entry_is_rejected(zip_server, tmp_path):
    base_url, serve_dir = zip_server
    with zipfile.ZipFile(serve_dir / "linked.zip", "w") as archive:
        archive.writestr("SKILL.md", "---\nname: demo\n---\n")
        link = zipfile.ZipInfo("link")
        link.external_attr = (0o120777 << 16) | 0x20
        archive.writestr(link, "SKILL.md")

    with pytest.raises(MlflowException, match="link"):
        pull_skill_version(_version(ZipSource(url=f"{base_url}/linked.zip")), tmp_path / "out")
    assert not os.path.lexists(tmp_path / "out")


def _make_destination(tmp_path, state):
    path = tmp_path / "dest"
    match state:
        case "file":
            path.write_text("x")
        case "non-empty-dir":
            path.mkdir()
            (path / "keep.txt").write_text("keep")
        case "hidden-entry-dir":
            path.mkdir()
            (path / ".keep").write_text("keep")
        case "symlink-to-empty-dir":
            (tmp_path / "real").mkdir()
            path.symlink_to(tmp_path / "real")
        case "symlink-to-file":
            (tmp_path / "real-file").write_text("x")
            path.symlink_to(tmp_path / "real-file")
        case "broken-symlink":
            path.symlink_to(tmp_path / "missing")
        case "fifo":
            os.mkfifo(path)
    return path


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        ("file", "exists and is not a directory"),
        ("non-empty-dir", "not empty"),
        ("hidden-entry-dir", "not empty"),
        ("symlink-to-empty-dir", "symbolic link"),
        ("symlink-to-file", "symbolic link"),
        ("broken-symlink", "symbolic link"),
        ("fifo", "exists and is not a directory"),
    ],
)
def test_rejected_destinations_are_untouched_and_nothing_is_fetched(state, reason, tmp_path):
    if state == "fifo" and not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs are not available on this platform")
    destination = _make_destination(tmp_path, state)
    before = os.lstat(destination)
    before_listing = sorted(os.listdir(tmp_path))

    with (
        mock.patch("mlflow.genai.skill_content.pull.fetch_source") as fetch,
        pytest.raises(MlflowException, match=reason) as exc,
    ):
        pull_skill_version(_version(GitSource(url="https://example.com/s.git")), destination)

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    fetch.assert_not_called()
    after = os.lstat(destination)
    assert (after.st_mode, after.st_ino, after.st_mtime_ns) == (
        before.st_mode,
        before.st_ino,
        before.st_mtime_ns,
    )
    assert sorted(os.listdir(tmp_path)) == before_listing


def test_destination_under_a_file_is_rejected(tmp_path):
    (tmp_path / "file").write_text("x")
    with pytest.raises(MlflowException, match="Cannot pull into"):
        pull_skill_version(
            _version(GitSource(url="https://example.com/s.git")), tmp_path / "file" / "out"
        )


def test_missing_parents_are_created_and_removed_again_on_failure(make_source, tmp_path):
    destination = tmp_path / "a" / "b" / "out"
    with pytest.raises(MlflowException, match="recorded digest"):
        pull_skill_version(_version(make_source("git", SUBPATH), digest="0" * 64), destination)
    assert not (tmp_path / "a").exists()

    pull_skill_version(_version(make_source("git", SUBPATH)), destination)
    assert (destination / "SKILL.md").exists()


def test_destination_appearing_during_fetch_is_not_overwritten(make_source, skill_tree, tmp_path):
    destination = tmp_path / "out"

    def racing_digest(root):
        destination.mkdir()
        (destination / "theirs.txt").write_text("theirs")
        return compute_tree_digest(root)

    version = _version(
        make_source("git", SUBPATH), digest=compute_tree_digest(skill_tree / SUBPATH)
    )
    with (
        mock.patch(
            "mlflow.genai.skill_content.pull.compute_tree_digest", side_effect=racing_digest
        ) as digest,
        pytest.raises(MlflowException, match="Failed to write pulled skill content"),
    ):
        pull_skill_version(version, destination)
    digest.assert_called_once()
    assert _listing(destination) == ["theirs.txt"]
    _assert_no_staging_left(tmp_path)


def test_failed_write_into_empty_directory_restores_it(make_source, tmp_path):
    destination = tmp_path / "empty"
    destination.mkdir()
    real_copy = shutil.copyfileobj
    calls = []

    def flaky_copy(src, dst):
        calls.append(dst)
        if len(calls) == 2:
            dst.write(b"half")
            raise OSError(28, "No space left on device")
        return real_copy(src, dst)

    with (
        mock.patch(
            "mlflow.genai.skill_content.pull.shutil.copyfileobj", side_effect=flaky_copy
        ) as copy,
        pytest.raises(MlflowException, match="No space left on device"),
    ):
        pull_skill_version(_version(make_source("git", SUBPATH)), destination)
    assert copy.call_count == 2
    assert list(destination.iterdir()) == []


def test_cross_device_publication_copies_and_cleans_up(make_source, skill_tree, tmp_path):
    destination = tmp_path / "out"
    with mock.patch.object(
        Path, "rename", side_effect=OSError(18, "Invalid cross-device link")
    ) as rename:
        pull_skill_version(_version(make_source("git", SUBPATH)), destination)
    rename.assert_called_once()
    assert compute_tree_digest(destination) == compute_tree_digest(skill_tree / SUBPATH)
    _assert_no_staging_left(tmp_path)


@pytest.mark.parametrize(
    "artifact_path",
    ["/etc", "./relative/dir", "~/secrets", "C:\\Users\\me"],
)
def test_mlflow_source_never_reads_the_local_filesystem(artifact_path, tmp_path):
    version = _version(MlflowSource(artifact_path=artifact_path))
    with (
        mock.patch("mlflow.genai.skill_content.pull.fetch_source") as fetch,
        pytest.raises(MlflowException, match="not an MLflow artifact location"),
    ):
        pull_skill_version(version, tmp_path / "out")
    fetch.assert_not_called()


@pytest.mark.parametrize("source", [None, "skills:/other/1"])
def test_versions_without_a_fetchable_source_are_rejected(source, tmp_path):
    version = SkillVersion(name="demo", version=1, source=source)
    with pytest.raises(MlflowException, match="no source that can be pulled"):
        pull_skill_version(version, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("organization", ["", "acme"])
@pytest.mark.parametrize(
    ("suffix", "method", "kwargs"),
    [
        ("/3", "get_skill_version", {"version": 3}),
        ("@production", "get_skill_version_by_alias", {"alias": "production"}),
        ("", "get_latest_skill_version", {}),
    ],
)
def test_resolve_skill_version(organization, suffix, method, kwargs):
    prefix = f"@{organization}/" if organization else ""
    client = mock.Mock()
    resolved = resolve_skill_version(f"skills:/{prefix}demo{suffix}", client)
    getattr(client, method).assert_called_once_with(
        name="demo", organization=organization, **kwargs
    )
    assert resolved is getattr(client, method).return_value


@pytest.mark.parametrize(
    ("uri", "message"),
    [
        ("models:/demo/1", "expected it to start with 'skills:/'"),
        ("skills:/", "missing name"),
        ("skills:/demo/01", "Invalid skill version '01'"),
        ("skills:/Demo", "Invalid skill name 'Demo'"),
        ("skills:/demo@latest", "'latest' alias name .* is reserved"),
    ],
)
def test_resolve_rejects_invalid_uris_before_any_request(uri, message):
    client = mock.Mock()
    with pytest.raises(MlflowException, match=message):
        resolve_skill_version(uri, client)
    assert client.mock_calls == []


def test_registry_authorization_failure_writes_nothing(tmp_path):
    client = mock.Mock(
        get_skill_version=mock.Mock(
            side_effect=MlflowException("Permission denied", error_code=PERMISSION_DENIED)
        )
    )
    with mock.patch("mlflow.genai.skill_content.pull.MlflowClient", return_value=client):
        with pytest.raises(MlflowException, match="Permission denied") as exc:
            mlflow.genai.pull("skills:/demo/1", destination=tmp_path / "out")
    assert exc.value.error_code == ErrorCode.Name(PERMISSION_DENIED)
    client.get_skill_version.assert_called_once()
    assert not (tmp_path / "out").exists()


def test_genai_pull_defaults_to_skill_name_in_cwd(make_source, skill_tree, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = skill_tree / SUBPATH
    version = _version(make_source("git", SUBPATH), digest=compute_tree_digest(expected))
    client = mock.Mock(get_latest_skill_version=mock.Mock(return_value=version))
    with mock.patch("mlflow.genai.skill_content.pull.MlflowClient", return_value=client):
        path = mlflow.genai.pull("skills:/demo")
    client.get_latest_skill_version.assert_called_once_with(name="demo", organization="")
    assert path == str(tmp_path / "demo")
    assert compute_tree_digest(path) == version.digest


def test_destination_with_parent_segments_is_normalized(make_source, tmp_path):
    (tmp_path / "a").mkdir()
    path = pull_skill_version(_version(make_source("git", SUBPATH)), tmp_path / "a" / ".." / "out")
    assert path == tmp_path / "out"
    assert (tmp_path / "out" / "SKILL.md").exists()
    _assert_no_staging_left(tmp_path)


@pytest.mark.parametrize("url", ["/srv/skills.git", "./skills.git", "~/skills.git"])
def test_git_source_that_is_a_local_path_is_refused(url, tmp_path):
    with (
        mock.patch("mlflow.genai.skill_content.pull.fetch_source") as fetch,
        pytest.raises(MlflowException, match="Git source that is a local path"),
    ):
        pull_skill_version(_version(GitSource(url=url)), tmp_path / "out")
    fetch.assert_not_called()


@pytest.mark.parametrize("existing", [False, True])
def test_parent_symlink_followed_by_dotdot_resolves_through_the_link(
    existing, make_source, tmp_path
):
    # current -> releases/v1, so current/../skills is releases/skills, not ./skills.
    (tmp_path / "releases" / "v1").mkdir(parents=True)
    (tmp_path / "current").symlink_to(tmp_path / "releases" / "v1")
    if existing:
        (tmp_path / "releases" / "skills").mkdir()
        (tmp_path / "releases" / "skills" / "keep.txt").write_text("keep")
    destination = tmp_path / "current" / ".." / "skills"

    if existing:
        with pytest.raises(MlflowException, match="not empty"):
            pull_skill_version(_version(make_source("git", SUBPATH)), destination)
        assert _listing(tmp_path / "releases" / "skills") == ["keep.txt"]
    else:
        path = pull_skill_version(_version(make_source("git", SUBPATH)), destination)
        assert path == (tmp_path / "releases" / "skills").resolve()
        assert (tmp_path / "releases" / "skills" / "SKILL.md").exists()
    assert not (tmp_path / "skills").exists()


class _UnauthorizedArtifactsHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(401)
        self.send_header("Content-Length", "0")
        self.end_headers()


def test_mlflow_artifact_auth_failure_does_not_expose_tracking_credentials(tmp_path):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _UnauthorizedArtifactsHandler)
    threading.Thread(target=server.serve_forever, name="skill-pull-401", daemon=True).start()
    tracking_uri = f"http://user:hunter2-secret@127.0.0.1:{server.server_address[1]}"
    mlflow.set_tracking_uri(tracking_uri)
    try:
        with pytest.raises(MlflowException, match="Failed to fetch skill content") as exc:
            pull_skill_version(
                _version(MlflowSource(artifact_path="mlflow-artifacts:/skills/demo/token")),
                tmp_path / "out",
            )
    finally:
        mlflow.set_tracking_uri(None)
        server.shutdown()
        server.server_close()
    assert "hunter2-secret" not in exc.value.message
    assert "hunter2-secret" not in "".join(traceback.format_exception(exc.value))
    assert "401" in exc.value.message
    assert exc.value.error_code == "UNAUTHENTICATED"
    assert not os.path.lexists(tmp_path / "out")


@pytest.mark.parametrize("racer", ["file", "directory"])
def test_entries_created_concurrently_in_empty_destination_are_never_replaced(
    racer, make_source, tmp_path
):
    # Another process creates names in the destination after the publish-time check.
    destination = tmp_path / "empty"
    destination.mkdir()
    real_copy = pull_module._copy_tree_exclusive

    def race_then_copy(source, target_dir, created):
        if target_dir == destination:
            match racer:
                case "file":
                    (destination / "SKILL.md").write_text("theirs")
                case "directory":
                    (destination / "scripts").mkdir()
                    (destination / "scripts" / "theirs.txt").write_text("theirs")
        return real_copy(source, target_dir, created)

    with (
        mock.patch.object(pull_module, "_copy_tree_exclusive", side_effect=race_then_copy) as copy,
        pytest.raises(MlflowException, match="Failed to write pulled skill content"),
    ):
        pull_skill_version(_version(make_source("git", SUBPATH)), destination)

    copy.assert_called()
    if racer == "file":
        assert _listing(destination) == ["SKILL.md"]
        assert (destination / "SKILL.md").read_text() == "theirs"
    else:
        assert _listing(destination) == ["scripts", "scripts/theirs.txt"]
    _assert_no_staging_left(tmp_path)


def test_zip_redirect_connection_failure_does_not_expose_signed_query(tmp_path, closed_port):
    signed_target = (
        f"http://127.0.0.1:{closed_port}/cdn/skills.zip"
        "?X-Amz-Credential=AKIAEXAMPLE&X-Amz-Signature=signed-secret"
    )

    class _RedirectToSigned(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", signed_target)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RedirectToSigned)
    threading.Thread(target=server.serve_forever, name="skill-pull-redirect", daemon=True).start()
    try:
        with pytest.raises(MlflowException, match="Failed to fetch skill content") as exc:
            pull_skill_version(
                _version(ZipSource(url=f"http://127.0.0.1:{server.server_address[1]}/s.zip")),
                tmp_path / "out",
            )
    finally:
        server.shutdown()
        server.server_close()
    assert "signed-secret" not in exc.value.message
    assert "signed-secret" not in "".join(traceback.format_exception(exc.value))
    assert "AKIAEXAMPLE" not in exc.value.message
    assert "/cdn/skills.zip" in exc.value.message
    assert exc.value.__cause__ is None
    assert exc.value.__suppress_context__
    assert exc.value.error_code == "TEMPORARILY_UNAVAILABLE"
    assert not os.path.lexists(tmp_path / "out")


def test_oci_blob_redirect_connection_failure_hides_signature_from_traceback(
    tmp_path, skill_tree, closed_port
):
    layer = package_skill_tree(skill_tree, tmp_path / "layer.tar.gz").read_bytes()
    layer_digest = _sha256(layer)
    manifest = json.dumps({
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {"mediaType": "application/vnd.oci.empty.v1+json", "digest": _sha256(b"{}")},
        "layers": [
            {
                "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                "digest": layer_digest,
                "size": len(layer),
            }
        ],
    }).encode()
    signed_target = (
        f"http://127.0.0.1:{closed_port}/blob"
        "?X-Amz-Credential=AKIAEXAMPLE&X-Amz-Signature=signed-secret"
    )

    class _RedirectingRegistry(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/v2/skills/demo/manifests/v1":
                self.send_response(200)
                self.send_header("Content-Type", "application/vnd.oci.image.manifest.v1+json")
                self.send_header("Content-Length", str(len(manifest)))
                self.end_headers()
                self.wfile.write(manifest)
                return
            # Registries hand blob pulls to a content server through a signed redirect.
            self.send_response(307)
            self.send_header("Location", signed_target)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RedirectingRegistry)
    threading.Thread(target=server.serve_forever, name="skill-pull-oci-307", daemon=True).start()
    try:
        with pytest.raises(MlflowException, match="Failed to fetch skill content") as exc:
            pull_skill_version(
                _version(
                    OCISource(image=f"oci://127.0.0.1:{server.server_address[1]}/skills/demo:v1")
                ),
                tmp_path / "out",
            )
    finally:
        server.shutdown()
        server.server_close()
    rendered = "".join(traceback.format_exception(exc.value))
    assert "signed-secret" not in rendered
    assert "AKIAEXAMPLE" not in rendered
    assert "signed-secret" not in exc.value.message
    assert exc.value.error_code == "TEMPORARILY_UNAVAILABLE"
    assert not os.path.lexists(tmp_path / "out")

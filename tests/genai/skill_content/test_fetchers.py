import http.server
import os
import shutil
import subprocess
import threading
import zipfile
from functools import partial
from pathlib import Path

import pytest
import requests

import mlflow
from mlflow.entities.skill_source import GitSource, OCISource, SkillSourceType, ZipSource
from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.digest import compute_tree_digest
from mlflow.genai.skill_content.fetchers import fetch_source
from mlflow.genai.skill_content.fetchers.git import _error_code_for_git
from mlflow.genai.skill_content.fetchers.zip import download_with_budget
from mlflow.protos.databricks_pb2 import ErrorCode

from tests.genai.skill_content.conftest import SKILL_MD


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
    (repo / "skills" / "demo" / "SKILL.md").write_text(SKILL_MD + "\nsecond revision\n")
    _git("commit", "-q", "-am", "v2", cwd=repo)
    return repo


class _RecordingHTTPHandler(http.server.SimpleHTTPRequestHandler):
    authorizations = []
    redirects = {}
    redirect_body = b""

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.authorizations.append(self.headers.get("Authorization"))
        if (target := self.redirects.get(self.path)) is not None:
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", str(len(self.redirect_body)))
            self.end_headers()
            try:
                self.wfile.write(self.redirect_body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        super().do_GET()


@pytest.fixture
def http_server(tmp_path):
    serve_dir = tmp_path / "www"
    serve_dir.mkdir()
    handler_cls = type("Handler", (_RecordingHTTPHandler,), {"authorizations": [], "redirects": {}})
    handler = partial(handler_cls, directory=str(serve_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, name="skill-content-http", daemon=True)
    thread.start()
    try:
        yield serve_dir, f"http://127.0.0.1:{server.server_address[1]}", handler_cls
    finally:
        server.shutdown()
        server.server_close()


# --- local ------------------------------------------------------------------------------------


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


def test_fetch_local_path_size_limit_applies_to_subpath(skill_tree):
    with pytest.raises(MlflowException, match="exceeds the size limit"):
        fetch_source(str(skill_tree), max_bytes=200)
    with fetch_source(str(skill_tree), subpath="skills/demo", max_bytes=200) as fetched:
        assert (fetched.root / "SKILL.md").exists()


# --- git --------------------------------------------------------------------------------------


def test_fetch_git_by_ref_and_head(git_repo, skill_tree):
    source = GitSource(url=f"file://{git_repo}", ref="v1", subpath="skills/demo")
    with fetch_source(source) as fetched:
        temp_root = fetched.root
        assert (fetched.root / "SKILL.md").read_text() == SKILL_MD
        assert not (fetched.root.parent.parent / ".git").exists()
        assert compute_tree_digest(fetched.root) == compute_tree_digest(
            skill_tree / "skills" / "demo"
        )
    assert not temp_root.exists()

    with fetch_source(f"file://{git_repo}", subpath="skills/demo") as fetched:
        assert fetched.resolved.source_type == SkillSourceType.GIT
        assert "second revision" in (fetched.root / "SKILL.md").read_text()


def test_fetch_git_missing_ref(git_repo):
    with pytest.raises(MlflowException, match="Failed to fetch skill content.*ref 'nope'") as exc:
        fetch_source(GitSource(url=f"file://{git_repo}", ref="nope"))
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_fetch_git_size_limit_applies_to_subpath(git_repo):
    with pytest.raises(MlflowException, match="exceeds the skill content size limit"):
        fetch_source(GitSource(url=f"file://{git_repo}"), max_bytes=200)
    with fetch_source(
        GitSource(url=f"file://{git_repo}", subpath="skills/demo"), max_bytes=200
    ) as f:
        assert (f.root / "SKILL.md").exists()


def test_fetch_git_rejects_option_like_ref(git_repo):
    with pytest.raises(MlflowException, match="must not start with '-'"):
        fetch_source(GitSource(url=f"file://{git_repo}", ref="--upload-pack=evil"))


def test_fetch_git_redacts_credentials_and_reports_availability(closed_port):
    url = f"https://user:s3cret-token@127.0.0.1:{closed_port}/skills.git"
    with pytest.raises(MlflowException, match="Failed to fetch skill content") as exc_info:
        fetch_source(url)
    message = str(exc_info.value)
    assert "s3cret-token" not in message
    assert "***@127.0.0.1" in message
    assert exc_info.value.error_code == "TEMPORARILY_UNAVAILABLE"


# --- zip --------------------------------------------------------------------------------------


@pytest.mark.no_mock_requests_get
def test_fetch_zip(http_server, skill_tree):
    serve_dir, base_url, _ = http_server
    shutil.make_archive(str(serve_dir / "skills"), "zip", root_dir=skill_tree)
    with fetch_source(ZipSource(url=f"{base_url}/skills.zip", subpath="skills/demo")) as fetched:
        assert compute_tree_digest(fetched.root) == compute_tree_digest(
            skill_tree / "skills" / "demo"
        )
        assert not (fetched.root.parent.parent.parent / "source.zip").exists()
        assert not (fetched.root.parent.parent / "big.bin").exists()


@pytest.mark.no_mock_requests_get
def test_fetch_zip_subpath_limits_budget(http_server, skill_tree):
    serve_dir, base_url, _ = http_server
    shutil.make_archive(str(serve_dir / "skills"), "zip", root_dir=skill_tree)
    with pytest.raises(MlflowException, match="exceeds the skill content size limit"):
        fetch_source(f"{base_url}/skills.zip", max_bytes=2000)
    with fetch_source(f"{base_url}/skills.zip", subpath="skills/demo", max_bytes=2000) as fetched:
        assert (fetched.root / "SKILL.md").exists()


@pytest.mark.no_mock_requests_get
def test_fetch_zip_sends_no_credentials(http_server, skill_tree, tmp_path, monkeypatch):
    serve_dir, base_url, handler = http_server
    shutil.make_archive(str(serve_dir / "skills"), "zip", root_dir=skill_tree)
    netrc = tmp_path / "netrc"
    netrc.write_text("machine 127.0.0.1 login netrc-user password netrc-pass\n")
    monkeypatch.setenv("NETRC", str(netrc))
    with fetch_source(f"{base_url}/skills.zip", subpath="skills/demo"):
        pass
    assert handler.authorizations == [None]

    with pytest.raises(MlflowException, match="publicly accessible"):
        fetch_source(base_url.replace("http://", "http://u:p@") + "/skills.zip")


@pytest.mark.no_mock_requests_get
def test_fetch_zip_redirect_never_uses_netrc(http_server, skill_tree, tmp_path, monkeypatch):
    # `requests` re-applies netrc credentials when it rebuilds a redirected request; the
    # public-only policy must hold on every hop, not just the first.
    serve_dir, base_url, handler = http_server
    shutil.make_archive(str(serve_dir / "skills"), "zip", root_dir=skill_tree)
    handler.redirects["/redirect.zip"] = "/skills.zip"
    netrc = tmp_path / "netrc"
    netrc.write_text("machine 127.0.0.1 login netrc-user password netrc-pass\n")
    monkeypatch.setenv("NETRC", str(netrc))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    with fetch_source(f"{base_url}/redirect.zip", subpath="skills/demo") as fetched:
        assert (fetched.root / "SKILL.md").exists()
    assert handler.authorizations == [None, None]


@pytest.mark.no_mock_requests_get
def test_fetch_zip_redirect_body_is_not_buffered(http_server, skill_tree, monkeypatch):
    # `requests` reads each redirect body before following it; the download budget must not
    # be bypassable through a redirect that carries a large body.
    serve_dir, base_url, handler = http_server
    shutil.make_archive(str(serve_dir / "skills"), "zip", root_dir=skill_tree)
    handler.redirects["/redirect.zip"] = "/skills.zip"
    handler.redirect_body = b"x" * 65536
    responses = []
    original = requests.adapters.HTTPAdapter.build_response

    def record_response(adapter, request, raw):
        response = original(adapter, request, raw)
        responses.append(response)
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "build_response", record_response)
    with fetch_source(f"{base_url}/redirect.zip", subpath="skills/demo", max_bytes=8192) as f:
        assert (f.root / "SKILL.md").exists()
    redirects = [response for response in responses if response.is_redirect]
    assert redirects
    # The closed redirect response yields no body: `_content` stays unset or reads as empty.
    assert all(not response._content for response in redirects)


@pytest.mark.no_mock_requests_get
def test_fetch_zip_errors(http_server, skill_tree):
    serve_dir, base_url, _ = http_server
    with pytest.raises(MlflowException, match="HTTP 404") as exc:
        fetch_source(f"{base_url}/missing.zip")
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"

    with zipfile.ZipFile(serve_dir / "evil.zip", "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(MlflowException, match="unsafe path"):
        fetch_source(f"{base_url}/evil.zip")


@pytest.mark.no_mock_requests_get
def test_fetch_zip_unreachable(closed_port):
    with pytest.raises(MlflowException, match="Failed to fetch skill content") as exc:
        fetch_source(f"http://127.0.0.1:{closed_port}/skills.zip")
    assert exc.value.error_code == "TEMPORARILY_UNAVAILABLE"


# --- oci --------------------------------------------------------------------------------------


def test_fetch_oci_not_supported_yet():
    # The source type resolves so callers get a precise message, but no registry is contacted.
    with pytest.raises(MlflowException, match="OCI sources are not supported yet"):
        fetch_source(OCISource(image="oci://ghcr.io/acme/skills:v1"))
    with pytest.raises(MlflowException, match="OCI sources are not supported yet"):
        fetch_source("oci://ghcr.io/acme/skills:v1")


# --- mlflow artifacts -------------------------------------------------------------------------


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


def test_fetch_mlflow_artifacts_subpath_downloads_only_subtree(skill_tree):
    with mlflow.start_run() as run:
        mlflow.log_artifacts(str(skill_tree), artifact_path="pkg")
    uri = f"runs:/{run.info.run_id}/pkg"
    with pytest.raises(MlflowException, match="exceeds"):
        fetch_source(uri, max_bytes=2000)
    with fetch_source(uri, subpath="skills/demo", max_bytes=2000) as fetched:
        assert (fetched.root / "SKILL.md").exists()
        assert not (fetched.root.parent.parent / "big.bin").exists()


def test_fetch_mlflow_artifacts_missing():
    with pytest.raises(MlflowException, match="Failed to fetch skill content") as exc:
        fetch_source("runs:/does-not-exist/skill")
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX shell hook fixture")
def test_fetch_git_does_not_execute_repository_hooks(tmp_path, monkeypatch):
    # A global `core.hooksPath=.githooks` would otherwise run a hook shipped inside the fetched
    # repository during checkout, before any validation.
    config = tmp_path / "gitconfig"
    config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    marker = tmp_path / "hook-ran"
    monkeypatch.setenv("SKILL_TEST_HOOK_MARKER", str(marker))
    repo = tmp_path / "fixture.git"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    (repo / "SKILL.md").write_text(SKILL_MD)
    hooks = repo / ".githooks"
    hooks.mkdir()
    hook = hooks / "post-checkout"
    hook.write_text('#!/bin/sh\nprintf "ran\\n" > "$SKILL_TEST_HOOK_MARKER"\n')
    hook.chmod(0o755)
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "fixture", cwd=repo)
    config.write_text("[core]\n    hooksPath = .githooks\n")

    with fetch_source(GitSource(url=repo.as_uri())) as fetched:
        assert (fetched.root / "SKILL.md").is_file()
    assert not marker.exists()


def test_fetch_git_same_commit_same_digest_regardless_of_autocrlf(tmp_path, monkeypatch):
    # The digest hashes committed bytes; a caller's checkout settings must not rewrite them.
    config = tmp_path / "gitconfig"
    config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = tmp_path / "fixture.git"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    (repo / "SKILL.md").write_bytes(b"---\nname: demo\n---\nHello\n")
    _git("add", "SKILL.md", cwd=repo)
    _git("commit", "-q", "-m", "fixture", cwd=repo)
    digests = []
    for setting in ("false", "true"):
        config.write_text(f"[core]\n    autocrlf = {setting}\n")
        with fetch_source(GitSource(url=repo.as_uri())) as fetched:
            assert (fetched.root / "SKILL.md").read_bytes() == b"---\nname: demo\n---\nHello\n"
            digests.append(compute_tree_digest(fetched.root))
    assert digests[0] == digests[1]


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX shell filter fixture")
def test_fetch_git_does_not_run_filter_drivers(tmp_path, monkeypatch):
    # A repository's .gitattributes can select any filter driver the caller has configured;
    # reading blobs from the object store instead of checking out means it never runs and the
    # committed bytes are what land on disk.
    marker = tmp_path / "filter-ran"
    script = tmp_path / "review-smudge"
    script.write_text(f'#!/bin/sh\ntouch "{marker}"\ncat\n')
    script.chmod(0o755)
    config = tmp_path / "gitconfig"
    config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = tmp_path / "fixture.git"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    (repo / "SKILL.md").write_text(SKILL_MD)
    (repo / "payload").write_bytes(b"raw\r\nbytes")
    (repo / ".gitattributes").write_text("payload filter=review\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "fixture", cwd=repo)
    config.write_text(f'[filter "review"]\n    smudge = {script}\n    required = true\n')

    with fetch_source(GitSource(url=repo.as_uri())) as fetched:
        assert (fetched.root / "payload").read_bytes() == b"raw\r\nbytes"
    assert not marker.exists()


def test_fetch_git_rejects_symlinks_and_preserves_exec_bit(tmp_path):
    repo = tmp_path / "fixture.git"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    (repo / "SKILL.md").write_text(SKILL_MD)
    (repo / "run.sh").write_text("#!/bin/sh\n")
    (repo / "run.sh").chmod(0o755)
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "fixture", cwd=repo)
    with fetch_source(GitSource(url=repo.as_uri())) as fetched:
        assert os.access(fetched.root / "run.sh", os.X_OK)

    (repo / "link.md").symlink_to("SKILL.md")
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "link", cwd=repo)
    with pytest.raises(MlflowException, match="symbolic links: 'link.md'"):
        fetch_source(GitSource(url=repo.as_uri()))


def test_fetch_git_subpath_must_exist_and_be_a_directory(git_repo):
    with pytest.raises(MlflowException, match="does not exist") as exc:
        fetch_source(GitSource(url=f"file://{git_repo}", subpath="skills/nope"))
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"
    with pytest.raises(MlflowException, match="must point to a directory"):
        fetch_source(GitSource(url=f"file://{git_repo}", subpath="skills/demo/SKILL.md"))


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        (
            "fatal: unable to access 'https://h/r.git/': The requested URL returned error: 403",
            "PERMISSION_DENIED",
        ),
        (
            "fatal: unable to access 'https://h/r.git/': The requested URL returned error: 401",
            "UNAUTHENTICATED",
        ),
        (
            "fatal: unable to access 'https://h/r.git/': The requested URL returned error: 404",
            "RESOURCE_DOES_NOT_EXIST",
        ),
        ("fatal: Authentication failed for 'https://h/r.git/'", "UNAUTHENTICATED"),
        (
            "fatal: unable to access 'https://h/r.git/': Could not resolve host: h",
            "TEMPORARILY_UNAVAILABLE",
        ),
        ("fatal: couldn't find remote ref nope", "RESOURCE_DOES_NOT_EXIST"),
    ],
)
def test_git_error_codes(detail, expected):
    assert ErrorCode.Name(_error_code_for_git(detail)) == expected


def test_fetch_git_partial_fetch_skips_blobs_outside_subpath(git_repo):
    # With partial clone enabled on the server, only the commit, trees, and the subpath's blobs
    # are transferred; the 5000-byte big.bin outside the subpath never comes down.
    _git("config", "uploadpack.allowFilter", "true", cwd=git_repo)
    with fetch_source(GitSource(url=f"file://{git_repo}", subpath="skills/demo")) as fetched:
        scratch = fetched.root.parent.parent.parent / "git-objects"
        counts = subprocess.run(
            ["git", "count-objects", "-v"], cwd=scratch, capture_output=True, text=True, check=True
        ).stdout
        size_pack_kb = int(
            next(line.split()[1] for line in counts.splitlines() if line.startswith("size-pack"))
        )
        assert size_pack_kb < 4
        assert (fetched.root / "SKILL.md").exists()


def test_download_with_budget_refuses_credentialed_urls(tmp_path):
    with pytest.raises(MlflowException, match="publicly accessible"):
        download_with_budget(
            "https://u:p@example.invalid/skills.zip", tmp_path / "x.zip", max_bytes=10
        )

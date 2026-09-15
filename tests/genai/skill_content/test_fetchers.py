import http.server
import os
import shutil
import subprocess
import threading
import zipfile
from functools import partial
from pathlib import Path
from unittest import mock

import pytest
import requests

import mlflow
from mlflow.entities.file_info import FileInfo
from mlflow.entities.skill_source import GitSource, SkillSourceType, ZipSource
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
        with fetch_source(str(skill_tree), subpath="skills/nope"):
            pass
    (skill_tree / "skills" / "demo" / "link").symlink_to(skill_tree / "README.md")
    with pytest.raises(MlflowException, match="symbolic links"):
        with fetch_source(str(skill_tree), subpath="skills/demo"):
            pass


def test_fetch_local_path_size_limit_applies_to_subpath(skill_tree):
    with pytest.raises(MlflowException, match="exceeds the size limit"):
        with fetch_source(str(skill_tree), max_bytes=200):
            pass
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
        with fetch_source(GitSource(url=f"file://{git_repo}", ref="nope")):
            pass
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_fetch_git_size_limit_applies_to_subpath(git_repo):
    with pytest.raises(MlflowException, match="exceeds the skill content size limit"):
        with fetch_source(GitSource(url=f"file://{git_repo}"), max_bytes=200):
            pass
    with fetch_source(
        GitSource(url=f"file://{git_repo}", subpath="skills/demo"), max_bytes=200
    ) as f:
        assert (f.root / "SKILL.md").exists()


def test_fetch_git_rejects_option_like_ref(git_repo):
    with pytest.raises(MlflowException, match="must not start with '-'"):
        with fetch_source(GitSource(url=f"file://{git_repo}", ref="--upload-pack=evil")):
            pass


def test_fetch_git_redacts_credentials_and_reports_availability(closed_port):
    url = f"https://user:s3cret-token@127.0.0.1:{closed_port}/skills.git"
    with pytest.raises(MlflowException, match="Failed to fetch skill content") as exc_info:
        with fetch_source(url):
            pass
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
        with fetch_source(f"{base_url}/skills.zip", max_bytes=2000):
            pass
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
        with fetch_source(base_url.replace("http://", "http://u:p@") + "/skills.zip"):
            pass


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
        with fetch_source(f"{base_url}/missing.zip"):
            pass
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"

    with zipfile.ZipFile(serve_dir / "evil.zip", "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(MlflowException, match="unsafe path"):
        with fetch_source(f"{base_url}/evil.zip"):
            pass


@pytest.mark.no_mock_requests_get
def test_fetch_zip_unreachable(closed_port):
    with pytest.raises(MlflowException, match="Failed to fetch skill content") as exc:
        with fetch_source(f"http://127.0.0.1:{closed_port}/skills.zip"):
            pass
    assert exc.value.error_code == "TEMPORARILY_UNAVAILABLE"


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
        with fetch_source(uri, max_bytes=2000):
            pass
    with fetch_source(uri, subpath="skills/demo", max_bytes=2000) as fetched:
        assert (fetched.root / "SKILL.md").exists()
        assert not (fetched.root.parent.parent / "big.bin").exists()


def test_fetch_mlflow_artifacts_rejects_oversized_tree_before_downloading(skill_tree):
    with mlflow.start_run() as run:
        mlflow.log_artifacts(str(skill_tree), artifact_path="pkg")
    uri = f"runs:/{run.info.run_id}/pkg"
    with mock.patch("mlflow.genai.skill_content.fetchers.artifacts.download_artifacts") as download:
        with pytest.raises(MlflowException, match="at least [0-9]+ bytes, which exceeds"):
            with fetch_source(uri, max_bytes=2000):
                pass
        download.assert_not_called()


def test_fetch_mlflow_artifacts_rejects_too_many_entries():
    listing = [FileInfo(path=f"skill/{i}.txt", is_dir=False, file_size=1) for i in range(10_001)]
    with (
        mock.patch(
            "mlflow.genai.skill_content.fetchers.artifacts.list_artifacts", return_value=listing
        ) as listed,
        mock.patch("mlflow.genai.skill_content.fetchers.artifacts.download_artifacts") as download,
    ):
        with pytest.raises(MlflowException, match="more than 10000 entries"):
            with fetch_source("runs:/run/skill"):
                pass
        listed.assert_called_once_with(artifact_uri="runs:/run/skill")
        download.assert_not_called()


def test_fetch_mlflow_artifacts_missing():
    with pytest.raises(MlflowException, match="Failed to fetch skill content") as exc:
        with fetch_source("runs:/does-not-exist/skill"):
            pass
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
        with fetch_source(GitSource(url=repo.as_uri())):
            pass


def test_fetch_git_subpath_must_exist_and_be_a_directory(git_repo):
    with pytest.raises(MlflowException, match="does not exist") as exc:
        with fetch_source(GitSource(url=f"file://{git_repo}", subpath="skills/nope")):
            pass
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"
    with pytest.raises(MlflowException, match="must point to a directory"):
        with fetch_source(GitSource(url=f"file://{git_repo}", subpath="skills/demo/SKILL.md")):
            pass


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
        assert scratch.is_dir()
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


def test_fetch_source_destination_keeps_content_after_exit(git_repo, tmp_path):
    destination = tmp_path / "cache" / "demo"
    with fetch_source(
        GitSource(url=f"file://{git_repo}", subpath="skills/demo"), destination=destination
    ) as fetched:
        assert fetched.root == destination / "skills" / "demo"
        scratch = fetched._tmpdir.name
        assert not (Path(scratch) / "content").exists()
    assert not Path(scratch).exists()
    assert (destination / "skills" / "demo" / "SKILL.md").read_text().startswith(SKILL_MD)
    assert not (destination / "big.bin").exists()


def test_fetch_source_destination_is_cleared_on_failure(git_repo, tmp_path):
    destination = tmp_path / "dest"
    destination.mkdir()
    with pytest.raises(MlflowException, match="exceeds"):
        with fetch_source(
            GitSource(url=f"file://{git_repo}"), destination=destination, max_bytes=200
        ):
            pass
    assert destination.is_dir()
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize("prepare", ["file", "non_empty"])
def test_fetch_source_destination_must_be_empty_directory(git_repo, tmp_path, prepare):
    destination = tmp_path / "dest"
    if prepare == "file":
        destination.write_text("x")
        message = "is not a directory"
    else:
        destination.mkdir()
        (destination / "stale").write_text("x")
        message = "must be empty"
    with pytest.raises(MlflowException, match=message):
        with fetch_source(GitSource(url=f"file://{git_repo}"), destination=destination):
            pass


def test_fetch_source_destination_rejected_for_local_sources(skill_tree, tmp_path):
    with pytest.raises(MlflowException, match="applies to remote sources only"):
        with fetch_source(str(skill_tree), destination=tmp_path / "dest"):
            pass
    assert not (tmp_path / "dest").exists()

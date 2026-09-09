import http.server
import shutil
import subprocess
import threading
import zipfile
from functools import partial
from pathlib import Path

import pytest

import mlflow
from mlflow.entities.skill_source import GitSource, SkillSourceType, ZipSource
from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.digest import compute_tree_digest
from mlflow.genai.skill_content.fetchers import fetch_source

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

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.authorizations.append(self.headers.get("Authorization"))
        if (target := self.redirects.get(self.path)) is not None:
            self.send_response(302)
            self.send_header("Location", target)
            self.end_headers()
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

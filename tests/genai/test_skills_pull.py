import shutil
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

import mlflow
from mlflow.cli import cli
from mlflow.entities.skill import SkillStatus
from mlflow.entities.skill_source import GitSource, SkillSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.digest import compute_tree_digest
from mlflow.tracking import MlflowClient

from tests.genai.skill_content.conftest import serve_git_repositories
from tests.server.skill_registry.live_server import serve_tracking_app

# tests/genai/conftest.py fakes `requests.get` in every test, for telemetry. The live server's
# readiness check calls `requests.get`, so without the real one it reports ready too early.
pytestmark = pytest.mark.no_mock_requests_get

SKILL_MD = "---\nname: review\ndescription: Reviews code\n---\n# Review\n"


def _git(*args, cwd):
    subprocess.check_call(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=cwd
    )


@pytest.fixture
def server_artifacts(tmp_path):
    # Where the server stores uploaded skill content, so a test can change it behind the server.
    return tmp_path / "server-artifacts"


@pytest.fixture
def tracking_server(db_uri, server_artifacts, monkeypatch):
    with serve_tracking_app(db_uri, server_artifacts, monkeypatch) as url:
        yield url


@pytest.fixture
def git_server(tmp_path):
    with serve_git_repositories(tmp_path) as url:
        yield url


@pytest.fixture
def client(tracking_server):
    mlflow.set_tracking_uri(tracking_server)
    try:
        yield MlflowClient()
    finally:
        mlflow.set_tracking_uri(None)


def _write_skill(root: Path, body: str) -> Path:
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(SKILL_MD + body)
    (root / "scripts").mkdir()
    (root / "scripts" / "run.py").write_text(f"print({body!r})\n")
    return root


def test_pull_uploaded_skill_by_version_alias_and_latest(client, tmp_path, request):
    name = "review"
    first = _write_skill(tmp_path / "v1", "first\n")
    second = _write_skill(tmp_path / "v2", "second\n")
    v1 = mlflow.genai.register_skill(source=str(first), name=name, organization="acme")
    v2 = mlflow.genai.register_skill(source=str(second), name=name, organization="acme")
    assert v1.source_type == SkillSourceType.MLFLOW
    client.set_skill_alias(name=name, alias="production", version=1, organization="acme")

    cases = [
        (f"skills:/@acme/{name}/1", first),
        (f"skills:/@acme/{name}@production", first),
        (f"skills:/@acme/{name}", second),
    ]
    for index, (uri, expected) in enumerate(cases):
        destination = tmp_path / f"out-{index}"
        assert mlflow.genai.pull(uri, destination=destination) == str(destination)
        assert compute_tree_digest(destination) == compute_tree_digest(expected)

    # Latest is resolved on every call, so a newer active version changes what it pulls.
    client.update_skill_version(name=name, version=v2.version, organization="acme", status="draft")
    destination = tmp_path / "latest-after-draft"
    mlflow.genai.pull(f"skills:/@acme/{name}", destination=destination)
    assert compute_tree_digest(destination) == compute_tree_digest(first)


def test_pull_registered_git_skill_and_detect_changed_source(client, tmp_path, git_server):
    name = "review"
    repo = tmp_path / "repo"
    _write_skill(repo / "skills" / name, "v1\n")
    (repo / "README.md").write_text("not part of the skill\n")
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "v1", cwd=repo)
    source = GitSource(url=f"{git_server}/{repo.name}", ref="main", subpath=f"skills/{name}")
    version = mlflow.genai.register_skill(source=source, name=name)
    assert version.digest == compute_tree_digest(repo / "skills" / name)

    destination = tmp_path / "pulled"
    mlflow.genai.pull(f"skills:/{name}/{version.version}", destination=destination)
    assert not (destination / "README.md").exists()
    assert compute_tree_digest(destination) == version.digest

    # The branch moves after registration: the recorded digest no longer matches.
    (repo / "skills" / name / "SKILL.md").write_text(SKILL_MD + "tampered\n")
    _git("commit", "-q", "-am", "v2", cwd=repo)
    moved = tmp_path / "moved"
    with pytest.raises(MlflowException, match="does not match its recorded digest"):
        mlflow.genai.pull(f"skills:/{name}/{version.version}", destination=moved)
    assert not moved.exists()


def test_deprecated_version_still_pulls(client, tmp_path, request):
    name = "review"
    skill = _write_skill(tmp_path / "skill", "deprecated\n")
    version = mlflow.genai.register_skill(source=str(skill), name=name)
    client.update_skill_version(name=name, version=version.version, status="deprecated")
    assert client.get_skill_version(name=name, version=1).status == SkillStatus.DEPRECATED
    destination = tmp_path / "out"
    mlflow.genai.pull(f"skills:/{name}/1", destination=destination)
    assert compute_tree_digest(destination) == version.digest


def test_pull_unknown_skill_writes_nothing(client, tmp_path):
    destination = tmp_path / "out"
    with pytest.raises(MlflowException, match="not found|does not exist|No ") as exc:
        mlflow.genai.pull("skills:/does-not-exist/1", destination=destination)
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"
    assert not destination.exists()


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("modified", "does not match its recorded digest"),
        ("removed", "Failed to fetch skill content"),
    ],
)
def test_uploaded_content_changed_on_the_server_fails_the_pull(
    change, match, client, server_artifacts, tmp_path
):
    skill = _write_skill(tmp_path / "skill", "uploaded\n")
    version = mlflow.genai.register_skill(source=str(skill), name="review")
    stored = server_artifacts / version.source.artifact_path.removeprefix("mlflow-artifacts:/")
    assert (stored / "SKILL.md").is_file()
    if change == "modified":
        (stored / "SKILL.md").write_text("changed after registration\n")
    else:
        shutil.rmtree(stored)
    destination = tmp_path / "out"

    with pytest.raises(MlflowException, match=match) as exc:
        mlflow.genai.pull(f"skills:/review/{version.version}", destination=destination)

    assert not destination.exists()
    if change == "modified":
        assert exc.value.error_code == "INVALID_STATE"


# If a version's uploaded files go missing from artifact storage outside MLflow (for example,
# deleted by hand) while its record remains, pull fails with INTERNAL_ERROR, unlike a missing
# source of any other type. No MLflow command does this; the test deletes the files itself.
@pytest.mark.xfail(strict=True, raises=AssertionError)
def test_removed_uploaded_content_is_reported_as_missing(client, server_artifacts, tmp_path):
    skill = _write_skill(tmp_path / "skill", "uploaded\n")
    version = mlflow.genai.register_skill(source=str(skill), name="review")
    shutil.rmtree(
        server_artifacts / version.source.artifact_path.removeprefix("mlflow-artifacts:/")
    )
    with pytest.raises(MlflowException, match="Failed to fetch skill content") as exc:
        mlflow.genai.pull(f"skills:/review/{version.version}", destination=tmp_path / "out")
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_cli_pull(client, tmp_path, request, monkeypatch):
    name = "review"
    skill = _write_skill(tmp_path / "skill", "cli\n")
    mlflow.genai.register_skill(source=str(skill), name=name)
    runner = CliRunner()

    destination = tmp_path / "positional"
    result = runner.invoke(cli, ["skills", "pull", f"skills:/{name}/1", "-d", str(destination)])
    assert result.exit_code == 0, result.output
    assert f"Pulled skills:/{name}/1 to {destination}" in result.output
    assert compute_tree_digest(destination) == compute_tree_digest(skill)

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli, ["skills", "pull", "--skill-uri", f"skills:/{name}"])
    assert result.exit_code == 0, result.output
    assert compute_tree_digest(tmp_path / name) == compute_tree_digest(skill)

    # The default destination now exists and is not empty, so a second pull is refused.
    result = runner.invoke(cli, ["skills", "pull", f"skills:/{name}"])
    assert result.exit_code == 1
    assert "is a directory that is not empty" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "Missing the skill URI"),
        (["skills:/a/1", "--skill-uri", "skills:/b/1"], "either as an argument or with"),
        (["models:/a/1"], "expected it to start with 'skills:/'"),
    ],
)
def test_cli_pull_usage_errors(args, message, tmp_path):
    result = CliRunner().invoke(cli, ["skills", "pull", *args, "-d", str(tmp_path / "out")])
    assert result.exit_code != 0
    assert message in result.output
    assert not (tmp_path / "out").exists()


def test_cli_pull_refuses_a_file_destination(tmp_path):
    destination = tmp_path / "out"
    destination.write_text("keep")
    result = CliRunner().invoke(cli, ["skills", "pull", "skills:/a/1", "-d", str(destination)])
    assert result.exit_code == 2
    assert "is a file" in result.output
    assert destination.read_text() == "keep"

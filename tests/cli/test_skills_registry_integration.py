import http.server
import json
import shutil
import subprocess
import threading
from functools import partial
from pathlib import Path

import pytest
from click.testing import CliRunner

import mlflow
from mlflow.cli.skills import commands
from mlflow.server import ARTIFACTS_DESTINATION_ENV_VAR, SERVE_ARTIFACTS_ENV_VAR, handlers
from mlflow.server.fastapi_app import app
from mlflow.server.handlers import initialize_backend_stores

from tests.helper_functions import get_safe_port
from tests.tracking.integration_test_utils import ServerThread

pytestmark = pytest.mark.notrackingurimock


@pytest.fixture
def runner():
    return CliRunner(catch_exceptions=False)


@pytest.fixture
def tracking_server(db_uri: str, tmp_path: Path, monkeypatch):
    handlers._tracking_store = None
    handlers._model_registry_store = None
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    monkeypatch.setenv(SERVE_ARTIFACTS_ENV_VAR, "true")
    monkeypatch.setenv(ARTIFACTS_DESTINATION_ENV_VAR, str(artifact_root))
    monkeypatch.setattr(handlers, "_artifact_repo", None)
    initialize_backend_stores(db_uri, default_artifact_root=artifact_root.as_uri())

    with ServerThread(app, get_safe_port()) as url:
        mlflow.set_tracking_uri(url)
        yield url
        mlflow.set_tracking_uri(None)


@pytest.fixture
def local_skill(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "reviewer"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: reviewer\ndescription: Reviews code\n---\n# Reviewer\n",
        encoding="utf-8",
    )
    return skill_dir


def _register(runner, tracking_server, local_skill, *, status=None):
    args = [
        "register",
        "--source",
        str(local_skill),
        "--name",
        "reviewer",
    ]
    if status is not None:
        args.extend(["--status", status])
    args.extend(["--output", "json"])
    return runner.invoke(
        commands,
        args,
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )


def _git(*args, cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _create_git_skill_repo(root: Path, names: list[str]) -> Path:
    for name in names:
        skill = root / "skills" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name} skill\n---\n# {name}\n",
            encoding="utf-8",
        )
    _git("init", "-q", "-b", "main", cwd=root)
    _git("add", ".", cwd=root)
    _git(
        "-c",
        "user.name=MLflow tests",
        "-c",
        "user.email=mlflow-tests@example.com",
        "commit",
        "-q",
        "-m",
        "add skills",
        cwd=root,
    )
    return root


@pytest.fixture
def source_http_server(tmp_path: Path):
    serve_dir = tmp_path / "source"
    serve_dir.mkdir()

    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

    handler = partial(_QuietHandler, directory=str(serve_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(
        target=server.serve_forever, daemon=True, name="skill-cli-source-http-server"
    )
    thread.start()
    try:
        yield serve_dir, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_cli_registers_local_skill_and_searches_parent(runner, tracking_server, local_skill):
    register_result = _register(runner, tracking_server, local_skill)

    assert register_result.exit_code == 0, register_result.output
    version = json.loads(register_result.output)["skill_versions"][0]
    assert version["version"] == 1
    assert version["status"] == "active"

    search_result = runner.invoke(
        commands,
        ["search", "--output", "json"],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )

    assert search_result.exit_code == 0, search_result.output
    payload = json.loads(search_result.output)
    assert [skill["name"] for skill in payload["skills"]] == ["reviewer"]


def test_cli_searches_skill_versions_with_pagination(runner, tracking_server, local_skill):
    assert _register(runner, tracking_server, local_skill).exit_code == 0
    assert _register(runner, tracking_server, local_skill, status="draft").exit_code == 0

    first_page = runner.invoke(
        commands,
        [
            "search-versions",
            "skills:/reviewer",
            "--max-results",
            "1",
            "--order-by",
            "version ASC",
            "--output",
            "json",
        ],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )

    assert first_page.exit_code == 0, first_page.output
    first_payload = json.loads(first_page.output)
    assert [version["version"] for version in first_payload["skill_versions"]] == [1]
    assert first_payload["next_page_token"]

    second_page = runner.invoke(
        commands,
        [
            "search-versions",
            "skills:/reviewer",
            "--max-results",
            "1",
            "--order-by",
            "version ASC",
            "--page-token",
            first_payload["next_page_token"],
            "--output",
            "json",
        ],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )

    assert second_page.exit_code == 0, second_page.output
    second_payload = json.loads(second_page.output)
    assert [version["version"] for version in second_payload["skill_versions"]] == [2]


def test_cli_updates_skill_version_aliases_and_tags(runner, tracking_server, local_skill):
    assert _register(runner, tracking_server, local_skill).exit_code == 0

    update_result = runner.invoke(
        commands,
        [
            "update",
            "skills:/reviewer",
            "--description",
            "Reviews pull requests",
            "--output",
            "json",
        ],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )
    assert update_result.exit_code == 0, update_result.output
    assert json.loads(update_result.output)["skill"]["description"] == "Reviews pull requests"

    for args in (
        ["set-alias", "skills:/reviewer", "--alias", "stable", "--version", "1"],
        ["set-tag", "skills:/reviewer", "--key", "team", "--value", "platform"],
        [
            "set-version-tag",
            "skills:/reviewer/1",
            "--key",
            "channel",
            "--value",
            "stable",
        ],
    ):
        result = runner.invoke(
            commands,
            args,
            env={"MLFLOW_TRACKING_URI": tracking_server},
        )
        assert result.exit_code == 0, result.output

    version_result = runner.invoke(
        commands,
        ["get-version", "skills:/reviewer@stable", "--output", "json"],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )
    assert version_result.exit_code == 0, version_result.output
    assert json.loads(version_result.output)["skill_versions"][0]["version"] == 1

    delete_alias_result = runner.invoke(
        commands,
        ["delete-alias", "skills:/reviewer", "--alias", "stable"],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )
    assert delete_alias_result.exit_code == 0, delete_alias_result.output

    parent = runner.invoke(
        commands,
        ["get", "skills:/reviewer", "--output", "json"],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )
    assert json.loads(parent.output)["skill"]["tags"] == {"team": "platform"}

    version = runner.invoke(
        commands,
        ["get-version", "skills:/reviewer/1", "--output", "json"],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )
    assert json.loads(version.output)["skill_versions"][0]["tags"] == {"channel": "stable"}

    for args in (
        ["delete-tag", "skills:/reviewer", "--key", "team"],
        ["delete-version-tag", "skills:/reviewer/1", "--key", "channel"],
    ):
        result = runner.invoke(commands, args, env={"MLFLOW_TRACKING_URI": tracking_server})
        assert result.exit_code == 0, result.output

    parent = runner.invoke(
        commands,
        ["get", "skills:/reviewer", "--output", "json"],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )
    assert json.loads(parent.output)["skill"]["tags"] == {}


def test_cli_imports_git_repository(runner, tracking_server, tmp_path):
    repo = tmp_path / "skills-repo"
    _create_git_skill_repo(repo, ["reviewer"])

    result = runner.invoke(
        commands,
        [
            "import",
            "--source",
            repo.as_uri(),
            "--ref",
            "main",
            "--subpath",
            "skills",
            "--organization",
            "acme",
            "--output",
            "json",
        ],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["skill_versions"][0]["name"] == "reviewer"
    assert payload["skill_versions"][0]["organization"] == "acme"


def test_cli_imports_selected_git_skill_as_draft(runner, tracking_server, tmp_path):
    repo = _create_git_skill_repo(tmp_path / "skills-repo", ["reviewer", "lint-check"])

    result = runner.invoke(
        commands,
        [
            "import",
            "--source",
            repo.as_uri(),
            "--ref",
            "main",
            "--subpath",
            "skills",
            "--skill-name",
            "reviewer",
            "--organization",
            "acme",
            "--status",
            "draft",
            "--output",
            "json",
        ],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [(version["name"], version["status"]) for version in payload["skill_versions"]] == [
        ("reviewer", "draft")
    ]

    search_result = runner.invoke(
        commands,
        ["search", "--output", "json"],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )
    assert search_result.exit_code == 0, search_result.output
    assert [skill["name"] for skill in json.loads(search_result.output)["skills"]] == ["reviewer"]


def test_cli_registers_git_source_and_reads_parent_and_version(runner, tracking_server, tmp_path):
    repo = _create_git_skill_repo(tmp_path / "skills-repo", ["reviewer"])

    register_result = runner.invoke(
        commands,
        [
            "register",
            "git",
            "--url",
            repo.as_uri(),
            "--ref",
            "main",
            "--subpath",
            "skills/reviewer",
            "--organization",
            "acme",
            "--name",
            "reviewer",
            "--output",
            "json",
        ],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )

    assert register_result.exit_code == 0, register_result.output
    assert json.loads(register_result.output)["skill_versions"][0]["source_type"] == "git"

    parent_result = runner.invoke(
        commands,
        [
            "get",
            "--skill-uri",
            "skills:/@acme/reviewer",
            "--output",
            "json",
        ],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )
    assert parent_result.exit_code == 0, parent_result.output
    assert json.loads(parent_result.output)["skill"]["organization"] == "acme"

    version_result = runner.invoke(
        commands,
        [
            "get-version",
            "--skill-uri",
            "skills:/@acme/reviewer/1",
            "--output",
            "json",
        ],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )
    assert version_result.exit_code == 0, version_result.output
    assert json.loads(version_result.output)["skill_versions"][0]["source_type"] == "git"


def test_cli_registers_git_source_and_infers_name(runner, tracking_server, tmp_path):
    repo = _create_git_skill_repo(tmp_path / "skills-repo", ["reviewer"])

    result = runner.invoke(
        commands,
        [
            "register",
            "git",
            "--url",
            repo.as_uri(),
            "--ref",
            "main",
            "--subpath",
            "skills/reviewer",
            "--output",
            "json",
        ],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["skill_versions"][0]["name"] == "reviewer"


@pytest.mark.no_mock_requests_get
def test_cli_registers_zip_source(runner, tracking_server, local_skill, source_http_server):
    serve_dir, base_url = source_http_server
    shutil.make_archive(str(serve_dir / "reviewer"), "zip", root_dir=local_skill)

    result = runner.invoke(
        commands,
        [
            "register",
            "zip",
            "--url",
            f"{base_url}/reviewer.zip",
            "--name",
            "reviewer",
            "--output",
            "json",
        ],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["skill_versions"][0]["name"] == "reviewer"
    assert payload["skill_versions"][0]["source_type"] == "zip"


def test_cli_reports_missing_skill_from_real_server(runner, tracking_server):
    result = runner.invoke(
        commands,
        ["get", "skills:/does-not-exist"],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )

    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_cli_updates_and_deletes_skill_version(runner, tracking_server, local_skill):
    assert _register(runner, tracking_server, local_skill).exit_code == 0

    update_result = runner.invoke(
        commands,
        [
            "update-version",
            "skills:/reviewer/1",
            "--status",
            "deprecated",
            "--output",
            "json",
        ],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )
    assert update_result.exit_code == 0, update_result.output
    assert json.loads(update_result.output)["skill_versions"][0]["status"] == "deprecated"

    delete_result = runner.invoke(
        commands,
        ["delete-version", "skills:/reviewer/1"],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )
    assert delete_result.exit_code == 0, delete_result.output

    version_result = runner.invoke(
        commands,
        ["get-version", "skills:/reviewer/1", "--output", "json"],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )
    assert version_result.exit_code != 0
    assert "not found" in version_result.output


def test_cli_supports_organization_qualified_uri(runner, tracking_server, local_skill):
    result = runner.invoke(
        commands,
        [
            "register",
            "--source",
            str(local_skill),
            "--name",
            "reviewer",
            "--organization",
            "acme",
            "--output",
            "json",
        ],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )
    assert result.exit_code == 0, result.output

    get_result = runner.invoke(
        commands,
        ["get", "skills:/@acme/reviewer", "--output", "json"],
        env={"MLFLOW_TRACKING_URI": tracking_server},
    )
    assert get_result.exit_code == 0, get_result.output
    assert json.loads(get_result.output)["skill"]["organization"] == "acme"

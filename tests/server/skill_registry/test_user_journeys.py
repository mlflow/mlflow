# User journeys through a live MLflow server.
#
# A journey is one story several people go through on the same server: a publisher registers a
# skill, a teammate looks at it in the UI, and consumers pull it. Each person uses the surface
# they would use for real (the CLI, the Python SDK, the UI's `/ajax-api` routes, `skills pull`),
# and each step depends on what the steps before it left on the server.
#
# These tests need a live server because what they check only holds across surfaces: what the
# CLI registers is what the SDK, the UI and a pull see, and a change made through the SDK reaches
# the next pull. The server is the full MLflow app on a real port, with a real SQLite store and
# local artifact storage.
#
# Rules of a single endpoint (validation, error codes, status transitions, alias rules, such as
# an alias belonging to one organization) are tested once, in the store, service and REST tests,
# and are not repeated here.
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import requests
from click.testing import CliRunner

import mlflow
from mlflow.cli.skills import commands
from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.digest import compute_tree_digest
from mlflow.tracking import MlflowClient
from mlflow.utils.workspace_context import WorkspaceContext
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME

from tests.server.skill_registry.live_server import serve_tracking_app

pytestmark = pytest.mark.notrackingurimock

# The MLflow UI reads through `/ajax-api`; programs use `/api`. Both serve the same router.
UI = "/ajax-api/3.0/mlflow/skills"
ARTIFACTS = "/api/2.0/mlflow-artifacts/artifacts"


def _serve(
    db_uri: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, workspaces: bool
) -> Iterator[str]:
    with serve_tracking_app(
        db_uri, tmp_path / "server-artifacts", monkeypatch, workspaces=workspaces
    ) as url:
        mlflow.set_tracking_uri(url)
        try:
            yield url
        finally:
            mlflow.set_tracking_uri(None)


@pytest.fixture
def server(db_uri: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    yield from _serve(db_uri, tmp_path, monkeypatch, workspaces=False)


@pytest.fixture
def workspace_server(db_uri: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    yield from _serve(db_uri, tmp_path, monkeypatch, workspaces=True)


def _write_skill(root: Path, body: str) -> Path:
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        f"---\nname: code-review\ndescription: Reviews code\n---\n# {body}\n", encoding="utf-8"
    )
    (root / "scripts").mkdir()
    (root / "scripts" / "run.py").write_text(f"print({body!r})\n", encoding="utf-8")
    return root


def _cli(server: str, *args: str) -> str:
    result = CliRunner().invoke(commands, list(args), env={"MLFLOW_TRACKING_URI": server})
    assert result.exit_code == 0, result.output
    return result.output


def _ok(response: requests.Response) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    return response.json()


def _pull_fails_as_missing(uri: str, destination: Path) -> None:
    with pytest.raises(MlflowException, match="not found|does not exist") as exc:
        mlflow.genai.pull(uri, destination=destination)
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_publish_consume_and_withdraw_a_skill(server, tmp_path):
    # A publisher creates `code-review` from the SDK and ships v1 from the CLI, with v2 staged as
    # a draft. A teammate reviews it in the UI. Consumers pull it from the SDK and the CLI, and
    # follow the publisher to v2 when v1 is withdrawn, without changing how they pull.
    client = MlflowClient()
    client.create_skill(name="code-review", description="Reviews pull requests")
    v1_dir = _write_skill(tmp_path / "v1", "first")
    v1 = json.loads(_cli(server, "register", "--source", str(v1_dir), "--output", "json"))
    assert [(v["version"], v["status"]) for v in v1["skill_versions"]] == [(1, "active")]
    v2_dir = _write_skill(tmp_path / "v2", "second")
    assert compute_tree_digest(v2_dir) != compute_tree_digest(v1_dir)
    v2 = mlflow.genai.register_skill(source=str(v2_dir), status="draft")
    client.set_skill_alias(name="code-review", alias="stable", version=1)

    # The UI shows the parent the SDK created and the version the CLI uploaded; the draft does
    # not take `latest`.
    skill = _ok(requests.get(f"{server}{UI}/code-review"))
    assert skill["description"] == "Reviews pull requests"
    assert skill["latest_version"] == 1
    assert skill["aliases"] == [{"alias": "stable", "version": 1}]

    # Consumers get exactly the files the publisher uploaded, by latest from the SDK and by alias
    # from the CLI.
    mlflow.genai.pull("skills:/code-review", destination=tmp_path / "sdk-v1")
    _cli(server, "pull", "skills:/code-review@stable", "-d", str(tmp_path / "cli-v1"))
    for pulled in (tmp_path / "sdk-v1", tmp_path / "cli-v1"):
        assert compute_tree_digest(pulled) == compute_tree_digest(v1_dir)

    # The publisher promotes v2. With both versions active, v2 takes `latest` at once, while
    # `stable` stays on v1 until the publisher moves it.
    client.update_skill_version(name="code-review", version=v2.version, status="active")
    skill = _ok(requests.get(f"{server}{UI}/code-review"))
    assert skill["latest_version"] == 2
    assert skill["aliases"] == [{"alias": "stable", "version": 1}]

    # The publisher moves `stable` to v2 and retires v1.
    client.set_skill_alias(name="code-review", alias="stable", version=v2.version)
    client.update_skill_version(name="code-review", version=1, status="deprecated")
    client.delete_skill_version(name="code-review", version=1)

    # The same pulls now deliver v2, and v1 can no longer be pulled by anyone.
    mlflow.genai.pull("skills:/code-review", destination=tmp_path / "sdk-v2")
    _cli(server, "pull", "skills:/code-review@stable", "-d", str(tmp_path / "cli-v2"))
    for pulled in (tmp_path / "sdk-v2", tmp_path / "cli-v2"):
        assert compute_tree_digest(pulled) == v2.digest == compute_tree_digest(v2_dir)
    _pull_fails_as_missing("skills:/code-review/1", tmp_path / "v1-again")

    versions = _ok(requests.get(f"{server}{UI}/code-review/versions"))["skill_versions"]
    assert [(v["version"], v["aliases"]) for v in versions] == [(2, ["stable"])]


def test_teams_in_two_organizations_publish_the_same_skill_name(server, tmp_path):
    # acme and globex each publish their own `code-review`. They are two skills: each is found
    # and pulled on its own, and acme deleting its skill leaves globex exactly as it was.
    acme_dir = _write_skill(tmp_path / "acme", "acme")
    globex_dir = _write_skill(tmp_path / "globex", "globex")
    mlflow.genai.register_skill(source=str(acme_dir), organization="acme")
    _cli(server, "register", "--source", str(globex_dir), "--organization", "globex")

    found = mlflow.genai.search_skills(filter_string="organization = 'acme'")
    assert [(s.organization, s.name) for s in found] == [("acme", "code-review")]

    for organization, expected in (("acme", acme_dir), ("globex", globex_dir)):
        destination = tmp_path / f"pulled-{organization}"
        mlflow.genai.pull(f"skills:/@{organization}/code-review", destination=destination)
        assert compute_tree_digest(destination) == compute_tree_digest(expected)
    assert compute_tree_digest(acme_dir) != compute_tree_digest(globex_dir)

    # globex is the bystander: a teammate watching it in the UI sees no change.
    def globex_in_ui():
        return (
            _ok(requests.get(f"{server}{UI}/@globex/code-review")),
            _ok(requests.get(f"{server}{UI}/@globex/code-review/versions")),
        )

    globex_before = globex_in_ui()
    MlflowClient().delete_skill(name="code-review", organization="acme")

    _pull_fails_as_missing("skills:/@acme/code-review", tmp_path / "acme-again")
    assert globex_in_ui() == globex_before
    mlflow.genai.pull("skills:/@globex/code-review", destination=tmp_path / "globex-again")
    assert compute_tree_digest(tmp_path / "globex-again") == compute_tree_digest(globex_dir)


def test_teams_in_separate_workspaces_never_see_each_others_skills(workspace_server, tmp_path):
    # Workspaces split one server between teams. Both teams upload a skill named `code-review`
    # from the SDK, each in its own workspace. Each team finds and pulls only its own.
    client = MlflowClient()
    for workspace in ("team-a", "team-b"):
        client.create_workspace(workspace)
    skill_dirs = {
        "team-a": _write_skill(tmp_path / "team-a", "team-a"),
        "team-b": _write_skill(tmp_path / "team-b", "team-b"),
    }
    uploaded = {}
    for workspace, skill_dir in skill_dirs.items():
        with WorkspaceContext(workspace):
            uploaded[workspace] = mlflow.genai.register_skill(source=str(skill_dir))

    for workspace, skill_dir in skill_dirs.items():
        with WorkspaceContext(workspace):
            found = mlflow.genai.search_skills()
            assert [(s.workspace, s.name) for s in found] == [(workspace, "code-review")]
            destination = tmp_path / f"pulled-{workspace}"
            mlflow.genai.pull("skills:/code-review", destination=destination)
            assert compute_tree_digest(destination) == compute_tree_digest(skill_dir)

    # Anyone who can read a version can read its content path, so the artifact API must hold
    # the workspace boundary too: team-a's content path does not resolve from team-b.
    content_path = uploaded["team-a"].source.artifact_path.removeprefix("mlflow-artifacts:/")
    content_url = f"{workspace_server}{ARTIFACTS}/{content_path}/SKILL.md"
    team_a = requests.get(content_url, headers={WORKSPACE_HEADER_NAME: "team-a"})
    assert team_a.status_code == 200, team_a.text
    assert team_a.content == (skill_dirs["team-a"] / "SKILL.md").read_bytes()
    team_b = requests.get(content_url, headers={WORKSPACE_HEADER_NAME: "team-b"})
    assert team_b.status_code == 404

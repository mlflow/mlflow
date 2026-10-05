# User journeys through a live MLflow server.
#
# Each test is one story a user goes through, told as a sequence of HTTP calls against the full
# MLflow app (`mlflow.server.fastapi_app:app`) served by uvicorn on a real port, backed by a real
# SQLite store and local artifact storage. Nothing in MLflow is mocked.
#
# What belongs here is only what no lower layer can show: state written by one step and observed
# by the next, over the wire, through the app as deployed (both route prefixes, the artifact
# routes beside the registry routes). Single-endpoint rules such as validation and error codes
# belong to `tests/server/test_skill_registry_api.py`, the store, and the registration service
# tests, and are not repeated here.
#
# Steps call the REST API directly for now. When the SDK, CLI and pull land, steps move to
# those surfaces; the stories stay the same.
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import requests

from mlflow.server import ARTIFACTS_DESTINATION_ENV_VAR, SERVE_ARTIFACTS_ENV_VAR, handlers
from mlflow.server.fastapi_app import app
from mlflow.server.handlers import initialize_backend_stores

from tests.helper_functions import get_safe_port
from tests.server.skill_registry.conftest import SKILL_FILES, skill_archive
from tests.tracking.integration_test_utils import ServerThread

# Programs use `/api`; the MLflow UI uses `/ajax-api`. Both serve the same router and store.
API = "/api/3.0/mlflow/skills"
UI = "/ajax-api/3.0/mlflow/skills"
ARTIFACTS = "/api/2.0/mlflow-artifacts/artifacts"

GIT_SOURCE = "https://github.com/acme/skills.git"


@pytest.fixture
def server(tmp_path: Path, db_uri: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    # The same setup `mlflow server --serve-artifacts` performs, applied to the in-process app.
    monkeypatch.setenv(SERVE_ARTIFACTS_ENV_VAR, "true")
    monkeypatch.setenv(ARTIFACTS_DESTINATION_ENV_VAR, str(tmp_path / "artifacts"))
    monkeypatch.setattr(handlers, "_tracking_store", None)
    monkeypatch.setattr(handlers, "_model_registry_store", None)
    monkeypatch.setattr(handlers, "_artifact_repo", None)
    initialize_backend_stores(db_uri, default_artifact_root="mlflow-artifacts:/")
    with ServerThread(app, get_safe_port()) as url:
        yield url


def _ok(response: requests.Response) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    return response.json()


def _not_found(response: requests.Response) -> None:
    assert response.status_code == 404, response.text
    assert response.json()["error_code"] == "RESOURCE_DOES_NOT_EXIST"


def _register_remote(server: str, name: str, **fields: Any) -> dict[str, Any]:
    return _ok(requests.post(f"{server}{API}/register", json={"name": name, **fields}))


def test_teams_in_different_organizations_publish_the_same_skill_name(server):
    # Two teams and the unscoped namespace each publish `code-review`. They are three skills:
    # each team finds, resolves and retires only its own.
    _register_remote(server, "code-review", source=GIT_SOURCE)
    _register_remote(server, "code-review", organization="acme", source=GIT_SOURCE, ref="acme-v1")
    _register_remote(
        server, "code-review", organization="globex", source=GIT_SOURCE, ref="globex-v1"
    )

    search = _ok(requests.get(f"{server}{API}", params={"filter_string": "organization = 'acme'"}))
    assert [(s["organization"], s["name"]) for s in search["skills"]] == [("acme", "code-review")]

    acme_v2 = _ok(
        requests.post(
            f"{server}{API}/@acme/code-review/versions",
            json={"source": GIT_SOURCE, "ref": "acme-v2"},
        )
    )
    assert acme_v2["version"] == 2

    # Each namespace resolves its own versions; acme's second version is not globex's.
    assert _ok(requests.get(f"{server}{API}/@acme/code-review/aliases/latest"))["ref"] == "acme-v2"
    globex_latest = _ok(requests.get(f"{server}{API}/@globex/code-review/aliases/latest"))
    assert (globex_latest["version"], globex_latest["ref"]) == (1, "globex-v1")
    assert _ok(requests.get(f"{server}{API}/code-review/aliases/latest"))["organization"] == ""

    # An alias is scoped to its namespace.
    _ok(
        requests.post(
            f"{server}{API}/@acme/code-review/aliases", json={"alias": "stable", "version": 1}
        )
    )
    assert _ok(requests.get(f"{server}{API}/@acme/code-review/aliases/stable"))["version"] == 1
    _not_found(requests.get(f"{server}{API}/@globex/code-review/aliases/stable"))
    _not_found(requests.get(f"{server}{API}/code-review/aliases/stable"))

    # Deleting acme's skill leaves the other two untouched.
    _ok(requests.delete(f"{server}{API}/@acme/code-review"))
    _not_found(requests.get(f"{server}{API}/@acme/code-review"))
    remaining = _ok(requests.get(f"{server}{API}"))["skills"]
    assert sorted(s["organization"] for s in remaining) == ["", "globex"]


def test_publish_promote_and_retire_versions(server):
    # A team ships a version, stages the next one as a draft, promotes it, and retires the old
    # one. Consumers follow `latest` and the team's `stable` alias throughout.
    _register_remote(server, "code-review", source=GIT_SOURCE, ref="v1")
    _register_remote(server, "code-review", source=GIT_SOURCE, ref="v2", status="draft")

    # A draft does not take `latest` from an active version.
    assert _ok(requests.get(f"{server}{API}/code-review/aliases/latest"))["version"] == 1
    _ok(requests.post(f"{server}{API}/code-review/aliases", json={"alias": "stable", "version": 1}))

    _ok(requests.patch(f"{server}{API}/code-review/versions/2", json={"status": "active"}))
    assert _ok(requests.get(f"{server}{API}/code-review/aliases/latest"))["version"] == 2
    # Promotion does not move an alias the team set; they move it themselves.
    assert _ok(requests.get(f"{server}{API}/code-review/aliases/stable"))["version"] == 1
    _ok(requests.post(f"{server}{API}/code-review/aliases", json={"alias": "stable", "version": 2}))

    # An active version is deprecated before it can be deleted.
    _ok(requests.patch(f"{server}{API}/code-review/versions/1", json={"status": "deprecated"}))
    _ok(requests.delete(f"{server}{API}/code-review/versions/1"))
    _not_found(requests.get(f"{server}{API}/code-review/versions/1"))
    versions = _ok(requests.get(f"{server}{API}/code-review/versions"))["skill_versions"]
    assert [(v["version"], v["aliases"]) for v in versions] == [(2, ["stable"])]

    # The UI, reading through its own prefix, sees the state the program left.
    skill = _ok(requests.get(f"{server}{UI}/code-review"))
    assert skill["latest_version"] == 2
    assert skill["aliases"] == [{"alias": "stable", "version": 2}]


def test_uploaded_skill_content_is_served_back_and_reclaimed_on_delete(server):
    # A user uploads a skill from their machine; an agent then resolves it and downloads its
    # files through the same server's artifact API. Deleting the skill removes the content.
    response = requests.post(
        f"{server}{API}/register",
        files={
            "metadata": ("metadata.json", '{"name": "reviewer"}', "application/json"),
            "content": ("content.tar.gz", skill_archive(), "application/gzip"),
        },
    )
    version = _ok(response)
    assert version["source_type"] == "mlflow"

    resolved = _ok(requests.get(f"{server}{API}/reviewer/aliases/latest"))
    content_path = resolved["source"].removeprefix("mlflow-artifacts:/")
    for name, expected in SKILL_FILES.items():
        download = requests.get(f"{server}{ARTIFACTS}/{content_path}/{name}")
        assert download.status_code == 200, download.text
        assert download.content == expected

    _ok(requests.delete(f"{server}{API}/reviewer"))
    _not_found(requests.get(f"{server}{UI}/reviewer"))
    gone = requests.get(f"{server}{ARTIFACTS}/{content_path}/SKILL.md")
    assert gone.status_code == 404

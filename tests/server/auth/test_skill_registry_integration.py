# Skill Registry permissions on a live `mlflow server --app-name basic-auth`.
#
# Only checks that need a live server stay here: logging in (401), the SDK's own HTTP client
# and credentials, grants made through the role API in separate workspaces, the artifact proxy,
# and pull. Which permission each route requires is checked without a network, through
# TestClient and the same permission middleware, in test_skill_registry_rest_auth.py.
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests

import mlflow.genai
from mlflow import MlflowClient, MlflowException
from mlflow.server import ARTIFACTS_DESTINATION_ENV_VAR, SERVE_ARTIFACTS_ENV_VAR
from mlflow.tracking._tracking_service.utils import _use_tracking_uri
from mlflow.utils.workspace_context import WorkspaceContext
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME

from tests.helper_functions import random_str
from tests.server.auth.auth_test_utils import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    User,
    create_user,
    grant_role_permission,
    write_isolated_auth_config,
)
from tests.tracking.integration_test_utils import _init_server

WORKSPACE = "skill-permissions"
OTHER_WORKSPACE = "skill-isolation"
PREFIX = "/api/3.0/mlflow/skills"
PREFIXES = [PREFIX, "/ajax-api/3.0/mlflow/skills"]
SOURCE = "https://example.com/skill.zip"


@dataclass
class Registry:
    url: str
    session: requests.Session
    users: dict[str, tuple[str, str]]
    artifact_root: Path

    def credentials(self, user):
        # A seeded user's key, an explicit (username, password) pair, or None for no credentials.
        if user is None or isinstance(user, tuple):
            return user
        return self.users[user]

    def request(self, method, path, *, user="admin", workspace=WORKSPACE, **kwargs):
        return self.session.request(
            method,
            f"{self.url}{path}",
            auth=self.credentials(user),
            headers={WORKSPACE_HEADER_NAME: workspace},
            timeout=30,
            **kwargs,
        )

    @contextmanager
    def client(self, user, workspace=WORKSPACE):
        with (
            pytest.MonkeyPatch.context() as patch,
            WorkspaceContext(workspace),
            _use_tracking_uri(self.url),
        ):
            patch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
            patch.setenv("MLFLOW_TRACKING_URI", self.url)
            with User(*(self.credentials(user) or ("", "")), patch):
                yield MlflowClient(tracking_uri=self.url)


@pytest.fixture(scope="module")
def registry(tmp_path_factory, cached_db):
    root = tmp_path_factory.mktemp("skill-auth-integration")
    # Reuse the session's migrated schema while keeping this server's data isolated.
    backend = root / "tracking.db"
    shutil.copy2(cached_db, backend)
    backend_uri = f"sqlite:///{backend}"
    artifact_root = root / "artifacts"
    config = write_isolated_auth_config(root, default_permission="NO_PERMISSIONS")
    with (
        _init_server(
            backend_uri=backend_uri,
            root_artifact_uri=artifact_root.as_uri(),
            app="mlflow.server.auth:create_app",
            extra_env={
                "_MLFLOW_SGI_NAME": "uvicorn",
                "MLFLOW_FLASK_SERVER_SECRET_KEY": "skill-integration-test-secret",
                "MLFLOW_AUTH_CONFIG_PATH": str(config),
                "MLFLOW_ENABLE_WORKSPACES": "true",
                "MLFLOW_WORKSPACE_STORE_URI": backend_uri,
                "MLFLOW_RBAC_SEED_DEFAULT_ROLES": "false",
                "MLFLOW_WORKSPACE": "default",
                SERVE_ARTIFACTS_ENV_VAR: "true",
                ARTIFACTS_DESTINATION_ENV_VAR: str(artifact_root),
            },
        ) as url,
        requests.Session() as session,
    ):
        session.trust_env = False
        result = Registry(url, session, {"admin": (ADMIN_USERNAME, ADMIN_PASSWORD)}, artifact_root)
        for workspace in (WORKSPACE, OTHER_WORKSPACE):
            response = result.request(
                "POST", "/api/3.0/mlflow/workspaces", json={"name": workspace}, workspace="default"
            )
            assert response.status_code == 201
        # Tests never add to these users; one that needs its own grants creates a fresh user.
        for permission in ("READ", "EDIT", "MANAGE", "NONE", "creator"):
            username, password = create_user(url)
            result.users[permission] = username, password
            if permission == "creator":
                grant_role_permission(url, username, "workspace", "*", "USE", WORKSPACE)
            if permission in ("READ", "EDIT", "MANAGE"):
                grant_role_permission(url, username, "skill", "*", permission, WORKSPACE)
        result.users["bad-password"] = result.users["READ"][0], "incorrect-password"
        yield result


@pytest.fixture
def skill(registry):
    name = f"skill-{random_str().lower()}"
    response = registry.request("POST", PREFIX, json={"name": name, "description": "original"})
    assert response.status_code == 200
    response = registry.request("POST", f"{PREFIX}/{name}/versions", json={"source": SOURCE})
    assert response.status_code == 200
    return name


def snapshot(registry, name, workspace=WORKSPACE):
    parent = registry.request("GET", f"{PREFIX}/{name}", workspace=workspace)
    assert parent.status_code == 200
    versions = registry.request("GET", f"{PREFIX}/{name}/versions", workspace=workspace)
    assert versions.status_code == 200
    return parent.json(), versions.json()


def stored_files(root):
    return sorted(path for path in root.rglob("*") if path.is_file())


def bulk_definition(name):
    return {
        "name": name,
        "source": "https://example.com/skills.git",
        "ref": "main",
        "digest": "a" * 64,
        "status": "draft",
    }


def write_skill(directory, name):
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Integration test skill\n---\nPrivate content.\n"
    )
    return directory


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("user", [None, "bad-password"])
def test_requests_without_valid_credentials_are_rejected(registry, skill, prefix, user):
    # Authentication runs before any lookup, so a Skill that does not exist is a 401, not a 404.
    assert registry.request("GET", f"{prefix}/does-not-exist", user=user).status_code == 401
    before = snapshot(registry, skill)
    response = registry.request(
        "PATCH", f"{prefix}/{skill}", user=user, json={"description": "updated"}
    )
    assert response.status_code == 401
    assert snapshot(registry, skill) == before


@pytest.mark.parametrize("user", [None, "bad-password"])
def test_sdk_reports_missing_or_wrong_credentials(registry, skill, user):
    with registry.client(user) as client:
        for call in (lambda: client.get_skill(name=skill), mlflow.genai.search_skills):
            with pytest.raises(MlflowException, match="not authenticated") as error:
                call()
            assert error.value.get_http_status_code() == 401


def test_sdk_reports_permission_denied_and_search_hides_unreadable_skills(registry, skill):
    with registry.client("NONE") as client:
        with pytest.raises(MlflowException, match="Permission denied") as error:
            client.get_skill(name=skill)
        assert error.value.get_http_status_code() == 403
        assert mlflow.genai.search_skills(include_skill_identities=[skill]) == []
    with registry.client("READ") as client:
        assert client.get_skill(name=skill).description == "original"
        found = mlflow.genai.search_skills(include_skill_identities=[skill])
        assert [s.name for s in found] == [skill]


def test_same_named_skills_do_not_share_workspace_permissions(registry, skill):
    response = registry.request(
        "POST", PREFIX, workspace=OTHER_WORKSPACE, json={"name": skill, "description": "private"}
    )
    assert response.status_code == 200
    response = registry.request(
        "POST", f"{PREFIX}/{skill}/versions", workspace=OTHER_WORKSPACE, json={"source": SOURCE}
    )
    assert response.status_code == 200
    before = snapshot(registry, skill)
    other_before = snapshot(registry, skill, OTHER_WORKSPACE)

    # MANAGE holds `skill:*` in WORKSPACE only; none of it carries over to OTHER_WORKSPACE.
    def request(method, path, **kwargs):
        return registry.request(method, path, user="MANAGE", workspace=OTHER_WORKSPACE, **kwargs)

    response = request("GET", PREFIX, params={"include_skill_identities": skill})
    assert response.status_code == 200
    assert response.json()["skills"] == []
    assert request("GET", f"{PREFIX}/{skill}").status_code == 403
    assert request("PATCH", f"{PREFIX}/{skill}", json={"description": "x"}).status_code == 403
    assert request("DELETE", f"{PREFIX}/{skill}").status_code == 403
    response = request("POST", f"{PREFIX}/bulk-register", json={"skills": [bulk_definition(skill)]})
    assert response.status_code == 403
    assert snapshot(registry, skill) == before
    assert snapshot(registry, skill, OTHER_WORKSPACE) == other_before


def test_search_filters_to_readable_skills_before_paginating(registry):
    prefix = f"scope-{random_str().lower()}"
    first = f"{prefix}-b"
    second = f"{prefix}-c"
    for org, name in [("", f"{prefix}-a-hidden"), ("", first), ("acme", first), ("", second)]:
        response = registry.request("POST", PREFIX, json={"name": name, "organization": org})
        assert response.status_code == 200
    reader = create_user(registry.url)
    for identity in (f"@acme/{first}", second):
        grant_role_permission(registry.url, reader[0], "skill", identity, "READ", WORKSPACE)

    # A grant on `@acme/<name>` does not cover the same name in the default organization.
    assert registry.request("GET", f"{PREFIX}/{first}", user=reader).status_code == 403
    assert registry.request("GET", f"{PREFIX}/@acme/{first}", user=reader).status_code == 200
    query = {"filter_string": f"name LIKE '{prefix}%'", "order_by": ["name ASC"], "max_results": 1}
    page = registry.request("GET", PREFIX, user=reader, params=query)
    assert page.status_code == 200
    assert [(s["organization"], s["name"]) for s in page.json()["skills"]] == [("acme", first)]
    token = page.json()["next_page_token"]
    assert token
    page = registry.request("GET", PREFIX, user=reader, params={**query, "page_token": token})
    assert [(s["organization"], s["name"]) for s in page.json()["skills"]] == [("", second)]
    assert page.json()["next_page_token"] is None
    # A caller's selector cannot widen what they may read, including same-named organizations.
    selected = registry.request(
        "GET", PREFIX, user=reader, params={"include_skill_identities": [first, f"@acme/{first}"]}
    )
    assert [(s["organization"], s["name"]) for s in selected.json()["skills"]] == [("acme", first)]
    response = registry.request("GET", PREFIX, user=reader, params={"include_skill_identities": ""})
    assert response.json()["skills"] == []


@pytest.mark.parametrize("user", [None, "READ", "EDIT"])
def test_local_registration_requires_edit_and_uploaded_content_requires_read(
    registry, skill, user, tmp_path
):
    source = write_skill(tmp_path / "source", "local-skill")
    before = snapshot(registry, skill)
    files_before = stored_files(registry.artifact_root)
    with registry.client(user):
        if user != "EDIT":
            with pytest.raises(MlflowException, match="Permission denied|not authenticated") as e:
                mlflow.genai.register_skill(source=str(source), name=skill, status="draft")
            assert e.value.get_http_status_code() == (401 if user is None else 403)
            # A rejected upload leaves no Skill version and no content behind.
            assert snapshot(registry, skill) == before
            assert stored_files(registry.artifact_root) == files_before
            return
        version = mlflow.genai.register_skill(source=str(source), name=skill, status="draft")
    assert version.version == 2
    source_uri = urlparse(version.source.artifact_path)
    assert source_uri.scheme == "mlflow-artifacts"
    artifact = f"/api/2.0/mlflow-artifacts/artifacts/{source_uri.path.lstrip('/')}/SKILL.md"
    for reader in ("READ", "EDIT", "MANAGE"):
        response = registry.request("GET", artifact, user=reader)
        assert response.status_code == 200
        assert response.text == (source / "SKILL.md").read_text()
    # Workspace USE permits creation, but does not grant READ on someone else's Skill.
    assert registry.request("GET", artifact, user="creator").status_code == 403
    assert registry.request("GET", artifact, user=None).status_code == 401


def test_creator_who_uploads_a_new_skill_can_manage_it(registry, tmp_path):
    name = f"upload-{random_str().lower()}"
    with registry.client("creator"):
        version = mlflow.genai.register_skill(source=str(write_skill(tmp_path / "source", name)))
    assert version.name == name
    assert version.created_by == registry.users["creator"][0]
    # The creator holds workspace USE only, so deleting the Skill proves it was granted MANAGE.
    assert registry.request("DELETE", f"{PREFIX}/{name}", user="creator").status_code == 200
    assert registry.request("GET", f"{PREFIX}/{name}").status_code == 404


@pytest.mark.parametrize("suffix", ["/2", "@stable", ""], ids=["version", "alias", "latest"])
@pytest.mark.parametrize("user", [None, "NONE", "READ"])
def test_pull_requires_read_and_denial_writes_nothing(registry, skill, user, suffix, tmp_path):
    source = write_skill(tmp_path / "source", "pull-skill")
    with registry.client("admin") as client:
        version = mlflow.genai.register_skill(source=str(source), name=skill)
        assert version.version == 2
        client.set_skill_alias(name=skill, alias="stable", version=2)
    destination = tmp_path / "pulled"
    with registry.client(user):
        if user == "READ":
            pulled = mlflow.genai.pull(f"skills:/{skill}{suffix}", destination=destination)
            assert pulled == str(destination)
            assert (destination / "SKILL.md").read_bytes() == (source / "SKILL.md").read_bytes()
            assert sorted(path.name for path in destination.iterdir()) == ["SKILL.md"]
        else:
            with pytest.raises(MlflowException, match="Permission denied|not authenticated") as e:
                mlflow.genai.pull(f"skills:/{skill}{suffix}", destination=destination)
            assert e.value.get_http_status_code() == (401 if user is None else 403)
            assert not destination.exists()

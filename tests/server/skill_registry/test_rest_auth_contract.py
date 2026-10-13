from dataclasses import dataclass
from pathlib import Path

import pytest
import requests

from mlflow.environment_variables import (
    MLFLOW_AUTH_CONFIG_PATH,
    MLFLOW_ENABLE_WORKSPACES,
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_WORKSPACE_STORE_URI,
)
from mlflow.server import ARTIFACTS_DESTINATION_ENV_VAR, SERVE_ARTIFACTS_ENV_VAR
from mlflow.utils.os import is_windows

from tests.helper_functions import random_str
from tests.server.auth.auth_test_utils import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    create_user,
    grant_role_permission,
    write_isolated_auth_config,
)
from tests.server.skill_registry.conftest import skill_archive, stored_files
from tests.tracking.integration_test_utils import _init_server

pytestmark = pytest.mark.notrackingurimock

_PREFIXES = ("/api/3.0/mlflow/skills", "/ajax-api/3.0/mlflow/skills")


@dataclass(frozen=True)
class AuthenticatedRegistry:
    base_url: str
    admin: tuple[str, str]
    reader: tuple[str, str]
    artifact_root: Path


@pytest.fixture(scope="module")
def authenticated_registry(tmp_path_factory) -> AuthenticatedRegistry:
    tmp_path = Path(tmp_path_factory.mktemp("skill-registry-rest-auth"))
    auth_config_path = write_isolated_auth_config(tmp_path)
    auth_config_path.write_text(
        auth_config_path.read_text().replace(
            "default_permission = READ", "default_permission = NO_PERMISSIONS"
        )
    )

    db_uri_path = tmp_path.joinpath("tracking.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + db_uri_path[len("file://") :]
    artifact_root = tmp_path / "artifacts"
    extra_env = {
        MLFLOW_AUTH_CONFIG_PATH.name: str(auth_config_path),
        MLFLOW_ENABLE_WORKSPACES.name: "true",
        MLFLOW_FLASK_SERVER_SECRET_KEY.name: "skill-registry-rest-auth-secret",
        MLFLOW_WORKSPACE_STORE_URI.name: backend_uri,
        SERVE_ARTIFACTS_ENV_VAR: "true",
        ARTIFACTS_DESTINATION_ENV_VAR: str(artifact_root),
        "_MLFLOW_SGI_NAME": "uvicorn",
    }

    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=artifact_root.as_uri(),
        extra_env=extra_env,
        app="mlflow.server.auth:create_app",
        server_type="fastapi",
    ) as base_url:
        reader = create_user(base_url)
        yield AuthenticatedRegistry(
            base_url=base_url,
            admin=(ADMIN_USERNAME, ADMIN_PASSWORD),
            reader=reader,
            artifact_root=artifact_root,
        )


def _url(registry: AuthenticatedRegistry, prefix: str, suffix: str = "") -> str:
    return f"{registry.base_url}{prefix}{suffix}"


def _create_skill(registry: AuthenticatedRegistry, prefix: str, name: str) -> None:
    response = requests.post(
        _url(registry, prefix),
        json={"name": name},
        auth=registry.admin,
    )
    assert response.status_code == 200, response.text


def _create_version(registry: AuthenticatedRegistry, prefix: str, name: str) -> None:
    response = requests.post(
        _url(registry, prefix, f"/{name}/versions"),
        json={
            "source": "https://github.com/acme/skills.git",
            "source_type": "git",
            "ref": "v1.0.0",
            "digest": "ab" * 32,
        },
        auth=registry.admin,
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_skill_read_requires_authentication(authenticated_registry, prefix):
    response = requests.get(_url(authenticated_registry, prefix, "/private"))

    assert response.status_code == 401, response.text


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_skill_read_enforces_resource_permission(authenticated_registry, prefix):
    registry = authenticated_registry
    reader = create_user(registry.base_url)
    name = f"private-{random_str()}"
    _create_skill(registry, prefix, name)

    response = requests.get(
        _url(registry, prefix, f"/{name}"),
        auth=reader,
    )
    assert response.status_code == 403, response.text

    grant_role_permission(
        registry.base_url,
        reader[0],
        "skill",
        name,
        "READ",
        auth=registry.admin,
    )
    response = requests.get(
        _url(registry, prefix, f"/{name}"),
        auth=reader,
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_skill_search_does_not_disclose_unreadable_skills(authenticated_registry, prefix):
    registry = authenticated_registry
    reader = create_user(registry.base_url)
    visible = f"visible-{random_str()}"
    hidden = f"hidden-{random_str()}"
    _create_skill(registry, prefix, visible)
    _create_skill(registry, prefix, hidden)
    grant_role_permission(
        registry.base_url,
        reader[0],
        "skill",
        visible,
        "READ",
        auth=registry.admin,
    )

    response = requests.get(_url(registry, prefix), auth=reader)

    assert response.status_code == 200, response.text
    assert [skill["name"] for skill in response.json()["skills"]] == [visible]


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_skill_mutations_require_write_permission(authenticated_registry, prefix):
    registry = authenticated_registry
    reader = create_user(registry.base_url)
    name = f"private-{random_str()}"
    _create_skill(registry, prefix, name)
    _create_version(registry, prefix, name)
    grant_role_permission(
        registry.base_url,
        reader[0],
        "skill",
        name,
        "READ",
        auth=registry.admin,
    )

    response = requests.patch(
        _url(registry, prefix, f"/{name}"),
        json={"description": "not allowed"},
        auth=reader,
    )
    assert response.status_code == 403, response.text
    response = requests.get(_url(registry, prefix, f"/{name}"), auth=registry.admin)
    assert response.status_code == 200, response.text
    assert response.json()["description"] != "not allowed"

    response = requests.delete(
        _url(registry, prefix, f"/{name}/versions/1"),
        auth=reader,
    )
    assert response.status_code == 403, response.text
    response = requests.get(
        _url(registry, prefix, f"/{name}/versions/1"),
        auth=registry.admin,
    )
    assert response.status_code == 200, response.text

    response = requests.post(
        _url(registry, prefix, f"/{name}/tags"),
        json={"key": "owner", "value": "reader"},
        auth=reader,
    )
    assert response.status_code == 403, response.text
    response = requests.get(_url(registry, prefix, f"/{name}"), auth=registry.admin)
    assert response.status_code == 200, response.text
    assert response.json()["tags"] == {}


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_skill_version_registration_requires_edit_permission(authenticated_registry, prefix):
    registry = authenticated_registry
    reader = create_user(registry.base_url)
    name = f"registration-{random_str()}"
    _create_skill(registry, prefix, name)
    grant_role_permission(
        registry.base_url,
        reader[0],
        "skill",
        name,
        "READ",
        auth=registry.admin,
    )

    response = requests.post(
        _url(registry, prefix, f"/{name}/versions"),
        json={
            "source": "https://github.com/acme/skills.git",
            "source_type": "git",
            "digest": "ab" * 32,
        },
        auth=reader,
    )

    assert response.status_code == 403, response.text
    response = requests.get(
        _url(registry, prefix, f"/{name}/versions"),
        auth=registry.admin,
    )
    assert response.json()["skill_versions"] == []


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_skill_alias_permissions_distinguish_edit_and_manage(authenticated_registry, prefix):
    registry = authenticated_registry
    reader = create_user(registry.base_url)
    name = f"alias-{random_str()}"
    _create_skill(registry, prefix, name)
    _create_version(registry, prefix, name)
    response = requests.post(
        _url(registry, prefix, f"/{name}/aliases"),
        json={"alias": "stable", "version": 1},
        auth=registry.admin,
    )
    assert response.status_code == 200, response.text
    grant_role_permission(
        registry.base_url,
        reader[0],
        "skill",
        name,
        "READ",
        auth=registry.admin,
    )

    response = requests.post(
        _url(registry, prefix, f"/{name}/aliases"),
        json={"alias": "candidate", "version": 1},
        auth=reader,
    )
    assert response.status_code == 403, response.text
    response = requests.get(
        _url(registry, prefix, f"/{name}/aliases/candidate"),
        auth=registry.admin,
    )
    assert response.status_code == 404, response.text
    response = requests.delete(
        _url(registry, prefix, f"/{name}/aliases/stable"),
        auth=reader,
    )
    assert response.status_code == 403, response.text
    response = requests.get(
        _url(registry, prefix, f"/{name}/aliases/stable"),
        auth=registry.admin,
    )
    assert response.status_code == 200, response.text

    grant_role_permission(
        registry.base_url,
        reader[0],
        "skill",
        name,
        "EDIT",
        auth=registry.admin,
    )
    response = requests.post(
        _url(registry, prefix, f"/{name}/aliases"),
        json={"alias": "candidate", "version": 1},
        auth=reader,
    )
    assert response.status_code == 200, response.text
    response = requests.delete(
        _url(registry, prefix, f"/{name}/aliases/stable"),
        auth=reader,
    )
    assert response.status_code == 403, response.text
    response = requests.get(
        _url(registry, prefix, f"/{name}/aliases/stable"),
        auth=registry.admin,
    )
    assert response.status_code == 200, response.text

    grant_role_permission(
        registry.base_url,
        reader[0],
        "skill",
        name,
        "MANAGE",
        auth=registry.admin,
    )
    response = requests.delete(
        _url(registry, prefix, f"/{name}/aliases/stable"),
        auth=reader,
    )
    assert response.status_code == 200, response.text
    response = requests.get(
        _url(registry, prefix, f"/{name}/aliases/stable"),
        auth=registry.admin,
    )
    assert response.status_code == 404, response.text


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_skill_permissions_are_organization_scoped(authenticated_registry, prefix):
    registry = authenticated_registry
    reader = create_user(registry.base_url)
    name = f"scoped-{random_str()}"
    for organization in ("acme", "other"):
        response = requests.post(
            _url(registry, prefix),
            json={"name": name, "organization": organization},
            auth=registry.admin,
        )
        assert response.status_code == 200, response.text
    grant_role_permission(
        registry.base_url,
        reader[0],
        "skill",
        f"@acme/{name}",
        "READ",
        auth=registry.admin,
    )

    response = requests.get(
        _url(registry, prefix, f"/@acme/{name}"),
        auth=reader,
    )
    assert response.status_code == 200, response.text
    response = requests.get(
        _url(registry, prefix, f"/@other/{name}"),
        auth=reader,
    )
    assert response.status_code == 403, response.text


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_skill_registration_requires_creation_permission(authenticated_registry, prefix):
    registry = authenticated_registry
    reader = create_user(registry.base_url)
    direct_name = f"not-created-{random_str()}"
    bulk_name = f"not-created-in-bulk-{random_str()}"
    response = requests.post(
        _url(registry, prefix),
        json={"name": direct_name},
        auth=reader,
    )
    assert response.status_code == 403, response.text

    response = requests.post(
        _url(registry, prefix, "/bulk-register"),
        json={
            "skills": [
                {
                    "name": bulk_name,
                    "source": "https://github.com/acme/skills.git",
                    "source_type": "git",
                    "digest": "cd" * 32,
                }
            ]
        },
        auth=reader,
    )
    assert response.status_code == 403, response.text

    for name in (direct_name, bulk_name):
        response = requests.get(_url(registry, prefix, f"/{name}"), auth=registry.admin)
        assert response.status_code == 404, response.text


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_authorized_bulk_registration_succeeds_with_workspace_permission(
    authenticated_registry, prefix
):
    registry = authenticated_registry
    reader = create_user(registry.base_url)
    name = f"bulk-{random_str()}"
    grant_role_permission(
        registry.base_url,
        reader[0],
        "workspace",
        "*",
        "USE",
        auth=registry.admin,
    )

    response = requests.post(
        _url(registry, prefix, "/bulk-register"),
        json={
            "skills": [
                {
                    "name": name,
                    "source": "https://github.com/acme/skills.git",
                    "source_type": "git",
                    "digest": "cd" * 32,
                }
            ]
        },
        auth=reader,
    )

    assert response.status_code == 200, response.text
    assert response.json()["skill_versions"][0]["name"] == name
    response = requests.get(_url(registry, prefix, f"/{name}"), auth=reader)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_denied_multipart_registration_does_not_persist_content(authenticated_registry, prefix):
    registry = authenticated_registry
    reader = create_user(registry.base_url)
    name = f"upload-{random_str()}"
    archive = skill_archive(
        files={
            "SKILL.md": (
                f"---\nname: {name}\ndescription: Reviews code\n---\n# Reviewer\n"
            ).encode()
        }
    ).getvalue()
    files_before = set(stored_files(registry.artifact_root))

    response = requests.post(
        _url(registry, prefix, f"/{name}/versions"),
        files={
            "metadata": ("metadata.json", "{}", "application/json"),
            "content": ("skill.tar.gz", archive, "application/gzip"),
        },
        auth=reader,
    )

    assert response.status_code == 403, response.text
    response = requests.get(_url(registry, prefix, f"/{name}"), auth=registry.admin)
    assert response.status_code == 404, response.text
    assert set(stored_files(registry.artifact_root)) == files_before

    authorized_name = f"authorized-upload-{random_str()}"
    authorized_archive = skill_archive(
        files={
            "SKILL.md": (
                f"---\nname: {authorized_name}\ndescription: Reviews code\n---\n# Reviewer\n"
            ).encode()
        }
    ).getvalue()
    response = requests.post(
        _url(registry, prefix, f"/{authorized_name}/versions"),
        files={
            "metadata": ("metadata.json", "{}", "application/json"),
            "content": ("skill.tar.gz", authorized_archive, "application/gzip"),
        },
        auth=registry.admin,
    )
    assert response.status_code == 200, response.text
    assert any(registry.artifact_root.rglob("*"))

import shutil
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

import pytest
import requests
from git import Actor, Repo

import mlflow.genai
from mlflow import MlflowClient, MlflowException
from mlflow.entities.skill_source import GitSource
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
ICON = {"src": "https://example.com/original.svg", "mimeType": "image/svg+xml"}
NEW_ICON = {"src": "https://example.com/updated.svg", "mimeType": "image/svg+xml"}
SOURCE = "https://example.com/skill.zip"


@dataclass
class Registry:
    url: str
    session: requests.Session
    users: dict[str, tuple[str, str]]

    def request(self, method, path, *, user="admin", workspace=WORKSPACE, **kwargs):
        return self.session.request(
            method,
            f"{self.url}{path}",
            auth=self.users[user] if user is not None else None,
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
            credentials = self.users[user] if user is not None else ("", "")
            with User(*credentials, patch):
                yield MlflowClient(tracking_uri=self.url)


@pytest.fixture(scope="module")
def registry(tmp_path_factory, cached_db):
    root = tmp_path_factory.mktemp("skill-auth-integration")
    # Reuse the session's migrated schema while keeping this server's data isolated.
    backend = root / "tracking.db"
    shutil.copy2(cached_db, backend)
    backend_uri = f"sqlite:///{backend}"
    config = write_isolated_auth_config(root)
    config.write_text(
        config.read_text().replace(
            "default_permission = READ", "default_permission = NO_PERMISSIONS"
        )
    )
    with (
        _init_server(
            backend_uri=backend_uri,
            root_artifact_uri=(root / "artifacts").as_uri(),
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
                ARTIFACTS_DESTINATION_ENV_VAR: str(root / "artifacts"),
            },
        ) as url,
        requests.Session() as session,
    ):
        session.trust_env = False
        result = Registry(url, session, {"admin": (ADMIN_USERNAME, ADMIN_PASSWORD)})
        for workspace in (WORKSPACE, OTHER_WORKSPACE):
            response = result.request(
                "POST", "/api/3.0/mlflow/workspaces", json={"name": workspace}, workspace="default"
            )
            assert response.status_code == 201
        for permission in ("READ", "EDIT", "MANAGE", "NONE", "creator", "DENY"):
            username, password = create_user(url)
            result.users[permission] = username, password
            if permission in ("creator", "DENY"):
                grant_role_permission(url, username, "workspace", "*", "USE", WORKSPACE)
            if permission in ("READ", "EDIT", "MANAGE", "DENY"):
                grant_role_permission(url, username, "skill", "*", permission, WORKSPACE)
        result.users["bad-password"] = result.users["READ"][0], "incorrect-password"
        yield result


@pytest.fixture(params=["", "acme"], ids=["default-organization", "named-organization"])
def organization(request):
    return request.param


def skill_path(name, organization):
    return f"/@{organization}/{name}" if organization else f"/{name}"


@pytest.fixture
def skill(registry, organization):
    name = f"skill-{random_str().lower()}"
    path = f"{PREFIX}{skill_path(name, organization)}"
    response = registry.request(
        "POST",
        PREFIX,
        json={
            "name": name,
            "organization": organization,
            "description": "original",
            "icons": [ICON],
        },
    )
    assert response.status_code == 200
    assert registry.request("POST", f"{path}/versions", json={"source": SOURCE}).status_code == 200
    assert (
        registry.request(
            "POST", f"{path}/tags", json={"key": "team", "value": "original"}
        ).status_code
        == 200
    )
    assert (
        registry.request(
            "POST", f"{path}/versions/1/tags", json={"key": "team", "value": "original"}
        ).status_code
        == 200
    )
    assert (
        registry.request(
            "POST", f"{path}/aliases", json={"alias": "stable", "version": 1}
        ).status_code
        == 200
    )
    yield name
    assert registry.request("DELETE", path).status_code in (200, 404)


@dataclass(frozen=True)
class Operation:
    sdk: str
    method: str
    suffix: str
    permission: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    body: dict[str, Any] | None = None


# Expected permissions are an explicit contract, independent of the middleware's policy table.
# Parent-tag deletion uses EDIT as merged in #25988; mlflow/rfcs#65 clarifies the RFC.
OPERATIONS = [
    Operation("get_skill", "GET", "", "READ"),
    Operation("get_skill_version", "GET", "/versions/1", "READ", {"version": 1}),
    Operation("get_skill_version_by_alias", "GET", "/aliases/stable", "READ", {"alias": "stable"}),
    Operation("get_latest_skill_version", "GET", "/aliases/latest", "READ"),
    Operation("search_skill_versions", "GET", "/versions", "READ"),
    Operation(
        "update_skill",
        "PATCH",
        "",
        "EDIT",
        {"description": "updated", "icons": [NEW_ICON]},
        {"description": "updated", "icons": [NEW_ICON]},
    ),
    Operation(
        "create_skill_version",
        "POST",
        "/versions",
        "EDIT",
        {"source": SOURCE, "status": "draft"},
        {"source": SOURCE, "status": "draft"},
    ),
    Operation(
        "update_skill_version",
        "PATCH",
        "/versions/1",
        "EDIT",
        {"version": 1, "status": "deprecated"},
        {"status": "deprecated"},
    ),
    Operation(
        "set_skill_tag",
        "POST",
        "/tags",
        "EDIT",
        {"key": "team", "value": "updated"},
        {"key": "team", "value": "updated"},
    ),
    Operation("delete_skill_tag", "DELETE", "/tags/team", "EDIT", {"key": "team"}),
    Operation(
        "set_skill_version_tag",
        "POST",
        "/versions/1/tags",
        "EDIT",
        {"version": 1, "key": "team", "value": "updated"},
        {"key": "team", "value": "updated"},
    ),
    Operation(
        "delete_skill_version_tag",
        "DELETE",
        "/versions/1/tags/team",
        "MANAGE",
        {"version": 1, "key": "team"},
    ),
    Operation(
        "set_skill_alias",
        "POST",
        "/aliases",
        "EDIT",
        {"alias": "candidate", "version": 1},
        {"alias": "candidate", "version": 1},
    ),
    Operation("delete_skill_alias", "DELETE", "/aliases/stable", "MANAGE", {"alias": "stable"}),
    Operation("delete_skill_version", "DELETE", "/versions/1", "MANAGE", {"version": 1}),
    Operation(
        "update_skill_version",
        "PATCH",
        "/versions/1",
        "MANAGE",
        {"version": 1, "status": "deleted"},
        {"status": "deleted"},
    ),
    Operation("delete_skill", "DELETE", "", "MANAGE"),
]


def operation_id(operation):
    return f"{operation.sdk}-{operation.permission}"


def snapshot(registry, name, organization, workspace=WORKSPACE):
    path = f"{PREFIX}{skill_path(name, organization)}"
    parent = registry.request("GET", path, workspace=workspace)
    assert parent.status_code == 200
    versions = registry.request("GET", f"{path}/versions", workspace=workspace)
    assert versions.status_code == 200
    return parent.json(), versions.json()


def prepare_operation(registry, name, organization, operation):
    if operation.sdk == "delete_skill_version" or operation.kwargs.get("status") == "deleted":
        path = f"{PREFIX}{skill_path(name, organization)}/versions/1"
        assert registry.request("PATCH", path, json={"status": "deprecated"}).status_code == 200


def assert_operation_persisted(registry, name, organization, operation):
    path = f"{PREFIX}{skill_path(name, organization)}"
    if operation.sdk == "delete_skill":
        assert registry.request("GET", path).status_code == 404
        return
    parent = registry.request("GET", path).json()
    if operation.sdk == "delete_skill_version" or operation.kwargs.get("status") == "deleted":
        assert registry.request("GET", f"{path}/versions/1").status_code == 404
        assert registry.request("GET", f"{path}/versions").json()["skill_versions"] == []
        assert parent["aliases"] == []
        return
    version = registry.request("GET", f"{path}/versions/1").json()
    match operation.sdk:
        case "get_skill":
            assert parent["icons"] == [ICON]
        case "update_skill":
            assert parent["description"] == "updated"
            assert parent["icons"] == [NEW_ICON]
        case "create_skill_version":
            assert registry.request("GET", f"{path}/versions/2").status_code == 200
        case "update_skill_version":
            assert version["status"] == operation.kwargs["status"]
        case "set_skill_tag":
            assert parent["tags"]["team"] == "updated"
        case "delete_skill_tag":
            assert "team" not in parent["tags"]
        case "set_skill_version_tag":
            assert version["tags"]["team"] == "updated"
        case "delete_skill_version_tag":
            assert "team" not in version["tags"]
        case "set_skill_alias":
            assert {"alias": "candidate", "version": 1} in parent["aliases"]
        case "delete_skill_alias":
            assert not any(alias["alias"] == "stable" for alias in parent["aliases"])


def is_allowed(permission, required):
    return (
        permission
        in {"READ": {"READ", "EDIT", "MANAGE"}, "EDIT": {"EDIT", "MANAGE"}, "MANAGE": {"MANAGE"}}[
            required
        ]
    )


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("permission", ["NONE", "READ", "EDIT", "MANAGE"])
@pytest.mark.parametrize("operation", OPERATIONS, ids=operation_id)
def test_rest_permission_matrix(registry, skill, organization, prefix, permission, operation):
    prepare_operation(registry, skill, organization, operation)
    before = snapshot(registry, skill, organization)
    response = registry.request(
        operation.method,
        f"{prefix}{skill_path(skill, organization)}{operation.suffix}",
        user=permission,
        json=operation.body,
    )
    if is_allowed(permission, operation.permission):
        assert response.status_code == 200
        assert_operation_persisted(registry, skill, organization, operation)
        if operation.sdk == "get_skill":
            assert response.json()["icons"] == [ICON]
    else:
        assert response.status_code == 403
        assert snapshot(registry, skill, organization) == before


@pytest.mark.parametrize("permission", ["NONE", "READ", "EDIT", "MANAGE"])
@pytest.mark.parametrize("operation", OPERATIONS, ids=operation_id)
def test_sdk_permission_matrix(registry, skill, organization, permission, operation):
    prepare_operation(registry, skill, organization, operation)
    before = snapshot(registry, skill, organization)
    with registry.client(permission) as client:
        if is_allowed(permission, operation.permission):
            result = getattr(client, operation.sdk)(
                name=skill, organization=organization, **operation.kwargs
            )
            if operation.sdk == "get_skill":
                assert result.icons == [ICON]
        else:
            with pytest.raises(MlflowException, match="Permission denied") as error:
                getattr(client, operation.sdk)(
                    name=skill, organization=organization, **operation.kwargs
                )
            assert error.value.get_http_status_code() == 403
    if is_allowed(permission, operation.permission):
        assert_operation_persisted(registry, skill, organization, operation)
    else:
        assert snapshot(registry, skill, organization) == before


@pytest.mark.parametrize("transport", ["rest", "ajax", "sdk", "genai"])
@pytest.mark.parametrize("permission", ["NONE", "READ", "EDIT", "MANAGE", "DENY"])
def test_discovery_and_icons_require_read(registry, skill, organization, permission, transport):
    identity = f"@{organization}/{skill}" if organization else skill
    if transport in ("rest", "ajax"):
        prefix = PREFIXES[transport == "ajax"]
        response = registry.request(
            "GET", prefix, user=permission, params={"include_skill_identities": identity}
        )
        assert response.status_code == 200
        skills = response.json()["skills"]
    else:
        with registry.client(permission) as client:
            search = client.search_skills if transport == "sdk" else mlflow.genai.search_skills
            skills = [asdict(s) for s in search(include_skill_identities=[identity])]
    if permission in ("NONE", "DENY"):
        assert skills == []
    else:
        assert len(skills) == 1
        assert skills[0]["name"] == skill
        assert skills[0]["organization"] == organization
        assert skills[0]["icons"] == [ICON]


def bulk_definition(name):
    return {
        "name": name,
        "source": "https://example.com/skills.git",
        "ref": "main",
        "digest": "a" * 64,
        "status": "draft",
    }


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("user", [None, "bad-password"])
def test_all_rest_operations_reject_unauthenticated_requests(
    registry, skill, organization, prefix, user
):
    before = snapshot(registry, skill, organization)
    path = f"{prefix}{skill_path(skill, organization)}"
    for operation in OPERATIONS:
        response = registry.request(
            operation.method, f"{path}{operation.suffix}", user=user, json=operation.body
        )
        assert response.status_code == 401
    assert registry.request("GET", prefix, user=user).status_code == 401
    assert (
        registry.request(
            "POST",
            prefix,
            user=user,
            json={"name": "unauthenticated", "organization": organization},
        ).status_code
        == 401
    )
    assert (
        registry.request(
            "POST",
            f"{prefix}/bulk-register",
            user=user,
            json={"organization": organization, "skills": [bulk_definition(skill)]},
        ).status_code
        == 401
    )
    assert snapshot(registry, skill, organization) == before


@pytest.mark.parametrize("user", [None, "bad-password"])
def test_all_sdk_operations_reject_unauthenticated_requests(registry, skill, organization, user):
    before = snapshot(registry, skill, organization)
    with registry.client(user) as client:
        for operation in OPERATIONS:
            with pytest.raises(MlflowException, match="not authenticated") as error:
                getattr(client, operation.sdk)(
                    name=skill, organization=organization, **operation.kwargs
                )
            assert error.value.get_http_status_code() == 401
        for call in (
            lambda: client.search_skills(),
            lambda: mlflow.genai.search_skills(),
            lambda: client.create_skill(name="unauthenticated", organization=organization),
            lambda: client.bulk_register_skills(
                skill_definitions=[bulk_definition(skill)], organization=organization
            ),
        ):
            with pytest.raises(MlflowException, match="not authenticated") as error:
                call()
            assert error.value.get_http_status_code() == 401
    assert snapshot(registry, skill, organization) == before


@pytest.mark.parametrize("transport", ["rest", "ajax", "sdk"])
@pytest.mark.parametrize("operation", ["create", "version", "bulk"])
@pytest.mark.parametrize("user", ["READ", "EDIT", "MANAGE", "creator", "DENY"])
def test_creation_requires_workspace_permission_and_grants_creator_manage(
    registry, organization, transport, operation, user
):
    name = f"created-{random_str().lower()}"
    path = skill_path(name, organization)
    if transport == "sdk":
        with registry.client(user) as client:

            def create():
                if operation == "create":
                    return client.create_skill(name=name, organization=organization, icons=[ICON])
                if operation == "version":
                    return client.create_skill_version(
                        name=name, organization=organization, source=SOURCE
                    )
                return client.bulk_register_skills(
                    skill_definitions=[bulk_definition(name)], organization=organization
                )

            if user == "creator":
                create()
            else:
                with pytest.raises(MlflowException, match="Permission denied") as error:
                    create()
                assert error.value.get_http_status_code() == 403
    else:
        prefix = PREFIXES[transport == "ajax"]
        if operation == "create":
            endpoint = prefix
            body = {"name": name, "organization": organization, "icons": [ICON]}
        elif operation == "version":
            endpoint = f"{prefix}{path}/versions"
            body = {"source": SOURCE}
        else:
            endpoint = f"{prefix}/bulk-register"
            body = {"organization": organization, "skills": [bulk_definition(name)]}
        response = registry.request("POST", endpoint, user=user, json=body)
        assert response.status_code == (200 if user == "creator" else 403)

    parent = registry.request("GET", f"{PREFIX}{path}")
    if user == "creator":
        assert parent.status_code == 200
        assert parent.json()["created_by"] == registry.users[user][0]
        if operation == "create":
            assert parent.json()["icons"] == [ICON]
        # This principal has workspace USE only; hard deletion proves creator MANAGE was granted.
        assert registry.request("DELETE", f"{PREFIX}{path}", user=user).status_code == 200
        assert registry.request("GET", f"{PREFIX}{path}").status_code == 404
    else:
        assert parent.status_code == 404


@pytest.mark.parametrize("transport", ["rest", "ajax", "sdk"])
@pytest.mark.parametrize("permission", ["NONE", "READ", "EDIT", "MANAGE"])
def test_bulk_registration_of_existing_skill_requires_edit(
    registry, skill, organization, transport, permission
):
    before = snapshot(registry, skill, organization)
    if transport == "sdk":
        with registry.client(permission) as client:
            if permission in ("EDIT", "MANAGE"):
                versions = client.bulk_register_skills(
                    skill_definitions=[bulk_definition(skill)], organization=organization
                )
                assert versions[0].version == 2
            else:
                with pytest.raises(MlflowException, match="Permission denied") as error:
                    client.bulk_register_skills(
                        skill_definitions=[bulk_definition(skill)], organization=organization
                    )
                assert error.value.get_http_status_code() == 403
    else:
        prefix = PREFIXES[transport == "ajax"]
        response = registry.request(
            "POST",
            f"{prefix}/bulk-register",
            user=permission,
            json={"organization": organization, "skills": [bulk_definition(skill)]},
        )
        assert response.status_code == (200 if permission in ("EDIT", "MANAGE") else 403)
    if permission in ("EDIT", "MANAGE"):
        version = registry.request("GET", f"{PREFIX}{skill_path(skill, organization)}/versions/2")
        assert version.status_code == 200
        assert version.json()["created_by"] == registry.users[permission][0]
    else:
        assert snapshot(registry, skill, organization) == before


@pytest.mark.parametrize("transport", ["rest", "ajax", "sdk"])
def test_same_named_skills_do_not_share_workspace_permissions(
    registry, skill, organization, transport, tmp_path
):
    path = f"{PREFIX}{skill_path(skill, organization)}"
    assert (
        registry.request(
            "POST",
            PREFIX,
            workspace=OTHER_WORKSPACE,
            json={
                "name": skill,
                "organization": organization,
                "description": "private workspace",
                "icons": [NEW_ICON],
            },
        ).status_code
        == 200
    )
    assert (
        registry.request(
            "POST", f"{path}/versions", workspace=OTHER_WORKSPACE, json={"source": SOURCE}
        ).status_code
        == 200
    )
    before = snapshot(registry, skill, organization)
    other_before = snapshot(registry, skill, organization, OTHER_WORKSPACE)
    identity = f"@{organization}/{skill}" if organization else skill
    try:
        if transport == "sdk":
            with registry.client("MANAGE", OTHER_WORKSPACE) as client:
                assert client.search_skills(include_skill_identities=[identity]) == []
                for operation in OPERATIONS:
                    with pytest.raises(MlflowException, match="Permission denied") as error:
                        getattr(client, operation.sdk)(
                            name=skill, organization=organization, **operation.kwargs
                        )
                    assert error.value.get_http_status_code() == 403
                with pytest.raises(MlflowException, match="Permission denied"):
                    client.bulk_register_skills(
                        skill_definitions=[bulk_definition(skill)], organization=organization
                    )
                with pytest.raises(MlflowException, match="Permission denied"):
                    mlflow.genai.pull(f"skills:/{identity}/1", destination=tmp_path / "private")
                assert not (tmp_path / "private").exists()
        else:
            prefix = PREFIXES[transport == "ajax"]
            response = registry.request(
                "GET",
                prefix,
                user="MANAGE",
                workspace=OTHER_WORKSPACE,
                params={"include_skill_identities": identity},
            )
            assert response.status_code == 200
            assert response.json()["skills"] == []
            for operation in OPERATIONS:
                response = registry.request(
                    operation.method,
                    f"{prefix}{skill_path(skill, organization)}{operation.suffix}",
                    user="MANAGE",
                    workspace=OTHER_WORKSPACE,
                    json=operation.body,
                )
                assert response.status_code == 403
            assert (
                registry.request(
                    "POST",
                    f"{prefix}/bulk-register",
                    user="MANAGE",
                    workspace=OTHER_WORKSPACE,
                    json={"organization": organization, "skills": [bulk_definition(skill)]},
                ).status_code
                == 403
            )
        assert snapshot(registry, skill, organization) == before
        assert snapshot(registry, skill, organization, OTHER_WORKSPACE) == other_before
    finally:
        assert registry.request("DELETE", path, workspace=OTHER_WORKSPACE).status_code == 200


@pytest.mark.parametrize("transport", ["rest", "sdk"])
def test_qualified_search_filters_before_pagination_and_isolates_workspace(registry, transport):
    prefix = f"scope-{random_str().lower()}"
    first = f"{prefix}-b"
    second = f"{prefix}-c"
    identities = [("", f"{prefix}-a-hidden"), ("", first), ("acme", first), ("", second)]
    credentials = create_user(registry.url)
    registry.users[credentials[0]] = credentials
    for org, name in identities:
        assert (
            registry.request("POST", PREFIX, json={"name": name, "organization": org}).status_code
            == 200
        )
    for identity in (f"@acme/{first}", second):
        grant_role_permission(registry.url, credentials[0], "skill", identity, "READ", WORKSPACE)
    try:
        query = {
            "filter_string": f"name LIKE '{prefix}%'",
            "order_by": ["name ASC"],
            "max_results": 1,
        }
        if transport == "rest":
            assert (
                registry.request("GET", f"{PREFIX}/{first}", user=credentials[0]).status_code == 403
            )
            assert (
                registry.request("GET", f"{PREFIX}/@acme/{first}", user=credentials[0]).status_code
                == 200
            )
            page = registry.request("GET", PREFIX, user=credentials[0], params=query)
            assert page.status_code == 200
            assert [(s["organization"], s["name"]) for s in page.json()["skills"]] == [
                ("acme", first)
            ]
            token = page.json()["next_page_token"]
            assert token
            page = registry.request(
                "GET", PREFIX, user=credentials[0], params={**query, "page_token": token}
            )
            assert [(s["organization"], s["name"]) for s in page.json()["skills"]] == [("", second)]
            assert page.json()["next_page_token"] is None
            assert (
                registry.request(
                    "GET",
                    PREFIX,
                    user=credentials[0],
                    workspace=OTHER_WORKSPACE,
                    params={**query, "page_token": token},
                ).status_code
                == 400
            )
            # A caller selector cannot widen the readable scope, including same-named orgs.
            selected = registry.request(
                "GET",
                PREFIX,
                user=credentials[0],
                params={"include_skill_identities": [first, f"@acme/{first}"]},
            )
            assert [(s["organization"], s["name"]) for s in selected.json()["skills"]] == [
                ("acme", first)
            ]
            assert (
                registry.request(
                    "GET", PREFIX, user=credentials[0], params={"include_skill_identities": ""}
                ).json()["skills"]
                == []
            )
        else:
            with registry.client(credentials[0]) as client:
                with pytest.raises(MlflowException, match="Permission denied"):
                    client.get_skill(name=first)
                assert client.get_skill(name=first, organization="acme").organization == "acme"
                page = client.search_skills(**query)
                assert [(s.organization, s.name) for s in page] == [("acme", first)]
                token = page.token
                assert token
                page = client.search_skills(**query, page_token=token)
                assert [(s.organization, s.name) for s in page] == [("", second)]
                assert page.token is None
                selected = client.search_skills(include_skill_identities=[first, f"@acme/{first}"])
                assert [(s.organization, s.name) for s in selected] == [("acme", first)]
                assert client.search_skills(include_skill_identities=[]) == []
            with registry.client(credentials[0], OTHER_WORKSPACE) as client:
                with pytest.raises(MlflowException, match="Page token") as error:
                    client.search_skills(**query, page_token=token)
                assert error.value.get_http_status_code() == 400
    finally:
        for org, name in identities:
            assert registry.request("DELETE", f"{PREFIX}{skill_path(name, org)}").status_code == 200


@pytest.mark.parametrize("transport", ["rest", "sdk"])
def test_bulk_permission_failure_leaves_no_partial_parents_or_versions(
    registry, skill, organization, transport
):
    credentials = create_user(registry.url)
    registry.users[credentials[0]] = credentials
    grant_role_permission(registry.url, credentials[0], "workspace", "*", "USE", WORKSPACE)
    identity = f"@{organization}/{skill}" if organization else skill
    grant_role_permission(registry.url, credentials[0], "skill", identity, "READ", WORKSPACE)
    name = f"atomic-{random_str().lower()}"
    definitions = [bulk_definition(name), bulk_definition(skill)]
    before = snapshot(registry, skill, organization)
    if transport == "rest":
        response = registry.request(
            "POST",
            f"{PREFIX}/bulk-register",
            user=credentials[0],
            json={"organization": organization, "skills": definitions},
        )
        assert response.status_code == 403
    else:
        with registry.client(credentials[0]) as client:
            with pytest.raises(MlflowException, match="Permission denied") as error:
                client.bulk_register_skills(
                    skill_definitions=definitions, organization=organization
                )
            assert error.value.get_http_status_code() == 403
    assert snapshot(registry, skill, organization) == before
    assert registry.request("GET", f"{PREFIX}{skill_path(name, organization)}").status_code == 404

    # The same batch succeeds once every existing parent is editable.
    grant_role_permission(registry.url, credentials[0], "skill", identity, "EDIT", WORKSPACE)
    if transport == "rest":
        response = registry.request(
            "POST",
            f"{PREFIX}/bulk-register",
            user=credentials[0],
            json={"organization": organization, "skills": definitions},
        )
        assert response.status_code == 200
    else:
        with registry.client(credentials[0]) as client:
            versions = client.bulk_register_skills(
                skill_definitions=definitions, organization=organization
            )
            assert [(v.name, v.version) for v in versions] == [(name, 1), (skill, 2)]
    assert (
        registry.request("GET", f"{PREFIX}{skill_path(skill, organization)}/versions/2").status_code
        == 200
    )
    # Only the newly created parent gets creator MANAGE; the existing one still has EDIT.
    assert (
        registry.request(
            "DELETE", f"{PREFIX}{skill_path(name, organization)}", user=credentials[0]
        ).status_code
        == 200
    )
    assert (
        registry.request(
            "DELETE", f"{PREFIX}{skill_path(skill, organization)}", user=credentials[0]
        ).status_code
        == 403
    )


def write_skill(directory, name):
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Integration test skill\n---\nPrivate content.\n"
    )
    return directory


@pytest.mark.parametrize("user", [None, "bad-password", "NONE", "READ", "EDIT", "MANAGE"])
def test_local_registration_enforces_edit_and_artifact_read(
    registry, skill, organization, user, tmp_path
):
    source = write_skill(tmp_path / "source", "local-skill")
    before = snapshot(registry, skill, organization)
    with registry.client(user):
        if user in ("EDIT", "MANAGE"):
            version = mlflow.genai.register_skill(
                source=str(source), name=skill, organization=organization, status="draft"
            )
            assert version.version == 2
        else:
            with pytest.raises(
                MlflowException, match="Permission denied|not authenticated"
            ) as error:
                mlflow.genai.register_skill(
                    source=str(source), name=skill, organization=organization, status="draft"
                )
            assert error.value.get_http_status_code() == (
                401 if user in (None, "bad-password") else 403
            )
    if user not in ("EDIT", "MANAGE"):
        assert snapshot(registry, skill, organization) == before
        return
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


@pytest.mark.parametrize(
    "user", [None, "bad-password", "READ", "EDIT", "MANAGE", "creator", "DENY"]
)
def test_local_registration_creation_permission_and_creator_grant(
    registry, organization, user, tmp_path
):
    name = f"upload-{random_str().lower()}"
    source = write_skill(tmp_path / "source", name)
    with registry.client(user):
        if user == "creator":
            version = mlflow.genai.register_skill(source=str(source), organization=organization)
            assert version.name == name
            assert version.created_by == registry.users[user][0]
        else:
            with pytest.raises(
                MlflowException, match="Permission denied|not authenticated"
            ) as error:
                mlflow.genai.register_skill(source=str(source), organization=organization)
            assert error.value.get_http_status_code() == (
                401 if user in (None, "bad-password") else 403
            )
    path = f"{PREFIX}{skill_path(name, organization)}"
    if user == "creator":
        assert registry.request("DELETE", path, user=user).status_code == 200
    assert registry.request("GET", path).status_code == 404


@pytest.mark.parametrize("user", [None, "bad-password", "NONE", "READ", "EDIT", "MANAGE"])
def test_standalone_git_import_requires_edit(registry, skill, organization, user, tmp_path):
    root = tmp_path / "git"
    write_skill(root / "skill", skill)
    with Repo.init(root) as repo:
        repo.index.add(["skill/SKILL.md"])
        actor = Actor("Test", "test@example.com")
        commit = repo.index.commit("Add standalone test Skill", author=actor, committer=actor)
        source = GitSource(url=root.as_uri(), ref=commit.hexsha)
    before = snapshot(registry, skill, organization)
    with registry.client(user):
        if user in ("EDIT", "MANAGE"):
            versions = mlflow.genai.import_skills(source=source, organization=organization)
            assert [(v.name, v.version) for v in versions] == [(skill, 2)]
        else:
            with pytest.raises(
                MlflowException, match="Permission denied|not authenticated"
            ) as error:
                mlflow.genai.import_skills(source=source, organization=organization)
            assert error.value.get_http_status_code() == (
                401 if user in (None, "bad-password") else 403
            )
    if user not in ("EDIT", "MANAGE"):
        assert snapshot(registry, skill, organization) == before


@pytest.mark.parametrize("suffix", ["/2", "@stable", ""], ids=["version", "alias", "latest"])
@pytest.mark.parametrize(
    "user", [None, "bad-password", "NONE", "READ", "EDIT", "MANAGE", "creator", "DENY"]
)
def test_pull_requires_read_and_denial_writes_nothing(
    registry, skill, organization, user, suffix, tmp_path
):
    source = write_skill(tmp_path / "source", "pull-skill")
    with registry.client("admin") as client:
        version = mlflow.genai.register_skill(
            source=str(source), name=skill, organization=organization
        )
        assert version.version == 2
        client.set_skill_alias(name=skill, organization=organization, alias="stable", version=2)
    identity = f"@{organization}/{skill}" if organization else skill
    destination = tmp_path / "pulled"
    with registry.client(user):
        if user in ("READ", "EDIT", "MANAGE"):
            assert mlflow.genai.pull(f"skills:/{identity}{suffix}", destination=destination) == str(
                destination
            )
            assert (destination / "SKILL.md").read_bytes() == (source / "SKILL.md").read_bytes()
            assert sorted(path.name for path in destination.iterdir()) == ["SKILL.md"]
        else:
            with pytest.raises(
                MlflowException, match="Permission denied|not authenticated"
            ) as error:
                mlflow.genai.pull(f"skills:/{identity}{suffix}", destination=destination)
            assert error.value.get_http_status_code() == (
                401 if user in (None, "bad-password") else 403
            )
            assert not destination.exists()

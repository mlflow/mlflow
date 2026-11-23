# Skill Registry permissions, checked without a network: TestClient hands each request to an
# app with the basic-auth permission middleware and the Skill routes, so no server is started.
# Checks that need a live server (logging in, the SDK's own client, pull) are in
# test_skill_registry_integration.py.
import asyncio
import io
import json
from dataclasses import dataclass
from typing import Any
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.requests import Request
from starlette.testclient import TestClient

from mlflow.entities import Workspace
from mlflow.entities.skill import SkillStatus
from mlflow.entities.skill_version import SkillVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import TEMPORARILY_UNAVAILABLE
from mlflow.server import auth, handlers, skill_registry_api, workspace_helpers
from mlflow.server.auth.permissions import (
    DENY,
    EDIT,
    MANAGE,
    NO_PERMISSIONS,
    READ,
    USE,
    Permission,
)
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore as AuthStore
from mlflow.server.fastapi_app import (
    add_fastapi_workspace_middleware,
    add_registry_exception_handlers,
)
from mlflow.server.skill_registry_api import (
    get_skill_registry_api_route_prefixes,
    skill_registry_router,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore as TrackingStore
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.store.workspace.sqlalchemy_store import SqlAlchemyStore as WorkspaceStore
from mlflow.utils.validation import _MAX_BULK_REGISTER_SKILLS
from mlflow.utils.workspace_context import ServerWorkspaceContext
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME

SKILLS = "/api/3.0/mlflow/skills"
SOURCE = "https://example.com/skill.zip"
ICONS = [{"src": "https://8.8.8.8/reviewer.png"}]


def test_every_skill_route_has_one_auth_policy():
    endpoints = [route.endpoint for route in skill_registry_router.routes]
    assert len(endpoints) == len(set(endpoints))
    assert set(endpoints) == set(auth._SKILL_ROUTE_OPERATIONS)
    assert set(auth._SKILL_ROUTE_OPERATIONS.values()) == {
        "read",
        "update",
        "manage",
        "create",
        "register",
        "search",
    }


def test_unmapped_skill_route_fails_closed(monkeypatch):
    monkeypatch.setattr(
        skill_registry_router,
        "routes",
        [*skill_registry_router.routes, APIRoute("/unmapped/operation", lambda: {})],
    )
    path = "/api/3.0/mlflow/skills/unmapped/operation"
    request = Request({"type": "http", "method": "GET", "path": path})
    assert not asyncio.run(auth._get_skill_registry_validator(path)("reader", request))


@pytest.mark.parametrize("path", ["reviewer", "@acme/reviewer"])
@pytest.mark.parametrize("allowed", [False, True])
def test_named_registration_authorizes_without_reading_upload(monkeypatch, path, allowed):
    path = f"/api/3.0/mlflow/skills/{path}/versions"

    async def unexpected_receive():
        raise AssertionError("Authorization read the upload body")

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [(b"content-type", b"multipart/form-data; boundary=boundary")],
        },
        unexpected_receive,
    )
    checked = []

    def check(username, organization, name, *, parent_exists):
        checked.append((username, organization, name, parent_exists))
        return allowed

    monkeypatch.setattr(auth, "_skill_exists_for_auth", lambda *args: True)
    monkeypatch.setattr(auth, "validate_can_register_skill", check)
    assert asyncio.run(auth._get_skill_registry_validator(path)("editor", request)) is allowed
    assert checked == [("editor", "acme" if "@acme" in path else "", "reviewer", True)]


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("", []),
        ("", {"name": 123}),
        ("", {"name": "reviewer", "organization": None}),
        ("/bulk-register", {"skills": {"name": "reviewer"}}),
        ("/bulk-register", {"skills": []}),
        ("/bulk-register", {"skills": ["reviewer"]}),
        ("/bulk-register", {"skills": [{"name": "reviewer"}, {"name": None}]}),
    ],
)
def test_creation_rejects_malformed_authorization_identity(workspace_registry, path, body):
    client, _, tracking_store, _ = workspace_registry
    response = client.post(f"/api/3.0/mlflow/skills{path}", json=body)
    assert response.status_code == 400, response.text
    assert list(tracking_store.search_skills()) == []


@pytest.mark.parametrize("path", ["", "/bulk-register"])
@pytest.mark.parametrize("body", [b'{"name":', b"null", b"[]"])
def test_invalid_creation_body_is_rejected_by_downstream_validation(workspace_registry, path, body):
    client, _, tracking_store, _ = workspace_registry
    with (
        mock.patch.object(auth, "_skill_exists_for_auth") as parent_lookup,
        mock.patch.object(auth, "validate_can_create_skill") as create_permission,
        mock.patch.object(auth, "validate_can_register_skill") as register_permission,
        mock.patch.object(tracking_store, "create_skill") as create,
        mock.patch.object(skill_registry_api, "bulk_register_skill_versions") as register,
    ):
        response = client.post(
            f"/api/3.0/mlflow/skills{path}",
            content=body,
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 400, response.text
    assert response.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    parent_lookup.assert_not_called()
    create_permission.assert_not_called()
    register_permission.assert_not_called()
    create.assert_not_called()
    register.assert_not_called()


def test_oversized_bulk_registration_skips_parent_lookups(workspace_registry):
    client, _, tracking_store, _ = workspace_registry
    skills = [
        {"name": f"skill-{i}", "source": "https://example.com/repo.git", "digest": "a" * 64}
        for i in range(_MAX_BULK_REGISTER_SKILLS + 1)
    ]
    with (
        mock.patch.object(auth, "_skill_exists_for_auth") as parent_lookup,
        mock.patch.object(auth, "validate_can_register_skill") as permission,
        mock.patch.object(skill_registry_api, "bulk_register_skill_versions") as register,
    ):
        response = client.post("/api/3.0/mlflow/skills/bulk-register", json={"skills": skills})
    assert response.status_code == 400, response.text
    assert response.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert str(_MAX_BULK_REGISTER_SKILLS) in response.json()["message"]
    parent_lookup.assert_not_called()
    permission.assert_not_called()
    register.assert_not_called()
    assert list(tracking_store.search_skills()) == []


@pytest.fixture
def workspace_registry(tmp_path, monkeypatch, db_uri):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "true")
    monkeypatch.setenv("MLFLOW_WORKSPACE", "default")
    auth_store = AuthStore()
    auth_store.init_db(f"sqlite:///{tmp_path / 'auth.db'}")
    tracking_store = WorkspaceAwareSqlAlchemyStore(db_uri, str(tmp_path / "artifacts"))
    creator = auth_store.create_user("creator", "strong-password", is_admin=False)
    auth_store.set_workspace_permission("default", creator.username, USE.name)
    monkeypatch.setattr(auth, "store", auth_store)
    monkeypatch.setattr(auth, "_get_tracking_store", lambda: tracking_store)
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda: tracking_store)
    monkeypatch.setattr(
        auth, "auth_config", auth.auth_config._replace(default_permission=NO_PERMISSIONS.name)
    )
    monkeypatch.setattr(auth, "_auth_initialized", True)
    # Requests act as the user named in `x-user`, or as `creator` when the header is absent.
    # `get_user` is bound here, so tests that count the store's user lookups don't count logins.
    get_user = auth_store.get_user
    monkeypatch.setattr(
        auth,
        "_authenticate_fastapi_request",
        lambda request: get_user(request.headers.get("x-user", creator.username)),
    )
    app = FastAPI()
    for prefix in get_skill_registry_api_route_prefixes():
        app.include_router(skill_registry_router, prefix=prefix)
    add_registry_exception_handlers(app)
    auth.add_fastapi_permission_middleware(app)
    try:
        with ServerWorkspaceContext("default"), TestClient(app) as client:
            yield client, auth_store, tracking_store, creator
    finally:
        auth_store.engine.dispose()
        tracking_store.engine.dispose()


def test_parent_skill_responses_include_allowed_actions(workspace_registry):
    client, auth_store, _, creator = workspace_registry
    prefix = "/api/3.0/mlflow/skills"
    manage_actions = ["USE", "UPDATE", "DELETE", "MANAGE"]
    cases = [
        ("", READ, []),
        ("acme", EDIT, ["USE", "UPDATE"]),
        ("other", MANAGE, manage_actions),
    ]
    for organization, permission, actions in cases:
        created = client.post(prefix, json={"name": "reviewer", "organization": organization})
        assert created.status_code == 200, created.text
        assert created.json()["allowed_actions"] == manage_actions
        identity = f"@{organization}/reviewer" if organization else "reviewer"
        auth_store.grant_user_permission(creator.username, "skill", identity, permission.name)

        response = client.get(f"{prefix}/{identity}")
        assert response.status_code == 200, response.text
        assert response.json()["allowed_actions"] == actions
        updated = client.patch(f"{prefix}/{identity}", json={"description": "updated"})
        if permission == READ:
            assert updated.status_code == 403
        else:
            assert updated.status_code == 200, updated.text
            assert updated.json()["allowed_actions"] == actions

    params = {"max_results": 1, "order_by": "organization ASC"}
    for index, (organization, _, actions) in enumerate(cases):
        response = client.get(prefix, params=params)
        assert response.status_code == 200, response.text
        page = response.json()
        assert len(page["skills"]) == 1
        assert page["skills"][0]["organization"] == organization
        assert page["skills"][0]["allowed_actions"] == actions
        token = page["next_page_token"]
        assert (token is not None) == (index < len(cases) - 1)
        params["page_token"] = token

    auth_store.grant_user_permission(creator.username, "skill", "reviewer", EDIT.name)
    updated = client.patch(f"{prefix}/reviewer", json={"description": "updated"})
    assert updated.status_code == 200, updated.text
    assert updated.json()["allowed_actions"] == ["USE", "UPDATE"]


@pytest.mark.parametrize("skill_count", [1, 100])
@pytest.mark.parametrize("workspaces_enabled", [False, True])
def test_search_allowed_actions_reuse_authorization_grants(
    workspace_registry, monkeypatch, skill_count, workspaces_enabled
):
    client, auth_store, tracking_store, creator = workspace_registry
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", str(workspaces_enabled).lower())
    auth_store.grant_user_permission(creator.username, "skill", "*", EDIT.name)
    for index in range(skill_count):
        tracking_store.create_skill(f"skill-{index}", organization="acme")

    with (
        mock.patch.object(auth_store, "get_user", wraps=auth_store.get_user) as get_user,
        mock.patch.object(auth_store, "list_grants", wraps=auth_store.list_grants) as list_grants,
        mock.patch.object(tracking_store, "get_skill", wraps=tracking_store.get_skill) as get_skill,
    ):
        response = client.get("/api/3.0/mlflow/skills")

    assert response.status_code == 200, response.text
    skills = response.json()["skills"]
    assert len(skills) == skill_count
    assert all(skill["allowed_actions"] == ["USE", "UPDATE"] for skill in skills)
    get_user.assert_called_once_with(creator.username)
    list_grants.assert_called_once_with(creator.id, "default", {"skill"})
    get_skill.assert_not_called()


@pytest.mark.parametrize(
    ("permission", "default_permission", "workspace_admin", "grant_default_access", "actions"),
    [
        (READ, EDIT, False, False, ["USE", "UPDATE"]),
        (None, READ, False, True, []),
        (None, EDIT, False, False, None),
        (DENY, EDIT, False, True, None),
        (DENY, NO_PERMISSIONS, True, False, ["USE", "UPDATE", "DELETE", "MANAGE"]),
    ],
)
def test_search_allowed_actions_preserve_permission_rules(
    workspace_registry,
    monkeypatch,
    permission,
    default_permission,
    workspace_admin,
    grant_default_access,
    actions,
):
    client, auth_store, tracking_store, creator = workspace_registry
    tracking_store.create_skill("reviewer", organization="acme")
    if permission is not None:
        auth_store.grant_user_permission(
            creator.username, "skill", "@acme/reviewer", permission.name
        )
    if workspace_admin:
        auth_store.set_workspace_permission("default", creator.username, MANAGE.name)
    monkeypatch.setattr(
        auth,
        "auth_config",
        auth.auth_config._replace(
            default_permission=default_permission.name,
            grant_default_workspace_access=grant_default_access,
        ),
    )

    response = client.get("/api/3.0/mlflow/skills")
    assert response.status_code == 200, response.text
    skills = response.json()["skills"]
    if actions is None:
        assert skills == []
    else:
        assert len(skills) == 1
        assert skills[0]["allowed_actions"] == actions


def test_skill_allowed_actions_are_workspace_scoped(workspace_registry, monkeypatch, db_uri):
    _, auth_store, tracking_store, creator = workspace_registry
    workspace_store = WorkspaceStore(db_uri)
    workspace_store.create_workspace(Workspace(name="other"))
    monkeypatch.setattr(workspace_helpers, "_get_workspace_store", lambda: workspace_store)
    cases = [("default", READ, []), ("other", EDIT, ["USE", "UPDATE"])]
    for workspace, permission, _ in cases:
        auth_store.set_workspace_permission(workspace, creator.username, USE.name)
        with ServerWorkspaceContext(workspace):
            tracking_store.create_skill("reviewer", organization="acme")
            auth_store.grant_user_permission(
                creator.username, "skill", "@acme/reviewer", permission.name
            )

    prefix = "/api/3.0/mlflow/skills"
    app = FastAPI()
    app.include_router(skill_registry_router, prefix=prefix)
    add_registry_exception_handlers(app)
    add_fastapi_workspace_middleware(app)
    auth.add_fastapi_permission_middleware(app)
    with TestClient(app) as client:
        for workspace, _, actions in cases:
            headers = {WORKSPACE_HEADER_NAME: workspace}
            response = client.get(f"{prefix}/@acme/reviewer", headers=headers)
            assert response.status_code == 200, response.text
            assert response.json()["workspace"] == workspace
            assert response.json()["allowed_actions"] == actions
            response = client.get(prefix, headers=headers)
            assert response.status_code == 200, response.text
            assert len(response.json()["skills"]) == 1
            assert response.json()["skills"][0]["allowed_actions"] == actions


@pytest.mark.parametrize("auth_enabled", [False, True])
def test_unrestricted_skill_responses_omit_allowed_actions(
    workspace_registry, monkeypatch, auth_enabled
):
    _, auth_store, _, _ = workspace_registry
    prefix = "/api/3.0/mlflow/skills"
    app = FastAPI()
    app.include_router(skill_registry_router, prefix=prefix)
    add_registry_exception_handlers(app)
    if auth_enabled:
        admin = auth_store.create_user("admin", "strong-password", is_admin=True)
        monkeypatch.setattr(auth, "_authenticate_fastapi_request", lambda request: admin)
        auth.add_fastapi_permission_middleware(app)
    with TestClient(app) as client:
        created = client.post(prefix, json={"name": "reviewer"})
        fetched = client.get(f"{prefix}/reviewer")
        updated = client.patch(f"{prefix}/reviewer", json={"description": "updated"})
        for response in (created, fetched, updated):
            assert response.status_code == 200, response.text
            assert "allowed_actions" not in response.json()
        response = client.get(prefix)
        assert response.status_code == 200, response.text
        assert len(response.json()["skills"]) == 1
        assert "allowed_actions" not in response.json()["skills"][0]


@pytest.mark.parametrize("admin_delete", [False, True])
def test_permission_cleanup_failure_is_logged_after_committed_delete(
    workspace_registry, monkeypatch, caplog, admin_delete
):
    client, auth_store, tracking_store, _ = workspace_registry
    prefix = "/api/3.0/mlflow/skills"
    reader = auth_store.create_user("reader", "strong-password")
    assert client.post(prefix, json={"name": "private"}).status_code == 200
    auth_store.grant_user_permission(reader.username, "skill", "private", READ.name)
    if admin_delete:
        admin = auth_store.create_user("admin", "strong-password", is_admin=True)
        monkeypatch.setattr(auth, "_authenticate_fastapi_request", lambda request: admin)

    def fail_cleanup(*args, **kwargs):
        assert list(tracking_store.search_skills()) == []
        raise MlflowException("Auth store unavailable", TEMPORARILY_UNAVAILABLE)

    with mock.patch.object(
        auth_store, "delete_grants_for_resource", side_effect=fail_cleanup
    ) as cleanup:
        response = client.delete(f"{prefix}/private")
    cleanup.assert_called_once()
    assert response.status_code == 200, response.text
    assert list(tracking_store.search_skills()) == []
    assert "after-request handler failed" in caplog.text
    assert "Auth store unavailable" in caplog.text
    # Accepted failure mode: cleanup cannot roll back the committed tracking deletion.
    assert (
        auth_store.get_role_permission_for_resource(reader.id, "skill", "private", "default")
        == READ
    )


@pytest.mark.parametrize("pattern", ["*", "@acme/private"])
@pytest.mark.parametrize("endpoint", ["create", "versions", "bulk-register", "multipart"])
def test_skill_deny_vetoes_creation(workspace_registry, monkeypatch, pattern, endpoint):
    client, auth_store, tracking_store, creator = workspace_registry
    auth_store.grant_user_permission(creator.username, "skill", pattern, DENY.name)
    prefix = "/api/3.0/mlflow/skills"

    def unexpected_write(*args, **kwargs):
        raise AssertionError("Denied creation reached registration or persistence")

    with monkeypatch.context() as patch:
        patch.setattr(skill_registry_api, "register_skill_version", unexpected_write)
        patch.setattr(skill_registry_api, "bulk_register_skill_versions", unexpected_write)
        patch.setattr(tracking_store, "create_skill", unexpected_write)
        if endpoint == "create":
            response = client.post(prefix, json={"name": "private", "organization": "acme"})
        elif endpoint == "bulk-register":
            response = client.post(
                f"{prefix}/bulk-register",
                json={
                    "organization": "acme",
                    "skills": [
                        {
                            "name": name,
                            "source": "https://example.com/repo.git",
                            "ref": "main",
                            "digest": "a" * 64,
                        }
                        for name in ("allowed", "private")
                    ],
                },
            )
        elif endpoint == "multipart":
            response = client.post(
                f"{prefix}/@acme/private/versions",
                files={
                    "metadata": (None, json.dumps({"name": "private", "organization": "acme"})),
                    "content": ("skill.tar.gz", b"denied upload", "application/gzip"),
                },
            )
        else:
            url = f"{prefix}/@acme/private/versions"
            response = client.post(
                url,
                json={
                    "name": "private",
                    "organization": "acme",
                    "source": "https://example.com/skill.zip",
                },
            )
    assert response.status_code == 403, response.text
    assert list(tracking_store.search_skills()) == []
    if pattern != "*":
        # The same name in another organization is not covered by this DENY.
        response = client.post(prefix, json={"name": "private", "organization": "other"})
        assert response.status_code == 200, response.text


def test_search_selector_intersects_current_authorization(workspace_registry):
    client, auth_store, tracking_store, creator = workspace_registry
    prefix = "/api/3.0/mlflow/skills"
    for organization in ("acme", "example", "other"):
        tracking_store.create_skill("reviewer", organization=organization)
    auth_store.grant_user_permission(creator.username, "skill", "@acme/reviewer", READ.name)
    auth_store.grant_user_permission(creator.username, "skill", "@example/reviewer", READ.name)
    selector = ["@acme/reviewer", "@example/reviewer", "@other/reviewer"]
    query = {"include_skill_identities": selector, "max_results": 1}
    first = client.get(prefix, params=query)
    assert first.status_code == 200, first.text
    assert [s["organization"] for s in first.json()["skills"]] == ["acme"]
    token = first.json()["next_page_token"]
    assert token is not None

    # New grants between pages do not invalidate the token.
    auth_store.grant_user_permission(creator.username, "skill", "@other/reviewer", READ.name)
    second = client.get(prefix, params={**query, "page_token": token})
    assert second.status_code == 200, second.text
    assert [s["organization"] for s in second.json()["skills"]] == ["example"]
    assert second.json()["next_page_token"] is not None

    # Revoking the next result is enforced with the same token and finite grants.
    auth_store.grant_user_permission(creator.username, "skill", "@example/reviewer", DENY.name)
    revoked = client.get(prefix, params={**query, "page_token": token})
    assert revoked.status_code == 200, revoked.text
    assert [s["organization"] for s in revoked.json()["skills"]] == ["other"]
    auth_store.grant_user_permission(creator.username, "skill", "@example/reviewer", READ.name)

    # DENY remains effective even when the caller explicitly selects the Skill.
    auth_store.grant_user_permission(creator.username, "skill", "*", READ.name)
    auth_store.grant_user_permission(creator.username, "skill", "@acme/reviewer", DENY.name)
    excluded = client.get(prefix, params={**query, "page_token": token})
    assert excluded.status_code == 200, excluded.text
    assert [s["organization"] for s in excluded.json()["skills"]] == ["other"]
    assert excluded.json()["next_page_token"] is None
    denied_only = client.get(
        prefix,
        params={
            "include_skill_identities": ["@acme/reviewer"],
            "max_results": 1,
        },
    )
    assert denied_only.json() == {"skills": [], "next_page_token": None}
    assert client.get(prefix, params={"include_skill_identities": ""}).json() == {
        "skills": [],
        "next_page_token": None,
    }


@pytest.mark.parametrize("prefix", ["/api/3.0/mlflow/skills", "/ajax-api/3.0/mlflow/skills"])
def test_skill_rest_enforces_grants_and_filters_before_pagination(
    tmp_path, monkeypatch, prefix, db_uri
):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "false")
    auth_store = AuthStore()
    auth_store.init_db(f"sqlite:///{tmp_path / 'auth.db'}")
    tracking_store = TrackingStore(db_uri, str(tmp_path / "artifacts"))
    owner = auth_store.create_user("owner", "strong-password", is_admin=False)
    reader = auth_store.create_user("reader", "strong-password", is_admin=False)
    admin = auth_store.create_user("admin2", "strong-password", is_admin=True)
    monkeypatch.setattr(auth, "store", auth_store)
    monkeypatch.setattr(auth, "_get_tracking_store", lambda: tracking_store)
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda: tracking_store)
    monkeypatch.setattr(
        auth, "auth_config", auth.auth_config._replace(default_permission=NO_PERMISSIONS.name)
    )
    monkeypatch.setattr(auth, "_auth_initialized", True)
    monkeypatch.setattr(
        auth,
        "_authenticate_fastapi_request",
        lambda request: auth_store.get_user(request.headers["x-user"]),
    )

    app = FastAPI()
    app.include_router(skill_registry_router, prefix=prefix)
    add_registry_exception_handlers(app)
    auth.add_fastapi_permission_middleware(app)
    client = TestClient(app)
    try:
        for organization in ("", "acme", "other"):
            response = client.post(
                prefix,
                json={"name": "reviewer", "organization": organization},
                headers={"x-user": owner.username},
            )
            assert response.status_code == 200, response.text

        # Authorization uses the path identity; the handler receives the untouched upload.
        captured_uploads = []

        def capture_registration(registration, **kwargs):
            captured_uploads.append(kwargs["content"].read())
            return SkillVersion(
                name=registration.name,
                organization=registration.organization,
                version=1,
                status=SkillStatus.ACTIVE,
            )

        upload = {
            "metadata": (
                "metadata.json",
                json.dumps({"name": "reviewer", "organization": "acme"}),
                "application/json",
            ),
            "content": ("skill.tar.gz", io.BytesIO(b"skill archive"), "application/gzip"),
        }
        with monkeypatch.context() as patch:
            patch.setattr(skill_registry_api, "register_skill_version", capture_registration)
            upload_response = client.post(
                f"{prefix}/@acme/reviewer/versions",
                files=upload,
                headers={"x-user": owner.username},
            )
        assert upload_response.status_code == 200, upload_response.text
        assert captured_uploads == [b"skill archive"]

        for name in ("register", "bulk-register"):
            special = f"{prefix}/{name}"
            owner_headers = {"x-user": owner.username}
            assert (
                client.post(prefix, json={"name": name}, headers=owner_headers).status_code == 200
            )
            assert client.get(special, headers=owner_headers).status_code == 200
            assert (
                client.patch(
                    special, json={"description": "ordinary skill"}, headers=owner_headers
                ).status_code
                == 200
            )
            assert client.delete(special, headers=owner_headers).status_code == 200

        assert (
            auth_store.get_role_permission_for_resource(
                owner.id, "skill", "@acme/reviewer", "default"
            ).name
            == MANAGE.name
        )
        target = f"{prefix}/@acme/reviewer"
        headers = {"x-user": reader.username}
        with monkeypatch.context() as patch:
            patch.setattr(skill_registry_api, "register_skill_version", capture_registration)
            denied_upload = client.post(
                f"{target}/versions",
                files={
                    "metadata": (
                        "metadata.json",
                        json.dumps({"name": "reviewer", "organization": "acme"}),
                        "application/json",
                    ),
                    "content": ("skill.tar.gz", io.BytesIO(b"denied"), "application/gzip"),
                },
                headers=headers,
            )
        assert denied_upload.status_code == 403
        assert captured_uploads == [b"skill archive"]
        assert client.get(target, headers=headers).status_code == 403
        assert client.post(f"{target}/versions", json={}, headers=headers).status_code == 403
        assert (
            client.post(
                f"{target}/versions",
                json={
                    "name": "reviewer",
                    "organization": "acme",
                    "source": "https://example.com/skill.zip",
                },
                headers=headers,
            ).status_code
            == 403
        )

        # Another creator can win after the route's preflight lookup. Recheck the
        # existing parent inside the tracking transaction before adding a version.
        register = skill_registry_api.register_skill_version

        def create_other_parent_first(registration, **kwargs):
            tracking_store.create_skill("race", created_by=owner.username)
            return register(registration, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(skill_registry_api, "register_skill_version", create_other_parent_first)
            raced = client.post(
                f"{prefix}/race/versions",
                json={"name": "race", "source": "https://example.com/skill.zip"},
                headers=headers,
            )
        assert raced.status_code == 409, raced.text
        assert list(tracking_store.search_skill_versions(name="race")) == []

        auth_store.grant_user_permission(reader.username, "skill", "@acme/reviewer", READ.name)
        response = client.get(target, headers=headers)
        assert response.status_code == 200
        assert response.json()["allowed_actions"] == []
        assert client.patch(target, json={"description": "x"}, headers=headers).status_code == 403
        page = client.get(prefix, params={"max_results": 1}, headers=headers)
        assert page.status_code == 200
        assert [(s["organization"], s["name"]) for s in page.json()["skills"]] == [
            ("acme", "reviewer")
        ]
        assert page.json()["next_page_token"] is None
        assert page.json()["skills"][0]["allowed_actions"] == []
        admin_page = client.get(prefix, headers={"x-user": admin.username})
        assert len(admin_page.json()["skills"]) == 4

        auth_store.grant_user_permission(reader.username, "skill", "@acme/reviewer", EDIT.name)
        auth_store.grant_user_permission(reader.username, "skill", "reviewer", EDIT.name)
        response = client.patch(target, json={"description": "x"}, headers=headers)
        assert response.status_code == 200
        assert response.json()["allowed_actions"] == ["USE", "UPDATE"]
        assert client.delete(target, headers=headers).status_code == 403

        # A status PATCH can soft-delete a version and remove its aliases. That
        # transition requires MANAGE even though ordinary status updates use EDIT.
        tracking_store.create_skill_version(
            name="reviewer",
            organization="acme",
            source_type="zip",
            source="https://example.com/skill.zip",
            status="draft",
        )
        tracking_store.set_skill_alias("reviewer", "stable", 1, organization="acme")
        version_path = f"{target}/versions/1"
        for tag_target in (target, f"{prefix}/reviewer"):
            assert (
                client.post(
                    f"{tag_target}/tags",
                    json={"key": "team", "value": "review"},
                    headers=headers,
                ).status_code
                == 200
            )
            assert client.delete(f"{tag_target}/tags/team", headers=headers).status_code == 200
        assert (
            client.post(
                f"{version_path}/tags", json={"key": "team", "value": "review"}, headers=headers
            ).status_code
            == 200
        )
        assert client.delete(f"{version_path}/tags/team", headers=headers).status_code == 403
        assert (
            client.delete(
                f"{version_path}/tags/team", headers={"x-user": owner.username}
            ).status_code
            == 200
        )
        assert client.delete(version_path, headers=headers).status_code == 403
        assert (
            client.patch(version_path, json={"status": "deleted"}, headers=headers).status_code
            == 403
        )
        version = tracking_store.get_skill_version("reviewer", 1, organization="acme")
        assert version.status == "draft"
        assert "stable" in version.aliases
        assert (
            client.patch(version_path, json={"status": "active"}, headers=headers).status_code
            == 200
        )
        assert (
            client.patch(version_path, json={"status": "draft"}, headers=headers).status_code == 200
        )
        assert (
            client.patch(
                version_path,
                json={"status": "deleted"},
                headers={"x-user": owner.username},
            ).status_code
            == 200
        )
        with pytest.raises(MlflowException, match="not found"):
            tracking_store.get_skill_version_by_alias("reviewer", "stable", organization="acme")

        tracking_store.create_skill_version(
            name="reviewer",
            organization="acme",
            source_type="zip",
            source="https://example.com/second.zip",
            status="draft",
        )
        assert (
            client.patch(
                f"{target}/versions/2",
                json={"status": "deleted"},
                headers={"x-user": admin.username},
            ).status_code
            == 200
        )

        # If the owner deletes an existing parent after registration preflight,
        # the transaction must not recreate it using the stale EDIT decision.
        for endpoint in ("versions", "bulk-register"):
            race_name = f"raced-{endpoint}"
            assert (
                client.post(
                    prefix,
                    json={"name": race_name},
                    headers={"x-user": owner.username},
                ).status_code
                == 200
            )
            auth_store.grant_user_permission(reader.username, "skill", race_name, EDIT.name)
            target_function = (
                "bulk_register_skill_versions"
                if endpoint == "bulk-register"
                else "register_skill_version"
            )
            original = getattr(skill_registry_api, target_function)

            def delete_after_preflight(*args, **kwargs):
                response = client.delete(
                    f"{prefix}/{race_name}", headers={"x-user": owner.username}
                )
                assert response.status_code == 200, response.text
                return original(*args, **kwargs)

            if endpoint == "bulk-register":
                url = f"{prefix}/bulk-register"
                body = {
                    "skills": [
                        {
                            "name": race_name,
                            "source": "https://example.com/repo.git",
                            "ref": "main",
                            "digest": "a" * 64,
                        }
                    ]
                }
            else:
                url = f"{prefix}/{race_name}/versions"
                body = {"source": "https://example.com/skill.zip"}
            with monkeypatch.context() as patch:
                patch.setattr(skill_registry_api, target_function, delete_after_preflight)
                response = client.post(url, json=body, headers=headers)
            assert response.status_code in (403, 409), response.text
            assert race_name not in [skill.name for skill in tracking_store.search_skills()]

        auth_store.grant_user_permission(reader.username, "skill", "@acme/reviewer", MANAGE.name)
        assert client.delete(target, headers=headers).status_code == 200
        assert (
            auth_store.get_role_permission_for_resource(
                reader.id, "skill", "@acme/reviewer", "default"
            )
            is None
        )
        assert (
            client.post(
                prefix, json={"name": "admin-cleanup"}, headers={"x-user": owner.username}
            ).status_code
            == 200
        )
        auth_store.grant_user_permission(reader.username, "skill", "admin-cleanup", READ.name)
        assert (
            client.delete(f"{prefix}/admin-cleanup", headers={"x-user": admin.username}).status_code
            == 200
        )
        assert (
            auth_store.get_role_permission_for_resource(
                reader.id, "skill", "admin-cleanup", "default"
            )
            is None
        )
    finally:
        auth_store.engine.dispose()
        tracking_store.engine.dispose()


def _identity(name, organization):
    return f"@{organization}/{name}" if organization else name


def _skill_url(name, organization, prefix=SKILLS):
    return f"{prefix}/{_identity(name, organization)}"


def _bulk_definition(name):
    return {
        "name": name,
        "source": "https://example.com/repo.git",
        "ref": "main",
        "digest": "a" * 64,
    }


def _create(client, endpoint, name, *, user):
    headers = {"x-user": user}
    if endpoint == "create":
        return client.post(SKILLS, json={"name": name}, headers=headers)
    if endpoint == "versions":
        return client.post(f"{SKILLS}/{name}/versions", json={"source": SOURCE}, headers=headers)
    return client.post(
        f"{SKILLS}/bulk-register", json={"skills": [_bulk_definition(name)]}, headers=headers
    )


def _seed_skill(tracking_store, name, organization):
    tracking_store.create_skill(
        name, organization=organization, description="original", icons=ICONS
    )
    tracking_store.create_skill_version(
        name=name, organization=organization, source_type="zip", source=SOURCE
    )
    tracking_store.set_skill_tag(name, "team", "original", organization=organization)
    tracking_store.set_skill_version_tag(name, 1, "team", "original", organization=organization)
    tracking_store.set_skill_alias(name, "stable", 1, organization=organization)


def _snapshot(tracking_store, name, organization):
    return (
        tracking_store.get_skill(name, organization=organization),
        list(tracking_store.search_skill_versions(name, organization=organization)),
    )


@pytest.mark.parametrize("endpoint", ["create", "versions", "bulk-register"])
@pytest.mark.parametrize("creator_kind", ["workspace-use", "workspace-manage", "admin"])
def test_registration_creator_grants_with_workspaces(workspace_registry, endpoint, creator_kind):
    client, auth_store, tracking_store, creator = workspace_registry
    if creator_kind == "workspace-manage":
        auth_store.set_workspace_permission("default", creator.username, MANAGE.name)
    is_admin = creator_kind == "admin"
    if is_admin:
        creator = auth_store.create_user("admin", "strong-password", is_admin=True)
    assert auth.validate_can_create_skill(creator.username) is (not is_admin)
    response = _create(client, endpoint, "private", user=creator.username)
    assert response.status_code == 200, response.text
    assert tracking_store.get_skill("private").created_by == creator.username
    permission = auth_store.get_role_permission_for_resource(
        creator.id, "skill", "private", "default"
    )
    assert (permission.name if permission else None) == (MANAGE.name if not is_admin else None)


@pytest.mark.parametrize("permission", [READ, EDIT, MANAGE], ids=lambda permission: permission.name)
def test_skill_grants_without_workspace_use_cannot_create_skills(workspace_registry, permission):
    client, auth_store, tracking_store, _ = workspace_registry
    user = auth_store.create_user("grantee", "strong-password")
    auth_store.grant_user_permission(user.username, "skill", "*", permission.name)
    for endpoint in ("create", "versions", "bulk-register"):
        response = _create(client, endpoint, "private", user=user.username)
        assert response.status_code == 403, response.text
    assert list(tracking_store.search_skills()) == []


@pytest.mark.parametrize("organization", ["", "acme"])
def test_mixed_bulk_batch_is_denied_as_a_whole(workspace_registry, organization):
    client, auth_store, tracking_store, _ = workspace_registry
    _seed_skill(tracking_store, "existing", organization)
    user = auth_store.create_user("bulk-writer", "strong-password")
    auth_store.set_workspace_permission("default", user.username, USE.name)
    existing = _identity("existing", organization)
    auth_store.grant_user_permission(user.username, "skill", existing, READ.name)
    headers = {"x-user": user.username}
    batch = {
        "organization": organization,
        "skills": [_bulk_definition("new"), _bulk_definition("existing")],
    }
    before = _snapshot(tracking_store, "existing", organization)

    # USE lets the caller create `new`, but READ on `existing` cannot add a version to it.
    response = client.post(f"{SKILLS}/bulk-register", json=batch, headers=headers)
    assert response.status_code == 403, response.text
    assert _snapshot(tracking_store, "existing", organization) == before
    assert [skill.name for skill in tracking_store.search_skills()] == ["existing"]

    auth_store.grant_user_permission(user.username, "skill", existing, EDIT.name)
    response = client.post(f"{SKILLS}/bulk-register", json=batch, headers=headers)
    assert response.status_code == 200, response.text
    versions = response.json()["skill_versions"]
    assert [(version["name"], version["version"]) for version in versions] == [
        ("new", 1),
        ("existing", 2),
    ]
    # Only the parent this batch created grants its creator MANAGE; `existing` stays at EDIT.
    assert client.delete(_skill_url("new", organization), headers=headers).status_code == 200
    assert client.delete(_skill_url("existing", organization), headers=headers).status_code == 403


def test_edit_on_an_existing_skill_lets_bulk_register_add_a_version(workspace_registry):
    client, auth_store, tracking_store, _ = workspace_registry
    _seed_skill(tracking_store, "existing", "")
    batch = {"skills": [_bulk_definition("existing")]}
    before = _snapshot(tracking_store, "existing", "")

    # Neither user holds workspace USE: adding a version needs EDIT on the Skill, not the right
    # to create Skills.
    response = client.post(
        f"{SKILLS}/bulk-register", json=batch, headers=_user_holding(auth_store, READ)
    )
    assert response.status_code == 403, response.text
    assert _snapshot(tracking_store, "existing", "") == before

    headers = _user_holding(auth_store, EDIT)
    response = client.post(f"{SKILLS}/bulk-register", json=batch, headers=headers)
    assert response.status_code == 200, response.text
    assert [version["version"] for version in response.json()["skill_versions"]] == [2]
    assert tracking_store.get_skill_version("existing", 2).created_by == headers["x-user"]


@dataclass(frozen=True)
class SkillRoute:
    operation: str
    method: str
    suffix: str
    required: Permission
    body: dict[str, Any] | None = None
    # An active version cannot be deleted by anyone, so these routes start from a deprecated one.
    deprecate_first: bool = False


# The permission each route needs, written out here rather than read from the middleware's own
# table. Deleting a parent tag needs EDIT but deleting a version tag needs MANAGE
# (https://github.com/mlflow/mlflow/pull/25988, https://github.com/mlflow/rfcs/pull/65), and a
# status PATCH to `deleted` needs MANAGE although other status changes need EDIT.
SKILL_ROUTES = [
    SkillRoute("get_skill", "GET", "", READ),
    SkillRoute("get_skill_version", "GET", "/versions/1", READ),
    SkillRoute("get_skill_version_by_alias", "GET", "/aliases/stable", READ),
    SkillRoute("get_latest_skill_version", "GET", "/aliases/latest", READ),
    SkillRoute("search_skill_versions", "GET", "/versions", READ),
    SkillRoute(
        "update_skill",
        "PATCH",
        "",
        EDIT,
        {"description": "updated", "icons": [{"src": "https://8.8.8.8/updated.png"}]},
    ),
    SkillRoute("create_skill_version", "POST", "/versions", EDIT, {"source": SOURCE}),
    SkillRoute("update_skill_version", "PATCH", "/versions/1", EDIT, {"status": "deprecated"}),
    SkillRoute("set_skill_tag", "POST", "/tags", EDIT, {"key": "team", "value": "updated"}),
    SkillRoute("delete_skill_tag", "DELETE", "/tags/team", EDIT),
    SkillRoute(
        "set_skill_version_tag", "POST", "/versions/1/tags", EDIT, {"key": "team", "value": "new"}
    ),
    SkillRoute("delete_skill_version_tag", "DELETE", "/versions/1/tags/team", MANAGE),
    SkillRoute("set_skill_alias", "POST", "/aliases", EDIT, {"alias": "candidate", "version": 1}),
    SkillRoute("delete_skill_alias", "DELETE", "/aliases/stable", MANAGE),
    SkillRoute("delete_skill_version", "DELETE", "/versions/1", MANAGE, deprecate_first=True),
    SkillRoute(
        "update_skill_version_to_deleted",
        "PATCH",
        "/versions/1",
        MANAGE,
        {"status": "deleted"},
        deprecate_first=True,
    ),
    SkillRoute("delete_skill", "DELETE", "", MANAGE),
]
_TIERS = [None, READ, EDIT, MANAGE]


def _user_holding(auth_store, permission):
    label = permission.name.lower() if permission else "no-grant"
    user = auth_store.create_user(f"{label}-user", "strong-password")
    if permission:
        auth_store.grant_user_permission(user.username, "skill", "*", permission.name)
    return {"x-user": user.username}


# Each route is tried with the two tiers on either side of what it needs: the highest that must
# be refused and the lowest that must be let through. Tiers are cumulative (MANAGE includes EDIT
# includes READ; test_auth_workspace.py::test_role_grant_on_skill_registry_parent_gates_capabilities
# checks that), so tiers further out add nothing.
@pytest.mark.parametrize(
    "organization", ["", "acme"], ids=["default-organization", "named-organization"]
)
@pytest.mark.parametrize("route", SKILL_ROUTES, ids=lambda route: route.operation)
def test_each_skill_route_requires_its_permission_tier(workspace_registry, route, organization):
    client, auth_store, tracking_store, _ = workspace_registry
    _seed_skill(tracking_store, "reviewer", organization)
    if route.deprecate_first:
        tracking_store.update_skill_version(
            "reviewer", 1, organization=organization, status="deprecated"
        )
    url = f"{_skill_url('reviewer', organization)}{route.suffix}"
    refused = _TIERS[_TIERS.index(route.required) - 1]
    before = _snapshot(tracking_store, "reviewer", organization)

    headers = _user_holding(auth_store, refused)
    response = client.request(route.method, url, json=route.body, headers=headers)
    assert response.status_code == 403, response.text
    assert _snapshot(tracking_store, "reviewer", organization) == before

    headers = _user_holding(auth_store, route.required)
    response = client.request(route.method, url, json=route.body, headers=headers)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "organization", ["", "acme"], ids=["default-organization", "named-organization"]
)
def test_workspace_use_alone_grants_nothing_on_someone_elses_skill(
    workspace_registry, organization
):
    client, _, tracking_store, _ = workspace_registry
    _seed_skill(tracking_store, "reviewer", organization)
    before = _snapshot(tracking_store, "reviewer", organization)
    # Requests without `x-user` act as `creator`, which holds workspace USE and nothing else.
    for prefix in get_skill_registry_api_route_prefixes():
        for route in SKILL_ROUTES:
            url = f"{_skill_url('reviewer', organization, prefix)}{route.suffix}"
            response = client.request(route.method, url, json=route.body)
            assert response.status_code == 403, (route.operation, response.text)
        identity = _identity("reviewer", organization)
        response = client.get(prefix, params={"include_skill_identities": identity})
        assert response.json()["skills"] == []
    assert _snapshot(tracking_store, "reviewer", organization) == before

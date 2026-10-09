import asyncio
import io
import json
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.requests import Request
from starlette.testclient import TestClient

from mlflow.entities.skill import SkillStatus
from mlflow.entities.skill_version import SkillVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import TEMPORARILY_UNAVAILABLE
from mlflow.server import auth, handlers, skill_registry_api
from mlflow.server.auth.permissions import DENY, EDIT, MANAGE, NO_PERMISSIONS, READ, USE
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore as AuthStore
from mlflow.server.fastapi_app import add_registry_exception_handlers
from mlflow.server.skill_registry_api import skill_registry_router
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore as TrackingStore
from mlflow.utils.validation import _MAX_BULK_REGISTER_SKILLS
from mlflow.utils.workspace_context import ServerWorkspaceContext


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
def workspace_registry(tmp_path, monkeypatch, db_uri, request):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "true")
    monkeypatch.setenv("MLFLOW_WORKSPACE", "default")
    auth_store = AuthStore()
    auth_store.init_db(
        db_uri if getattr(request, "param", False) else f"sqlite:///{tmp_path / 'auth.db'}"
    )
    tracking_store = TrackingStore(db_uri, str(tmp_path / "artifacts"))
    creator = auth_store.create_user("creator", "strong-password", is_admin=False)
    auth_store.set_workspace_permission("default", creator.username, USE.name)
    monkeypatch.setattr(auth, "store", auth_store)
    monkeypatch.setattr(auth, "_get_tracking_store", lambda: tracking_store)
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda: tracking_store)
    monkeypatch.setattr(
        auth, "auth_config", auth.auth_config._replace(default_permission=NO_PERMISSIONS.name)
    )
    monkeypatch.setattr(auth, "_auth_initialized", True)
    monkeypatch.setattr(auth, "_authenticate_fastapi_request", lambda request: creator)
    app = FastAPI()
    app.include_router(skill_registry_router, prefix="/api/3.0/mlflow/skills")
    add_registry_exception_handlers(app)
    auth.add_fastapi_permission_middleware(app)
    try:
        with ServerWorkspaceContext("default"), TestClient(app) as client:
            yield client, auth_store, tracking_store, creator
    finally:
        auth_store.engine.dispose()
        tracking_store.engine.dispose()


@pytest.mark.parametrize("workspace_registry", [False, True], indirect=True)
@pytest.mark.parametrize("admin_delete", [False, True])
def test_permission_cleanup_failure_rolls_back_delete_and_can_be_retried(
    workspace_registry, monkeypatch, admin_delete
):
    client, auth_store, tracking_store, creator = workspace_registry
    prefix = "/api/3.0/mlflow/skills"
    reader = auth_store.create_user("reader", "strong-password")
    auth_store.set_workspace_permission("default", reader.username, USE.name)
    assert client.post(prefix, json={"name": "private"}).status_code == 200
    auth_store.grant_user_permission(reader.username, "skill", "private", READ.name)
    if admin_delete:
        admin = auth_store.create_user("admin", "strong-password", is_admin=True)
        monkeypatch.setattr(auth, "_authenticate_fastapi_request", lambda request: admin)

    with mock.patch.object(
        auth_store,
        "delete_grants_for_resource_in_session",
        side_effect=MlflowException("Auth store unavailable", TEMPORARILY_UNAVAILABLE),
    ) as cleanup:
        response = client.delete(f"{prefix}/private")
    cleanup.assert_called_once()
    assert response.status_code == 503, response.text
    assert tracking_store.get_skill("private").created_by == creator.username
    assert (
        auth_store.get_role_permission_for_resource(reader.id, "skill", "private", "default")
        == READ
    )

    # Retrying DELETE still addresses the original parent and clears grants before reuse.
    response = client.delete(f"{prefix}/private")
    assert response.status_code == 200, response.text
    assert list(tracking_store.search_skills()) == []
    assert (
        auth_store.get_role_permission_for_resource(reader.id, "skill", "private", "default")
        is None
    )
    monkeypatch.setattr(auth, "_authenticate_fastapi_request", lambda request: creator)
    assert client.post(prefix, json={"name": "private"}).status_code == 200
    monkeypatch.setattr(auth, "_authenticate_fastapi_request", lambda request: reader)
    assert client.get(f"{prefix}/private").status_code == 403


@pytest.mark.parametrize("workspace_registry", [False, True], indirect=True)
def test_tracking_rollback_after_permission_cleanup(workspace_registry, monkeypatch, request):
    client, auth_store, tracking_store, _ = workspace_registry
    prefix = "/api/3.0/mlflow/skills"
    reader = auth_store.create_user("reader", "strong-password")
    assert client.post(prefix, json={"name": "private"}).status_code == 200
    auth_store.grant_user_permission(reader.username, "skill", "private", READ.name)
    cleanup = auth._skill_delete_before_commit

    def fail_after_cleanup(organization, name, session):
        cleanup(organization, name, session)
        raise MlflowException("Tracking transaction failed", TEMPORARILY_UNAVAILABLE)

    monkeypatch.setattr(auth, "_skill_delete_before_commit", fail_after_cleanup)
    response = client.delete(f"{prefix}/private")
    assert response.status_code == 503, response.text
    assert tracking_store.get_skill("private").name == "private"
    # A shared database rolls back both changes; a separate auth database retains revocation.
    expected_permission = READ if request.node.callspec.params["workspace_registry"] else None
    assert (
        auth_store.get_role_permission_for_resource(reader.id, "skill", "private", "default")
        == expected_permission
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
        assert client.get(target, headers=headers).status_code == 200
        assert client.patch(target, json={"description": "x"}, headers=headers).status_code == 403
        page = client.get(prefix, params={"max_results": 1}, headers=headers)
        assert page.status_code == 200
        assert [(s["organization"], s["name"]) for s in page.json()["skills"]] == [
            ("acme", "reviewer")
        ]
        assert page.json()["next_page_token"] is None
        admin_page = client.get(prefix, headers={"x-user": admin.username})
        assert len(admin_page.json()["skills"]) == 4

        auth_store.grant_user_permission(reader.username, "skill", "@acme/reviewer", EDIT.name)
        assert client.patch(target, json={"description": "x"}, headers=headers).status_code == 200
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
        assert (
            client.post(
                f"{target}/tags", json={"key": "team", "value": "review"}, headers=headers
            ).status_code
            == 200
        )
        assert client.delete(f"{target}/tags/team", headers=headers).status_code == 200
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


@pytest.mark.parametrize("endpoint", ["versions", "bulk-register"])
@pytest.mark.parametrize("is_admin", [False, True])
def test_registration_creator_grants_with_workspaces(
    tmp_path, monkeypatch, endpoint, is_admin, db_uri
):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "true")
    monkeypatch.setenv("MLFLOW_WORKSPACE", "default")
    auth_store = AuthStore()
    auth_store.init_db(f"sqlite:///{tmp_path / 'auth.db'}")
    tracking_store = TrackingStore(db_uri, str(tmp_path / "artifacts"))
    creator = auth_store.create_user("creator", "strong-password", is_admin=is_admin)
    if not is_admin:
        auth_store.set_workspace_permission("default", creator.username, MANAGE.name)
    monkeypatch.setattr(auth, "store", auth_store)
    monkeypatch.setattr(auth, "_get_tracking_store", lambda: tracking_store)
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda: tracking_store)
    monkeypatch.setattr(
        auth,
        "auth_config",
        auth.auth_config._replace(
            default_permission=NO_PERMISSIONS.name, grant_default_workspace_access=False
        ),
    )
    monkeypatch.setattr(auth, "_auth_initialized", True)
    monkeypatch.setattr(
        auth,
        "_authenticate_fastapi_request",
        lambda request: auth_store.get_user(request.headers["x-user"]),
    )
    app = FastAPI()
    app.include_router(skill_registry_router, prefix="/api/3.0/mlflow/skills")
    add_registry_exception_handlers(app)
    auth.add_fastapi_permission_middleware(app)
    client = TestClient(app)
    prefix = "/api/3.0/mlflow/skills"
    headers = {"x-user": creator.username}
    try:
        with ServerWorkspaceContext("default"):
            assert auth.validate_can_create_skill(creator.username) is (not is_admin)
            if endpoint == "bulk-register":
                response = client.post(
                    f"{prefix}/bulk-register",
                    json={
                        "skills": [
                            {
                                "name": "private",
                                "source": "https://example.com/repo.git",
                                "ref": "main",
                                "digest": "a" * 64,
                            }
                        ]
                    },
                    headers=headers,
                )
            else:
                url = f"{prefix}/private/versions"
                body = {"source": "https://example.com/skill.zip"}
                response = client.post(url, json=body, headers=headers)
            assert response.status_code == 200, response.text
            assert tracking_store.get_skill("private").created_by == creator.username
            permission = auth_store.get_role_permission_for_resource(
                creator.id, "skill", "private", "default"
            )
            assert (permission.name if permission else None) == (
                MANAGE.name if not is_admin else None
            )
    finally:
        auth_store.engine.dispose()
        tracking_store.engine.dispose()

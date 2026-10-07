import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from mlflow.exceptions import MlflowException
from mlflow.server import auth, handlers, skill_registry_api
from mlflow.server.auth.permissions import EDIT, MANAGE, NO_PERMISSIONS, READ
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore as AuthStore
from mlflow.server.fastapi_app import add_registry_exception_handlers
from mlflow.server.skill_registry_api import skill_registry_router
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore as TrackingStore


def test_skill_rest_enforces_grants_and_filters_before_pagination(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "false")
    auth_store = AuthStore()
    auth_store.init_db(f"sqlite:///{tmp_path / 'auth.db'}")
    tracking_store = TrackingStore(
        f"sqlite:///{tmp_path / 'tracking.db'}", str(tmp_path / "artifacts")
    )
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
    app.include_router(skill_registry_router, prefix="/api/3.0/mlflow/skills")
    add_registry_exception_handlers(app)
    auth.add_fastapi_permission_middleware(app)
    client = TestClient(app)
    prefix = "/api/3.0/mlflow/skills"
    try:
        for organization in ("", "acme", "other"):
            response = client.post(
                prefix,
                json={"name": "reviewer", "organization": organization},
                headers={"x-user": owner.username},
            )
            assert response.status_code == 200, response.text

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
        assert client.get(target, headers=headers).status_code == 403
        assert client.post(f"{target}/versions", json={}, headers=headers).status_code == 403
        assert (
            client.post(
                f"{prefix}/register",
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
                f"{prefix}/register",
                json={"name": "race", "source": "https://example.com/skill.zip"},
                headers=headers,
            )
        assert raced.status_code == 403, raced.text
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

        # If the owner deletes an existing parent after registration preflight,
        # the transaction must not recreate it using the stale EDIT decision.
        for endpoint in ("register", "versions", "bulk-register"):
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
                url = (
                    f"{prefix}/{race_name}/versions"
                    if endpoint == "versions"
                    else f"{prefix}/register"
                )
                body = {"source": "https://example.com/skill.zip"}
                if endpoint == "register":
                    body["name"] = race_name
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
    finally:
        auth_store.engine.dispose()
        tracking_store.engine.dispose()

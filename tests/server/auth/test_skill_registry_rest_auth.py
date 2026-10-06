from fastapi import FastAPI
from starlette.testclient import TestClient

from mlflow.server import auth, handlers
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
        assert client.post(
            f"{prefix}/register",
            json={
                "name": "reviewer",
                "organization": "acme",
                "source": "https://example.com/skill.zip",
            },
            headers=headers,
        ).status_code == 403

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
        assert len(admin_page.json()["skills"]) == 3

        auth_store.grant_user_permission(
            reader.username, "skill", "@acme/reviewer", EDIT.name
        )
        assert client.patch(target, json={"description": "x"}, headers=headers).status_code == 200
        assert client.delete(target, headers=headers).status_code == 403

        auth_store.grant_user_permission(
            reader.username, "skill", "@acme/reviewer", MANAGE.name
        )
        assert client.delete(target, headers=headers).status_code == 200
        assert auth_store.get_role_permission_for_resource(
            reader.id, "skill", "@acme/reviewer", "default"
        ) is None
    finally:
        auth_store.engine.dispose()
        tracking_store.engine.dispose()

import base64
import json
import socket
import threading
import time
from unittest.mock import patch

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from itsdangerous import TimestampSigner
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse

import mlflow.assistant.config as config_module
from mlflow.assistant.config import AssistantConfig, ProviderConfig, set_config_user
from mlflow.assistant.providers.base import clear_config_cache
from mlflow.entities import GatewayEndpointModelConfig, GatewayModelLinkageType
from mlflow.entities.workspace import Workspace
from mlflow.server.assistant.api import assistant_router
from mlflow.server.gateway_api import gateway_router
from mlflow.store.workspace.sqlalchemy_store import SqlAlchemyStore as WorkspaceStore
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.utils.workspace_context import ServerWorkspaceContext, get_request_workspace
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME

SESSION_SECRET = "assistant-gateway-test-session-secret"


def _credentials(username, auth_method):
    match auth_method:
        case "session":
            payload = base64.b64encode(json.dumps({"username": username}).encode())
            cookie = TimestampSigner(SESSION_SECRET).sign(payload).decode()
            return {"Cookie": f"session={cookie}"}
        case "basic":
            token = base64.b64encode(f"{username}:{username}-token".encode()).decode()
            return {"Authorization": f"Basic {token}"}
        case "bearer":
            return {"Authorization": f"Bearer {username}-token"}


@pytest.fixture
def gateway_auth_server(monkeypatch, tmp_path, db_uri):
    """A third-party auth app around the real Assistant and Gateway HTTP routes.

    Signed sessions and tokens model the plugin contract without making MLflow's test suite
    depend on a particular third-party auth package or an external identity provider.
    """
    monkeypatch.delenv("_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("_MLFLOW_SERVER_TLS_LISTENER", raising=False)
    monkeypatch.delenv("_MLFLOW_STATIC_PREFIX", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", db_uri)
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "true")
    monkeypatch.setenv("MLFLOW_WORKSPACE_STORE_URI", db_uri)
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", "gateway-test-passphrase")
    monkeypatch.setattr(config_module, "MLFLOW_ASSISTANT_HOME", tmp_path / "assistant")
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "assistant" / "config.json")
    clear_config_cache()

    workspace_store = WorkspaceStore(db_uri)
    workspace_store.create_workspace(Workspace(name="team-a"))
    workspace_store.create_workspace(Workspace(name="team-b"))
    with ServerWorkspaceContext("team-a"):
        store = _get_store()
        secret = store.create_gateway_secret(
            secret_name="model-key", secret_value={"api_key": "provider-secret"}, provider="openai"
        )
        model = store.create_gateway_model_definition(
            name="model", secret_id=secret.secret_id, provider="openai", model_name="gpt-4o"
        )
        store.create_gateway_endpoint(
            name="chat-endpoint",
            model_configs=[
                GatewayEndpointModelConfig(
                    model_definition_id=model.model_definition_id,
                    linkage_type=GatewayModelLinkageType.PRIMARY,
                    weight=1.0,
                )
            ],
            usage_tracking=False,
        )

    for username in ("alice", "bob"):
        set_config_user(username)
        AssistantConfig(
            providers={"mlflow_gateway": ProviderConfig(selected=True, model="chat-endpoint")}
        ).save(write_projects=False)
    set_config_user(None)

    tokens = {
        _credentials(username, method)["Authorization"]: username
        for username in ("alice", "bob")
        for method in ("basic", "bearer")
    }
    grants = {("alice", "team-a", "chat-endpoint")}
    gateway_requests = []
    app = FastAPI()
    app.include_router(assistant_router)
    app.include_router(gateway_router)

    @app.middleware("http")
    async def authenticate_and_authorize(request: Request, call_next):
        username = tokens.get(request.headers.get("authorization")) or request.session.get(
            "username"
        )
        if username not in {"alice", "bob"}:
            return JSONResponse({"detail": "Authentication required"}, status_code=401)
        request.state.username = username
        workspace = request.headers.get(WORKSPACE_HEADER_NAME)
        with ServerWorkspaceContext(workspace):
            if request.url.path.startswith("/gateway/"):
                body = await request.json()
                allowed = (username, workspace, body["model"]) in grants
                gateway_requests.append({
                    "username": username,
                    "workspace": get_request_workspace(),
                    "allowed": allowed,
                    "authorization": request.headers.get("authorization"),
                    "cookie": request.headers.get("cookie"),
                })
                if not allowed:
                    return JSONResponse({"detail": "Permission denied"}, status_code=403)
            return await call_next(request)

    app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="warning"))
        thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [sock]},
            name="assistant-gateway-auth-test",
            daemon=True,
        )
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started
            with httpx.Client(
                base_url=f"http://127.0.0.1:{sock.getsockname()[1]}", timeout=10, trust_env=False
            ) as client:
                yield client, gateway_requests
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            clear_config_cache()
            assert not thread.is_alive()


@pytest.mark.parametrize("auth_method", ["session", "basic", "bearer"])
@pytest.mark.parametrize(
    ("username", "workspace", "allowed"),
    [("alice", "team-a", True), ("bob", "team-a", False), ("alice", "team-b", False)],
)
def test_gateway_authentication_and_permissions(
    gateway_auth_server, auth_method, username, workspace, allowed
):
    client, gateway_requests = gateway_auth_server
    credentials = _credentials(username, auth_method)
    client.headers.update({**credentials, WORKSPACE_HEADER_NAME: workspace})
    provider_workspaces = []

    async def model_stream():
        provider_workspaces.append(get_request_workspace())
        chunk = {
            "id": "completion",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "MODEL_RESPONSE"},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    with patch(
        "mlflow.gateway.providers.openai.send_stream_request", return_value=model_stream()
    ) as model_request:
        message = client.post("/ajax-api/3.0/mlflow/assistant/message", json={"message": "Hi"})
        assert message.status_code == 200
        response = client.get(message.json()["stream_url"])

    assert response.status_code == 200
    assert gateway_requests == [
        {
            "username": username,
            "workspace": workspace,
            "allowed": allowed,
            "authorization": credentials.get("Authorization"),
            "cookie": credentials.get("Cookie"),
        }
    ]
    if allowed:
        assert "MODEL_RESPONSE" in response.text
        assert "event: error" not in response.text
        model_request.assert_called_once()
        assert model_request.call_args.kwargs["headers"] == {
            "authorization": "Bearer provider-secret"
        }
        assert provider_workspaces == [workspace]
    else:
        assert "error 403" in response.text
        assert "MODEL_RESPONSE" not in response.text
        model_request.assert_not_called()
        assert provider_workspaces == []

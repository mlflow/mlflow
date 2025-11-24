"""Integration tests for FastAPI authorization with the K8s workspace provider.

These tests ensure OTEL and job APIs enforce workspace-aware authentication.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kubernetes_workspace_provider.auth import (
    DEFAULT_REMOTE_GROUPS_HEADER,
    DEFAULT_REMOTE_GROUPS_SEPARATOR,
    DEFAULT_REMOTE_USER_HEADER,
    PATH_AUTHORIZATION_RULES,
    AuthorizationMode,
    KubernetesAuthConfig,
    KubernetesAuthMiddleware,
    KubernetesAuthorizer,
    _compile_authorization_rules,
    _validate_fastapi_route_authorization,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from mlflow.exceptions import MlflowException
from mlflow.tracing.utils.otlp import OTLP_TRACES_PATH
from mlflow.utils import workspace_context
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME


@pytest.fixture(autouse=True)
def _compile_rules(monkeypatch):
    """Ensure authorization rules are populated before each test."""
    # Limit endpoint discovery to avoid unrelated Flask routes during tests
    monkeypatch.setattr("kubernetes_workspace_provider.auth.get_endpoints", lambda _: [])
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth.mlflow_app.url_map.iter_rules",
        lambda: [],
    )
    _compile_authorization_rules()


def test_otel_endpoints_in_auth_rules():
    """Verify OTEL endpoints are registered in PATH_AUTHORIZATION_RULES."""
    # Check that OTEL endpoints are registered
    assert (OTLP_TRACES_PATH, "POST") in PATH_AUTHORIZATION_RULES

    # Verify they have the correct authorization rule
    rule = PATH_AUTHORIZATION_RULES[(OTLP_TRACES_PATH, "POST")]
    assert (rule.verb, rule.resource) == ("update", "experiments")


def test_trace_get_endpoints_in_auth_rules():
    """Ensure REST trace retrieval endpoints are mapped."""
    paths = [
        "/api/3.0/mlflow/traces/get",
        "/ajax-api/3.0/mlflow/traces/get",
    ]

    for path in paths:
        rule = PATH_AUTHORIZATION_RULES[(path, "GET")]
        assert (rule.verb, rule.resource) == ("get", "experiments")


def test_job_api_endpoints_in_auth_rules():
    """Ensure Job API endpoints are explicitly mapped."""
    cases = [
        ("/ajax-api/3.0/jobs", "POST", "create"),
        ("/ajax-api/3.0/jobs/<job_id>", "GET", "get"),
        ("/ajax-api/3.0/jobs/search", "POST", "list"),
    ]

    for path, method, verb in cases:
        rule = PATH_AUTHORIZATION_RULES[(path, method)]
        assert (rule.verb, rule.resource) == (verb, "jobs")


@pytest.fixture
def mock_authorizer():
    """Create a mock KubernetesAuthorizer."""
    authorizer = Mock(spec=KubernetesAuthorizer)
    authorizer.is_allowed.return_value = True  # Default to allowed
    return authorizer


@pytest.fixture
def mock_config():
    """Create a mock KubernetesAuthConfig."""
    config = Mock(spec=KubernetesAuthConfig)
    config.username_claim = "sub"
    config.cache_ttl_seconds = 300.0
    config.authorization_mode = AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW
    config.user_header = DEFAULT_REMOTE_USER_HEADER
    config.groups_header = DEFAULT_REMOTE_GROUPS_HEADER
    config.groups_separator = DEFAULT_REMOTE_GROUPS_SEPARATOR
    return config


@pytest.fixture
def fastapi_app_with_k8s_auth(mock_authorizer, mock_config):
    """Create a FastAPI app with Kubernetes auth middleware."""
    app = FastAPI()

    app.add_middleware(
        KubernetesAuthMiddleware,
        authorizer=mock_authorizer,
        config_values=mock_config,
    )

    class _WorkspaceContextMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            workspace_header = request.headers.get(WORKSPACE_HEADER_NAME)
            if not workspace_header:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": f"Missing {WORKSPACE_HEADER_NAME} header"}},
                )
            if request.url.path == OTLP_TRACES_PATH and not request.headers.get(
                "X-MLflow-Experiment-Id"
            ):
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": "Missing X-MLflow-Experiment-Id header"}},
                )
            token = workspace_context.set_current_workspace(workspace_header)
            try:
                return await call_next(request)
            finally:
                workspace_context.reset_workspace(token)

    app.add_middleware(_WorkspaceContextMiddleware)

    # Add a mock OTEL endpoint
    @app.post(OTLP_TRACES_PATH)
    async def mock_otel_endpoint(request: Request):
        workspace_name = workspace_context.get_request_workspace()
        return {
            "status": "ok",
            "workspace": workspace_name,
        }

    # Add a mock Flask-style endpoint (should NOT be processed by K8s middleware)
    @app.post("/api/2.0/mlflow/experiments/create")
    async def mock_flask_endpoint(request: Request):
        # This should be called without auth processing
        return {"status": "flask_endpoint"}

    # Add a mock Job API endpoint (should be processed by K8s middleware)
    @app.get("/ajax-api/3.0/jobs/123")
    async def mock_job_endpoint(request: Request):
        workspace_name = workspace_context.get_request_workspace()
        return {
            "status": "job_endpoint",
            "workspace": workspace_name,
        }

    return app


def test_otel_endpoint_requires_auth(fastapi_app_with_k8s_auth):
    """OTEL endpoints must receive an auth header."""
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.post(
        OTLP_TRACES_PATH,
        headers={
            "X-MLflow-Experiment-Id": "exp123",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )
    assert response.status_code == 401
    assert (
        "Missing Authorization header or X-Forwarded-Access-Token header"
        in response.json()["error"]["message"]
    )


def test_otel_endpoint_requires_bearer_token(fastapi_app_with_k8s_auth):
    """OTEL endpoints enforce Bearer token formatting."""
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.post(
        OTLP_TRACES_PATH,
        headers={
            "Authorization": "Basic invalid",
            "X-MLflow-Experiment-Id": "exp123",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )
    assert response.status_code == 401
    assert "Bearer" in response.json()["error"]["message"]


def test_otel_endpoint_requires_experiment_id(fastapi_app_with_k8s_auth):
    """OTEL endpoints require the experiment header."""
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.post(
        OTLP_TRACES_PATH,
        headers={
            "Authorization": "Bearer valid-token",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )
    assert response.status_code == 400
    assert "X-MLflow-Experiment-Id" in response.json()["error"]["message"]


def test_otel_endpoint_requires_workspace_header(fastapi_app_with_k8s_auth):
    """OTEL endpoints require the workspace header."""
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.post(
        OTLP_TRACES_PATH,
        headers={
            "Authorization": "Bearer valid-token",
            "X-MLflow-Experiment-Id": "exp123",
        },
    )

    assert response.status_code == 400
    assert WORKSPACE_HEADER_NAME in response.json()["error"]["message"]


def test_otel_endpoint_with_valid_auth(fastapi_app_with_k8s_auth, mock_authorizer):
    """Successful OTEL request propagates workspace info."""
    client = TestClient(fastapi_app_with_k8s_auth)

    with patch("kubernetes_workspace_provider.auth._parse_jwt_subject", return_value="test-user"):
        response = client.post(
            OTLP_TRACES_PATH,
            headers={
                "Authorization": "Bearer valid-token",
                "X-MLflow-Experiment-Id": "exp123",
                WORKSPACE_HEADER_NAME: "team-a",
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["workspace"] == "team-a"

    mock_authorizer.is_allowed.assert_called_once()
    identity, resource, verb, namespace = mock_authorizer.is_allowed.call_args[0]
    assert identity.token == "valid-token"
    assert (resource, verb, namespace) == ("experiments", "update", "team-a")


def test_otel_endpoint_accepts_forwarded_access_token(
    fastapi_app_with_k8s_auth, mock_authorizer
) -> None:
    """OTEL endpoints accept the forwarded token header."""
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.post(
        OTLP_TRACES_PATH,
        headers={
            "X-Forwarded-Access-Token": "forwarded-token",
            "X-MLflow-Experiment-Id": "exp123",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    mock_authorizer.is_allowed.assert_called_once()
    identity, resource, verb, namespace = mock_authorizer.is_allowed.call_args[0]
    assert identity.token == "forwarded-token"
    assert (resource, verb, namespace) == ("experiments", "update", "team-a")


def test_otel_endpoint_prefers_forwarded_token_on_invalid_authorization(
    fastapi_app_with_k8s_auth, mock_authorizer
) -> None:
    """Forwarded token is used when Authorization header is invalid."""
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.post(
        OTLP_TRACES_PATH,
        headers={
            "Authorization": "Token invalid",
            "X-Forwarded-Access-Token": "forwarded-token",
            "X-MLflow-Experiment-Id": "exp123",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )

    assert response.status_code == 200
    mock_authorizer.is_allowed.assert_called_once()
    identity, resource, verb, namespace = mock_authorizer.is_allowed.call_args[0]
    assert identity.token == "forwarded-token"
    assert (resource, verb, namespace) == ("experiments", "update", "team-a")


def test_otel_endpoint_permission_denied(fastapi_app_with_k8s_auth, mock_authorizer):
    """Permission denial propagates to clients."""
    mock_authorizer.is_allowed.return_value = False
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.post(
        OTLP_TRACES_PATH,
        headers={
            "Authorization": "Bearer valid-token",
            "X-MLflow-Experiment-Id": "exp123",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )

    assert response.status_code == 403
    assert "Permission denied" in response.json()["error"]["message"]


def test_flask_endpoints_bypass_fastapi_middleware(
    fastapi_app_with_k8s_auth, mock_authorizer
) -> None:
    """Flask endpoints should not be processed by the FastAPI middleware."""
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.post(
        "/api/2.0/mlflow/experiments/create",
        headers={WORKSPACE_HEADER_NAME: "team-a"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "flask_endpoint"
    mock_authorizer.is_allowed.assert_not_called()


def test_job_api_endpoints_require_auth(fastapi_app_with_k8s_auth, mock_authorizer):
    """Job API endpoints must enforce authentication."""
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.get(
        "/ajax-api/3.0/jobs/123",
        headers={WORKSPACE_HEADER_NAME: "team-a"},
    )
    assert response.status_code == 401
    assert (
        "Missing Authorization header or X-Forwarded-Access-Token header"
        in response.json()["error"]["message"]
    )

    mock_authorizer.reset_mock()
    with patch("kubernetes_workspace_provider.auth._parse_jwt_subject", return_value="test-user"):
        response = client.get(
            "/ajax-api/3.0/jobs/123",
            headers={
                "Authorization": "Bearer valid-token",
                WORKSPACE_HEADER_NAME: "team-a",
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "job_endpoint"
    assert response.json()["workspace"] == "team-a"
    mock_authorizer.is_allowed.assert_called_once()
    identity, resource, verb, namespace = mock_authorizer.is_allowed.call_args[0]
    assert identity.token == "valid-token"
    assert (resource, verb, namespace) == ("jobs", "get", "team-a")


def test_job_api_endpoints_accept_forwarded_access_token(
    fastapi_app_with_k8s_auth, mock_authorizer
) -> None:
    """Ensure Job API GETs authorize successfully using only the forwarded token header."""
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.get(
        "/ajax-api/3.0/jobs/123",
        headers={
            "X-Forwarded-Access-Token": "forwarded-token",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "job_endpoint"
    mock_authorizer.is_allowed.assert_called_once()
    identity, resource, verb, namespace = mock_authorizer.is_allowed.call_args[0]
    assert identity.token == "forwarded-token"
    assert (resource, verb, namespace) == ("jobs", "get", "team-a")


def test_job_api_endpoints_prefer_forwarded_token_on_invalid_authorization(
    fastapi_app_with_k8s_auth, mock_authorizer
) -> None:
    """Job API falls back to forwarded token when Authorization invalid."""
    client = TestClient(fastapi_app_with_k8s_auth)

    response = client.get(
        "/ajax-api/3.0/jobs/123",
        headers={
            "Authorization": "Token invalid",
            "X-Forwarded-Access-Token": "forwarded-token",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )

    assert response.status_code == 200
    mock_authorizer.is_allowed.assert_called_once()
    identity, resource, verb, namespace = mock_authorizer.is_allowed.call_args[0]
    assert identity.token == "forwarded-token"
    assert (resource, verb, namespace) == ("jobs", "get", "team-a")


def test_job_api_missing_workspace_context_returns_error(mock_authorizer, mock_config) -> None:
    """Missing workspace context should be surfaced as a server error."""
    app = FastAPI()
    app.add_middleware(
        KubernetesAuthMiddleware,
        authorizer=mock_authorizer,
        config_values=mock_config,
    )

    @app.get("/ajax-api/3.0/jobs/123")
    async def job_endpoint():
        return {"status": "job_endpoint"}

    client = TestClient(app)

    response = client.get(
        "/ajax-api/3.0/jobs/123",
        headers={
            "Authorization": "Bearer valid-token",
            WORKSPACE_HEADER_NAME: "team-a",
        },
    )

    assert response.status_code == 500
    assert "Workspace context" in response.json()["error"]["message"]


def test_fastapi_route_validation_passes_for_known_routes(fastapi_app_with_k8s_auth):
    _validate_fastapi_route_authorization(fastapi_app_with_k8s_auth)


def test_fastapi_route_validation_fails_for_missing_rule():
    app = FastAPI()

    @app.get("/ajax-api/3.0/jobs-new")
    async def _missing():
        return {}

    with pytest.raises(MlflowException, match="FastAPI endpoints"):
        _validate_fastapi_route_authorization(app)


# Example usage documentation
"""
Example: Using OTEL endpoints with Kubernetes authorization

1. Start MLflow server with the Kubernetes workspace provider (via workspace store URI) and auth:
   ```bash
   mlflow server \\
     --app-name kubernetes-auth \\
     --enable-workspaces \\
     --workspace-store-uri "kubernetes://?label_selector=mlflow-enabled%3Dtrue&default_workspace=team-a"
   ```

2. Configure OTEL exporter to send traces with authentication:
   ```python
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

   # Create exporter with Bearer token
   exporter = OTLPSpanExporter(
       endpoint="http://mlflow-server:5000/v1/traces",
      headers={
          "Authorization": "Bearer <k8s-service-account-token>",
          "X-MLflow-Experiment-Id": "experiment-123",
          "X-MLFLOW-WORKSPACE": "team-a",
      }
   )
   ```

3. To send to another workspace, update the header:
   ```python
   exporter = OTLPSpanExporter(
        endpoint="http://mlflow-server:5000/v1/traces",
      headers={
          "Authorization": "Bearer <k8s-service-account-token>",
          "X-MLflow-Experiment-Id": "experiment-123",
          "X-MLFLOW-WORKSPACE": "team-b",
      }
   )
   ```

4. Required Kubernetes RBAC permissions:
   ```yaml
   apiVersion: rbac.authorization.k8s.io/v1
   kind: Role
   metadata:
     name: mlflow-trace-ingester
     namespace: team-a  # workspace namespace
   rules:
     - apiGroups: ["mlflow.kubeflow.org"]
       resources: ["experiments"]
       verbs: ["create"]  # Required for trace ingestion
   ```
"""

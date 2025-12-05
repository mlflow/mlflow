"""Tests for Kubernetes auth behavior with Flask and workspace helpers.

This suite exercises request overrides, authorization rules, and caching logic.
"""

import base64
import json
import os
import time
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from flask import Flask, g, request
from kubernetes_workspace_provider.auth import (
    AUTHORIZATION_MODE_ENV,
    GRAPHQL_OPERATION_RULES,
    K8S_GRAPHQL_OPERATION_RESOURCE_MAP,
    K8S_GRAPHQL_OPERATION_VERB_MAP,
    PATH_AUTHORIZATION_RULES,
    REMOTE_GROUPS_HEADER_ENV,
    REMOTE_USER_HEADER_ENV,
    RESOURCE_EXPERIMENTS,
    RESOURCE_WORKSPACES,
    AuthorizationMode,
    AuthorizationRule,
    KubernetesAuthConfig,
    KubernetesAuthorizer,
    _AuthorizationCache,
    _authorize_request,
    _CacheEntry,
    _compile_authorization_rules,
    _find_authorization_rule,
    _is_unprotected_path,
    _override_run_user,
    _parse_jwt_subject,
    _parse_remote_groups,
    _RequestIdentity,
)

from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.protos.service_pb2 import (
    CreateRun,
    CreateWorkspace,
    DeleteWorkspace,
    GetWorkspace,
    ListWorkspaces,
    UpdateWorkspace,
)
from mlflow.utils import workspace_context


@pytest.fixture(autouse=True)
def _compile_rules(monkeypatch):
    """Ensure authorization rules are populated before each test."""
    if os.environ.get("K8S_AUTH_TEST_SKIP_COMPILE") == "1":
        return

    # Limit endpoint discovery to avoid unrelated Flask routes during tests
    def _fake_get_endpoints(resolver):
        return [
            ("/api/2.0/mlflow/runs/create", resolver(CreateRun), ["POST"]),
            ("/api/2.0/mlflow/workspaces", resolver(ListWorkspaces), ["GET"]),
            ("/api/2.0/mlflow/workspaces", resolver(CreateWorkspace), ["POST"]),
            (
                "/api/2.0/mlflow/workspaces/<workspace_name>",
                resolver(GetWorkspace),
                ["GET"],
            ),
            (
                "/api/2.0/mlflow/workspaces/<workspace_name>",
                resolver(UpdateWorkspace),
                ["PATCH"],
            ),
            (
                "/api/2.0/mlflow/workspaces/<workspace_name>",
                resolver(DeleteWorkspace),
                ["DELETE"],
            ),
        ]

    monkeypatch.setattr("kubernetes_workspace_provider.auth.get_endpoints", _fake_get_endpoints)
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth.mlflow_app.url_map.iter_rules",
        lambda: [],
    )
    _compile_authorization_rules()


def _make_jwt_token(payload: dict[str, object]) -> str:
    header = base64.urlsafe_b64encode(b"{}").decode().rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{header}.{body}.signature"


def test_parse_jwt_subject_returns_claim_value_when_present():
    token = _make_jwt_token({"sub": "alice"})
    assert _parse_jwt_subject(token, "sub") == "alice"


def test_parse_jwt_subject_missing_claim_returns_none():
    token = _make_jwt_token({"sub": "alice"})
    assert _parse_jwt_subject(token, "email") is None


def test_parse_jwt_subject_empty_or_non_string_claim_returns_none():
    assert _parse_jwt_subject(_make_jwt_token({"sub": ""}), "sub") is None
    assert _parse_jwt_subject(_make_jwt_token({"sub": {"nested": True}}), "sub") is None


def test_parse_jwt_subject_token_without_payload_segment_returns_none():
    assert _parse_jwt_subject("malformed-token", "sub") is None


def test_parse_jwt_subject_invalid_base64_payload_returns_none():
    header = base64.urlsafe_b64encode(b"{}").decode().rstrip("=")
    token = f"{header}.@@not-base64@@.signature"
    assert _parse_jwt_subject(token, "sub") is None


def test_parse_jwt_subject_invalid_json_payload_returns_none():
    header = base64.urlsafe_b64encode(b"{}").decode().rstrip("=")
    body = base64.urlsafe_b64encode(b"not json").decode().rstrip("=")
    token = f"{header}.{body}.signature"
    assert _parse_jwt_subject(token, "sub") is None


def test_request_identity_subject_hash_self_subject_access_review():
    identity = _RequestIdentity(token="token-value")
    digest = identity.subject_hash(AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW)
    assert digest == sha256(b"token-value").hexdigest()


def test_request_identity_subject_hash_subject_access_review_normalizes(monkeypatch):
    identity = _RequestIdentity(user="  alice  ", groups=("group-b", "group-a"))
    digest = identity.subject_hash(AuthorizationMode.SUBJECT_ACCESS_REVIEW)
    expected = sha256(b"alice\x00group-a\x00group-b").hexdigest()
    assert digest == expected


def test_request_identity_subject_hash_missing_user_raises():
    identity = _RequestIdentity(user=None)
    with pytest.raises(MlflowException, match="X-Remote-User header required"):
        identity.subject_hash(
            AuthorizationMode.SUBJECT_ACCESS_REVIEW,
            missing_user_label="X-Remote-User header required",
        )


@pytest.mark.parametrize(
    ("header_value", "separator", "expected"),
    [
        (None, "|", ()),
        ("", "|", ()),
        ("group-a|group-b", "", ("group-a|group-b",)),
        (" group-a | group-b ", "|", ("group-a", "group-b")),
        ("one,two", ",", ("one", "two")),
    ],
)
def test_parse_remote_groups_variations(header_value, separator, expected):
    assert _parse_remote_groups(header_value, separator) == expected


def test_request_identity_subject_hash_missing_token_in_ssar():
    identity = _RequestIdentity(token=None)
    with pytest.raises(
        MlflowException,
        match="Bearer token is required for SelfSubjectAccessReview mode.",
    ) as exc:
        identity.subject_hash(AuthorizationMode.SELF_SUBJECT_ACCESS_REVIEW)

    assert exc.value.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.UNAUTHENTICATED)


def test_kubernetes_auth_config_invalid_mode(monkeypatch):
    monkeypatch.setenv(AUTHORIZATION_MODE_ENV, "invalid-mode")
    with pytest.raises(MlflowException, match="must be one of"):
        KubernetesAuthConfig.from_env()


def test_kubernetes_auth_config_empty_user_header(monkeypatch):
    monkeypatch.setenv(REMOTE_USER_HEADER_ENV, "   ")
    with pytest.raises(MlflowException, match="cannot be empty") as exc:
        KubernetesAuthConfig.from_env()

    assert exc.value.error_code == databricks_pb2.ErrorCode.Name(
        databricks_pb2.INVALID_PARAMETER_VALUE
    )


def test_kubernetes_auth_config_empty_groups_header(monkeypatch):
    monkeypatch.setenv(REMOTE_GROUPS_HEADER_ENV, "")
    with pytest.raises(MlflowException, match="cannot be empty"):
        KubernetesAuthConfig.from_env()


def test_override_run_user_with_json_request():
    """Test that _override_run_user correctly modifies JSON requests."""
    app = Flask(__name__)

    @app.route("/test", methods=["POST"])
    def test_endpoint():
        # Get the modified JSON data
        data = request.get_json(silent=True)
        # Also test with silent=False to ensure both cached values work
        data2 = request.get_json(silent=False)
        assert data == data2
        return {"received": data, "user_id": data.get("user_id")}

    with app.test_request_context(
        "/test",
        method="POST",
        data=json.dumps({"experiment_id": "123"}),
        content_type="application/json",
    ):
        # Simulate the auth handler modifying the request
        _override_run_user("test-user")

        # Verify the request was modified correctly
        modified_data = request.get_json(silent=True)
        assert modified_data["experiment_id"] == "123"
        assert modified_data["user_id"] == "test-user"

        # Test that both silent=True and silent=False work
        data_silent_false = request.get_json(silent=False)
        assert data_silent_false == modified_data

        # Verify the raw data was updated
        request.environ["wsgi.input"].seek(0)
        raw_data = request.environ["wsgi.input"].read()
        parsed_raw = json.loads(raw_data)
        assert parsed_raw["user_id"] == "test-user"


def test_override_run_user_with_empty_request():
    """Test _override_run_user with an empty JSON request."""
    app = Flask(__name__)

    with app.test_request_context(
        "/test",
        method="POST",
        data="{}",
        content_type="application/json",
    ):
        _override_run_user("test-user")

        modified_data = request.get_json(silent=True)
        assert modified_data == {"user_id": "test-user"}


def test_override_run_user_with_non_json_request():
    """Test that _override_run_user ignores non-JSON requests."""
    app = Flask(__name__)

    with app.test_request_context(
        "/test",
        method="POST",
        data="not json data",
        content_type="text/plain",
    ):
        original_data = request.data
        _override_run_user("test-user")

        # Data should not be modified for non-JSON requests
        assert request.data == original_data


def test_override_run_user_preserves_other_fields():
    """Test that _override_run_user preserves all other JSON fields."""
    app = Flask(__name__)

    original_payload = {
        "experiment_id": "exp123",
        "tags": [{"key": "tag1", "value": "val1"}],
        "nested": {"field": "value"},
        "user_id": "original-user",  # This should be overwritten
    }

    with app.test_request_context(
        "/test",
        method="POST",
        data=json.dumps(original_payload),
        content_type="application/json",
    ):
        _override_run_user("new-user")

        modified_data = request.get_json(silent=True)
        assert modified_data["user_id"] == "new-user"
        assert modified_data["experiment_id"] == "exp123"
        assert modified_data["tags"] == [{"key": "tag1", "value": "val1"}]
        assert modified_data["nested"] == {"field": "value"}


def test_flask_create_run_request_processing(monkeypatch):
    """Test that CreateRun requests are processed correctly with user override."""
    from kubernetes_workspace_provider.auth import create_app

    # Create a minimal Flask app with auth
    app = Flask(__name__)

    @app.route("/api/2.0/mlflow/runs/create", methods=["POST"])
    def create_run():
        # This simulates the MLflow handler
        data = request.get_json(silent=True)
        return {"run": {"info": {"run_id": "test-run-id", "user_id": data.get("user_id")}}}

    @app.before_request
    def _set_rbac_context():
        g._workspace_token = workspace_context.set_current_workspace("default")

    @app.teardown_request
    def _reset_rbac_context(_response):
        token = getattr(g, "_workspace_token", None)
        if token is not None:
            workspace_context.reset_workspace(token)
            delattr(g, "_workspace_token")

    # Set up the app with Kubernetes auth
    monkeypatch.setenv("MLFLOW_K8S_AUTH_CACHE_TTL_SECONDS", "300")
    fake_config = SimpleNamespace(
        host="https://cluster.local",
        ssl_ca_cert=None,
        verify_ssl=True,
        proxy=None,
        no_proxy=None,
        proxy_headers=None,
        safe_chars_for_path_param=None,
        connection_pool_maxsize=10,
    )
    with (
        patch(
            "kubernetes_workspace_provider.auth.KubernetesAuthorizer.is_allowed",
            return_value=True,
        ),
        patch(
            "kubernetes_workspace_provider.auth._load_kubernetes_configuration",
            return_value=fake_config,
        ),
    ):
        create_app(app)

        # Test the request
        client = app.test_client()

        with patch(
            "kubernetes_workspace_provider.auth._parse_jwt_subject", return_value="k8s-user"
        ):
            response = client.post(
                "/api/2.0/mlflow/runs/create",
                json={"experiment_id": "0"},
                headers={"Authorization": "Bearer test-token"},
            )

    # The response should have the overridden user
    assert response.status_code == 200
    assert response.json["run"]["info"]["user_id"] == "k8s-user"


def _build_workspace_app(monkeypatch):
    from kubernetes_workspace_provider.auth import create_app

    app = Flask(__name__)

    @app.route("/api/2.0/mlflow/workspaces", methods=["GET"])
    def list_workspaces():
        return {"workspaces": [{"name": "team-a"}]}

    @app.route("/api/2.0/mlflow/workspaces", methods=["POST"])
    def create_workspace_endpoint():
        payload = request.get_json()
        return (
            {"workspace": {"name": payload.get("name")}},
            201,
        )

    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth.KubernetesAuthorizer.is_allowed",
        lambda self, identity, resource, verb, namespace: True,
    )
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth._load_kubernetes_configuration",
        lambda: None,
    )
    monkeypatch.setenv("MLFLOW_K8S_AUTH_CACHE_TTL_SECONDS", "300")

    create_app(app)
    return app.test_client()


def test_list_workspaces_without_context(monkeypatch):
    client = _build_workspace_app(monkeypatch)

    with patch(
        "kubernetes_workspace_provider.auth._parse_jwt_subject",
        return_value="k8s-user",
    ):
        response = client.get(
            "/api/2.0/mlflow/workspaces",
            headers={"Authorization": "Bearer list-token"},
        )

    assert response.status_code == 200
    assert response.json["workspaces"] == [{"name": "team-a"}]


def test_create_workspace_requests_are_denied(monkeypatch):
    mock_is_allowed = Mock(return_value=True)
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth.KubernetesAuthorizer.is_allowed",
        mock_is_allowed,
    )
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth._load_kubernetes_configuration",
        lambda: None,
    )
    monkeypatch.setenv("MLFLOW_K8S_AUTH_CACHE_TTL_SECONDS", "300")

    from kubernetes_workspace_provider.auth import create_app

    app = Flask(__name__)

    @app.route("/api/2.0/mlflow/workspaces", methods=["POST"])
    def create_workspace_endpoint():
        payload = request.get_json()
        return {"workspace": {"name": payload.get("name")}}, 201

    create_app(app)
    client = app.test_client()

    with patch(
        "kubernetes_workspace_provider.auth._parse_jwt_subject",
        return_value="k8s-user",
    ):
        response = client.post(
            "/api/2.0/mlflow/workspaces",
            json={"name": "team-new"},
            headers={"Authorization": "Bearer create-token"},
        )

    assert response.status_code == 403
    assert "Workspace create" in response.json["message"]
    mock_is_allowed.assert_not_called()


def _build_flask_auth_app(monkeypatch, *, is_allowed=True):
    from kubernetes_workspace_provider.auth import create_app

    app = Flask(__name__)

    @app.route("/api/2.0/mlflow/runs/create", methods=["POST"])
    def create_run():
        data = request.get_json(silent=True)
        return {"run": {"info": {"run_id": "test-run-id", "user_id": data.get("user_id")}}}

    @app.before_request
    def _set_rbac_context():
        g._workspace_token = workspace_context.set_current_workspace("default")

    @app.teardown_request
    def _reset_rbac_context(_response):
        token = getattr(g, "_workspace_token", None)
        if token is not None:
            workspace_context.reset_workspace(token)
            delattr(g, "_workspace_token")

    mock_is_allowed = Mock(return_value=is_allowed)
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth.KubernetesAuthorizer.is_allowed", mock_is_allowed
    )
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth._load_kubernetes_configuration", lambda: None
    )
    monkeypatch.setenv("MLFLOW_K8S_AUTH_CACHE_TTL_SECONDS", "300")

    create_app(app)
    return app.test_client(), mock_is_allowed


def test_missing_authorization_header_returns_401(monkeypatch):
    client, mock_is_allowed = _build_flask_auth_app(monkeypatch)

    response = client.post("/api/2.0/mlflow/runs/create", json={"experiment_id": "0"})
    assert response.status_code == 401
    assert (
        "Missing Authorization header or X-Forwarded-Access-Token header"
        in response.json["message"]
    )
    mock_is_allowed.assert_not_called()


def test_invalid_bearer_scheme_returns_401(monkeypatch):
    client, mock_is_allowed = _build_flask_auth_app(monkeypatch)

    response = client.post(
        "/api/2.0/mlflow/runs/create",
        json={"experiment_id": "0"},
        headers={"Authorization": "Token bad-token"},
    )
    assert response.status_code == 401
    assert "Bearer" in response.json["message"]
    mock_is_allowed.assert_not_called()


def test_forwarded_access_token_header_allows_request(monkeypatch):
    client, mock_is_allowed = _build_flask_auth_app(monkeypatch)

    with patch("kubernetes_workspace_provider.auth._parse_jwt_subject", return_value="k8s-user"):
        response = client.post(
            "/api/2.0/mlflow/runs/create",
            json={"experiment_id": "0"},
            headers={"X-Forwarded-Access-Token": "test-token"},
        )

    assert response.status_code == 200
    mock_is_allowed.assert_called_once()
    identity, resource, verb, namespace = mock_is_allowed.call_args[0]
    assert identity.token == "test-token"


def test_invalid_authorization_header_with_forwarded_token(monkeypatch):
    client, mock_is_allowed = _build_flask_auth_app(monkeypatch)

    with patch("kubernetes_workspace_provider.auth._parse_jwt_subject", return_value="k8s-user"):
        response = client.post(
            "/api/2.0/mlflow/runs/create",
            json={"experiment_id": "0"},
            headers={
                "Authorization": "Basic bad-token",
                "X-Forwarded-Access-Token": "forwarded-token",
            },
        )

    assert response.status_code == 200
    mock_is_allowed.assert_called_once()
    identity, resource, verb, namespace = mock_is_allowed.call_args[0]
    assert identity.token == "forwarded-token"


def test_permission_denied_returns_403(monkeypatch):
    client, mock_is_allowed = _build_flask_auth_app(monkeypatch, is_allowed=False)

    with patch("kubernetes_workspace_provider.auth._parse_jwt_subject", return_value="k8s-user"):
        response = client.post(
            "/api/2.0/mlflow/runs/create",
            json={"experiment_id": "0"},
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 403
    assert "Permission denied" in response.json["message"]
    mock_is_allowed.assert_called_once()
    identity, resource, verb, namespace = mock_is_allowed.call_args[0]
    assert identity.token == "test-token"
    assert (resource, verb, namespace) == ("experiments", "update", "default")


def test_workspace_scope_string_is_normalized(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.return_value = True

    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth._find_authorization_rule",
        lambda path, method: AuthorizationRule("list", resource="experiments"),
    )
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )

    config = KubernetesAuthConfig()
    result = _authorize_request(
        authorization_header="Bearer valid-token",
        forwarded_access_token=None,
        remote_user_header_value=None,
        remote_groups_header_value=None,
        path="/ajax-api/2.0/mlflow/experiments/search",
        method="GET",
        authorizer=authorizer,
        config_values=config,
        workspace=" team-a ",
    )

    call_args = authorizer.is_allowed.call_args[0]
    identity_arg = call_args[0]
    assert identity_arg.token == "valid-token"
    assert call_args[1:] == ("experiments", "list", "team-a")
    assert result.username == "k8s-user"


def test_workspace_listing_allows_missing_context(monkeypatch):
    authorizer = Mock()
    rule = AuthorizationRule(
        None,
        resource=RESOURCE_WORKSPACES,
        apply_workspace_filter=True,
        requires_workspace=False,
    )
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth._find_authorization_rule",
        lambda path, method: rule,
    )
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )

    config = KubernetesAuthConfig()
    result = _authorize_request(
        authorization_header="Bearer list-token",
        forwarded_access_token=None,
        remote_user_header_value=None,
        remote_groups_header_value=None,
        path="/api/2.0/mlflow/workspaces",
        method="GET",
        authorizer=authorizer,
        config_values=config,
        workspace=None,
    )

    assert result.username == "k8s-user"
    assert result.rule.apply_workspace_filter
    authorizer.is_allowed.assert_not_called()


def test_subject_access_review_mode_uses_remote_headers(monkeypatch):
    authorizer = Mock()
    authorizer.is_allowed.return_value = True
    rule = AuthorizationRule("list", resource=RESOURCE_EXPERIMENTS)
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth._find_authorization_rule",
        lambda path, method: rule,
    )

    config = KubernetesAuthConfig(authorization_mode=AuthorizationMode.SUBJECT_ACCESS_REVIEW)

    result = _authorize_request(
        authorization_header=None,
        forwarded_access_token=None,
        remote_user_header_value="proxy-user",
        remote_groups_header_value="group-a|group-b",
        path="/ajax-api/2.0/mlflow/experiments/search",
        method="GET",
        authorizer=authorizer,
        config_values=config,
        workspace="team-a",
    )

    identity, resource, verb, namespace = authorizer.is_allowed.call_args[0]
    assert identity.token is None
    assert identity.user == "proxy-user"
    assert identity.groups == ("group-a", "group-b")
    assert (resource, verb, namespace) == ("experiments", "list", "team-a")
    assert result.username == "proxy-user"


def test_subject_access_review_mode_requires_user_header(monkeypatch):
    authorizer = Mock()
    rule = AuthorizationRule("get", resource=RESOURCE_EXPERIMENTS)
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth._find_authorization_rule",
        lambda path, method: rule,
    )

    config = KubernetesAuthConfig(authorization_mode=AuthorizationMode.SUBJECT_ACCESS_REVIEW)

    with pytest.raises(MlflowException, match="Missing required") as exc:
        _authorize_request(
            authorization_header=None,
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value="group-a|group-b",
            path="/ajax-api/2.0/mlflow/experiments/get",
            method="GET",
            authorizer=authorizer,
            config_values=config,
            workspace="team-a",
        )

    assert exc.value.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.UNAUTHENTICATED)
    authorizer.is_allowed.assert_not_called()


def test_workspace_scope_falls_back_to_view_args(monkeypatch):
    app = Flask(__name__)
    authorizer = Mock()
    authorizer.can_access_workspace.return_value = True
    rule = AuthorizationRule(None, resource=RESOURCE_WORKSPACES)
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth._find_authorization_rule",
        lambda path, method: rule,
    )
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )

    with app.test_request_context("/api/2.0/mlflow/workspaces/team-a", method="GET"):
        request.view_args = {"workspace_name": "team-a"}
        _authorize_request(
            authorization_header="Bearer scope-token",
            forwarded_access_token=None,
            remote_user_header_value=None,
            remote_groups_header_value=None,
            path="/api/2.0/mlflow/workspaces/team-a",
            method="GET",
            authorizer=authorizer,
            config_values=KubernetesAuthConfig(),
            workspace=None,
        )

    args = authorizer.can_access_workspace.call_args[0]
    assert args[0].token == "scope-token"
    assert args[1:] == ("team-a",)
    kwargs = authorizer.can_access_workspace.call_args[1]
    assert kwargs == {"verb": "get"}


def test_workspace_create_requests_are_denied(monkeypatch):
    app = Flask(__name__)
    authorizer = Mock()
    rule = AuthorizationRule("create", resource=RESOURCE_WORKSPACES, deny=True)
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth._find_authorization_rule",
        lambda path, method: rule,
    )
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth._parse_jwt_subject",
        lambda token, claim: "k8s-user",
    )

    with app.test_request_context(
        "/api/2.0/mlflow/workspaces",
        method="POST",
        data=json.dumps({"name": "team-new"}),
        content_type="application/json",
    ):
        with pytest.raises(
            MlflowException, match="Workspace create, update, and delete operations"
        ) as exc:
            _authorize_request(
                authorization_header="Bearer create-token",
                forwarded_access_token=None,
                remote_user_header_value=None,
                remote_groups_header_value=None,
                path="/api/2.0/mlflow/workspaces",
                method="POST",
                authorizer=authorizer,
                config_values=KubernetesAuthConfig(),
                workspace=None,
            )

    assert exc.value.error_code == databricks_pb2.ErrorCode.Name(databricks_pb2.PERMISSION_DENIED)
    authorizer.is_allowed.assert_not_called()


def test_compile_rules_raise_for_uncovered_endpoint(monkeypatch):
    import kubernetes_workspace_provider.auth as auth_mod

    monkeypatch.setenv("K8S_AUTH_TEST_SKIP_COMPILE", "1")

    auth_mod._RULES_COMPILED = False
    auth_mod._AUTH_RULES.clear()
    auth_mod._AUTH_REGEX_RULES.clear()
    auth_mod._HANDLER_RULES.clear()

    def _fake_endpoints(resolver):
        def _handler():
            return None

        return [("/api/2.0/mlflow/uncovered", _handler, ["GET"])]

    monkeypatch.setattr("kubernetes_workspace_provider.auth.get_endpoints", _fake_endpoints)
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth.mlflow_app.url_map.iter_rules",
        lambda: [],
    )

    with pytest.raises(MlflowException, match="/api/2.0/mlflow/uncovered") as exc:
        auth_mod._compile_authorization_rules()

    assert "/api/2.0/mlflow/uncovered" in str(exc.value)


def test_graphql_operation_map_matches_constant():
    assert set(K8S_GRAPHQL_OPERATION_RESOURCE_MAP) == set(K8S_GRAPHQL_OPERATION_VERB_MAP)
    for operation_name in K8S_GRAPHQL_OPERATION_RESOURCE_MAP:
        assert operation_name in GRAPHQL_OPERATION_RULES
        assert (
            GRAPHQL_OPERATION_RULES[operation_name].verb
            == K8S_GRAPHQL_OPERATION_VERB_MAP[operation_name]
        )


def test_graphql_unknown_operation_defaults_to_read_only():
    app = Flask(__name__)
    with app.test_request_context(
        "/graphql",
        method="POST",
        json={"operationName": "NewGraphQLOperation"},
    ):
        rule = _find_authorization_rule("/graphql", "POST")
    assert rule.verb == "get"


def test_gateway_proxy_routes_require_verbs():
    get_rule = PATH_AUTHORIZATION_RULES[("/api/2.0/mlflow/gateway-proxy", "GET")]
    post_rule = PATH_AUTHORIZATION_RULES[("/ajax-api/2.0/mlflow/gateway-proxy", "POST")]
    assert get_rule.verb == "get"
    assert post_rule.verb == "update"


def test_misc_path_authorization_rules_cover_recent_endpoints():
    assert PATH_AUTHORIZATION_RULES[("/version", "GET")].verb is None
    assert (
        PATH_AUTHORIZATION_RULES[("/ajax-api/2.0/mlflow/metrics/get-history-bulk", "GET")].verb
        == "list"
    )
    assert (
        PATH_AUTHORIZATION_RULES[
            ("/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval", "GET")
        ].verb
        == "list"
    )
    assert (
        PATH_AUTHORIZATION_RULES[("/ajax-api/2.0/mlflow/runs/create-promptlab-run", "POST")].verb
        == "update"
    )
    assert (
        PATH_AUTHORIZATION_RULES[
            ("/ajax-api/2.0/mlflow/logged-models/<model_id>/artifacts/files", "GET")
        ].verb
        == "get"
    )


def test_server_features_endpoints_are_unprotected():
    assert _is_unprotected_path("/api/2.0/mlflow/server-features")
    assert _is_unprotected_path("/ajax-api/2.0/mlflow/server-features")


def test_authorization_cache_does_not_drop_new_entries_during_cleanup():
    cache = _AuthorizationCache(ttl_seconds=0.1)
    key = ("token", "namespace", "resource", "verb")

    class _InstrumentedLock:
        def __init__(self):
            self.on_release_read = None

        def acquire_read(self):
            return None

        def release_read(self):
            if self.on_release_read:
                callback = self.on_release_read
                self.on_release_read = None
                callback()

        def acquire_write(self):
            return None

        def release_write(self):
            return None

    cache._lock = _InstrumentedLock()  # type: ignore[assignment]
    cache._entries[key] = _CacheEntry(allowed=True, expires_at=time.time() - 10)

    def _insert_new_entry():
        cache._entries[key] = _CacheEntry(allowed=False, expires_at=time.time() + 100)

    cache._lock.on_release_read = _insert_new_entry  # type: ignore[attr-defined]

    # First call observes the expired entry and triggers cleanup
    assert cache.get(key) is None
    # New entry should remain available
    assert cache.get(key) is False


def test_experiment_permissions_are_checked_first(monkeypatch):
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth._load_kubernetes_configuration",
        lambda: SimpleNamespace(
            host=None,
            ssl_ca_cert=None,
            verify_ssl=True,
            proxy=None,
            no_proxy=None,
            proxy_headers=None,
            safe_chars_for_path_param=None,
            connection_pool_maxsize=None,
        ),
    )

    config = KubernetesAuthConfig(cache_ttl_seconds=1)
    authorizer = KubernetesAuthorizer(config)

    def _fake_permission(identity, resource, verb, namespace):
        return resource == "experiments" and namespace == "team-a"

    authorizer.is_allowed = Mock(side_effect=_fake_permission)  # type: ignore[method-assign]

    identity = _RequestIdentity(token="token")
    accessible = authorizer.accessible_workspaces(identity, ["team-a", "team-b"])

    assert accessible == {"team-a"}
    first_call = authorizer.is_allowed.call_args_list[0][0]
    assert first_call[1] == "experiments"


def test_can_access_workspace_iterates_priority_resources(monkeypatch):
    monkeypatch.setattr(
        "kubernetes_workspace_provider.auth._load_kubernetes_configuration",
        lambda: SimpleNamespace(
            host=None,
            ssl_ca_cert=None,
            verify_ssl=True,
            proxy=None,
            no_proxy=None,
            proxy_headers=None,
            safe_chars_for_path_param=None,
            connection_pool_maxsize=None,
        ),
    )

    config = KubernetesAuthConfig(cache_ttl_seconds=1)
    authorizer = KubernetesAuthorizer(config)

    def _fake_permission(identity, resource, verb, namespace):
        return resource == "registeredmodels" and namespace == "team-a" and verb == "get"

    authorizer.is_allowed = Mock(side_effect=_fake_permission)  # type: ignore[method-assign]

    identity = _RequestIdentity(token="token")
    assert authorizer.can_access_workspace(identity, "team-a", verb="get") is True

    calls = authorizer.is_allowed.call_args_list
    assert calls[0][0][1] == "experiments"
    assert calls[1][0][1] == "registeredmodels"

    authorizer.is_allowed.reset_mock()
    assert authorizer.can_access_workspace(identity, "team-b", verb="get") is False

    calls = authorizer.is_allowed.call_args_list
    assert len(calls) == 3
    assert [c[0][1] for c in calls] == ["experiments", "registeredmodels", "jobs"]

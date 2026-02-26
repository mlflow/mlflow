from pathlib import Path
from unittest import mock

import pytest

kubernetes = pytest.importorskip("kubernetes")

from mlflow.exceptions import MlflowException
from mlflow.tracking.request_auth.kubernetes_request_auth_provider import (
    AUTHORIZATION_HEADER_NAME,
    WORKSPACE_HEADER_NAME,
    KubernetesAuth,
    KubernetesRequestAuthProvider,
    _get_credentials,
    _get_credentials_from_kubeconfig,
    _get_credentials_from_service_account,
    _read_file_if_exists,
)
from mlflow.utils.workspace_context import get_request_workspace, set_workspace

# Tests for _read_file_if_exists


def test_read_file_returns_none_when_file_does_not_exist(tmp_path):
    non_existent = tmp_path / "does_not_exist"
    assert _read_file_if_exists(non_existent) is None


def test_read_file_returns_stripped_content_when_file_exists(tmp_path):
    test_file = tmp_path / "test_file"
    test_file.write_text("  test-namespace  \n")
    assert _read_file_if_exists(test_file) == "test-namespace"


def test_read_file_returns_none_on_permission_error(tmp_path):
    test_file = tmp_path / "test_file"
    test_file.write_text("content")
    with (
        mock.patch.object(Path, "exists", return_value=True),
        mock.patch.object(Path, "read_text", side_effect=PermissionError("denied")),
    ):
        assert _read_file_if_exists(test_file) is None


# Tests for _get_credentials_from_service_account


def test_service_account_returns_credentials_when_both_files_exist(tmp_path):
    namespace_file = tmp_path / "namespace"
    namespace_file.write_text("test-namespace")
    token_file = tmp_path / "token"
    token_file.write_text("test-token")

    with (
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_NAMESPACE_PATH",
            namespace_file,
        ),
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_TOKEN_PATH",
            token_file,
        ),
    ):
        result = _get_credentials_from_service_account()
        assert result == ("test-namespace", "Bearer test-token")


def test_service_account_returns_none_when_namespace_missing(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("test-token")

    with (
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_NAMESPACE_PATH",
            tmp_path / "nonexistent",
        ),
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_TOKEN_PATH",
            token_file,
        ),
    ):
        result = _get_credentials_from_service_account()
        assert result is None


def test_service_account_returns_none_when_token_missing(tmp_path):
    namespace_file = tmp_path / "namespace"
    namespace_file.write_text("test-namespace")

    with (
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_NAMESPACE_PATH",
            namespace_file,
        ),
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_TOKEN_PATH",
            tmp_path / "nonexistent",
        ),
    ):
        result = _get_credentials_from_service_account()
        assert result is None


# Tests for _get_credentials_from_kubeconfig


def test_kubeconfig_returns_credentials_when_both_available():
    active_context = {"name": "my-context", "context": {"namespace": "my-namespace"}}
    mock_api_client = mock.MagicMock()
    mock_api_client.default_headers = {"Authorization": "Bearer test-token"}

    with (
        mock.patch("kubernetes.config.load_kube_config"),
        mock.patch(
            "kubernetes.config.list_kube_config_contexts", return_value=([], active_context)
        ),
        mock.patch("kubernetes.client.ApiClient", return_value=mock_api_client),
    ):
        result = _get_credentials_from_kubeconfig()
        assert result == ("my-namespace", "Bearer test-token")


def test_kubeconfig_returns_none_when_no_namespace():
    active_context = {"name": "my-context", "context": {}}
    mock_api_client = mock.MagicMock()
    mock_api_client.default_headers = {"Authorization": "Bearer test-token"}

    with (
        mock.patch("kubernetes.config.load_kube_config"),
        mock.patch(
            "kubernetes.config.list_kube_config_contexts", return_value=([], active_context)
        ),
        mock.patch("kubernetes.client.ApiClient", return_value=mock_api_client),
    ):
        result = _get_credentials_from_kubeconfig()
        assert result is None


def test_kubeconfig_returns_none_when_no_token():
    active_context = {"name": "my-context", "context": {"namespace": "my-namespace"}}
    mock_api_client = mock.MagicMock()
    mock_api_client.default_headers = {}
    mock_api_client.configuration.api_key = {}

    with (
        mock.patch("kubernetes.config.load_kube_config"),
        mock.patch(
            "kubernetes.config.list_kube_config_contexts", return_value=([], active_context)
        ),
        mock.patch("kubernetes.client.ApiClient", return_value=mock_api_client),
    ):
        result = _get_credentials_from_kubeconfig()
        assert result is None


def test_kubeconfig_returns_none_when_no_active_context():
    with (
        mock.patch("kubernetes.config.load_kube_config"),
        mock.patch("kubernetes.config.list_kube_config_contexts", return_value=([], None)),
    ):
        result = _get_credentials_from_kubeconfig()
        assert result is None


def test_kubeconfig_uses_lowercase_authorization_header():
    active_context = {"name": "my-context", "context": {"namespace": "my-namespace"}}
    mock_api_client = mock.MagicMock()
    mock_api_client.default_headers = {"authorization": "Bearer lowercase-token"}

    with (
        mock.patch("kubernetes.config.load_kube_config"),
        mock.patch(
            "kubernetes.config.list_kube_config_contexts", return_value=([], active_context)
        ),
        mock.patch("kubernetes.client.ApiClient", return_value=mock_api_client),
    ):
        result = _get_credentials_from_kubeconfig()
        assert result == ("my-namespace", "Bearer lowercase-token")


def test_kubeconfig_falls_back_to_api_key():
    active_context = {"name": "my-context", "context": {"namespace": "my-namespace"}}
    mock_api_client = mock.MagicMock()
    mock_api_client.default_headers = {}
    mock_api_client.configuration.api_key = {"authorization": "fallback-token"}

    with (
        mock.patch("kubernetes.config.load_kube_config"),
        mock.patch(
            "kubernetes.config.list_kube_config_contexts", return_value=([], active_context)
        ),
        mock.patch("kubernetes.client.ApiClient", return_value=mock_api_client),
    ):
        result = _get_credentials_from_kubeconfig()
        assert result == ("my-namespace", "Bearer fallback-token")


def test_kubeconfig_strips_bearer_prefix_from_api_key():
    active_context = {"name": "my-context", "context": {"namespace": "my-namespace"}}
    mock_api_client = mock.MagicMock()
    mock_api_client.default_headers = {}
    mock_api_client.configuration.api_key = {"authorization": "Bearer prefixed-token"}

    with (
        mock.patch("kubernetes.config.load_kube_config"),
        mock.patch(
            "kubernetes.config.list_kube_config_contexts", return_value=([], active_context)
        ),
        mock.patch("kubernetes.client.ApiClient", return_value=mock_api_client),
    ):
        result = _get_credentials_from_kubeconfig()
        assert result == ("my-namespace", "Bearer prefixed-token")


# Tests for _get_credentials


def test_get_credentials_prefers_service_account(tmp_path):
    namespace_file = tmp_path / "namespace"
    namespace_file.write_text("sa-namespace")
    token_file = tmp_path / "token"
    token_file.write_text("sa-token")

    active_context = {"name": "my-context", "context": {"namespace": "kubeconfig-namespace"}}
    mock_api_client = mock.MagicMock()
    mock_api_client.default_headers = {"Authorization": "Bearer kubeconfig-token"}

    with (
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_NAMESPACE_PATH",
            namespace_file,
        ),
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_TOKEN_PATH",
            token_file,
        ),
        mock.patch("kubernetes.config.load_kube_config"),
        mock.patch(
            "kubernetes.config.list_kube_config_contexts", return_value=([], active_context)
        ),
        mock.patch("kubernetes.client.ApiClient", return_value=mock_api_client),
    ):
        result = _get_credentials()
        assert result == ("sa-namespace", "Bearer sa-token")


def test_get_credentials_falls_back_to_kubeconfig(tmp_path):
    active_context = {"name": "my-context", "context": {"namespace": "kubeconfig-namespace"}}
    mock_api_client = mock.MagicMock()
    mock_api_client.default_headers = {"Authorization": "Bearer kubeconfig-token"}

    with (
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_NAMESPACE_PATH",
            tmp_path / "nonexistent",
        ),
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_TOKEN_PATH",
            tmp_path / "nonexistent",
        ),
        mock.patch("kubernetes.config.load_kube_config"),
        mock.patch(
            "kubernetes.config.list_kube_config_contexts", return_value=([], active_context)
        ),
        mock.patch("kubernetes.client.ApiClient", return_value=mock_api_client),
    ):
        result = _get_credentials()
        assert result == ("kubeconfig-namespace", "Bearer kubeconfig-token")


def test_get_credentials_returns_none_when_nothing_available(tmp_path):
    with (
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_NAMESPACE_PATH",
            tmp_path / "nonexistent",
        ),
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_TOKEN_PATH",
            tmp_path / "nonexistent",
        ),
        mock.patch("kubernetes.config.load_kube_config"),
        mock.patch("kubernetes.config.list_kube_config_contexts", return_value=([], None)),
    ):
        result = _get_credentials()
        assert result is None


# Tests for KubernetesAuth


def test_auth_adds_headers_to_request(tmp_path):
    namespace_file = tmp_path / "namespace"
    namespace_file.write_text("test-namespace")
    token_file = tmp_path / "token"
    token_file.write_text("test-token")

    mock_request = mock.MagicMock()
    mock_request.headers = {}

    with (
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_NAMESPACE_PATH",
            namespace_file,
        ),
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_TOKEN_PATH",
            token_file,
        ),
    ):
        auth = KubernetesAuth()
        result = auth(mock_request)

        assert result is mock_request
        assert mock_request.headers[WORKSPACE_HEADER_NAME] == "test-namespace"
        assert mock_request.headers[AUTHORIZATION_HEADER_NAME] == "Bearer test-token"


def test_auth_skips_when_both_headers_already_set():
    mock_request = mock.MagicMock()
    mock_request.headers = {
        WORKSPACE_HEADER_NAME: "existing-workspace",
        AUTHORIZATION_HEADER_NAME: "existing-auth",
    }

    auth = KubernetesAuth()
    result = auth(mock_request)

    assert result is mock_request
    assert mock_request.headers[WORKSPACE_HEADER_NAME] == "existing-workspace"
    assert mock_request.headers[AUTHORIZATION_HEADER_NAME] == "existing-auth"


def test_auth_does_not_override_existing_workspace_header(tmp_path):
    namespace_file = tmp_path / "namespace"
    namespace_file.write_text("test-namespace")
    token_file = tmp_path / "token"
    token_file.write_text("test-token")

    mock_request = mock.MagicMock()
    mock_request.headers = {
        WORKSPACE_HEADER_NAME: "existing-workspace",
    }

    with (
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_NAMESPACE_PATH",
            namespace_file,
        ),
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_TOKEN_PATH",
            token_file,
        ),
    ):
        auth = KubernetesAuth()
        result = auth(mock_request)

        assert result is mock_request
        assert mock_request.headers[WORKSPACE_HEADER_NAME] == "existing-workspace"
        assert mock_request.headers[AUTHORIZATION_HEADER_NAME] == "Bearer test-token"


def test_auth_does_not_override_existing_authorization_header(tmp_path):
    namespace_file = tmp_path / "namespace"
    namespace_file.write_text("test-namespace")
    token_file = tmp_path / "token"
    token_file.write_text("test-token")

    mock_request = mock.MagicMock()
    mock_request.headers = {
        AUTHORIZATION_HEADER_NAME: "existing-auth",
    }

    with (
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_NAMESPACE_PATH",
            namespace_file,
        ),
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_TOKEN_PATH",
            token_file,
        ),
    ):
        auth = KubernetesAuth()
        result = auth(mock_request)

        assert result is mock_request
        assert mock_request.headers[WORKSPACE_HEADER_NAME] == "test-namespace"
        assert mock_request.headers[AUTHORIZATION_HEADER_NAME] == "existing-auth"


def test_auth_raises_when_no_credentials(tmp_path):
    mock_request = mock.MagicMock()
    mock_request.headers = {}

    with (
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_NAMESPACE_PATH",
            tmp_path / "nonexistent",
        ),
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_TOKEN_PATH",
            tmp_path / "nonexistent",
        ),
        mock.patch("kubernetes.config.load_kube_config"),
        mock.patch("kubernetes.config.list_kube_config_contexts", return_value=([], None)),
    ):
        auth = KubernetesAuth()
        with pytest.raises(MlflowException, match="Could not determine Kubernetes credentials"):
            auth(mock_request)


# Tests for KubernetesRequestAuthProvider


def test_provider_get_name_returns_kubernetes():
    provider = KubernetesRequestAuthProvider()
    assert provider.get_name() == "kubernetes"


def test_provider_get_auth_returns_kubernetes_auth():
    provider = KubernetesRequestAuthProvider()
    auth = provider.get_auth()
    assert isinstance(auth, KubernetesAuth)


def test_provider_get_auth_raises_when_kubernetes_not_installed():
    provider = KubernetesRequestAuthProvider()
    with mock.patch.dict("sys.modules", {"kubernetes": None}):
        with pytest.raises(MlflowException, match="kubernetes.*not installed"):
            provider.get_auth()


# Tests for workspace context integration


def test_auth_sets_workspace_context(tmp_path):
    namespace_file = tmp_path / "namespace"
    namespace_file.write_text("test-namespace")
    token_file = tmp_path / "token"
    token_file.write_text("test-token")

    mock_request = mock.MagicMock()
    mock_request.headers = {}

    with (
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_NAMESPACE_PATH",
            namespace_file,
        ),
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_TOKEN_PATH",
            token_file,
        ),
    ):
        auth = KubernetesAuth()
        auth(mock_request)

    assert get_request_workspace() == "test-namespace"


def test_auth_does_not_override_existing_workspace_context(tmp_path):
    namespace_file = tmp_path / "namespace"
    namespace_file.write_text("test-namespace")
    token_file = tmp_path / "token"
    token_file.write_text("test-token")

    mock_request = mock.MagicMock()
    mock_request.headers = {}

    set_workspace("pre-existing-workspace")

    with (
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_NAMESPACE_PATH",
            namespace_file,
        ),
        mock.patch(
            "mlflow.tracking.request_auth.kubernetes_request_auth_provider."
            "_SERVICE_ACCOUNT_TOKEN_PATH",
            token_file,
        ),
    ):
        auth = KubernetesAuth()
        auth(mock_request)

    assert get_request_workspace() == "pre-existing-workspace"


def test_auth_early_return_does_not_set_workspace_context():
    mock_request = mock.MagicMock()
    mock_request.headers = {
        WORKSPACE_HEADER_NAME: "existing-workspace",
        AUTHORIZATION_HEADER_NAME: "existing-auth",
    }

    auth = KubernetesAuth()
    auth(mock_request)

    assert get_request_workspace() is None

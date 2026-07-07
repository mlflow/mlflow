from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("kubernetes")

from kubernetes.config.config_exception import ConfigException

import mlflow.tracking.request_auth.kubernetes_request_auth_provider as _k8s_auth
from mlflow.exceptions import MlflowException
from mlflow.tracking.request_auth.kubernetes_request_auth_provider import (
    AUTHORIZATION_HEADER_NAME,
    WORKSPACE_HEADER_NAME,
    KubernetesAuth,
    KubernetesNamespacedRequestAuthProvider,
    KubernetesRequestAuthProvider,
    _get_namespace,
    _get_token,
    _get_token_from_kubeconfig,
    _read_file_if_exists,
)
from mlflow.utils.workspace_context import get_request_workspace, set_workspace

_MODULE = "mlflow.tracking.request_auth.kubernetes_request_auth_provider"


@pytest.fixture(autouse=True)
def _clear_caches():
    for cache in (
        _k8s_auth._file_cache,
        _k8s_auth._kubeconfig_token_cache,
        _k8s_auth._kubeconfig_namespace_cache,
    ):
        if cache is not None:
            cache.clear()
    yield
    for cache in (
        _k8s_auth._file_cache,
        _k8s_auth._kubeconfig_token_cache,
        _k8s_auth._kubeconfig_namespace_cache,
    ):
        if cache is not None:
            cache.clear()
    set_workspace(None)


@contextmanager
def _patch_service_account(tmp_path, *, namespace=None, token=None):
    ns_path = tmp_path / "sa-namespace"
    tk_path = tmp_path / "sa-token"
    if namespace is not None:
        ns_path.write_text(namespace)
    if token is not None:
        tk_path.write_text(token)
    with (
        mock.patch(f"{_MODULE}._SERVICE_ACCOUNT_NAMESPACE_PATH", ns_path),
        mock.patch(f"{_MODULE}._SERVICE_ACCOUNT_TOKEN_PATH", tk_path),
    ):
        yield


@contextmanager
def _patch_kubeconfig(*, namespace=None, token=None, load_error=None):
    active_context = (
        {"name": "ctx", "context": {"namespace": namespace}}
        if namespace is not None
        else {"name": "ctx", "context": {}}
    )
    mock_api_client = mock.MagicMock()
    mock_api_client.__enter__.return_value = mock_api_client
    mock_api_client.default_headers = {"Authorization": f"Bearer {token}"} if token else {}
    mock_api_client.configuration.api_key = {}
    with (
        mock.patch(
            "kubernetes.config.load_kube_config",
            side_effect=load_error,
        ),
        mock.patch(
            "kubernetes.config.list_kube_config_contexts",
            return_value=([], active_context),
        ),
        mock.patch("kubernetes.client.ApiClient", return_value=mock_api_client),
    ):
        yield


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        ("missing", None),
        ("  content  \n", "content"),
        ("permission_error", None),
    ],
    ids=["missing-file", "strips-whitespace", "permission-error"],
)
def test_read_file_if_exists(tmp_path, setup, expected):
    path = tmp_path / "file"
    if setup == "missing":
        pass
    elif setup == "permission_error":
        path.write_text("x")
        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "read_text", side_effect=PermissionError("denied")),
        ):
            assert _read_file_if_exists(path) is None
            return
    else:
        path.write_text(setup)
    assert _read_file_if_exists(path) == expected


def test_read_file_caches_result(tmp_path):
    path = tmp_path / "cached"
    path.write_text("cached-content")
    with mock.patch.object(Path, "read_text", wraps=path.read_text) as mock_read:
        assert _read_file_if_exists(path) == "cached-content"
        assert _read_file_if_exists(path) == "cached-content"
        mock_read.assert_called_once()


@pytest.mark.parametrize(
    ("sa_namespace", "kubeconfig_namespace", "expected"),
    [
        ("sa-ns", "kube-ns", "sa-ns"),
        (None, "kube-ns", "kube-ns"),
        (None, None, None),
    ],
    ids=["prefers-service-account", "falls-back-to-kubeconfig", "none-available"],
)
def test_get_namespace_source_priority(tmp_path, sa_namespace, kubeconfig_namespace, expected):
    with (
        _patch_service_account(tmp_path, namespace=sa_namespace),
        _patch_kubeconfig(namespace=kubeconfig_namespace),
    ):
        assert _get_namespace() == expected


@pytest.mark.parametrize(
    ("default_headers", "api_key", "expected"),
    [
        ({"Authorization": "Bearer tok"}, {}, "Bearer tok"),
        ({"authorization": "Bearer lower"}, {}, "Bearer lower"),
        ({}, {"authorization": "raw-tok"}, "Bearer raw-tok"),
        ({}, {"authorization": "Bearer prefixed"}, "Bearer prefixed"),
        ({}, {}, None),
    ],
    ids=[
        "default-header-bearer",
        "lowercase-header",
        "api-key-fallback",
        "api-key-strips-bearer",
        "no-token",
    ],
)
def test_token_from_kubeconfig_extraction(default_headers, api_key, expected):
    mock_api_client = mock.MagicMock()
    mock_api_client.__enter__.return_value = mock_api_client
    mock_api_client.default_headers = default_headers
    mock_api_client.configuration.api_key = api_key
    with (
        mock.patch("kubernetes.config.load_kube_config"),
        mock.patch("kubernetes.client.ApiClient", return_value=mock_api_client),
        mock.patch(
            "kubernetes.config.list_kube_config_contexts",
            return_value=([], {"name": "ctx"}),
        ),
    ):
        assert _get_token_from_kubeconfig() == expected


def test_token_from_kubeconfig_returns_none_on_load_failure():
    with mock.patch(
        "kubernetes.config.load_kube_config",
        side_effect=ConfigException("no kubeconfig"),
    ):
        assert _get_token_from_kubeconfig() is None


def test_token_from_kubeconfig_caches_by_context():
    mock_api_client = mock.MagicMock()
    mock_api_client.__enter__.return_value = mock_api_client
    mock_api_client.default_headers = {"Authorization": "Bearer cached"}

    with (
        mock.patch("kubernetes.config.load_kube_config") as mock_load,
        mock.patch("kubernetes.client.ApiClient", return_value=mock_api_client),
        mock.patch(
            "kubernetes.config.list_kube_config_contexts",
            return_value=([], {"name": "ctx"}),
        ),
    ):
        assert _get_token_from_kubeconfig() == "Bearer cached"
        assert _get_token_from_kubeconfig() == "Bearer cached"
        mock_load.assert_called_once()


def test_token_from_kubeconfig_cache_invalidates_on_context_change():
    clients = [mock.MagicMock(), mock.MagicMock()]
    for c in clients:
        c.__enter__.return_value = c
    clients[0].default_headers = {"Authorization": "Bearer token-a"}
    clients[1].default_headers = {"Authorization": "Bearer token-b"}

    with (
        mock.patch("kubernetes.config.load_kube_config") as mock_load,
        mock.patch("kubernetes.client.ApiClient", side_effect=clients),
        mock.patch(
            "kubernetes.config.list_kube_config_contexts",
            side_effect=[
                ([], {"name": "ctx-a"}),
                ([], {"name": "ctx-b"}),
            ],
        ),
    ):
        assert _get_token_from_kubeconfig() == "Bearer token-a"
        assert _get_token_from_kubeconfig() == "Bearer token-b"
        assert mock_load.call_count == 2


@pytest.mark.parametrize(
    ("sa_token", "kubeconfig_token", "expected"),
    [
        ("sa-tok", "kube-tok", "Bearer sa-tok"),
        (None, "kube-tok", "Bearer kube-tok"),
        (None, None, None),
    ],
    ids=["prefers-service-account", "falls-back-to-kubeconfig", "none-available"],
)
def test_get_token_source_priority(tmp_path, sa_token, kubeconfig_token, expected):
    with (
        _patch_service_account(tmp_path, token=sa_token),
        _patch_kubeconfig(
            token=kubeconfig_token,
            load_error=ConfigException("no config") if kubeconfig_token is None else None,
        ),
    ):
        assert _get_token() == expected


@pytest.mark.parametrize(
    (
        "enable_workspaces",
        "initial_headers",
        "sa_namespace",
        "sa_token",
        "expected_workspace",
        "expected_auth",
    ),
    [
        # workspaces disabled — does not inject workspace header
        (False, {}, None, "tok", None, "Bearer tok"),
        (False, {WORKSPACE_HEADER_NAME: "caller-ws"}, None, "tok", "caller-ws", "Bearer tok"),
        (False, {AUTHORIZATION_HEADER_NAME: "preset"}, None, None, None, "preset"),
        # workspaces enabled — both headers
        (True, {}, "ns", "tok", "ns", "Bearer tok"),
        (
            True,
            {WORKSPACE_HEADER_NAME: "pre", AUTHORIZATION_HEADER_NAME: "pre-auth"},
            "ns",
            "tok",
            "pre",
            "pre-auth",
        ),
        (True, {WORKSPACE_HEADER_NAME: "pre"}, None, "tok", "pre", "Bearer tok"),
        (True, {AUTHORIZATION_HEADER_NAME: "pre-auth"}, "ns", None, "ns", "pre-auth"),
    ],
    ids=[
        "disabled-auth-only",
        "disabled-ignores-caller-workspace",
        "disabled-preserves-preset-auth",
        "enabled-both-headers",
        "enabled-both-preset",
        "enabled-workspace-preset-auth-resolved",
        "enabled-auth-preset-workspace-resolved",
    ],
)
def test_auth_headers(
    tmp_path,
    enable_workspaces,
    initial_headers,
    sa_namespace,
    sa_token,
    expected_workspace,
    expected_auth,
):
    request = mock.MagicMock()
    request.headers = dict(initial_headers)

    with _patch_service_account(tmp_path, namespace=sa_namespace, token=sa_token):
        auth = KubernetesAuth(enable_workspaces=enable_workspaces)
        result = auth(request)

    assert result is request
    assert request.headers.get(AUTHORIZATION_HEADER_NAME) == expected_auth
    if expected_workspace is None:
        assert WORKSPACE_HEADER_NAME not in request.headers
    else:
        assert request.headers[WORKSPACE_HEADER_NAME] == expected_workspace


@pytest.mark.parametrize(
    ("enable_workspaces", "initial_headers", "sa_namespace", "sa_token", "error_match"),
    [
        (True, {}, None, "tok", "Could not determine Kubernetes namespace"),
        (
            True,
            {WORKSPACE_HEADER_NAME: "ws"},
            None,
            None,
            "Could not determine Kubernetes credentials",
        ),
        (False, {}, None, None, "Could not determine Kubernetes credentials"),
    ],
    ids=[
        "enabled-missing-namespace",
        "enabled-missing-token",
        "disabled-missing-token",
    ],
)
def test_auth_raises_on_missing_credentials(
    tmp_path, enable_workspaces, initial_headers, sa_namespace, sa_token, error_match
):
    request = mock.MagicMock()
    request.headers = dict(initial_headers)

    with (
        _patch_service_account(tmp_path, namespace=sa_namespace, token=sa_token),
        _patch_kubeconfig(load_error=ConfigException("no config")),
        pytest.raises(MlflowException, match=error_match),
    ):
        KubernetesAuth(enable_workspaces=enable_workspaces)(request)


@pytest.mark.parametrize(
    ("enable_workspaces", "pre_existing_workspace", "expected_context", "expected_ws_header"),
    [
        (True, None, "ns", "ns"),
        (True, "pre-existing", "pre-existing", "pre-existing"),
        (False, None, None, None),
    ],
    ids=["sets-workspace-context", "preserves-existing-context", "disabled-no-context"],
)
def test_auth_workspace_context(
    tmp_path, enable_workspaces, pre_existing_workspace, expected_context, expected_ws_header
):
    if pre_existing_workspace:
        set_workspace(pre_existing_workspace)

    request = mock.MagicMock()
    request.headers = {}

    with _patch_service_account(tmp_path, namespace="ns", token="tok"):
        KubernetesAuth(enable_workspaces=enable_workspaces)(request)

    assert get_request_workspace() == expected_context
    if expected_ws_header is not None:
        assert request.headers[WORKSPACE_HEADER_NAME] == expected_ws_header


@pytest.mark.parametrize(
    ("provider_cls", "expected_name", "expected_workspaces"),
    [
        (KubernetesRequestAuthProvider, "kubernetes", False),
        (KubernetesNamespacedRequestAuthProvider, "kubernetes-namespaced", True),
    ],
    ids=["kubernetes", "kubernetes-namespaced"],
)
def test_provider_registration(provider_cls, expected_name, expected_workspaces):
    provider = provider_cls()
    assert provider.get_name() == expected_name
    auth = provider.get_auth()
    assert isinstance(auth, KubernetesAuth)
    assert auth._enable_workspaces is expected_workspaces


@pytest.mark.parametrize(
    "provider_cls",
    [KubernetesRequestAuthProvider, KubernetesNamespacedRequestAuthProvider],
    ids=["kubernetes", "kubernetes-namespaced"],
)
def test_provider_raises_without_kubernetes(provider_cls):
    with (
        mock.patch.dict("sys.modules", {"kubernetes": None}),
        pytest.raises(MlflowException, match="kubernetes.*not installed"),
    ):
        provider_cls().get_auth()

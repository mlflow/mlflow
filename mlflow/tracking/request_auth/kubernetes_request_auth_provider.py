"""Request auth provider for Kubernetes environments.

This module provides two auth plugins activated via ``MLFLOW_TRACKING_AUTH``:

- ``kubernetes`` — adds only the ``Authorization`` header (bearer token).
- ``kubernetes-namespaced`` — adds both ``Authorization`` and ``X-MLFLOW-WORKSPACE``
  (derived from the Kubernetes namespace).

Requires the ``kubernetes`` package. Install it with:
    pip install mlflow[kubernetes]
"""

import logging
import os
import threading
from pathlib import Path

from mlflow.exceptions import MlflowException
from mlflow.tracking.request_auth.abstract_request_auth_provider import RequestAuthProvider
from mlflow.utils.workspace_context import get_request_workspace, set_workspace
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME

_FILE_CACHE_TTL = 60

_file_cache = None
_file_cache_lock = threading.Lock()

_kubeconfig_token_cache = None
_kubeconfig_token_cache_lock = threading.Lock()

_kubeconfig_namespace_cache = None
_kubeconfig_namespace_cache_lock = threading.Lock()

_logger = logging.getLogger(__name__)

# Kubernetes service account paths
_SERVICE_ACCOUNT_NAMESPACE_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
_SERVICE_ACCOUNT_TOKEN_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")

AUTHORIZATION_HEADER_NAME = "Authorization"


def _ensure_caches():
    global _file_cache, _kubeconfig_token_cache, _kubeconfig_namespace_cache
    if _file_cache is not None:
        return
    from cachetools import TTLCache

    _file_cache = TTLCache(maxsize=10, ttl=_FILE_CACHE_TTL)
    _kubeconfig_token_cache = TTLCache(maxsize=4, ttl=_FILE_CACHE_TTL)
    _kubeconfig_namespace_cache = TTLCache(maxsize=4, ttl=_FILE_CACHE_TTL)


def _cache_lookup(cache, lock, key, compute_fn):
    with lock:
        try:
            return cache[key]
        except KeyError:
            pass
    value = compute_fn()
    with lock:
        try:
            cache[key] = value
        except ValueError:
            pass  # value exceeds maxsize — skip caching, still return it
    return value


def _check_kubernetes_installed():
    try:
        from kubernetes import client, config  # noqa: F401
    except ImportError:
        raise MlflowException(
            "The 'kubernetes' package is required to use the Kubernetes auth providers "
            "(MLFLOW_TRACKING_AUTH=kubernetes or MLFLOW_TRACKING_AUTH=kubernetes-namespaced), "
            "but it is not installed or is incomplete. "
            "Install it with: pip install mlflow[kubernetes]"
        )


def _read_file_uncached(path: Path) -> str | None:
    try:
        if path.exists():
            return path.read_text().strip() or None
    except OSError as e:
        _logger.debug("Could not read file %s: %s", path, e)
    return None


def _read_file_if_exists(path: Path) -> str | None:
    _ensure_caches()
    return _cache_lookup(
        _file_cache, _file_cache_lock, str(path), lambda: _read_file_uncached(path)
    )


def _get_namespace() -> str | None:
    """Get workspace namespace. Tries service account file then kubeconfig context."""
    if namespace := _read_file_if_exists(_SERVICE_ACCOUNT_NAMESPACE_PATH):
        return namespace
    return _get_namespace_from_kubeconfig()


def _get_namespace_from_kubeconfig_uncached() -> str | None:
    from kubernetes import config
    from kubernetes.config.config_exception import ConfigException

    try:
        config.load_kube_config()
    except (OSError, ConfigException) as e:
        _logger.warning("Could not load kubeconfig: %s", e)
        return None

    try:
        _, active_context = config.list_kube_config_contexts()
    except (OSError, ConfigException) as e:
        _logger.warning("Could not list kubeconfig contexts: %s", e)
        return None
    if not active_context:
        return None

    return active_context.get("context", {}).get("namespace", "").strip() or None


def _get_namespace_from_kubeconfig() -> str | None:
    _ensure_caches()
    key = _get_kubeconfig_cache_key()
    return _cache_lookup(
        _kubeconfig_namespace_cache,
        _kubeconfig_namespace_cache_lock,
        key,
        _get_namespace_from_kubeconfig_uncached,
    )


def _get_token() -> str | None:
    """Get authorization token. Tries service account file then kubeconfig."""
    if token := _read_file_if_exists(_SERVICE_ACCOUNT_TOKEN_PATH):
        return f"Bearer {token}"
    return _get_token_from_kubeconfig()


def _get_kubeconfig_cache_key() -> str:
    """Build a cache key from the kubeconfig path and active context name."""
    from kubernetes import config
    from kubernetes.config.config_exception import ConfigException

    kubeconfig_path = os.environ.get("KUBECONFIG", "DEFAULT")
    try:
        _, active_context = config.list_kube_config_contexts()
    except (OSError, ConfigException):
        return kubeconfig_path
    context_name = active_context.get("name", "") if active_context else ""
    return f"{kubeconfig_path}:{context_name}"


def _get_token_from_kubeconfig_uncached() -> str | None:
    from kubernetes import client, config
    from kubernetes.config.config_exception import ConfigException

    try:
        config.load_kube_config()
    except (OSError, ConfigException) as e:
        _logger.warning("Could not load kubeconfig: %s", e)
        return None

    with client.ApiClient() as api_client:
        token = None

        # 1) Try default_headers first (where most auth flows put the resolved token)
        auth = api_client.default_headers.get("Authorization") or api_client.default_headers.get(
            "authorization"
        )
        if isinstance(auth, str) and auth.lower().startswith("bearer "):
            token = auth[7:].strip()

        # 2) Fallback: configuration.api_key (some versions/auth flows store it here)
        if not token:
            api_key = api_client.configuration.api_key.get("authorization")
            if isinstance(api_key, str):
                token = api_key.strip()
                if token.lower().startswith("bearer "):
                    token = token[7:].strip()

        if not token:
            return None

        return f"Bearer {token}"


def _get_token_from_kubeconfig() -> str | None:
    _ensure_caches()
    key = _get_kubeconfig_cache_key()
    return _cache_lookup(
        _kubeconfig_token_cache,
        _kubeconfig_token_cache_lock,
        key,
        _get_token_from_kubeconfig_uncached,
    )


class KubernetesAuth:
    """Custom authentication class for Kubernetes environments.

    This class is callable and will be invoked by the requests library
    to add authentication headers to each request.

    Args:
        enable_workspaces: When True, also adds the X-MLFLOW-WORKSPACE header
            and sets the workspace context. Defaults to False.
    """

    def __init__(self, *, enable_workspaces: bool = False):
        self._enable_workspaces = enable_workspaces

    def __call__(self, request):
        """Add Kubernetes authentication headers to the request.

        Args:
            request: The prepared request object from the requests library.

        Returns:
            The modified request object with authentication headers.

        Raises:
            MlflowException: If workspace or authorization cannot be determined.
        """
        if self._enable_workspaces and WORKSPACE_HEADER_NAME not in request.headers:
            # Prefer an explicitly configured request workspace over the Kubernetes namespace.
            if workspace := get_request_workspace():
                request.headers[WORKSPACE_HEADER_NAME] = workspace
            else:
                namespace = _get_namespace()
                if not namespace:
                    raise MlflowException(
                        "Could not determine Kubernetes namespace. "
                        "Ensure you are running in a Kubernetes pod with a service account "
                        "or have a valid kubeconfig with a namespace set in the active context."
                    )
                request.headers[WORKSPACE_HEADER_NAME] = namespace

        if AUTHORIZATION_HEADER_NAME not in request.headers:
            token = _get_token()
            if not token:
                raise MlflowException(
                    "Could not determine Kubernetes credentials. "
                    "Ensure you are running in a Kubernetes pod with a service account "
                    "or have a valid kubeconfig with credentials set."
                )
            request.headers[AUTHORIZATION_HEADER_NAME] = token

        # Propagate the workspace globally so _log_url includes the workspace
        # query parameter and subprocesses inherit it via the environment variable.
        if self._enable_workspaces and not get_request_workspace():
            set_workspace(request.headers.get(WORKSPACE_HEADER_NAME))

        return request


class KubernetesRequestAuthProvider(RequestAuthProvider):
    """Provides token-only authentication for Kubernetes environments.

    This provider adds only the Authorization header from Kubernetes credentials.
    Use ``MLFLOW_TRACKING_AUTH=kubernetes`` to enable.
    """

    def get_name(self) -> str:
        return "kubernetes"

    def get_auth(self):
        _check_kubernetes_installed()
        return KubernetesAuth(enable_workspaces=False)


class KubernetesNamespacedRequestAuthProvider(RequestAuthProvider):
    """Provides authentication with workspace headers for Kubernetes environments.

    This provider adds both headers based on Kubernetes environment:
    - Authorization: Set from service account token file or kubeconfig credentials
    - X-MLFLOW-WORKSPACE: Set from service account namespace file or kubeconfig context

    Each header is resolved independently — the namespace and token may come from
    different sources. Either header can also be pre-set by the caller.

    Use ``MLFLOW_TRACKING_AUTH=kubernetes-namespaced`` to enable.
    """

    def get_name(self) -> str:
        return "kubernetes-namespaced"

    def get_auth(self):
        _check_kubernetes_installed()
        return KubernetesAuth(enable_workspaces=True)

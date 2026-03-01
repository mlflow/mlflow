"""Request auth provider for Kubernetes environments.

This provider automatically adds workspace and authorization headers when running
in Kubernetes environments or when kubeconfig is available.

To use this provider, set the environment variable:
    MLFLOW_TRACKING_AUTH=kubernetes

Requires the ``kubernetes`` package. Install it with:
    pip install mlflow[kubernetes]
"""

import logging
import threading
from pathlib import Path

from cachetools import TTLCache, cached

from mlflow.exceptions import MlflowException
from mlflow.tracking.request_auth.abstract_request_auth_provider import RequestAuthProvider
from mlflow.utils.workspace_context import get_request_workspace, set_workspace

# Cache for file reads (1 minute TTL)
_FILE_CACHE_TTL = 60
_file_cache: TTLCache = TTLCache(maxsize=10, ttl=_FILE_CACHE_TTL)
_file_cache_lock = threading.Lock()

# Cache for kubeconfig token (1 minute TTL) — avoids repeated exec-based auth (EKS, GKE, AKS)
_kubeconfig_token_cache: TTLCache = TTLCache(maxsize=1, ttl=_FILE_CACHE_TTL)
_kubeconfig_token_cache_lock = threading.Lock()

_logger = logging.getLogger(__name__)

# Kubernetes service account paths
_SERVICE_ACCOUNT_NAMESPACE_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
_SERVICE_ACCOUNT_TOKEN_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")

# Header names
WORKSPACE_HEADER_NAME = "X-MLFLOW-WORKSPACE"
AUTHORIZATION_HEADER_NAME = "Authorization"


def _check_kubernetes_installed():
    try:
        from kubernetes import client, config  # noqa: F401
    except ImportError:
        raise MlflowException(
            "The 'kubernetes' package is required to use the Kubernetes auth provider "
            "(MLFLOW_TRACKING_AUTH=kubernetes), but it is not installed or is incomplete. "
            "Install it with: pip install mlflow[kubernetes]"
        )


@cached(cache=_file_cache, lock=_file_cache_lock, key=lambda path: str(path))
def _read_file_if_exists(path: Path) -> str | None:
    """Read a file and return its contents stripped, or None if it doesn't exist."""
    try:
        if path.exists():
            return path.read_text().strip() or None
    except OSError as e:
        _logger.debug("Could not read file %s: %s", path, e)
    return None


def _get_namespace() -> str | None:
    """Get workspace namespace. Tries service account file then kubeconfig context."""
    if namespace := _read_file_if_exists(_SERVICE_ACCOUNT_NAMESPACE_PATH):
        return namespace
    return _get_namespace_from_kubeconfig()


def _get_namespace_from_kubeconfig() -> str | None:
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


def _get_token() -> str | None:
    """Get authorization token. Tries service account file then kubeconfig."""
    if token := _read_file_if_exists(_SERVICE_ACCOUNT_TOKEN_PATH):
        return f"Bearer {token}"
    return _get_token_from_kubeconfig()


# No-arg function: default hashkey produces () as the cache key, giving a single cached entry.
@cached(cache=_kubeconfig_token_cache, lock=_kubeconfig_token_cache_lock)
def _get_token_from_kubeconfig() -> str | None:
    from kubernetes import client, config
    from kubernetes.config.config_exception import ConfigException

    try:
        config.load_kube_config()
    except (OSError, ConfigException) as e:
        _logger.warning("Could not load kubeconfig: %s", e)
        return None

    api_client = client.ApiClient()
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


class KubernetesAuth:
    """Custom authentication class for Kubernetes environments.

    This class is callable and will be invoked by the requests library
    to add authentication headers to each request.
    """

    def __call__(self, request):
        """Add Kubernetes authentication headers to the request.

        Args:
            request: The prepared request object from the requests library.

        Returns:
            The modified request object with authentication headers.

        Raises:
            MlflowException: If workspace or authorization cannot be determined.
        """
        if WORKSPACE_HEADER_NAME not in request.headers:
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

        if not get_request_workspace():
            set_workspace(request.headers.get(WORKSPACE_HEADER_NAME))

        return request


class KubernetesRequestAuthProvider(RequestAuthProvider):
    """Provides authentication for Kubernetes environments.

    This provider adds headers based on Kubernetes environment:
    - X-MLFLOW-WORKSPACE: Set from service account namespace file or kubeconfig context
    - Authorization: Set from service account token file or kubeconfig credentials

    Each header is resolved independently — the namespace and token may come from
    different sources. Either header can also be pre-set by the caller.

    To enable this provider, set:
        MLFLOW_TRACKING_AUTH=kubernetes
    """

    def get_name(self) -> str:
        return "kubernetes"

    def get_auth(self):
        _check_kubernetes_installed()
        return KubernetesAuth()

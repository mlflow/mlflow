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


def _get_credentials_from_service_account() -> tuple[str, str] | None:
    """Get namespace and token from mounted service account files.

    Returns:
        Tuple of (namespace, token) if both are available, None otherwise.
    """
    namespace = _read_file_if_exists(_SERVICE_ACCOUNT_NAMESPACE_PATH)
    token = _read_file_if_exists(_SERVICE_ACCOUNT_TOKEN_PATH)

    if namespace and token:
        return namespace, f"Bearer {token}"
    return None


def _get_credentials_from_kubeconfig() -> tuple[str, str] | None:
    """Get namespace and token from kubeconfig.

    Uses ApiClient so kubeconfig exec auth is resolved when possible
    (EKS, GKE, AKS, OpenShift, OIDC).

    Returns:
        Tuple of (namespace, token) if both are available, None otherwise.
    """
    from kubernetes import client, config
    from kubernetes.config.config_exception import ConfigException

    try:
        config.load_kube_config()
    except (OSError, ConfigException) as e:
        _logger.warning("Could not load kubeconfig: %s", e)
        return None

    # Get namespace from context
    try:
        _, active_context = config.list_kube_config_contexts()
    except (OSError, ConfigException) as e:
        _logger.warning("Could not list kubeconfig contexts: %s", e)
        return None
    if not active_context:
        return None

    namespace = active_context.get("context", {}).get("namespace", "").strip() or None
    if not namespace:
        return None

    # Get token from ApiClient
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

    return namespace, f"Bearer {token}"


def _get_credentials() -> tuple[str, str] | None:
    """Get workspace and authorization credentials.

    Tries service account files first, then falls back to kubeconfig.
    Both values must come from the same source for consistency.

    Returns:
        Tuple of (namespace, authorization) if available, None otherwise.
    """
    # Try service account files first (running in a pod)
    if creds := _get_credentials_from_service_account():
        return creds

    # Fallback to kubeconfig
    return _get_credentials_from_kubeconfig()


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
        # Skip if both headers are already set
        if (
            WORKSPACE_HEADER_NAME in request.headers
            and AUTHORIZATION_HEADER_NAME in request.headers
        ):
            return request

        credentials = _get_credentials()
        if not credentials:
            raise MlflowException(
                "Could not determine Kubernetes credentials. "
                "Ensure you are running in a Kubernetes pod with a service account "
                "or have a valid kubeconfig with a namespace and credentials set."
            )

        namespace, authorization = credentials

        if WORKSPACE_HEADER_NAME not in request.headers:
            request.headers[WORKSPACE_HEADER_NAME] = namespace

        if AUTHORIZATION_HEADER_NAME not in request.headers:
            request.headers[AUTHORIZATION_HEADER_NAME] = authorization

        if not get_request_workspace():
            set_workspace(request.headers.get(WORKSPACE_HEADER_NAME, namespace))

        return request


class KubernetesRequestAuthProvider(RequestAuthProvider):
    """Provides authentication for Kubernetes environments.

    This provider adds headers based on Kubernetes environment:
    - X-MLFLOW-WORKSPACE: Set from service account namespace file or kubeconfig context
    - Authorization: Set from service account token file or kubeconfig credentials

    Credentials are sourced consistently - either both from mounted service account
    files (when running in a pod) or both from kubeconfig (when running locally).

    To enable this provider, set:
        MLFLOW_TRACKING_AUTH=kubernetes
    """

    def get_name(self) -> str:
        return "kubernetes"

    def get_auth(self):
        _check_kubernetes_installed()
        return KubernetesAuth()

import warnings

from mlflow.utils.plugins import get_entry_points

REQUEST_AUTH_PROVIDER_ENTRYPOINT = "mlflow.request_auth_provider"


class RequestAuthProviderRegistry:
    def __init__(self):
        self._registry = []

    def register(self, request_auth_provider):
        self._registry.append(request_auth_provider())

    def register_entrypoints(self):
        for entrypoint in get_entry_points(REQUEST_AUTH_PROVIDER_ENTRYPOINT):
            try:
                self.register(entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    'Failure attempting to register request auth provider "{}": {}'.format(
                        entrypoint.name, str(exc)
                    ),
                    stacklevel=2,
                )

    def __iter__(self):
        return iter(self._registry)


_request_auth_provider_registry = RequestAuthProviderRegistry()
_request_auth_provider_registry.register_entrypoints()


def fetch_auth(request_auth):
    """
    Find the request auth from registered providers based on the auth provider's name.
    The auth provider's name can be provided through environment variable `MLFLOW_TRACKING_AUTH`.

    This function iterates through all request auth providers in the registry. Additional context
    providers can be registered as described in
    :py:class:`mlflow.tracking.request_auth.RequestAuthProvider`.

    Args:
        request_auth: The name of request auth provider.

    Returns:
        The auth object.
    """

    for auth_provider in _request_auth_provider_registry:
        if auth_provider.get_name() == request_auth:
            return auth_provider.get_auth()

    warnings.warn(
        f"Could not find any registered plugin for {request_auth}. "
        "No authentication header will be added. Please check your "
        "provider documentation for installing the right plugin or "
        "correct provider name."
    )

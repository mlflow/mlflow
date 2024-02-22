from abc import ABC, abstractmethod

from mlflow.utils.annotations import developer_stable


@developer_stable
class RequestAuthProvider(ABC):
    """
    Abstract base class for specifying custom request auth to add to outgoing requests

    When a request is sent, MLflow will iterate through all registered RequestAuthProviders.
    For each provider where ``get_name`` matches auth provider name, MLflow calls the ``get_auth``
    method on the provider to compute request auth.

    The resulting request auth will then be added and sent with the request.
    """

    @abstractmethod
    def get_name(self):
        """Get the name of the request auth provider.

        Returns:
            str of request auth provider name.
        """
        pass

    @abstractmethod
    def get_auth(self):
        """
        Generate request auth object (e.g., `requests.auth import HTTPBasicAuth`). See
        https://requests.readthedocs.io/en/latest/user/authentication/ for more details.

        Returns:
            request auth object.
        """
        pass

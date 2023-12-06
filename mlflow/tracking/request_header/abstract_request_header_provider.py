from abc import ABCMeta, abstractmethod

from mlflow.utils.annotations import developer_stable


@developer_stable
class RequestHeaderProvider(metaclass=ABCMeta):
    """
    Abstract base class for specifying custom request headers to add to outgoing requests
    (e.g. request headers specifying the environment from which mlflow is running).

    When a request is sent, MLflow will iterate through all registered RequestHeaderProviders.
    For each provider where ``in_context`` returns ``True``, MLflow calls the ``request_headers``
    method on the provider to compute request headers.

    All resulting request headers will then be merged together and sent with the request.
    """

    @abstractmethod
    def in_context(self):
        """
        Determine if MLflow is running in this context.

        :return: bool indicating if in this context
        """
        pass

    @abstractmethod
    def request_headers(self):
        """
        Generate context-specific request headers.

        :return: dict of request headers
        """
        pass

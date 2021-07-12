from abc import ABCMeta, abstractmethod


class RequestHeaderProvider(object):
    """
    Abstract base class for specifying custom request headers to add to outgoing requests
    (e.g. request headers specifying the environment from which mlflux is running).

    When a request is sent, mlflux will iterate through all registered RequestHeaderProviders.
    For each provider where ``in_context`` returns ``True``, mlflux calls the ``request_headers``
    method on the provider to compute request headers.

    All resulting request headers will then be merged together and sent with the request.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def in_context(self):
        """
        Determine if mlflux is running in this context.

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

from mlflow.tracking.request_header.abstract_request_header_provider import RequestHeaderProvider
from mlflow import __version__

_USER_AGENT = "User-Agent"
_DEFAULT_HEADERS = {_USER_AGENT: "mlflow-python-client/%s" % __version__}


class DefaultRequestHeaderProvider(RequestHeaderProvider):
    """
    Provides default request headers for outgoing request.
    """

    def in_context(self):
        return True

    def request_headers(self):
        return dict(**_DEFAULT_HEADERS)

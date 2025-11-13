from mlflow import __version__
from mlflow.tracking.request_header.abstract_request_header_provider import RequestHeaderProvider

_USER_AGENT = "User-Agent"
_CLIENT_VERSION = "X-MLflow-Client-Version"
# We need to specify client version in separate header as user agent is overwritten in SDK call path
_DEFAULT_HEADERS = {
    _USER_AGENT: f"mlflow-python-client/{__version__}",
    _CLIENT_VERSION: f"{__version__}",
}


class DefaultRequestHeaderProvider(RequestHeaderProvider):
    """
    Provides default request headers for outgoing request.
    """

    def in_context(self):
        return True

    def request_headers(self):
        return dict(**_DEFAULT_HEADERS)

from mlflow.tracking.request_header.abstract_request_header_provider import RequestHeaderProvider


class PluginRequestHeaderProvider(RequestHeaderProvider):
    """RequestHeaderProvider provided through plugin system"""

    def in_context(self):
        return False

    def request_headers(self):
        return {"test": "header"}

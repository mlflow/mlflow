from mlflow.tracking.request_auth.abstract_request_auth_provider import RequestAuthProvider


class PluginRequestAuthProvider(RequestAuthProvider):
    """RequestAuthProvider provided through plugin system"""

    def get_name(self):
        return "test_auth_provider_name"

    def get_auth(self):
        return {"auth_name": "test_auth_provider_name"}

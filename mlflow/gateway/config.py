from mlflow import MlflowException


class GatewayConfigSingleton:
    _instance = None
    _gateway_config = None

    @staticmethod
    def getInstance():
        """Static access method."""
        if GatewayConfigSingleton._instance is None:
            GatewayConfigSingleton()
        return GatewayConfigSingleton._instance

    def __init__(self):
        """Virtually private constructor."""
        if GatewayConfigSingleton._instance is not None:
            raise MlflowException("The GatewayConfigSingleton can only be instantiated once.")
        else:
            GatewayConfigSingleton._instance = self

    @property
    def gateway_config(self):
        return self._gateway_config

    @gateway_config.setter
    def gateway_config(self, config):
        self._gateway_config = config

    def update_config(self, config):
        self.gateway_config = config

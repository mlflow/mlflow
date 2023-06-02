from multiprocessing import Process
from typing import Optional
from uvicorn import Server, Config
from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import _CONFIGURATION_FILE
from mlflow.gateway.config import GatewayConfigSingleton
from mlflow.gateway.handlers import _load_gateway_config, RouteConfig
from mlflow.protos.databricks_pb2 import BAD_REQUEST


class GatewayServer(Process):
    """
    Process management wrapper around the uvicorn server process to provide API-driven
    control over the initialized process
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.server = Server(config)

    def stop(self):
        """
        Stops the uvicorn server by terminating its process
        """
        self.terminate()

    def run(self):
        """
        Starts the uvicorn server
        """
        self.server.run()


class ServerManager:
    def __init__(self, app, host, port):
        self.app = app
        self.host = host
        self.port = port
        self.gateway_config = None
        self.server_process = None

    def _validate_config(self, config_path):
        self.gateway_config = _load_gateway_config(config_path)
        # Update the global singleton config with route definitions
        GatewayConfigSingleton.getInstance().update_config(self.gateway_config)

    def start_server(self, config_path, **uvicorn_kwargs):
        server_config = Config(app=self.app, host=self.host, port=self.port, **uvicorn_kwargs)

        self._validate_config(config_path)
        self.server_process = GatewayServer(config=server_config)
        self.server_process.start()

    def _stop_server(self):
        if self.server_process:
            self.server_process.stop()
            self.server_process = None

    def update_server(self, config_path, **uvicorn_kwargs):
        if self.server_process:
            self._stop_server()
            self.start_server(config_path, **uvicorn_kwargs)
        else:
            raise MlflowException(
                "No server to update. Please start the server before trying to update.",
                error_code=BAD_REQUEST,
            )


# Global server manager
server_manager: Optional[ServerManager] = None


def start_service(config_path: str, app: str, host: str, port: int, **uvicorn_kwargs):
    global server_manager
    server_manager = ServerManager(app, host, port)
    server_manager.start_server(config_path, **uvicorn_kwargs)


def update_service(config_path: str, **uvicorn_kwargs):
    global server_manager
    if server_manager is not None:
        server_manager.update_server(config_path, **uvicorn_kwargs)
    else:
        raise Exception("No server to update. Please start the server before trying to update.")

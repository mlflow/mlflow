from multiprocessing import Process
import pathlib
import time
from typing import Optional
from uvicorn import Server, Config
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import _CONFIGURATION_FILE
from mlflow.gateway.handlers import _load_gateway_config
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


class ConfigHandler(FileSystemEventHandler):
    def __init__(self, config_path, app, host, port):
        self.config_path = config_path
        self.app = app
        self.host = host
        self.port = port
        self.gateway_config = None
        self.server_process = None
        self.restart_server()

    def _validate_config(self):
        self.gateway_config = _load_gateway_config(self.config_path)

        # TODO: run validations of the provided configuration and issue a warning if the config
        # is invalid or required fields are not set
        # TODO: this is a placeholder for now

        return Config(app=self.app, host=self.host, port=self.port)

    def restart_server(self):
        config = self._validate_config()
        if self.server_process:
            self.server_process.stop()
        self.server_process = GatewayServer(config=config)
        self.server_process.run()

    def on_modified(self, event):
        if not event.is_directory and event.src_path == self.config_path:
            self.restart_server()


# Global state managers
observer: Optional[Observer] = None
handler: Optional[ConfigHandler] = None


def _start_gateway(config_path: str, app: str, host: str, port: int, poll_interval: int = 1):
    """
    Initiate the gateway service and watchdog monitoring on the configuration path location
    """
    global observer
    global handler

    if observer is not None:
        raise MlflowException(
            "Unable to start an already running gateway server. "
            "Please stop the gateway if you would like to start it again.",
            error_code=BAD_REQUEST,
        )

    handler = ConfigHandler(config_path, app, host, port)
    observer = Observer()
    observer_directory = str(pathlib.Path(config_path).parent)
    observer.schedule(handler, path=observer_directory, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        _stop_gateway()


def _stop_gateway():
    """
    Stop the gateway service and associated watchdog services
    """
    global observer
    global handler

    if observer is not None:
        observer.stop()
        observer.join()
        observer = None
    if handler is not None and handler.server_process is not None:
        # Gracefully terminate the process
        handler.server_process.stop()
        handler = None

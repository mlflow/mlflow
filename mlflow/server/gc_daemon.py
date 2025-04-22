import time
from logging import getLogger
from multiprocessing import Process
from typing import Final

from mlflow.utils.process import ShellCommandException, _exec_cmd

_logger = getLogger(__name__)

MAX_CONSECUTIVE_FAILURES: Final = 10


def run_gc_daemon(interval: int, backend_store_uri: str, artifacts_destination: str) -> Process:
    """
    Starts a background process that periodically runs the mlflow gc command.

    Args:
        interval: The interval (in seconds) at which to run garbage collection.
        backend_store_uri: URI of the backend store from which to delete runs.
        artifacts_destination: The base artifact location from which to resolve
            artifact upload/download/list requests.
    """

    def _gc_process(interval, backend_store_uri, artifacts_destination):
        gc_cmd = [
            "mlflow",
            "gc",
            "--backend-store-uri",
            backend_store_uri,
            "--artifacts-destination",
            artifacts_destination,
        ]
        failures = 0
        while failures < MAX_CONSECUTIVE_FAILURES:
            time.sleep(interval)
            try:
                _exec_cmd(gc_cmd, throw_on_error=True, stream_output=True)
                failures = 0
            except ShellCommandException:
                failures += 1
                _logger.exception(
                    "Automatic garbage collection failed (%s/%s)",
                    failures,
                    MAX_CONSECUTIVE_FAILURES,
                )

        _logger.error("Automatic garbage collection disabled after repeated failures")

    gc_process = Process(
        target=_gc_process, args=(interval, backend_store_uri, artifacts_destination)
    )
    gc_process.daemon = True
    gc_process.start()
    return gc_process

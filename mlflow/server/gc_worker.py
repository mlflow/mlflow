import logging
import subprocess
import sys
import threading
import time

_logger = logging.getLogger(__name__)


def _gc_once(older_than: str) -> None:
    """Run ``mlflow gc`` once via a subprocess."""
    cmd = [sys.executable, "mlflow", "gc"]
    if older_than:
        cmd += ["--older-than", older_than]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        _logger.debug("Failed to run mlflow gc: %s", exc)


def start_gc_worker(interval: int, older_than: str) -> threading.Thread:
    """Start a background thread that runs garbage collection periodically."""

    def loop():
        while True:
            try:
                _gc_once(older_than)
            except Exception as exc:  # pragma: no cover
                _logger.debug("Error running mlflow gc: %s", exc)
            time.sleep(interval)

    thread = threading.Thread(target=loop, daemon=True, name="MLflowGCWorker")
    thread.start()
    _logger.info("Started MLflow GC worker with interval %s seconds", interval)
    return thread

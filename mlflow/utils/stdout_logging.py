import sys
import threading
import time
from contextlib import contextmanager
from io import StringIO

import mlflow


class TeeStringIO:
    """A file-like object that writes to both original stdout and a StringIO buffer."""

    def __init__(self, original_stdout, string_buffer):
        self.original_stdout = original_stdout
        self.string_buffer = string_buffer

    def write(self, data):
        # Write to both original stdout and our buffer
        self.original_stdout.write(data)
        self.string_buffer.write(data)
        return len(data)

    def flush(self):
        self.original_stdout.flush()
        self.string_buffer.flush()

    def __getattr__(self, name):
        # Delegate other attributes to original stdout
        return getattr(self.original_stdout, name)


@contextmanager
def log_stdout_stream(interval_seconds=5):
    """
    A context manager to stream stdout to an MLflow artifact.

    This context manager redirects `sys.stdout` to an in-memory buffer.
    A background thread periodically flushes this buffer and logs its
    contents to an MLflow artifact file named 'stdout.log'.

    Args:
        interval_seconds (int): The interval in seconds at which to log
                                the stdout buffer to MLflow.

    Example:
        import time
        import mlflow

        with mlflow.start_run():
            with log_stdout_stream():
                print("This is the start of my script.")
                time.sleep(6)
                print("This message will appear in the first log upload.")
                time.sleep(6)
                print("And this will be in the second.")
            # The context manager will automatically handle final log upload
            # and cleanup.
        print("Stdout is now back to normal.")
    """
    if not mlflow.active_run():
        raise RuntimeError("An active MLflow run is required to stream stdout.")

    original_stdout = sys.stdout
    stdout_buffer = StringIO()
    tee_stdout = TeeStringIO(original_stdout, stdout_buffer)
    sys.stdout = tee_stdout

    stop_event = threading.Event()
    log_thread = None

    def _log_loop():
        while not stop_event.is_set():
            time.sleep(interval_seconds)
            _log_current_stdout()

    def _log_current_stdout():
        content = stdout_buffer.getvalue()

        if content:
            mlflow.log_text(content, "stdout.log")

    try:
        log_thread = threading.Thread(target=_log_loop, name="mlflow-stdout-logging")
        log_thread.daemon = True
        log_thread.start()
        yield
    finally:
        if log_thread:
            stop_event.set()
            log_thread.join()

        # Final flush and log to capture any remaining output
        _log_current_stdout()

        # Restore stdout
        sys.stdout = original_stdout
        stdout_buffer.close()

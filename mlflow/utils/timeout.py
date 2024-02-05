import signal
from contextlib import contextmanager

from mlflow.utils.process import _IS_UNIX


class MLflowTimeoutError(Exception):
    pass


@contextmanager
def run_with_timeout(seconds):
    """
    Context manager to runs a block of code with a timeout. If the block of code takes longer
    than `seconds` to execute, a `TimeoutError` is raised.
    NB: This function uses Unix signals to implement the timeout, so it is not thread-safe.
    Also it does not work on non-Unix platforms such as Windows.

    E.g.
        ```
        with run_with_timeout(5):
            model.predict(data)
        ```
    """
    assert _IS_UNIX, "Timeouts are not implemented yet for non-Unix platforms"

    def signal_handler(signum, frame):
        raise MLflowTimeoutError(f"Operation timed out after {seconds} seconds")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)  # Disable the alarm after the operation completes or times out

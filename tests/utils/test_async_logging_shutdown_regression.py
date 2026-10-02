"""
Process-level regression test for mlflow/mlflow#20575.

Async-logging worker pools used to be `concurrent.futures.ThreadPoolExecutor`s,
whose workers are non-daemon and tracked in
`concurrent.futures.thread._threads_queues`. Python's `_python_exit` joins those
workers with no timeout *before* any `atexit.register` hook runs, so a worker
blocked on network I/O made the interpreter hang and never reach
`_at_exit_callback`. The fix replaces the worker pools with daemon threads via
`mlflow.utils.async_logging.daemon_thread_pool.DaemonThreadPool`.

This test launches a subprocess that submits a job that sleeps for 120s and
then exits. With the fix, the process exits cleanly within a few seconds.
Without the fix, it hangs and the subprocess timeout fires.
"""

import subprocess
import sys
import textwrap
import time

_HANG_WORKER_SCRIPT = textwrap.dedent(
    """
    import threading
    import time

    import mlflow.utils.async_logging.async_logging_queue as alq
    alq._AT_EXIT_TIMEOUT_SECONDS = 1.0

    from mlflow.utils.async_logging.async_logging_queue import AsyncLoggingQueue
    from mlflow.entities.metric import Metric

    started = threading.Event()


    def blocking(run_id, metrics, params, tags):
        started.set()
        time.sleep(120)


    q = AsyncLoggingQueue(logging_func=blocking)
    q.activate()
    q.log_batch_async(
        run_id="r",
        metrics=[Metric(key="m", value=1.0, timestamp=int(time.time() * 1000), step=0)],
        params=[],
        tags=[],
    )
    assert started.wait(timeout=5)
    print("READY", flush=True)
    """
)


def test_subprocess_exits_when_async_worker_blocked():
    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", _HANG_WORKER_SCRIPT],
        capture_output=True,
        text=True,
        timeout=15,
    )
    elapsed = time.monotonic() - start

    assert result.returncode == 0, (
        f"subprocess exited with code {result.returncode}\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}"
    )
    assert "READY" in result.stdout
    assert elapsed < 15, f"subprocess took {elapsed:.2f}s, expected < 15s"

import errno
import json
import multiprocessing
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class JobResult:
    succeeded: bool
    result: str | None = None  # serialized JSON string
    is_transient_error: bool | None = None
    error: str | None = None

    @classmethod
    def from_error(cls, e: Exception) -> "JobResult":
        from mlflow.server.jobs import TransientError

        if isinstance(e, TransientError):
            return JobResult(succeeded=False, is_transient_error=True, error=repr(e.origin_error))
        return JobResult(
            succeeded=False,
            is_transient_error=False,
            error=repr(e),
        )


def _exit_when_orphaned(poll_interval: float = 1) -> None:
    while True:
        if os.getppid() == 1:
            os._exit(1)
        time.sleep(poll_interval)


def _job_subproc_entry(
    func: Callable[..., Any],
    kwargs: dict[str, Any],
    result_queue: multiprocessing.Queue,
) -> None:
    """Child process entrypoint: run func and put result or exception into queue."""

    # ensure the subprocess is killed when parent process dies.
    threading.Thread(
        target=_exit_when_orphaned,
        name="exit_when_orphaned",
        daemon=True,
    ).start()

    try:
        value = func(**kwargs)
        result_queue.put(
            JobResult(
                succeeded=True,
                result=json.dumps(value),
            )
        )
    except Exception as e:
        # multiprocess uses pickle which can't serialize any kind of python objects.
        # so serialize exception class to serializable JobResult before putting it to result queue.
        result_queue.put(JobResult.from_error(e))


def execute_function_with_timeout(
    func: Callable[..., Any],
    kwargs: dict[str, Any],
    timeout: float | None = None,
) -> JobResult:
    """
    Run `func(**kwargs)` in a spawned subprocess.
    Returns an instance of `JobResult`.

    Raises:
      - TimeoutError if not finished within `timeout`
    """
    if timeout:
        # NOTE: Use 'spawn' instead of 'fork' because
        #  we should avoid forking sqlalchemy engine,
        #  otherwise connection pool, sockets, locks used by the sqlalchemy engine are forked
        #  and deadlock / race conditions might occur.
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue(maxsize=1)
        subproc = ctx.Process(target=_job_subproc_entry, args=(func, kwargs, result_queue))
        subproc.daemon = True
        subproc.start()

        subproc.join(timeout=timeout)
        if not subproc.is_alive():
            return result_queue.get()

        # timeout case
        subproc.kill()
        subproc.join()
        raise TimeoutError()

    try:
        raw_result = func(**kwargs)
        return JobResult(succeeded=True, result=json.dumps(raw_result))
    except Exception as e:
        return JobResult.from_error(e)


def is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # doesn't actually kill
    except OSError as e:
        if e.errno == errno.ESRCH:  # No such process
            return False
        elif e.errno == errno.EPERM:  # Process exists, but no permission
            return True
        else:
            raise
    else:
        return True

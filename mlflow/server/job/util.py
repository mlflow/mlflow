from typing import Any, Callable
import multiprocessing
import os
import signal
from dataclasses import dataclass
import json
from mlflow.utils import _get_fully_qualified_class_name


@dataclass
class JobResult:
    succeeded: bool
    result: str | None = None  # serialized JSON string
    error_class: str | None = None
    error: str | None = None


def _job_subproc_entry(
    func: Callable[..., Any],
    kwargs: dict[str, Any],
    result_queue,
) -> None:
    """Child process entrypoint: run func and put result or exception into queue."""

    try:
        value = func(**kwargs)
        result_queue.put(JobResult(
            succeeded=True,
            result=json.dumps(value),
        ))
    except Exception as e:
        # multiprocess uses pickle which can't serialize any kind of python objects.
        # so serialize exception class to string before putting it to result queue.
        result_queue.put(JobResult(
            succeeded=False,
            error_class=_get_fully_qualified_class_name(e),
            error=str(e),
        ))


def _hard_kill(proc) -> None:
    """Try hard-kill on Unix if terminate() didn't end it."""
    try:
        os.kill(proc.pid, signal.SIGKILL)
    except Exception:
        pass


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
        _hard_kill(subproc)
        raise TimeoutError()

    try:
        raw_result = func(**kwargs)
        return JobResult(
            succeeded=True,
            result=json.dumps(raw_result)
        )
    except Exception as e:
        return JobResult(
            succeeded=False,
            error_class=_get_fully_qualified_class_name(e),
            error=repr(e),
        )

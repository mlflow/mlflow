import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from mlflow.tracing.export.async_export_queue import AsyncTraceExportQueue, Task


def test_async_queue_handle_tasks():
    queue = AsyncTraceExportQueue()

    counter = 0

    def _increment(delta):
        nonlocal counter
        counter += delta

    for _ in range(10):
        task = Task(handler=_increment, args=(1,))
        queue.put(task)

    queue.flush(terminate=True)
    assert counter == 10


@mock.patch("atexit.register")
def test_async_queue_activate_thread_safe(mock_atexit):
    queue = AsyncTraceExportQueue()

    def _count_threads():
        main_thread = threading.main_thread()
        return sum(
            t.is_alive()
            for t in threading.enumerate()
            if t is not main_thread and t.getName().startswith("MLflowTraceLogging")
        )

    # 1. Validate activation
    with ThreadPoolExecutor(max_workers=10) as executor:
        for _ in range(10):
            executor.submit(queue.activate)

    assert queue.is_active()
    assert _count_threads() > 0  # Logging thread + max 5 worker threads
    mock_atexit.assert_called_once()
    mock_atexit.reset_mock()

    # 2. Validate flush (continue)
    queue.flush(terminate=False)
    assert queue.is_active()
    assert _count_threads() > 0  # New threads should be created
    mock_atexit.assert_not_called()  # Exit callback should not be registered again

    # 3. Validate flush with termination
    with ThreadPoolExecutor(max_workers=10) as executor:
        for _ in range(10):
            executor.submit(queue.flush(terminate=True))

    assert not queue.is_active()
    assert _count_threads() == 0


def test_async_queue_drop_task_when_full(monkeypatch):
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_QUEUE_SIZE", "3")
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_WORKERS", "1")

    queue = AsyncTraceExportQueue()

    processed_tasks = 0

    # Create a slow handler to keep tasks in the queue
    def slow_handler():
        time.sleep(0.5)

        nonlocal processed_tasks
        processed_tasks += 1

    for _ in range(10):
        task = Task(handler=slow_handler, args=())
        queue.put(task)

    queue.flush(terminate=True)

    # One more task than the queue size might be processed, because the first task
    # can be drained from the queue immediately, which creates a slot for another task
    assert processed_tasks <= 4

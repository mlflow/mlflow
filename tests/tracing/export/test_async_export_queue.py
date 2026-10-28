import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest

from mlflow.tracing.export.async_export_queue import AsyncTraceExportQueue, Task

from tests.tracing.helper import skip_when_testing_trace_sdk


def test_async_queue_handle_tasks():
    queue = AsyncTraceExportQueue()

    counter = 0

    def increment(delta):
        nonlocal counter
        counter += delta

    for _ in range(10):
        task = Task(handler=increment, args=(1,))
        queue.put(task)

    queue.flush(terminate=True)
    assert counter == 10


def exporter_process(counter):
    # This process exits before waiting for the tasks to finish

    queue = AsyncTraceExportQueue()

    def increment(counter):
        time.sleep(1)
        with counter.get_lock():
            counter.value += 1

    for _ in range(10):
        task = Task(handler=increment, args=(counter,))
        queue.put(task)


@skip_when_testing_trace_sdk
def test_async_queue_complete_task_process_finished():
    multiprocessing.set_start_method("spawn", force=True)
    counter = multiprocessing.Value("i", 0)
    process = multiprocessing.Process(target=exporter_process, args=(counter,))
    process.start()
    process.join(timeout=15)

    assert counter.value == 10


def test_async_queue_activate_thread_safe():
    with mock.patch("atexit.register") as mock_atexit:
        queue = AsyncTraceExportQueue()

        def count_threads():
            main_thread = threading.main_thread()
            return sum(
                t.is_alive()
                for t in threading.enumerate()
                if t is not main_thread and t.name.startswith("MLflowTraceLogging")
            )

        # 1. Validate activation
        with ThreadPoolExecutor(
            max_workers=10, thread_name_prefix="test-async-export-queue-activate"
        ) as executor:
            for _ in range(10):
                executor.submit(queue.activate)
        assert count_threads() > 0  # Logging thread + max 5 worker threads
        mock_atexit.assert_called_once()
        mock_atexit.reset_mock()

        # 2. Validate flush (continue)
        queue.flush(terminate=False)
        assert queue.is_active()
        assert count_threads() > 0  # New threads should be created
        mock_atexit.assert_not_called()  # Exit callback should not be registered again

        # 3. Validate flush with termination
        with ThreadPoolExecutor(
            max_workers=10, thread_name_prefix="test-async-export-queue-flush"
        ) as executor:
            for _ in range(10):
                executor.submit(queue.flush(terminate=True))
        assert count_threads() == 0


def test_put_after_terminate_executes_synchronously():
    queue = AsyncTraceExportQueue()

    calls = []
    queue.put(Task(handler=calls.append, args=(1,)))
    queue.flush(terminate=True)

    assert not queue.is_active()
    assert queue._stop_event.is_set()

    # Calling put() after termination must not deadlock; task must run synchronously.
    queue.put(Task(handler=calls.append, args=(2,)))

    assert calls == [1, 2]


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


@pytest.mark.parametrize("terminate", [False, True])
def test_concurrent_put_during_flush_does_not_deadlock(terminate):
    queue = AsyncTraceExportQueue()

    calls = []
    consumer_joined = threading.Event()
    allow_flush_to_continue = threading.Event()

    queue.activate()

    original_join = queue._consumer_thread.join

    def controlled_join(*args, **kwargs):
        result = original_join(*args, **kwargs)
        consumer_joined.set()
        allow_flush_to_continue.wait(timeout=5)
        return result

    queue._consumer_thread.join = controlled_join

    flush_thread = threading.Thread(
        target=queue.flush,
        kwargs={"terminate": terminate},
        name="test-concurrent-put-flush",
        daemon=True,
    )
    flush_thread.start()

    assert consumer_joined.wait(timeout=5)

    # The consumer has stopped while flush() is still in progress.
    # A concurrent put() must not enqueue work that no consumer can process.
    queue.put(Task(handler=calls.append, args=(1,)))

    allow_flush_to_continue.set()
    flush_thread.join(timeout=5)

    assert not flush_thread.is_alive()
    assert calls == [1]
    assert queue.is_active() is not terminate

    if not terminate:
        queue.flush(terminate=True)


def test_flush_waits_for_in_progress_flush():
    queue = AsyncTraceExportQueue()
    release = threading.Event()
    completed = []

    def handler(value):
        assert release.wait(timeout=10)
        completed.append(value)

    for value in range(3):
        queue.put(Task(handler=handler, args=(value,)))

    first_flush_thread = threading.Thread(
        target=queue.flush,
        kwargs={"terminate": True},
        name="test-first-concurrent-flush",
        daemon=True,
    )
    first_flush_thread.start()

    deadline = time.monotonic() + 5
    while queue.is_active() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not queue.is_active()

    completed_when_second_flush_returns = []

    def second_flush():
        queue.flush(terminate=True)
        completed_when_second_flush_returns.append(len(completed))

    second_flush_thread = threading.Thread(
        target=second_flush,
        name="test-second-concurrent-flush",
        daemon=True,
    )
    second_flush_thread.start()

    second_flush_thread.join(timeout=0.1)

    release.set()
    first_flush_thread.join(timeout=5)
    second_flush_thread.join(timeout=5)

    assert not first_flush_thread.is_alive()
    assert not second_flush_thread.is_alive()
    assert completed_when_second_flush_returns == [3]

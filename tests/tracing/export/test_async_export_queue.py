import contextvars
import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

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


def test_put_propagates_caller_contextvars_to_worker():
    # ThreadPoolExecutor does not propagate ContextVars to worker threads.
    # AsyncTraceExportQueue.put() must snapshot the caller's context so handlers
    # see request-scoped state such as the active workspace set by the server
    # middleware (regression test for https://github.com/mlflow/mlflow/issues/23748).
    test_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("test_var", default=None)
    test_var.set("caller-value")

    seen_in_worker: dict[str, str | None] = {}

    def handler():
        seen_in_worker["value"] = test_var.get()

    queue = AsyncTraceExportQueue()
    queue.put(Task(handler=handler, args=()))
    queue.flush(terminate=True)

    assert seen_in_worker["value"] == "caller-value"


def test_put_honors_explicitly_attached_context():
    test_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("test_var", default=None)

    def captured_context_factory() -> contextvars.Context:
        ctx = contextvars.copy_context()
        ctx.run(test_var.set, "from-explicit-context")
        return ctx

    explicit_context = captured_context_factory()
    test_var.set("caller-thread-value")

    seen_in_worker: dict[str, str | None] = {}

    def handler():
        seen_in_worker["value"] = test_var.get()

    queue = AsyncTraceExportQueue()
    queue.put(Task(handler=handler, args=(), context=explicit_context))
    queue.flush(terminate=True)

    assert seen_in_worker["value"] == "from-explicit-context"


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

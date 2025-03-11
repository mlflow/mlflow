from pathlib import Path
from threading import Thread

import pytest

from mlflow import MlflowClient
from mlflow.entities import Metric
from mlflow.utils.autologging_utils.logging_and_warnings import _WarningsController
from mlflow.utils.autologging_utils.metrics_queue import (
    _metrics_queue,
    _metrics_queue_lock,
    flush_metrics_queue,
)


def test_flush_metrics_queue_is_thread_safe():
    """
    Autologging augments TensorBoard event logging hooks with MLflow `log_metric` API
    calls. To prevent these API calls from blocking TensorBoard event logs, `log_metric`
    API calls are scheduled via `_flush_queue` on a background thread. Accordingly, this test
    verifies that `_flush_queue` is thread safe.
    """

    client = MlflowClient()
    run = client.create_run(experiment_id="0")
    metric_queue_item = (run.info.run_id, Metric("foo", 0.1, 100, 1))
    _metrics_queue.append(metric_queue_item)

    # Verify that, if another thread holds a lock on the metric queue leveraged by
    # _flush_queue, _flush_queue terminates and does not modify the queue
    _metrics_queue_lock.acquire()
    flush_thread1 = Thread(target=flush_metrics_queue)
    flush_thread1.start()
    flush_thread1.join()
    assert len(_metrics_queue) == 1
    assert _metrics_queue[0] == metric_queue_item
    _metrics_queue_lock.release()

    # Verify that, if no other thread holds a lock on the metric queue leveraged by
    # _flush_queue, _flush_queue flushes the queue as expected
    flush_thread2 = Thread(target=flush_metrics_queue)
    flush_thread2.start()
    flush_thread2.join()
    assert len(_metrics_queue) == 0


def test_no_recursion_error_after_fix(monkeypatch):
    """
    This test verifies that _patched_showwarning does not trigger a RecursionError even if
    Path.__str__ is monkey-patched to recursively call itself.
    """
    controller = _WarningsController()

    def recursive_str(self):
        return recursive_str(self)

    monkeypatch.setattr(Path, "__str__", recursive_str)

    called = False

    def dummy_showwarning(message, category, filename, lineno, *args, **kwargs):
        nonlocal called
        called = True

    controller._original_showwarning = dummy_showwarning

    try:
        controller._patched_showwarning(
            message="Test warning", category=UserWarning, filename="dummy", lineno=1
        )
    except RecursionError:
        pytest.fail("RecursionError was raised despite the fix being applied.")

    assert called, "The dummy original showwarning was not called as expected."

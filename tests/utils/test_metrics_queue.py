from threading import Event, Thread
from unittest.mock import call

import pytest

from mlflow.entities import Metric
from mlflow.utils.autologging_utils import metrics_queue


@pytest.mark.parametrize("count", [0, 1, 500, 10000])
def test_group_metrics_preserves_run_order_and_metric_identity(count):
    items = [(f"run-{i % 3}", Metric("loss", i / 10, 0, i)) for i in range(count)]

    grouped = metrics_queue._assoc_list_to_map(items)

    assert list(grouped) == list(dict.fromkeys(run for run, _ in items))
    for run, metrics in grouped.items():
        expected = [metric for run_id, metric in items if run_id == run]
        assert len(metrics) == len(expected)
        assert all(actual is reference for actual, reference in zip(metrics, expected))


def test_flush_preserves_metrics_appended_while_logging(monkeypatch, mocker):
    first = Metric("loss", 0.1, 100, 1)
    second = Metric("loss", 0.2, 200, 2)
    queue = [("run", first)]
    monkeypatch.setattr(metrics_queue, "_metrics_queue", queue)
    client = mocker.patch.object(metrics_queue, "MlflowClient").return_value
    logging_started = Event()
    continue_logging = Event()

    def log_batch(*args, **kwargs):
        logging_started.set()
        assert continue_logging.wait(timeout=10)

    client.log_batch.side_effect = log_batch
    worker = Thread(
        name="test-append-during-metrics-flush", target=metrics_queue.flush_metrics_queue
    )
    worker.start()
    try:
        assert logging_started.wait(timeout=10)
        metrics_queue.add_to_metrics_queue("loss", 0.2, 2, 200, "run")
    finally:
        continue_logging.set()
        worker.join(timeout=10)
    assert not worker.is_alive()
    assert metrics_queue._metrics_queue is queue
    assert queue == [("run", second)]

    metrics_queue.flush_metrics_queue()

    assert queue == []
    assert client.log_batch.call_args_list == [
        call("run", metrics=[first], params=[], tags=[]),
        call("run", metrics=[second], params=[], tags=[]),
    ]


def test_flush_preserves_duplicates_and_run_order(monkeypatch, mocker):
    metric = Metric("loss", 0.1, 100, 1)
    queue = [("second", metric), ("first", metric), ("second", metric)]
    monkeypatch.setattr(metrics_queue, "_metrics_queue", queue)
    client = mocker.patch.object(metrics_queue, "MlflowClient").return_value

    metrics_queue.flush_metrics_queue()

    assert queue == []
    assert client.log_batch.call_args_list == [
        call("second", metrics=[metric, metric], params=[], tags=[]),
        call("first", metrics=[metric], params=[], tags=[]),
    ]


def test_empty_flush_does_not_create_client(monkeypatch, mocker):
    monkeypatch.setattr(metrics_queue, "_metrics_queue", [])
    client = mocker.patch.object(metrics_queue, "MlflowClient")

    metrics_queue.flush_metrics_queue()

    client.assert_not_called()

import io
import pickle
import random
import threading
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from mlflow import MlflowException
from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.entities.run_tag import RunTag
from mlflow.utils.async_logging.async_logging_queue import AsyncLoggingQueue, QueueStatus

METRIC_PER_BATCH = 250
TAGS_PER_BATCH = 1
PARAMS_PER_BATCH = 1
TOTAL_BATCHES = 5


class RunData:
    def __init__(self, throw_exception_on_batch_number=None) -> None:
        if throw_exception_on_batch_number is None:
            throw_exception_on_batch_number = []
        self.received_run_id = ""
        self.received_metrics = []
        self.received_tags = []
        self.received_params = []
        self.batch_count = 0
        self.throw_exception_on_batch_number = (
            throw_exception_on_batch_number if throw_exception_on_batch_number else []
        )

    def consume_queue_data(self, run_id, metrics, tags, params):
        self.batch_count += 1
        if self.batch_count in self.throw_exception_on_batch_number:
            raise MlflowException("Failed to log run data")
        self.received_run_id = run_id
        self.received_metrics.extend(metrics or [])
        self.received_params.extend(params or [])
        self.received_tags.extend(tags or [])


def test_single_thread_publish_consume_queue(monkeypatch):
    monkeypatch.setenv("MLFLOW_ASYNC_LOGGING_BUFFERING_SECONDS", "3")

    with patch.object(
        AsyncLoggingQueue, "_batch_logging_worker_threadpool", create=True
    ) as mock_worker_threadpool, patch.object(
        AsyncLoggingQueue, "_batch_status_check_threadpool", create=True
    ) as mock_check_threadpool:
        mock_worker_threadpool.submit = MagicMock()
        mock_check_threadpool.submit = MagicMock()
        mock_worker_threadpool.shutdown = MagicMock()
        mock_check_threadpool.shutdown = MagicMock()

        run_id = "test_run_id"
        run_data = RunData()
        async_logging_queue = AsyncLoggingQueue(run_data.consume_queue_data)
        async_logging_queue.activate()
        async_logging_queue._batch_logging_worker_threadpool = mock_worker_threadpool
        async_logging_queue._batch_status_check_threadpool = mock_check_threadpool

        for params, tags, metrics in _get_run_data():
            async_logging_queue.log_batch_async(
                run_id=run_id, metrics=metrics, tags=tags, params=params
            )
        async_logging_queue.flush()
        # 2 batches are sent to the worker thread pool due to grouping, otherwise it would be 5.
        assert mock_worker_threadpool.submit.call_count == 2
        assert async_logging_queue.is_idle()
        assert mock_check_threadpool.shutdown.call_count == 1
        assert mock_worker_threadpool.shutdown.call_count == 1


def test_grouping_batch_in_time_window():
    run_id = "test_run_id"
    run_data = RunData()
    async_logging_queue = AsyncLoggingQueue(run_data.consume_queue_data)
    async_logging_queue.activate()
    metrics_sent = []
    tags_sent = []
    params_sent = []

    for params, tags, metrics in _get_run_data():
        async_logging_queue.log_batch_async(
            run_id=run_id, metrics=metrics, tags=tags, params=params
        )
        metrics_sent += metrics
        tags_sent += tags
        params_sent += params

    async_logging_queue.flush()

    _assert_sent_received_data(
        metrics_sent,
        params_sent,
        tags_sent,
        run_data.received_metrics,
        run_data.received_params,
        run_data.received_tags,
    )


def test_queue_activation():
    run_id = "test_run_id"
    run_data = RunData()
    async_logging_queue = AsyncLoggingQueue(run_data.consume_queue_data)
    assert async_logging_queue.is_idle()

    metrics = [
        Metric(
            key=f"batch metrics async-{val}",
            value=val,
            timestamp=val,
            step=0,
        )
        for val in range(METRIC_PER_BATCH)
    ]
    with pytest.raises(MlflowException, match="AsyncLoggingQueue is not activated."):
        async_logging_queue.log_batch_async(run_id=run_id, metrics=metrics, tags=[], params=[])

    async_logging_queue.activate()
    assert async_logging_queue.is_active()


def test_end_async_logging():
    run_id = "test_run_id"
    run_data = RunData()
    async_logging_queue = AsyncLoggingQueue(run_data.consume_queue_data)
    async_logging_queue.activate()

    metrics = [
        Metric(
            key=f"batch metrics async-{val}",
            value=val,
            timestamp=val,
            step=0,
        )
        for val in range(METRIC_PER_BATCH)
    ]
    async_logging_queue.log_batch_async(run_id=run_id, metrics=metrics, tags=[], params=[])
    async_logging_queue.end_async_logging()
    assert async_logging_queue._status == QueueStatus.TEAR_DOWN
    # end_async_logging should not shutdown the threadpool
    assert not async_logging_queue._batch_logging_worker_threadpool._shutdown
    assert not async_logging_queue._batch_status_check_threadpool._shutdown

    async_logging_queue.flush()
    assert async_logging_queue.is_idle()


def test_partial_logging_failed():
    run_id = "test_run_id"
    run_data = RunData(throw_exception_on_batch_number=[3, 4])

    async_logging_queue = AsyncLoggingQueue(run_data.consume_queue_data)
    async_logging_queue.activate()

    metrics_sent = []
    tags_sent = []
    params_sent = []

    run_operations = []
    batch_id = 1
    for params, tags, metrics in _get_run_data():
        if batch_id in [3, 4]:
            with pytest.raises(MlflowException, match="Failed to log run data"):
                async_logging_queue.log_batch_async(
                    run_id=run_id, metrics=metrics, tags=tags, params=params
                ).wait()
        else:
            run_operations.append(
                async_logging_queue.log_batch_async(
                    run_id=run_id, metrics=metrics, tags=tags, params=params
                )
            )
            metrics_sent += metrics
            tags_sent += tags
            params_sent += params

        batch_id += 1

    for run_operation in run_operations:
        run_operation.wait()

    _assert_sent_received_data(
        metrics_sent,
        params_sent,
        tags_sent,
        run_data.received_metrics,
        run_data.received_params,
        run_data.received_tags,
    )


def test_publish_multithread_consume_single_thread():
    run_id = "test_run_id"
    run_data = RunData(throw_exception_on_batch_number=[])

    async_logging_queue = AsyncLoggingQueue(run_data.consume_queue_data)
    async_logging_queue.activate()

    run_operations = []
    t1 = threading.Thread(
        target=_send_metrics_tags_params, args=(async_logging_queue, run_id, run_operations)
    )
    t2 = threading.Thread(
        target=_send_metrics_tags_params, args=(async_logging_queue, run_id, run_operations)
    )

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    for run_operation in run_operations:
        run_operation.wait()

    assert len(run_data.received_metrics) == 2 * METRIC_PER_BATCH * TOTAL_BATCHES
    assert len(run_data.received_tags) == 2 * TAGS_PER_BATCH * TOTAL_BATCHES
    assert len(run_data.received_params) == 2 * PARAMS_PER_BATCH * TOTAL_BATCHES


class Consumer:
    def __init__(self) -> None:
        self.metrics = []
        self.tags = []
        self.params = []

    def consume_queue_data(self, run_id, metrics, tags, params):
        time.sleep(0.5)
        self.metrics.extend(metrics or [])
        self.params.extend(params or [])
        self.tags.extend(tags or [])


def test_async_logging_queue_pickle():
    run_id = "test_run_id"
    consumer = Consumer()
    async_logging_queue = AsyncLoggingQueue(consumer.consume_queue_data)

    # Pickle the queue without activating it.
    buffer = io.BytesIO()
    pickle.dump(async_logging_queue, buffer)
    deserialized_queue = pickle.loads(buffer.getvalue())  # Type: AsyncLoggingQueue

    # activate the queue and then try to pickle it
    async_logging_queue.activate()

    run_operations = []
    for val in range(0, 10):
        run_operations.append(
            async_logging_queue.log_batch_async(
                run_id=run_id,
                metrics=[Metric("metric", val, timestamp=time.time(), step=1)],
                tags=[],
                params=[],
            )
        )

    # Pickle the queue
    buffer = io.BytesIO()
    pickle.dump(async_logging_queue, buffer)

    deserialized_queue = pickle.loads(buffer.getvalue())  # Type: AsyncLoggingQueue
    assert deserialized_queue._queue.empty()
    assert deserialized_queue._lock is not None
    assert deserialized_queue._status is QueueStatus.IDLE

    for run_operation in run_operations:
        run_operation.wait()

    assert len(consumer.metrics) == 10

    # try to log using deserialized queue after activating it.
    deserialized_queue.activate()
    assert deserialized_queue.is_active()

    run_operations = []

    for val in range(0, 10):
        run_operations.append(
            deserialized_queue.log_batch_async(
                run_id=run_id,
                metrics=[Metric("metric", val, timestamp=time.time(), step=1)],
                tags=[],
                params=[],
            )
        )

    for run_operation in run_operations:
        run_operation.wait()

    assert len(deserialized_queue._logging_func.__self__.metrics) == 10


def _send_metrics_tags_params(run_data_queueing_processor, run_id, run_operations=None):
    if run_operations is None:
        run_operations = []
    metrics_sent = []
    tags_sent = []
    params_sent = []

    for params, tags, metrics in _get_run_data():
        run_operations.append(
            run_data_queueing_processor.log_batch_async(
                run_id=run_id, metrics=metrics, tags=tags, params=params
            )
        )

        time.sleep(random.randint(1, 3))
        metrics_sent += metrics
        tags_sent += tags
        params_sent += params


def _get_run_data(total_batches=TOTAL_BATCHES):
    for num in range(0, total_batches):
        guid8 = str(uuid.uuid4())[:8]
        params = [
            Param(f"batch param-{guid8}-{val}", value=str(time.time()))
            for val in range(PARAMS_PER_BATCH)
        ]
        tags = [
            RunTag(f"batch tag-{guid8}-{val}", value=str(time.time()))
            for val in range(TAGS_PER_BATCH)
        ]
        metrics = [
            Metric(
                key=f"batch metrics async-{num}",
                value=val,
                timestamp=int(time.time() * 1000),
                step=0,
            )
            for val in range(METRIC_PER_BATCH)
        ]
        yield params, tags, metrics


def _assert_sent_received_data(
    metrics_sent, params_sent, tags_sent, received_metrics, received_params, received_tags
):
    for num in range(1, len(metrics_sent)):
        assert metrics_sent[num].key == received_metrics[num].key
        assert metrics_sent[num].value == received_metrics[num].value
        assert metrics_sent[num].timestamp == received_metrics[num].timestamp
        assert metrics_sent[num].step == received_metrics[num].step

    for num in range(1, len(tags_sent)):
        assert tags_sent[num].key == received_tags[num].key
        assert tags_sent[num].value == received_tags[num].value

    for num in range(1, len(params_sent)):
        assert params_sent[num].key == received_params[num].key
        assert params_sent[num].value == received_params[num].value

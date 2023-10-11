import random
import threading
import time
import uuid

from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.entities.run_tag import RunTag
from mlflow.utils.async_logging.async_logging_queue import AsyncLoggingQueue

METRIC_PER_BATCH = 250
TAGS_PER_BATCH = 1
PARAMS_PER_BATCH = 1
TOTAL_BATCHES = 5


class RunData:
    def __init__(self, throw_exception_on_batch_number=[]) -> None:
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
        self.received_run_id = run_id
        self.received_metrics += metrics or []
        self.received_params += params or []
        self.received_tags += tags or []

        if self.batch_count in self.throw_exception_on_batch_number:
            raise Exception("Failed to process batch number: " + str(self.batch_count))


def test_single_thread_publish_consume_queue():
    run_id = "test_run_id"
    run_data = RunData()
    run_data_queueing_processor = AsyncLoggingQueue(run_data.consume_queue_data)

    metrics_sent = []
    tags_sent = []
    params_sent = []

    run_operations = []
    for params, tags, metrics in _get_run_data():
        run_operations.append(
            run_data_queueing_processor.log_batch_async(
                run_id=run_id, metrics=metrics, tags=tags, params=params
            )
        )
        metrics_sent += metrics
        tags_sent += tags
        params_sent += params

    for run_operation in run_operations:
        run_operation.wait()

    # Stop the run data processing thread.
    run_data_queueing_processor.continue_to_process_data = False

    _assert_sent_received_data(
        metrics_sent,
        params_sent,
        tags_sent,
        run_data.received_metrics,
        run_data.received_params,
        run_data.received_tags,
    )


def test_single_thread_publish_certain_batches_failed_to_be_sent_from_queue():
    run_id = "test_run_id"
    run_data = RunData(throw_exception_on_batch_number=[3, 4])
    run_data_queueing_processor = AsyncLoggingQueue(run_data.consume_queue_data)

    metrics_sent = []
    tags_sent = []
    params_sent = []

    run_operations = []
    for params, tags, metrics in _get_run_data():
        run_operations.append(
            run_data_queueing_processor.log_batch_async(
                run_id=run_id, metrics=metrics, tags=tags, params=params
            )
        )
        metrics_sent += metrics
        tags_sent += tags
        params_sent += params

    exceptions = []
    for run_operation in run_operations:
        try:
            run_operation.wait()
        except Exception as e:
            exceptions.append(e)

    assert len(exceptions) == 2
    assert "Failed to process batch number: 3" in str(exceptions[0])
    assert "Failed to process batch number: 4" in str(exceptions[1])
    # Stop the run data processing thread.
    run_data_queueing_processor.continue_to_process_data = False

    num = 0
    for run_operation in run_operations:
        if num == 2 or num == 3:
            run_operation.await_completion()

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
    run_data_queueing_processor = AsyncLoggingQueue(run_data.consume_queue_data)
    run_operations = []
    t1 = threading.Thread(
        target=_send_metrics_tags_params, args=(run_data_queueing_processor, run_id, run_operations)
    )
    t2 = threading.Thread(
        target=_send_metrics_tags_params, args=(run_data_queueing_processor, run_id, run_operations)
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


def _send_metrics_tags_params(run_data_queueing_processor, run_id, run_operations=[]):
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

import atexit
import time
import uuid

from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.entities.run_tag import RunTag
from mlflow.utils.run_data_queuing_processor import RunDataQueuingProcessor


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
    run_data_queueing_processor = RunDataQueuingProcessor(run_data.consume_queue_data)

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

    run_operations[1].await_completion()
    run_operations[5].await_completion()
    # This ensures the callback registered by RunDataQueuingProcessor is called on
    # simluated exit of the process
    atexit._run_exitfuncs()

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
    run_data_queueing_processor = RunDataQueuingProcessor(run_data.consume_queue_data)

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

    try:
        run_operations[2].await_completion()
    except Exception as e:
        assert "Failed to process batch number: 3" in str(e)

    try:
        run_operations[3].await_completion()
    except Exception as e:
        assert "Failed to process batch number: 4" in str(e)

    # This ensures the callback registered by RunDataQueuingProcessor is called on
    # simluated exit of the process
    atexit._run_exitfuncs()

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


def _get_run_data():
    for num in range(0, 10):
        guid8 = str(uuid.uuid4())[:8]
        params = [Param(f"batch param-{guid8}-{val}", value=str(time.time())) for val in range(1)]
        tags = [RunTag(f"batch tag-{guid8}-{val}", value=str(time.time())) for val in range(1)]
        metrics = [
            Metric(
                key=f"batch metrics async-{num}",
                value=val,
                timestamp=int(time.time() * 1000),
                step=0,
            )
            for val in range(250)
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

import time
import pytest
from unittest import mock

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _truncate_dict
from mlflow.utils.autologging_utils import MlflowAutologgingQueueingClient
from mlflow.utils.validation import (
    MAX_ENTITY_KEY_LENGTH,
    MAX_PARAM_VAL_LENGTH,
    MAX_TAG_VAL_LENGTH,
    MAX_PARAMS_TAGS_PER_BATCH,
    MAX_METRICS_PER_BATCH,
)


pytestmark = pytest.mark.large


def get_run_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    # Ignore tags mlflow logs by default (e.g. "mlflow.user")
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    return data.params, data.metrics, tags


def test_client_truncates_param_keys_and_values():
    client = MlflowAutologgingQueueingClient()
    params_to_log = {
        "a" * (MAX_ENTITY_KEY_LENGTH + 5): "b" * (MAX_PARAM_VAL_LENGTH + 5),
        "a" * (MAX_ENTITY_KEY_LENGTH + 50): "b" * (MAX_PARAM_VAL_LENGTH + 50),
    }

    with mlflow.start_run() as run:
        client.log_params(run_id=run.info.run_id, params=params_to_log)
        client.flush()

    run_params = get_run_data(run.info.run_id)[0]
    assert run_params == _truncate_dict(
        params_to_log,
        max_key_length=MAX_ENTITY_KEY_LENGTH,
        max_value_length=MAX_PARAM_VAL_LENGTH,
    )


def test_client_truncates_tag_keys_and_values():
    client = MlflowAutologgingQueueingClient()
    tags_to_log = {
        "a" * (MAX_ENTITY_KEY_LENGTH + 5): "b" * (MAX_PARAM_VAL_LENGTH + 5),
        "c" * (MAX_ENTITY_KEY_LENGTH + 50): "d" * (MAX_PARAM_VAL_LENGTH + 50),
    }

    with mlflow.start_run() as run:
        client.set_tags(run_id=run.info.run_id, tags=tags_to_log)
        client.flush()

    run_tags = get_run_data(run.info.run_id)[2]
    assert run_tags == _truncate_dict(
        tags_to_log,
        max_key_length=MAX_ENTITY_KEY_LENGTH,
        max_value_length=MAX_TAG_VAL_LENGTH,
    )


def test_client_truncates_metric_keys():
    client = MlflowAutologgingQueueingClient()
    metrics_to_log = {
        "a" * (MAX_ENTITY_KEY_LENGTH + 5): 1,
        "b" * (MAX_ENTITY_KEY_LENGTH + 50): 2,
    }

    with mlflow.start_run() as run:
        client.log_metrics(run_id=run.info.run_id, metrics=metrics_to_log)
        client.flush()

    run_metrics = get_run_data(run.info.run_id)[1]
    assert run_metrics == _truncate_dict(metrics_to_log, max_key_length=MAX_ENTITY_KEY_LENGTH)


def test_client_logs_expected_run_data():
    client = MlflowAutologgingQueueingClient()

    params_to_log = {
        "param_key_{}".format(i): "param_val_{}".format(i)
        for i in range((2 * MAX_PARAMS_TAGS_PER_BATCH) + 1)
    }
    tags_to_log = {
        "tag_key_{}".format(i): "tag_val_{}".format(i)
        for i in range((2 * MAX_PARAMS_TAGS_PER_BATCH) + 1)
    }
    metrics_to_log = {"metric_key_{}".format(i): i for i in range((4 * MAX_METRICS_PER_BATCH) + 1)}

    with mlflow.start_run() as run:
        client.log_params(run_id=run.info.run_id, params=params_to_log)
        client.set_tags(run_id=run.info.run_id, tags=tags_to_log)
        client.log_metrics(run_id=run.info.run_id, metrics=metrics_to_log)
        client.flush()

    run_params, run_metrics, run_tags = get_run_data(run.info.run_id)
    assert run_params == params_to_log
    assert run_metrics == metrics_to_log
    assert run_tags == tags_to_log


def test_client_logs_metric_steps_correctly():
    client = MlflowAutologgingQueueingClient()

    with mlflow.start_run() as run:
        for step in range(3):
            client.log_metrics(run_id=run.info.run_id, metrics={"a": 1}, step=step)
        client.flush()

    metric_history = MlflowClient().get_metric_history(run_id=run.info.run_id, key="a")
    assert len(metric_history) == 3
    assert [metric.step for metric in metric_history] == list(range(3))


def test_client_run_creation_and_termination_are_successful():
    experiment_name = "test_run_creation_termination"
    MlflowClient().create_experiment(experiment_name)
    experiment_id = MlflowClient().get_experiment_by_name(experiment_name).experiment_id

    client = MlflowAutologgingQueueingClient()
    pending_run_id = client.create_run(experiment_id=experiment_id, start_time=5, tags={"a": "b"})
    client.set_terminated(run_id=pending_run_id, status="FINISHED", end_time=6)
    client.flush()

    runs = mlflow.search_runs(experiment_ids=[experiment_id], output_format="list")
    assert len(runs) == 1
    run = runs[0]
    assert run.info.start_time == 5
    assert run.info.end_time == 6
    assert run.info.status == "FINISHED"
    assert {"a": "b"}.items() <= run.data.tags.items()


def test_client_asynchronous_flush_operates_correctly():
    original_log_batch = MlflowClient().log_batch

    def mock_log_batch(run_id, metrics, params, tags):  # pylint: disable=unused-argument
        # Sleep to simulate a long-running logging operation
        time.sleep(3)
        return original_log_batch(run_id, metrics, params, tags)

    with mock.patch(
        "mlflow.utils.autologging_utils.client.MlflowClient.log_batch"
    ) as log_batch_mock:
        log_batch_mock.side_effect = mock_log_batch

        with mlflow.start_run() as run:
            client = MlflowAutologgingQueueingClient()
            client.log_params(run_id=run.info.run_id, params={"a": "b"})
            run_operations = client.flush(synchronous=False)

            # Parameter data should not be available because the asynchronous logging
            # operation is still inflight
            logged_params_1 = get_run_data(run.info.run_id)[0]
            assert not logged_params_1

            run_operations.await_completion()

            # Parameter data should now be available after waiting for completion of the
            # asynchronous logging operation
            logged_params_2 = get_run_data(run.info.run_id)[0]
            assert logged_params_2 == {"a": "b"}


def test_client_synchronous_flush_operates_correctly():
    original_log_batch = MlflowClient().log_batch

    def mock_log_batch(run_id, metrics, params, tags):  # pylint: disable=unused-argument
        # Sleep to simulate a long-running logging operation
        time.sleep(3)
        return original_log_batch(run_id, metrics, params, tags)

    with mock.patch(
        "mlflow.utils.autologging_utils.client.MlflowClient.log_batch"
    ) as log_batch_mock:
        log_batch_mock.side_effect = mock_log_batch

        with mlflow.start_run() as run:
            client = MlflowAutologgingQueueingClient()
            client.log_params(run_id=run.info.run_id, params={"a": "b"})
            client.flush(synchronous=True)

            # Parameter data should be available after the synchronous flush call returns
            logged_params = get_run_data(run.info.run_id)[0]
            assert logged_params == {"a": "b"}


def test_flush_clears_pending_operations():
    with mock.patch(
        "mlflow.utils.autologging_utils.client.MlflowClient", autospec=True
    ) as mlflow_client_mock:
        client = MlflowAutologgingQueueingClient()

        pending_run_id = client.create_run(experiment_id=5)
        client.log_params(run_id=pending_run_id, params={"a": "b"})
        client.log_metrics(run_id=pending_run_id, metrics={"c": 1})
        client.set_terminated(run_id=pending_run_id, status="FINISHED")
        client.flush()

        logging_call_count_1 = len(mlflow_client_mock.method_calls)
        # Verify that at least 3 calls have been made to MLflow logging APIs as a result
        # of the flush (i.e. log_batch, create_run, and set_terminated)
        assert logging_call_count_1 >= 3

        client.flush()

        logging_call_count_2 = len(mlflow_client_mock.method_calls)
        # Verify that performing a second flush did not result in any additional logging API calls,
        # since no new run content was added prior to the flush
        assert logging_call_count_2 == logging_call_count_1


def test_client_correctly_operates_as_context_manager_for_synchronous_flush():
    params_to_log = {"a": "b"}
    metrics_to_log = {"c": 1}
    tags_to_log = {"d": "e"}

    with mlflow.start_run(), MlflowAutologgingQueueingClient() as client:
        run_id_1 = mlflow.active_run().info.run_id
        client.log_params(run_id_1, params_to_log)
        client.log_metrics(run_id_1, metrics_to_log)
        client.set_tags(run_id_1, tags_to_log)

    run_params_1, run_metrics_1, run_tags_1 = get_run_data(run_id_1)
    assert run_params_1 == params_to_log
    assert run_metrics_1 == metrics_to_log
    assert run_tags_1 == tags_to_log

    exc_to_raise = Exception("test exception")
    with pytest.raises(Exception, match=str(exc_to_raise)) as raised_exc_info:
        with mlflow.start_run(), MlflowAutologgingQueueingClient() as client:
            run_id_2 = mlflow.active_run().info.run_id
            client.log_params(run_id_2, params_to_log)
            client.log_metrics(run_id_2, metrics_to_log)
            client.set_tags(run_id_2, tags_to_log)
            raise exc_to_raise

    assert raised_exc_info.value == exc_to_raise
    # Verify that no run content was logged because the context exited with an exception
    run_params_2, run_metrics_2, run_tags_2 = get_run_data(run_id_2)
    assert not run_params_2
    assert not run_metrics_2
    assert not run_tags_2


def test_logging_failures_are_handled_as_expected():
    experiment_name = "test_run_creation_termination"
    MlflowClient().create_experiment(experiment_name)
    experiment_id = MlflowClient().get_experiment_by_name(experiment_name).experiment_id

    with mock.patch(
        "mlflow.utils.autologging_utils.client.MlflowClient.log_batch"
    ) as log_batch_mock:
        log_batch_mock.side_effect = Exception("Batch logging failed!")

        client = MlflowAutologgingQueueingClient()
        pending_run_id = client.create_run(experiment_id=experiment_id)
        client.log_metrics(run_id=pending_run_id, metrics={"a": 1})
        client.set_terminated(run_id=pending_run_id, status="KILLED")

        with pytest.raises(MlflowException, match="Batch logging failed!") as exc:
            client.flush()

        runs = mlflow.search_runs(experiment_ids=[experiment_id], output_format="list")
        assert len(runs) == 1
        run = runs[0]
        # Verify that metrics are absent due to the failure of batch logging
        assert not run.data.metrics
        # Verify that the run termination operation was still performed successfully
        assert run.info.status == "KILLED"

        assert "Failed to perform one or more operations on the run with ID {run_id}".format(
            run_id=run.info.run_id
        ) in str(exc.value)
        assert "Batch logging failed!" in str(exc.value)

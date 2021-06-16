# from unittest import mock

import mlflow
from mlflow.utils import _truncate_dict
from mlflow.utils.validation import (
    MAX_ENTITY_KEY_LENGTH,
    MAX_PARAM_VAL_LENGTH,
    MAX_TAG_VAL_LENGTH,
    MAX_PARAMS_TAGS_PER_BATCH,
    MAX_METRICS_PER_BATCH,
)
from mlflow.utils.autologging_utils import MlflowAutologgingQueueingClient


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
        "a" * (MAX_ENTITY_KEY_LENGTH + 50): "b" * (MAX_PARAM_VAL_LENGTH + 50)
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
    assert run_metrics == _truncate_dict(
        metrics_to_log,
        max_key_length=MAX_ENTITY_KEY_LENGTH,
    )


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
    metrics_to_log = {
        "metric_key_{}".format(i): i
        for i in range((2 * MAX_METRICS_PER_BATCH) + 1)
    }

    with mlflow.start_run() as run:
        client.log_params(
            run_id=run.info.run_id,
            params=params_to_log,
        )
        client.set_tags(
            run_id=run.info.run_id,
            tags=tags_to_log,
        )
        client.log_metrics(
            run_id=run.info.run_id,
            metrics=metrics_to_log,
        )
        client.flush()

    run_params, run_metrics, run_tags = get_run_data(run.info.run_id)
    assert run_params == params_to_log
    assert run_metrics == metrics_to_log
    assert run_tags == tags_to_log

# def test_client_run_creation_and_termination_successful():
#   pass

# def test_client_asynchronous_flush_operates_correctly():

# def test_flush_clears_pending_operations():
#     client = MlflowAutologgingQueueingClient()
#
#     with mlflow.start_run() as run:
#         client.log_params(run_id=run.info.run_id, params={"A": "B"})
#         client.flush()

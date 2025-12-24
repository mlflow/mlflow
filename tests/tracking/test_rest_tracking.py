"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures we can use the tracking API to communicate with it.
"""

import json
import logging
import math
import os
import pathlib
import posixpath
import subprocess
import sys
import time
import urllib.parse
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from unittest import mock

import flask
import pandas as pd
import pytest
import requests
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

import mlflow.experiments
import mlflow.pyfunc
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities import (
    Dataset,
    DatasetInput,
    GatewayResourceType,
    InputTag,
    Metric,
    Param,
    RunInputs,
    RunTag,
    Span,
    SpanEvent,
    SpanStatusCode,
    ViewType,
)
from mlflow.entities.logged_model_input import LoggedModelInput
from mlflow.entities.logged_model_output import LoggedModelOutput
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.span import SpanAttributeKey
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_metrics import (
    AggregationType,
    MetricAggregation,
    MetricViewType,
)
from mlflow.entities.trace_state import TraceState
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import (
    _MLFLOW_GO_STORE_TESTING,
    MLFLOW_SERVER_GRAPHQL_MAX_ALIASES,
    MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS,
    MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT,
)
from mlflow.exceptions import MlflowException, RestException
from mlflow.genai.datasets import (
    add_dataset_to_experiments,
    create_dataset,
    remove_dataset_from_experiments,
)
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.server import handlers
from mlflow.server.fastapi_app import app
from mlflow.server.handlers import initialize_backend_stores
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.analysis import TraceFilterCorrelationResult
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import (
    TRACE_SCHEMA_VERSION_KEY,
    TraceMetricDimensionKey,
    TraceMetricKey,
)
from mlflow.tracing.utils import build_otel_context
from mlflow.utils import mlflow_tags
from mlflow.utils.file_utils import TempDir, path_to_local_file_uri
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATASET_CONTEXT,
    MLFLOW_GIT_COMMIT,
    MLFLOW_PARENT_RUN_ID,
    MLFLOW_PROJECT_ENTRY_POINT,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_USER,
)
from mlflow.utils.os import is_windows
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.providers import _PROVIDER_BACKEND_AVAILABLE
from mlflow.utils.time import get_current_time_millis

from tests.helper_functions import get_safe_port
from tests.integration.utils import invoke_cli_runner
from tests.tracking.integration_test_utils import (
    ServerThread,
    _init_server,
    _send_rest_tracking_post_request,
)

_logger = logging.getLogger(__name__)


@pytest.fixture(params=["file", "sqlalchemy"])
def store_type(request):
    """Provides the store type for parameterized tests."""
    return request.param


@pytest.fixture
def mlflow_client(store_type: str, tmp_path: Path, db_uri: str, monkeypatch):
    """Provides an MLflow Tracking API client pointed at the local tracking server."""
    # Set passphrase for secrets management (required for encryption)
    monkeypatch.setenv(
        "MLFLOW_CRYPTO_KEK_PASSPHRASE", "test-passphrase-at-least-32-characters-long"
    )

    if store_type == "file":
        backend_uri = tmp_path.joinpath("file").as_uri()
    elif store_type == "sqlalchemy":
        backend_uri = db_uri

    # Force-reset backend stores before each test.
    handlers._tracking_store = None
    handlers._model_registry_store = None
    initialize_backend_stores(backend_uri, default_artifact_root=tmp_path.as_uri())

    with ServerThread(app, get_safe_port()) as url:
        yield MlflowClient(url)


@pytest.fixture
def mlflow_client_with_secrets(tmp_path: Path, monkeypatch):
    """Provides an MLflow Tracking API client with fresh database for secrets management.

    Creates a fresh SQLite database for each test to avoid encryption state pollution.
    This is necessary because the KEK encryption state can persist across tests when
    using a shared cached database.
    """
    # Set passphrase for secrets management (required for encryption)
    monkeypatch.setenv(
        "MLFLOW_CRYPTO_KEK_PASSPHRASE", "test-passphrase-at-least-32-characters-long"
    )

    # Create fresh database for this test (not using cached_db)
    backend_uri = f"sqlite:///{tmp_path}/mlflow.db"
    artifact_uri = (tmp_path / "artifacts").as_uri()

    # Initialize the store (which creates tables)
    store = SqlAlchemyStore(backend_uri, artifact_uri)
    store.engine.dispose()

    # Force-reset backend stores before each test
    handlers._tracking_store = None
    handlers._model_registry_store = None
    initialize_backend_stores(backend_uri, default_artifact_root=artifact_uri)

    with ServerThread(app, get_safe_port()) as url:
        yield MlflowClient(url)


@pytest.fixture
def cli_env(mlflow_client):
    """Provides an environment for the MLflow CLI pointed at the local tracking server."""
    return {
        "LC_ALL": "en_US.UTF-8",
        "LANG": "en_US.UTF-8",
        "MLFLOW_TRACKING_URI": mlflow_client.tracking_uri,
    }


def create_experiments(client, names):
    return [client.create_experiment(n) for n in names]


def test_create_get_search_experiment(mlflow_client):
    experiment_id = mlflow_client.create_experiment(
        "My Experiment",
        artifact_location="my_location",
        tags={"key1": "val1", "key2": "val2"},
    )
    exp = mlflow_client.get_experiment(experiment_id)
    assert exp.name == "My Experiment"
    if is_windows():
        assert exp.artifact_location == pathlib.Path.cwd().joinpath("my_location").as_uri()
    else:
        assert exp.artifact_location == str(pathlib.Path.cwd().joinpath("my_location"))
    assert len(exp.tags) == 2
    assert exp.tags["key1"] == "val1"
    assert exp.tags["key2"] == "val2"

    experiments = mlflow_client.search_experiments()
    assert {e.name for e in experiments} == {"My Experiment", "Default"}
    mlflow_client.delete_experiment(experiment_id)
    assert {e.name for e in mlflow_client.search_experiments()} == {"Default"}
    assert {e.name for e in mlflow_client.search_experiments(view_type=ViewType.ACTIVE_ONLY)} == {
        "Default"
    }
    assert {e.name for e in mlflow_client.search_experiments(view_type=ViewType.DELETED_ONLY)} == {
        "My Experiment"
    }
    assert {e.name for e in mlflow_client.search_experiments(view_type=ViewType.ALL)} == {
        "My Experiment",
        "Default",
    }
    active_exps_paginated = mlflow_client.search_experiments(max_results=1)
    assert {e.name for e in active_exps_paginated} == {"Default"}
    assert active_exps_paginated.token is None

    all_exps_paginated = mlflow_client.search_experiments(max_results=1, view_type=ViewType.ALL)
    first_page_names = {e.name for e in all_exps_paginated}
    all_exps_second_page = mlflow_client.search_experiments(
        max_results=1, view_type=ViewType.ALL, page_token=all_exps_paginated.token
    )
    second_page_names = {e.name for e in all_exps_second_page}
    assert len(first_page_names) == 1
    assert len(second_page_names) == 1
    assert first_page_names.union(second_page_names) == {"Default", "My Experiment"}


def test_create_experiment_validation(mlflow_client):
    def assert_bad_request(payload, expected_error_message):
        response = _send_rest_tracking_post_request(
            mlflow_client.tracking_uri,
            "/api/2.0/mlflow/experiments/create",
            payload,
        )
        assert response.status_code == 400
        assert expected_error_message in response.text

    assert_bad_request(
        {
            "name": 123,
        },
        "Invalid value 123 for parameter 'name'",
    )
    assert_bad_request({}, "Missing value for required parameter 'name'.")
    assert_bad_request(
        {
            "name": "experiment name",
            "artifact_location": 9.0,
            "tags": [{"key": "key", "value": "value"}],
        },
        "Invalid value 9.0 for parameter 'artifact_location'",
    )
    assert_bad_request(
        {
            "name": "experiment name",
            "artifact_location": "my_location",
            "tags": "5",
        },
        "Invalid value \\\"5\\\" for parameter 'tags'",
    )


def test_delete_restore_experiment(mlflow_client):
    experiment_id = mlflow_client.create_experiment("Deleterious")
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == "active"
    mlflow_client.delete_experiment(experiment_id)
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == "deleted"
    mlflow_client.restore_experiment(experiment_id)
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == "active"


def test_delete_restore_experiment_cli(mlflow_client, cli_env):
    experiment_name = "DeleteriousCLI"
    invoke_cli_runner(
        mlflow.experiments.commands,
        ["create", "--experiment-name", experiment_name],
        env=cli_env,
    )
    experiment_id = mlflow_client.get_experiment_by_name(experiment_name).experiment_id
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == "active"
    invoke_cli_runner(
        mlflow.experiments.commands, ["delete", "-x", str(experiment_id)], env=cli_env
    )
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == "deleted"
    invoke_cli_runner(
        mlflow.experiments.commands, ["restore", "-x", str(experiment_id)], env=cli_env
    )
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == "active"


def test_rename_experiment(mlflow_client):
    experiment_id = mlflow_client.create_experiment("BadName")
    assert mlflow_client.get_experiment(experiment_id).name == "BadName"
    mlflow_client.rename_experiment(experiment_id, "GoodName")
    assert mlflow_client.get_experiment(experiment_id).name == "GoodName"


def test_rename_experiment_cli(mlflow_client, cli_env):
    bad_experiment_name = "CLIBadName"
    good_experiment_name = "CLIGoodName"

    invoke_cli_runner(
        mlflow.experiments.commands, ["create", "-n", bad_experiment_name], env=cli_env
    )
    experiment_id = mlflow_client.get_experiment_by_name(bad_experiment_name).experiment_id
    assert mlflow_client.get_experiment(experiment_id).name == bad_experiment_name
    invoke_cli_runner(
        mlflow.experiments.commands,
        [
            "rename",
            "--experiment-id",
            str(experiment_id),
            "--new-name",
            good_experiment_name,
        ],
        env=cli_env,
    )
    assert mlflow_client.get_experiment(experiment_id).name == good_experiment_name


@pytest.mark.parametrize("parent_run_id_kwarg", [None, "my-parent-id"])
def test_create_run_all_args(mlflow_client, parent_run_id_kwarg):
    user = "username"
    source_name = "Hello"
    entry_point = "entry"
    source_version = "abc"
    create_run_kwargs = {
        "start_time": 456,
        "run_name": "my name",
        "tags": {
            MLFLOW_USER: user,
            MLFLOW_SOURCE_TYPE: "LOCAL",
            MLFLOW_SOURCE_NAME: source_name,
            MLFLOW_PROJECT_ENTRY_POINT: entry_point,
            MLFLOW_GIT_COMMIT: source_version,
            MLFLOW_PARENT_RUN_ID: "7",
            "my": "tag",
            "other": "tag",
        },
    }
    experiment_id = mlflow_client.create_experiment(
        f"Run A Lot (parent_run_id={parent_run_id_kwarg})"
    )
    created_run = mlflow_client.create_run(experiment_id, **create_run_kwargs)
    run_id = created_run.info.run_id
    _logger.info(f"Run id={run_id}")
    fetched_run = mlflow_client.get_run(run_id)
    for run in [created_run, fetched_run]:
        assert run.info.run_id == run_id
        assert run.info.experiment_id == experiment_id
        assert run.info.user_id == user
        assert run.info.start_time == create_run_kwargs["start_time"]
        assert run.info.run_name == "my name"
        for tag in create_run_kwargs["tags"]:
            assert tag in run.data.tags
        assert run.data.tags.get(MLFLOW_USER) == user
        assert run.data.tags.get(MLFLOW_PARENT_RUN_ID) == parent_run_id_kwarg or "7"
        assert [run.info for run in mlflow_client.search_runs([experiment_id])] == [run.info]


def test_create_run_defaults(mlflow_client):
    experiment_id = mlflow_client.create_experiment("Run A Little")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    run = mlflow_client.get_run(run_id)
    assert run.info.run_id == run_id
    assert run.info.experiment_id == experiment_id
    assert run.info.user_id == "unknown"


def test_log_metrics_params_tags(mlflow_client):
    experiment_id = mlflow_client.create_experiment("Oh My")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    mlflow_client.log_metric(run_id, key="metric", value=123.456, timestamp=789, step=2)
    mlflow_client.log_metric(run_id, key="nan_metric", value=float("nan"))
    mlflow_client.log_metric(run_id, key="inf_metric", value=float("inf"))
    mlflow_client.log_metric(run_id, key="-inf_metric", value=-float("inf"))
    mlflow_client.log_metric(run_id, key="stepless-metric", value=987.654, timestamp=321)
    mlflow_client.log_param(run_id, "param", "value")
    mlflow_client.set_tag(run_id, "taggity", "do-dah")
    run = mlflow_client.get_run(run_id)
    assert run.data.metrics.get("metric") == 123.456
    assert math.isnan(run.data.metrics.get("nan_metric"))
    assert run.data.metrics.get("inf_metric") >= 1.7976931348623157e308
    assert run.data.metrics.get("-inf_metric") <= -1.7976931348623157e308
    assert run.data.metrics.get("stepless-metric") == 987.654
    assert run.data.params.get("param") == "value"
    assert run.data.tags.get("taggity") == "do-dah"
    metric_history0 = mlflow_client.get_metric_history(run_id, "metric")
    assert len(metric_history0) == 1
    metric0 = metric_history0[0]
    assert metric0.key == "metric"
    assert metric0.value == 123.456
    assert metric0.timestamp == 789
    assert metric0.step == 2
    metric_history1 = mlflow_client.get_metric_history(run_id, "stepless-metric")
    assert len(metric_history1) == 1
    metric1 = metric_history1[0]
    assert metric1.key == "stepless-metric"
    assert metric1.value == 987.654
    assert metric1.timestamp == 321
    assert metric1.step == 0

    metric_history = mlflow_client.get_metric_history(run_id, "a_test_accuracy")
    assert metric_history == []


def test_log_metric_validation(mlflow_client):
    experiment_id = mlflow_client.create_experiment("metrics validation")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    def assert_bad_request(payload, expected_error_message):
        response = _send_rest_tracking_post_request(
            mlflow_client.tracking_uri,
            "/api/2.0/mlflow/runs/log-metric",
            payload,
        )
        assert response.status_code == 400
        assert expected_error_message in response.text

    assert_bad_request(
        {
            "run_id": 31,
            "key": "metric",
            "value": 41,
            "timestamp": 59,
            "step": 26,
        },
        "Invalid value 31 for parameter 'run_id' supplied",
    )
    assert_bad_request(
        {
            "run_id": run_id,
            "key": 31,
            "value": 41,
            "timestamp": 59,
            "step": 26,
        },
        "Invalid value 31 for parameter 'key' supplied",
    )
    assert_bad_request(
        {
            "run_id": run_id,
            "key": "foo",
            "value": 31,
            "timestamp": 59,
            "step": "foo",
        },
        "Invalid value \\\"foo\\\" for parameter 'step' supplied",
    )
    assert_bad_request(
        {
            "run_id": run_id,
            "key": "foo",
            "value": 31,
            "timestamp": "foo",
            "step": 41,
        },
        "Invalid value \\\"foo\\\" for parameter 'timestamp' supplied",
    )
    assert_bad_request(
        {
            "run_id": None,
            "key": "foo",
            "value": 31,
            "timestamp": 59,
            "step": 41,
        },
        "Missing value for required parameter 'run_id'",
    )
    assert_bad_request(
        {
            "run_id": run_id,
            # Missing key
            "value": 31,
            "timestamp": 59,
            "step": 41,
        },
        "Missing value for required parameter 'key'",
    )
    assert_bad_request(
        {
            "run_id": run_id,
            "key": None,
            "value": 31,
            "timestamp": 59,
            "step": 41,
        },
        "Missing value for required parameter 'key'",
    )


def test_log_metric_model(mlflow_client: MlflowClient):
    experiment_id = mlflow_client.create_experiment("metrics validation")
    run = mlflow_client.create_run(experiment_id)
    model = mlflow_client.create_logged_model(experiment_id)
    mlflow_client.log_metric(
        run.info.run_id,
        key="metric",
        value=0.5,
        timestamp=123456789,
        step=1,
        dataset_name="name",
        dataset_digest="digest",
        model_id=model.model_id,
    )

    model = mlflow_client.get_logged_model(model.model_id)
    assert model.metrics == [
        Metric(
            key="metric",
            value=0.5,
            timestamp=123456789,
            step=1,
            model_id=model.model_id,
            dataset_name="name",
            dataset_digest="digest",
            run_id=run.info.run_id,
        )
    ]


def test_log_param_validation(mlflow_client):
    experiment_id = mlflow_client.create_experiment("params validation")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    def assert_bad_request(payload, expected_error_message):
        response = _send_rest_tracking_post_request(
            mlflow_client.tracking_uri,
            "/api/2.0/mlflow/runs/log-parameter",
            payload,
        )
        assert response.status_code == 400
        assert expected_error_message in response.text

    assert_bad_request(
        {
            "run_id": 31,
            "key": "param",
            "value": 41,
        },
        "Invalid value 31 for parameter 'run_id' supplied",
    )
    assert_bad_request(
        {
            "run_id": run_id,
            "key": 31,
            "value": 41,
        },
        "Invalid value 31 for parameter 'key' supplied",
    )


def test_log_param_with_empty_string_as_value(mlflow_client):
    experiment_id = mlflow_client.create_experiment(
        test_log_param_with_empty_string_as_value.__name__
    )
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    mlflow_client.log_param(run_id, "param_key", "")
    assert {"param_key": ""}.items() <= mlflow_client.get_run(run_id).data.params.items()


def test_set_tag_with_empty_string_as_value(mlflow_client):
    experiment_id = mlflow_client.create_experiment(
        test_set_tag_with_empty_string_as_value.__name__
    )
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    mlflow_client.set_tag(run_id, "tag_key", "")
    assert {"tag_key": ""}.items() <= mlflow_client.get_run(run_id).data.tags.items()


def test_log_batch_containing_params_and_tags_with_empty_string_values(mlflow_client):
    experiment_id = mlflow_client.create_experiment(
        test_log_batch_containing_params_and_tags_with_empty_string_values.__name__
    )
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    mlflow_client.log_batch(
        run_id=run_id,
        params=[Param("param_key", "")],
        tags=[RunTag("tag_key", "")],
    )
    assert {"param_key": ""}.items() <= mlflow_client.get_run(run_id).data.params.items()
    assert {"tag_key": ""}.items() <= mlflow_client.get_run(run_id).data.tags.items()


def test_set_tag_validation(mlflow_client):
    experiment_id = mlflow_client.create_experiment("tags validation")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    def assert_bad_request(payload, expected_error_message):
        response = _send_rest_tracking_post_request(
            mlflow_client.tracking_uri,
            "/api/2.0/mlflow/runs/set-tag",
            payload,
        )
        assert response.status_code == 400
        assert expected_error_message in response.text

    assert_bad_request(
        {
            "run_id": 31,
            "key": "tag",
            "value": 41,
        },
        "Invalid value 31 for parameter 'run_id' supplied",
    )
    assert_bad_request(
        {
            "run_id": run_id,
            "key": "param",
            "value": 41,
        },
        "Invalid value 41 for parameter 'value' supplied",
    )
    assert_bad_request(
        {
            "run_id": run_id,
            # Missing key
            "value": "value",
        },
        "Missing value for required parameter 'key'",
    )

    response = _send_rest_tracking_post_request(
        mlflow_client.tracking_uri,
        "/api/2.0/mlflow/runs/set-tag",
        {
            "run_uuid": run_id,
            "key": "key",
            "value": "value",
        },
    )
    assert response.status_code == 200


def test_path_validation(mlflow_client):
    experiment_id = mlflow_client.create_experiment("tags validation")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    invalid_path = "../path"

    def assert_response(resp):
        assert resp.status_code == 400
        assert response.json() == {
            "error_code": "INVALID_PARAMETER_VALUE",
            "message": "Invalid path",
        }

    response = requests.get(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/artifacts/list",
        params={"run_id": run_id, "path": invalid_path},
    )
    assert_response(response)

    response = requests.get(
        f"{mlflow_client.tracking_uri}/get-artifact",
        params={"run_id": run_id, "path": invalid_path},
    )
    assert_response(response)

    response = requests.get(
        f"{mlflow_client.tracking_uri}//model-versions/get-artifact",
        params={"name": "model", "version": 1, "path": invalid_path},
    )
    assert_response(response)


def test_set_experiment_tag(mlflow_client):
    experiment_id = mlflow_client.create_experiment("SetExperimentTagTest")
    mlflow_client.set_experiment_tag(experiment_id, "dataset", "imagenet1K")
    experiment = mlflow_client.get_experiment(experiment_id)
    assert "dataset" in experiment.tags
    assert experiment.tags["dataset"] == "imagenet1K"
    # test that updating a tag works
    mlflow_client.set_experiment_tag(experiment_id, "dataset", "birdbike")
    experiment = mlflow_client.get_experiment(experiment_id)
    assert "dataset" in experiment.tags
    assert experiment.tags["dataset"] == "birdbike"
    # test that setting a tag on 1 experiment does not impact another experiment.
    experiment_id_2 = mlflow_client.create_experiment("SetExperimentTagTest2")
    experiment2 = mlflow_client.get_experiment(experiment_id_2)
    assert len(experiment2.tags) == 0
    # test that setting a tag on different experiments maintain different values across experiments
    mlflow_client.set_experiment_tag(experiment_id_2, "dataset", "birds200")
    experiment = mlflow_client.get_experiment(experiment_id)
    experiment2 = mlflow_client.get_experiment(experiment_id_2)
    assert "dataset" in experiment.tags
    assert experiment.tags["dataset"] == "birdbike"
    assert "dataset" in experiment2.tags
    assert experiment2.tags["dataset"] == "birds200"
    # test can set multi-line tags
    mlflow_client.set_experiment_tag(experiment_id, "multiline tag", "value2\nvalue2\nvalue2")
    experiment = mlflow_client.get_experiment(experiment_id)
    assert "multiline tag" in experiment.tags
    assert experiment.tags["multiline tag"] == "value2\nvalue2\nvalue2"


def test_set_experiment_tag_with_empty_string_as_value(mlflow_client):
    experiment_id = mlflow_client.create_experiment(
        test_set_experiment_tag_with_empty_string_as_value.__name__
    )
    mlflow_client.set_experiment_tag(experiment_id, "tag_key", "")
    assert {"tag_key": ""}.items() <= mlflow_client.get_experiment(experiment_id).tags.items()


def test_delete_experiment_tag(mlflow_client):
    experiment_id = mlflow_client.create_experiment("DeleteExperimentTagTest")
    mlflow_client.set_experiment_tag(experiment_id, "dataset", "imagenet1K")
    experiment = mlflow_client.get_experiment(experiment_id)
    assert experiment.tags["dataset"] == "imagenet1K"
    # test that deleting a tag works
    mlflow_client.delete_experiment_tag(experiment_id, "dataset")
    experiment = mlflow_client.get_experiment(experiment_id)
    assert "dataset" not in experiment.tags


def test_delete_tag(mlflow_client):
    experiment_id = mlflow_client.create_experiment("DeleteTagExperiment")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    mlflow_client.log_metric(run_id, key="metric", value=123.456, timestamp=789, step=2)
    mlflow_client.log_metric(run_id, key="stepless-metric", value=987.654, timestamp=321)
    mlflow_client.log_param(run_id, "param", "value")
    mlflow_client.set_tag(run_id, "taggity", "do-dah")
    run = mlflow_client.get_run(run_id)
    assert "taggity" in run.data.tags
    assert run.data.tags["taggity"] == "do-dah"
    mlflow_client.delete_tag(run_id, "taggity")
    run = mlflow_client.get_run(run_id)
    assert "taggity" not in run.data.tags
    with pytest.raises(MlflowException, match=r"Run .+ not found"):
        mlflow_client.delete_tag("fake_run_id", "taggity")
    with pytest.raises(MlflowException, match="No tag with name: fakeTag"):
        mlflow_client.delete_tag(run_id, "fakeTag")
    mlflow_client.delete_run(run_id)
    with pytest.raises(MlflowException, match=f"The run {run_id} must be in"):
        mlflow_client.delete_tag(run_id, "taggity")


def test_log_batch(mlflow_client):
    experiment_id = mlflow_client.create_experiment("Batch em up")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    mlflow_client.log_batch(
        run_id=run_id,
        metrics=[Metric("metric", 123.456, 789, 3)],
        params=[Param("param", "value")],
        tags=[RunTag("taggity", "do-dah")],
    )
    run = mlflow_client.get_run(run_id)
    assert run.data.metrics.get("metric") == 123.456
    assert run.data.params.get("param") == "value"
    assert run.data.tags.get("taggity") == "do-dah"
    metric_history = mlflow_client.get_metric_history(run_id, "metric")
    assert len(metric_history) == 1
    metric = metric_history[0]
    assert metric.key == "metric"
    assert metric.value == 123.456
    assert metric.timestamp == 789
    assert metric.step == 3


def test_log_batch_validation(mlflow_client):
    experiment_id = mlflow_client.create_experiment("log_batch validation")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    def assert_bad_request(payload, expected_error_message):
        response = _send_rest_tracking_post_request(
            mlflow_client.tracking_uri,
            "/api/2.0/mlflow/runs/log-batch",
            payload,
        )
        assert response.status_code == 400
        assert expected_error_message in response.text

    for request_parameter in ["metrics", "params", "tags"]:
        assert_bad_request(
            {
                "run_id": run_id,
                request_parameter: "foo",
            },
            f"Invalid value \\\"foo\\\" for parameter '{request_parameter}' supplied",
        )

    ## Should 400 if missing timestamp
    assert_bad_request(
        {"run_id": run_id, "metrics": [{"key": "mae", "value": 2.5}]},
        "Missing value for required parameter 'metrics[0].timestamp'",
    )

    ## Should 200 if timestamp provided but step is not
    response = _send_rest_tracking_post_request(
        mlflow_client.tracking_uri,
        "/api/2.0/mlflow/runs/log-batch",
        {
            "run_id": run_id,
            "metrics": [{"key": "mae", "value": 2.5, "timestamp": 123456789}],
        },
    )

    assert response.status_code == 200


@pytest.mark.xfail(reason="Tracking server does not support logged-model endpoints yet")
@pytest.mark.allow_infer_pip_requirements_fallback
def test_log_model(mlflow_client):
    experiment_id = mlflow_client.create_experiment("Log models")
    with TempDir(chdr=True):
        model_paths = [f"model/path/{i}" for i in range(3)]
        mlflow.set_tracking_uri(mlflow_client.tracking_uri)
        with mlflow.start_run(experiment_id=experiment_id) as run:
            for i, m in enumerate(model_paths):
                mlflow.pyfunc.log_model(name=m, loader_module="mlflow.pyfunc")
                mlflow.pyfunc.save_model(
                    m,
                    mlflow_model=Model(artifact_path=m, run_id=run.info.run_id),
                    loader_module="mlflow.pyfunc",
                )
                model = Model.load(os.path.join(m, "MLmodel"))
                run = mlflow.get_run(run.info.run_id)
                tag = run.data.tags["mlflow.log-model.history"]
                models = json.loads(tag)
                model.utc_time_created = models[i]["utc_time_created"]

                history_model_meta = models[i].copy()
                original_model_uuid = history_model_meta.pop("model_uuid")
                model_meta = model.get_tags_dict().copy()
                new_model_uuid = model_meta.pop("model_uuid")
                assert history_model_meta == model_meta
                assert original_model_uuid != new_model_uuid
                assert len(models) == i + 1
                for j in range(0, i + 1):
                    assert models[j]["artifact_path"] == model_paths[j]


def test_set_terminated_defaults(mlflow_client):
    experiment_id = mlflow_client.create_experiment("Terminator 1")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    assert mlflow_client.get_run(run_id).info.status == "RUNNING"
    assert mlflow_client.get_run(run_id).info.end_time is None
    mlflow_client.set_terminated(run_id)
    assert mlflow_client.get_run(run_id).info.status == "FINISHED"
    assert mlflow_client.get_run(run_id).info.end_time <= get_current_time_millis()


def test_set_terminated_status(mlflow_client):
    experiment_id = mlflow_client.create_experiment("Terminator 2")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    assert mlflow_client.get_run(run_id).info.status == "RUNNING"
    assert mlflow_client.get_run(run_id).info.end_time is None
    mlflow_client.set_terminated(run_id, "FAILED")
    assert mlflow_client.get_run(run_id).info.status == "FAILED"
    assert mlflow_client.get_run(run_id).info.end_time <= get_current_time_millis()


def test_artifacts(mlflow_client, tmp_path):
    experiment_id = mlflow_client.create_experiment("Art In Fact")
    experiment_info = mlflow_client.get_experiment(experiment_id)
    assert experiment_info.artifact_location.startswith(path_to_local_file_uri(str(tmp_path)))
    artifact_path = urllib.parse.urlparse(experiment_info.artifact_location).path
    assert posixpath.split(artifact_path)[-1] == experiment_id

    created_run = mlflow_client.create_run(experiment_id)
    assert created_run.info.artifact_uri.startswith(experiment_info.artifact_location)
    run_id = created_run.info.run_id
    src_dir = tmp_path.joinpath("test_artifacts_src")
    src_dir.mkdir()
    src_file = os.path.join(src_dir, "my.file")
    with open(src_file, "w") as f:
        f.write("Hello, World!")
    mlflow_client.log_artifact(run_id, src_file, None)
    mlflow_client.log_artifacts(run_id, src_dir, "dir")

    root_artifacts_list = mlflow_client.list_artifacts(run_id)
    assert {a.path for a in root_artifacts_list} == {"my.file", "dir"}

    dir_artifacts_list = mlflow_client.list_artifacts(run_id, "dir")
    assert {a.path for a in dir_artifacts_list} == {"dir/my.file"}

    all_artifacts = download_artifacts(
        run_id=run_id, artifact_path=".", tracking_uri=mlflow_client.tracking_uri
    )
    with open(f"{all_artifacts}/my.file") as f:
        assert f.read() == "Hello, World!"
    with open(f"{all_artifacts}/dir/my.file") as f:
        assert f.read() == "Hello, World!"

    dir_artifacts = download_artifacts(
        run_id=run_id, artifact_path="dir", tracking_uri=mlflow_client.tracking_uri
    )
    with open(f"{dir_artifacts}/my.file") as f:
        assert f.read() == "Hello, World!"


def test_search_pagination(mlflow_client):
    experiment_id = mlflow_client.create_experiment("search_pagination")
    runs = [mlflow_client.create_run(experiment_id, start_time=1).info.run_id for _ in range(0, 10)]
    runs = sorted(runs)
    result = mlflow_client.search_runs([experiment_id], max_results=4, page_token=None)
    assert [r.info.run_id for r in result] == runs[0:4]
    assert result.token is not None
    result = mlflow_client.search_runs([experiment_id], max_results=4, page_token=result.token)
    assert [r.info.run_id for r in result] == runs[4:8]
    assert result.token is not None
    result = mlflow_client.search_runs([experiment_id], max_results=4, page_token=result.token)
    assert [r.info.run_id for r in result] == runs[8:]
    assert result.token is None


def test_search_validation(mlflow_client):
    experiment_id = mlflow_client.create_experiment("search_validation")
    with pytest.raises(
        MlflowException,
        match=r"Invalid value 123456789 for parameter 'max_results' supplied",
    ):
        mlflow_client.search_runs([experiment_id], max_results=123456789)


def test_get_experiment_by_name(mlflow_client):
    name = "test_get_experiment_by_name"
    experiment_id = mlflow_client.create_experiment(name)
    res = mlflow_client.get_experiment_by_name(name)
    assert res.experiment_id == experiment_id
    assert res.name == name
    assert mlflow_client.get_experiment_by_name("idontexist") is None


def test_get_experiment(mlflow_client):
    name = "test_get_experiment"
    experiment_id = mlflow_client.create_experiment(name)
    res = mlflow_client.get_experiment(experiment_id)
    assert res.experiment_id == experiment_id
    assert res.name == name


def test_search_experiments(mlflow_client):
    # To ensure the default experiment and non-default experiments have different creation_time
    # for deterministic search results, send a request to the server and initialize the tracking
    # store.
    assert mlflow_client.search_experiments()[0].name == "Default"

    experiments = [
        ("a", {"key": "value"}),
        ("ab", {"key": "vaLue"}),
        ("Abc", None),
    ]
    experiment_ids = []
    for name, tags in experiments:
        # sleep for windows file system current_time precision in Python to enforce
        # deterministic ordering based on last_update_time (creation_time due to no
        # mutation of experiment state)
        time.sleep(0.001)
        experiment_ids.append(mlflow_client.create_experiment(name, tags=tags))

    # filter_string
    experiments = mlflow_client.search_experiments(filter_string="attribute.name = 'a'")
    assert [e.name for e in experiments] == ["a"]
    experiments = mlflow_client.search_experiments(filter_string="attribute.name != 'a'")
    assert [e.name for e in experiments] == ["Abc", "ab", "Default"]
    experiments = mlflow_client.search_experiments(filter_string="name LIKE 'a%'")
    assert [e.name for e in experiments] == ["ab", "a"]
    experiments = mlflow_client.search_experiments(filter_string="tag.key = 'value'")
    assert [e.name for e in experiments] == ["a"]
    experiments = mlflow_client.search_experiments(filter_string="tag.key != 'value'")
    assert [e.name for e in experiments] == ["ab"]
    experiments = mlflow_client.search_experiments(filter_string="tag.key ILIKE '%alu%'")
    assert [e.name for e in experiments] == ["ab", "a"]

    # order_by
    experiments = mlflow_client.search_experiments(order_by=["name DESC"])
    assert [e.name for e in experiments] == ["ab", "a", "Default", "Abc"]

    # max_results
    experiments = mlflow_client.search_experiments(max_results=2)
    assert [e.name for e in experiments] == ["Abc", "ab"]
    # page_token
    experiments = mlflow_client.search_experiments(page_token=experiments.token)
    assert [e.name for e in experiments] == ["a", "Default"]

    # view_type
    time.sleep(0.001)
    mlflow_client.delete_experiment(experiment_ids[1])
    experiments = mlflow_client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
    assert [e.name for e in experiments] == ["Abc", "a", "Default"]
    experiments = mlflow_client.search_experiments(view_type=ViewType.DELETED_ONLY)
    assert [e.name for e in experiments] == ["ab"]
    experiments = mlflow_client.search_experiments(view_type=ViewType.ALL)
    assert [e.name for e in experiments] == ["Abc", "ab", "a", "Default"]


def test_get_metric_history_bulk_rejects_invalid_requests(mlflow_client):
    def assert_response(resp, message_part):
        assert resp.status_code == 400
        response_json = resp.json()
        assert response_json.get("error_code") == "INVALID_PARAMETER_VALUE"
        assert message_part in response_json.get("message", "")

    response_no_run_ids_field = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        params={"metric_key": "key"},
    )
    assert_response(
        response_no_run_ids_field,
        "GetMetricHistoryBulk request must specify at least one run_id",
    )

    response_empty_run_ids = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        params={"run_id": [], "metric_key": "key"},
    )
    assert_response(
        response_empty_run_ids,
        "GetMetricHistoryBulk request must specify at least one run_id",
    )

    response_too_many_run_ids = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        params={"run_id": [f"id_{i}" for i in range(1000)], "metric_key": "key"},
    )
    assert_response(
        response_too_many_run_ids,
        "GetMetricHistoryBulk request cannot specify more than",
    )

    response_no_metric_key_field = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        params={"run_id": ["123"]},
    )
    assert_response(
        response_no_metric_key_field,
        "GetMetricHistoryBulk request must specify a metric_key",
    )


def test_get_metric_history_bulk_returns_expected_metrics_in_expected_order(
    mlflow_client,
):
    experiment_id = mlflow_client.create_experiment("get metric history bulk")
    created_run1 = mlflow_client.create_run(experiment_id)
    run_id1 = created_run1.info.run_id
    created_run2 = mlflow_client.create_run(experiment_id)
    run_id2 = created_run2.info.run_id
    created_run3 = mlflow_client.create_run(experiment_id)
    run_id3 = created_run3.info.run_id

    metricA_history = [
        {"key": "metricA", "timestamp": 1, "step": 2, "value": 10.0},
        {"key": "metricA", "timestamp": 1, "step": 3, "value": 11.0},
        {"key": "metricA", "timestamp": 1, "step": 3, "value": 12.0},
        {"key": "metricA", "timestamp": 2, "step": 3, "value": 12.0},
    ]
    for metric in metricA_history:
        mlflow_client.log_metric(run_id1, **metric)
        metric_for_run2 = dict(metric)
        metric_for_run2["value"] += 1.0
        mlflow_client.log_metric(run_id2, **metric_for_run2)

    metricB_history = [
        {"key": "metricB", "timestamp": 7, "step": -2, "value": -100.0},
        {"key": "metricB", "timestamp": 8, "step": 0, "value": 0.0},
        {"key": "metricB", "timestamp": 8, "step": 0, "value": 1.0},
        {"key": "metricB", "timestamp": 9, "step": 1, "value": 12.0},
    ]
    for metric in metricB_history:
        mlflow_client.log_metric(run_id1, **metric)
        metric_for_run2 = dict(metric)
        metric_for_run2["value"] += 1.0
        mlflow_client.log_metric(run_id2, **metric_for_run2)

    response_run1_metricA = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        params={"run_id": [run_id1], "metric_key": "metricA"},
    )
    assert response_run1_metricA.status_code == 200
    assert response_run1_metricA.json().get("metrics") == [
        {**metric, "run_id": run_id1} for metric in metricA_history
    ]

    response_run2_metricB = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        params={"run_id": [run_id2], "metric_key": "metricB"},
    )
    assert response_run2_metricB.status_code == 200
    assert response_run2_metricB.json().get("metrics") == [
        {**metric, "run_id": run_id2, "value": metric["value"] + 1.0} for metric in metricB_history
    ]

    response_run1_run2_metricA = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        params={"run_id": [run_id1, run_id2], "metric_key": "metricA"},
    )
    assert response_run1_run2_metricA.status_code == 200
    assert response_run1_run2_metricA.json().get("metrics") == sorted(
        [{**metric, "run_id": run_id1} for metric in metricA_history]
        + [
            {**metric, "run_id": run_id2, "value": metric["value"] + 1.0}
            for metric in metricA_history
        ],
        key=lambda metric: metric["run_id"],
    )

    response_run1_run2_run_3_metricB = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        params={"run_id": [run_id1, run_id2, run_id3], "metric_key": "metricB"},
    )
    assert response_run1_run2_run_3_metricB.status_code == 200
    assert response_run1_run2_run_3_metricB.json().get("metrics") == sorted(
        [{**metric, "run_id": run_id1} for metric in metricB_history]
        + [
            {**metric, "run_id": run_id2, "value": metric["value"] + 1.0}
            for metric in metricB_history
        ],
        key=lambda metric: metric["run_id"],
    )


def test_get_metric_history_bulk_respects_max_results(mlflow_client):
    experiment_id = mlflow_client.create_experiment("get metric history bulk")
    run_id = mlflow_client.create_run(experiment_id).info.run_id
    max_results = 2

    metricA_history = [
        {"key": "metricA", "timestamp": 1, "step": 2, "value": 10.0},
        {"key": "metricA", "timestamp": 1, "step": 3, "value": 11.0},
        {"key": "metricA", "timestamp": 1, "step": 3, "value": 12.0},
        {"key": "metricA", "timestamp": 2, "step": 3, "value": 12.0},
    ]
    for metric in metricA_history:
        mlflow_client.log_metric(run_id, **metric)

    response_limited = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        params={
            "run_id": [run_id],
            "metric_key": "metricA",
            "max_results": max_results,
        },
    )
    assert response_limited.status_code == 200
    assert response_limited.json().get("metrics") == [
        {**metric, "run_id": run_id} for metric in metricA_history[:max_results]
    ]


def test_get_metric_history_bulk_calls_optimized_impl_when_expected(tmp_path):
    from mlflow.server.handlers import get_metric_history_bulk_handler

    path = path_to_local_file_uri(str(tmp_path.joinpath("sqlalchemy.db")))
    uri = ("sqlite://" if sys.platform == "win32" else "sqlite:////") + path[len("file://") :]
    mock_store = mock.Mock(wraps=SqlAlchemyStore(uri, str(tmp_path)))

    flask_app = flask.Flask("test_flask_app")

    class MockRequestArgs:
        def __init__(self, args_dict):
            self.args_dict = args_dict

        def to_dict(
            self,
            flat,
        ):
            return self.args_dict

        def get(self, key, default=None):
            return self.args_dict.get(key, default)

    with (
        mock.patch("mlflow.server.handlers._get_tracking_store", return_value=mock_store),
        flask_app.test_request_context() as mock_context,
    ):
        run_ids = [str(i) for i in range(10)]
        mock_context.request.args = MockRequestArgs(
            {
                "run_id": run_ids,
                "metric_key": "mock_key",
            }
        )

        get_metric_history_bulk_handler()

        mock_store.get_metric_history_bulk.assert_called_once_with(
            run_ids=run_ids,
            metric_key="mock_key",
            max_results=25000,
        )


def test_get_metric_history_respects_max_results(mlflow_client):
    experiment_id = mlflow_client.create_experiment("test max_results")
    run = mlflow_client.create_run(experiment_id)
    run_id = run.info.run_id

    metric_history = [
        {"key": "test_metric", "value": float(i), "step": i, "timestamp": 1000 + i}
        for i in range(5)
    ]
    for metric in metric_history:
        mlflow_client.log_metric(run_id, **metric)

    # Test without max_results - should return all metrics
    all_metrics = mlflow_client.get_metric_history(run_id, "test_metric")
    assert len(all_metrics) == 5

    # Test with max_results=3 - should return only 3 metrics
    response = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history",
        params={"run_id": run_id, "metric_key": "test_metric", "max_results": 3},
    )
    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["metrics"]) == 3

    returned_metrics = response_data["metrics"]
    for i, metric in enumerate(returned_metrics):
        assert metric["key"] == "test_metric"
        assert metric["value"] == float(i)
        if _MLFLOW_GO_STORE_TESTING.get():
            assert int(metric["step"]) == i
        else:
            assert metric["step"] == i


def test_get_metric_history_with_page_token(mlflow_client):
    experiment_id = mlflow_client.create_experiment("test page_token")
    run = mlflow_client.create_run(experiment_id)
    run_id = run.info.run_id

    metric_history = [
        {"key": "test_metric", "value": float(i), "step": i, "timestamp": 1000 + i}
        for i in range(10)
    ]
    for metric in metric_history:
        mlflow_client.log_metric(run_id, **metric)

    page_size = 4

    first_response = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history",
        params={
            "run_id": run_id,
            "metric_key": "test_metric",
            "max_results": page_size,
        },
    )
    assert first_response.status_code == 200
    first_data = first_response.json()
    first_metrics = first_data["metrics"]
    first_token = first_data.get("next_page_token")

    assert first_token is not None
    assert len(first_metrics) == 4

    second_response = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history",
        params={
            "run_id": run_id,
            "metric_key": "test_metric",
            "max_results": page_size,
            "page_token": first_token,
        },
    )
    assert second_response.status_code == 200
    second_data = second_response.json()
    second_metrics = second_data["metrics"]
    second_token = second_data.get("next_page_token")

    assert second_token is not None
    assert len(second_metrics) == 4

    third_response = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history",
        params={
            "run_id": run_id,
            "metric_key": "test_metric",
            "max_results": page_size,
            "page_token": second_token,
        },
    )
    assert third_response.status_code == 200
    third_data = third_response.json()
    third_metrics = third_data["metrics"]
    third_token = third_data.get("next_page_token")

    assert third_token is None
    assert len(third_metrics) == 2

    all_paginated_metrics = first_metrics + second_metrics + third_metrics
    assert len(all_paginated_metrics) == 10

    for i, metric in enumerate(all_paginated_metrics):
        assert metric["key"] == "test_metric"
        assert metric["value"] == float(i)
        if _MLFLOW_GO_STORE_TESTING.get():
            assert int(metric["step"]) == i
        else:
            assert metric["step"] == i
        if _MLFLOW_GO_STORE_TESTING.get():
            assert int(metric["timestamp"]) == 1000 + i
        else:
            assert metric["timestamp"] == 1000 + i

    # Test with invalid page_token
    response = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history",
        params={
            "run_id": run_id,
            "metric_key": "test_metric",
            "page_token": "invalid_token",
        },
    )
    assert response.status_code == 400
    response_data = response.json()
    assert "INVALID_PARAMETER_VALUE" in response_data.get("error_code", "")


def test_get_metric_history_bulk_interval_rejects_invalid_requests(mlflow_client):
    def assert_response(resp, message_part):
        assert resp.status_code == 400
        response_json = resp.json()
        assert response_json.get("error_code") == "INVALID_PARAMETER_VALUE"
        assert message_part in response_json.get("message", "")

    url = f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval"

    assert_response(
        requests.get(url, params={"metric_key": "key"}),
        "Missing value for required parameter 'run_ids'.",
    )

    assert_response(
        requests.get(url, params={"run_ids": [], "metric_key": "key"}),
        "Missing value for required parameter 'run_ids'.",
    )

    assert_response(
        requests.get(
            url,
            params={"run_ids": [f"id_{i}" for i in range(1000)], "metric_key": "key"},
        ),
        "GetMetricHistoryBulkInterval request must specify at most 100 run_ids.",
    )

    assert_response(
        requests.get(url, params={"run_ids": ["123"], "metric_key": "key", "max_results": 0}),
        "max_results must be between 1 and 2500",
    )

    assert_response(
        requests.get(url, params={"run_ids": ["123"], "metric_key": ""}),
        "Missing value for required parameter 'metric_key'",
    )

    assert_response(
        requests.get(url, params={"run_ids": ["123"], "max_results": 5}),
        "Missing value for required parameter 'metric_key'",
    )

    assert_response(
        requests.get(
            url,
            params={
                "run_ids": ["123"],
                "metric_key": "key",
                "start_step": 1,
                "end_step": 0,
                "max_results": 5,
            },
        ),
        "end_step must be greater than start_step. ",
    )

    assert_response(
        requests.get(
            url,
            params={
                "run_ids": ["123"],
                "metric_key": "key",
                "start_step": 1,
                "max_results": 5,
            },
        ),
        "If either start step or end step are specified, both must be specified.",
    )


def test_get_metric_history_bulk_interval_respects_max_results(mlflow_client):
    experiment_id = mlflow_client.create_experiment("get metric history bulk")
    run_id1 = mlflow_client.create_run(experiment_id).info.run_id
    metric_history = [
        {"key": "metricA", "timestamp": 1, "step": i, "value": 10.0} for i in range(10)
    ]
    for metric in metric_history:
        mlflow_client.log_metric(run_id1, **metric)

    url = f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval"
    response_limited = requests.get(
        url,
        params={"run_ids": [run_id1], "metric_key": "metricA", "max_results": 5},
    )
    assert response_limited.status_code == 200
    expected_steps = [0, 2, 4, 6, 8, 9]
    expected_metrics = [
        {**metric, "run_id": run_id1}
        for metric in metric_history
        if metric["step"] in expected_steps
    ]
    assert response_limited.json().get("metrics") == expected_metrics

    # with start_step and end_step
    response_limited = requests.get(
        url,
        params={
            "run_ids": [run_id1],
            "metric_key": "metricA",
            "start_step": 0,
            "end_step": 4,
            "max_results": 5,
        },
    )
    assert response_limited.status_code == 200
    assert response_limited.json().get("metrics") == [
        {**metric, "run_id": run_id1} for metric in metric_history[:5]
    ]

    # multiple runs
    run_id2 = mlflow_client.create_run(experiment_id).info.run_id
    metric_history2 = [
        {"key": "metricA", "timestamp": 1, "step": i, "value": 10.0} for i in range(20)
    ]
    for metric in metric_history2:
        mlflow_client.log_metric(run_id2, **metric)
    response_limited = requests.get(
        url,
        params={
            "run_ids": [run_id1, run_id2],
            "metric_key": "metricA",
            "max_results": 5,
        },
    )
    expected_steps = [0, 4, 8, 9, 12, 16, 19]
    expected_metrics = []
    for run_id, metric_history in [
        (run_id1, metric_history),
        (run_id2, metric_history2),
    ]:
        expected_metrics.extend(
            [
                {**metric, "run_id": run_id}
                for metric in metric_history
                if metric["step"] in expected_steps
            ]
        )
    assert response_limited.json().get("metrics") == expected_metrics

    # test metrics with same steps
    metric_history_timestamp2 = [
        {"key": "metricA", "timestamp": 2, "step": i, "value": 10.0} for i in range(10)
    ]
    for metric in metric_history_timestamp2:
        mlflow_client.log_metric(run_id1, **metric)

    response_limited = requests.get(
        url,
        params={"run_ids": [run_id1], "metric_key": "metricA", "max_results": 5},
    )
    assert response_limited.status_code == 200
    expected_steps = [0, 2, 4, 6, 8, 9]
    expected_metrics = [
        {"key": "metricA", "timestamp": j, "step": i, "value": 10.0, "run_id": run_id1}
        for i in expected_steps
        for j in [1, 2]
    ]
    assert response_limited.json().get("metrics") == expected_metrics


def test_search_dataset_handler_rejects_invalid_requests(mlflow_client):
    def assert_response(resp, message_part):
        assert resp.status_code == 400
        response_json = resp.json()
        assert response_json.get("error_code") == "INVALID_PARAMETER_VALUE"
        assert message_part in response_json.get("message", "")

    response_no_experiment_id_field = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/experiments/search-datasets",
        json={},
    )
    assert_response(
        response_no_experiment_id_field,
        "SearchDatasets request must specify at least one experiment_id.",
    )

    response_empty_experiment_id_field = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/experiments/search-datasets",
        json={"experiment_ids": []},
    )
    assert_response(
        response_empty_experiment_id_field,
        "SearchDatasets request must specify at least one experiment_id.",
    )

    response_too_many_experiment_ids = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/experiments/search-datasets",
        json={"experiment_ids": [f"id_{i}" for i in range(1000)]},
    )
    assert_response(
        response_too_many_experiment_ids,
        "SearchDatasets request cannot specify more than",
    )


def test_search_dataset_handler_returns_expected_results(mlflow_client):
    experiment_id = mlflow_client.create_experiment("log inputs test")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    dataset1 = Dataset(
        name="name1",
        digest="digest1",
        source_type="source_type1",
        source="source1",
    )
    dataset_inputs1 = [
        DatasetInput(
            dataset=dataset1,
            tags=[InputTag(key=MLFLOW_DATASET_CONTEXT, value="training")],
        )
    ]
    mlflow_client.log_inputs(run_id, dataset_inputs1)

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/experiments/search-datasets",
        json={"experiment_ids": [experiment_id]},
    )
    expected = {
        "experiment_id": experiment_id,
        "name": "name1",
        "digest": "digest1",
        "context": "training",
    }

    assert response.status_code == 200
    assert response.json().get("dataset_summaries") == [expected]


def test_create_model_version_with_path_source(mlflow_client):
    name = "model"
    mlflow_client.create_registered_model(name)
    exp_id = mlflow_client.create_experiment("test")
    run = mlflow_client.create_run(experiment_id=exp_id)

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": run.info.artifact_uri[len("file://") :],
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    # run_id is not specified
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": run.info.artifact_uri[len("file://") :],
        },
    )
    assert response.status_code == 400
    assert "To use a local path as a model version" in response.json()["message"]

    # run_id is specified but source is not in the run's artifact directory
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "/tmp",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "To use a local path as a model version" in response.json()["message"]


def test_create_model_version_with_non_local_source(mlflow_client):
    name = "model"
    mlflow_client.create_registered_model(name)
    exp_id = mlflow_client.create_experiment("test")
    run = mlflow_client.create_run(experiment_id=exp_id)

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": run.info.artifact_uri[len("file://") :],
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    # Test that remote uri's supplied as a source with absolute paths work fine
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "mlflow-artifacts:/models",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    # A single trailing slash
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "mlflow-artifacts:/models/",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    # Multiple trailing slashes
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "mlflow-artifacts:/models///",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    # Multiple slashes
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "mlflow-artifacts:/models/foo///bar",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "mlflow-artifacts://host:9000/models",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    # Multiple dots
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "mlflow-artifacts://host:9000/models/artifact/..../",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    # Test that invalid remote uri's cannot be created
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "mlflow-artifacts://host:9000/models/../../../",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "If supplying a source as an http, https," in response.json()["message"]

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "http://host:9000/models/../../../",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "If supplying a source as an http, https," in response.json()["message"]

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "https://host/api/2.0/mlflow-artifacts/artifacts/../../../",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "If supplying a source as an http, https," in response.json()["message"]

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "s3a://my_bucket/api/2.0/mlflow-artifacts/artifacts/../../../",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "If supplying a source as an http, https," in response.json()["message"]

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "ftp://host:8888/api/2.0/mlflow-artifacts/artifacts/../../../",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "If supplying a source as an http, https," in response.json()["message"]

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "mlflow-artifacts://host:9000/models/..%2f..%2fartifacts",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "If supplying a source as an http, https," in response.json()["message"]

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "mlflow-artifacts://host:9000/models/artifact%00",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "If supplying a source as an http, https," in response.json()["message"]

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": f"dbfs:/{run.info.run_id}/artifacts/a%3f/../../../../../../../../../../",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "Invalid model version source" in response.json()["message"]

    model = mlflow_client.create_logged_model(experiment_id=exp_id)
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": model.artifact_location,
            "model_id": model.model_id,
        },
    )
    assert response.status_code == 200

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": model.model_uri,
            "model_id": model.model_id,
        },
    )
    assert response.status_code == 200

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "file:///path/to/model",
            "model_id": model.model_id,
        },
    )
    assert response.status_code == 400


def test_create_model_version_with_file_uri(mlflow_client):
    name = "test"
    mlflow_client.create_registered_model(name)
    exp_id = mlflow_client.create_experiment("test")
    run = mlflow_client.create_run(experiment_id=exp_id)
    assert run.info.artifact_uri.startswith("file://")
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": run.info.artifact_uri,
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": f"{run.info.artifact_uri}/model",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": f"{run.info.artifact_uri}/.",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": f"{run.info.artifact_uri}/model/..",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    # run_id is not specified
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": run.info.artifact_uri,
        },
    )
    assert response.status_code == 400
    assert "To use a local path as a model version" in response.json()["message"]

    # run_id is specified but source is not in the run's artifact directory
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "file:///tmp",
        },
    )
    assert response.status_code == 400
    assert "To use a local path as a model version" in response.json()["message"]

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "file://123.456.789.123/path/to/source",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 500, response.json()
    assert "is not a valid remote uri" in response.json()["message"]


def test_create_model_version_with_validation_regex(tmp_path: Path):
    port = get_safe_port()
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--port",
            str(port),
            "--backend-store-uri",
            f"sqlite:///{tmp_path / 'mlflow.db'}",
        ],
        env=(
            os.environ.copy()
            | {
                "MLFLOW_CREATE_MODEL_VERSION_SOURCE_VALIDATION_REGEX": r"^mlflow-artifacts:/.*$",
            }
        ),
    ) as proc:
        try:
            # Wait for the server to start
            for _ in range(10):
                try:
                    if requests.get(f"http://localhost:{port}/health").ok:
                        break
                except requests.ConnectionError:
                    time.sleep(1)
            else:
                raise RuntimeError("Failed to connect to the MLflow server")

            # Test that the validation regex works as expected
            client = MlflowClient(f"http://localhost:{port}")
            name = "test"
            client.create_registered_model(name)
            # Invalid source
            with pytest.raises(MlflowException, match="Invalid model version source"):
                client.create_model_version(name, source="s3://path/to/model")
            # Valid source
            experiment_id = client.create_experiment("test")
            run = client.create_run(experiment_id=experiment_id)
            assert run.info.artifact_uri.startswith("mlflow-artifacts:/")
            client.create_model_version(
                name, source=f"{run.info.artifact_uri}/model", run_id=run.info.run_id
            )
        finally:
            proc.terminate()
            proc.wait()


@pytest.mark.xfail(reason="Tracking server does not support logged-model endpoints yet")
def test_logging_model_with_local_artifact_uri(mlflow_client):
    from sklearn.linear_model import LogisticRegression

    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    with mlflow.start_run() as run:
        assert run.info.artifact_uri.startswith("file://")
        mlflow.sklearn.log_model(LogisticRegression(), name="model", registered_model_name="rmn")
        mlflow.pyfunc.load_model("models:/rmn/1")


def test_log_input(mlflow_client, tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    path = tmp_path / "temp.csv"
    df.to_csv(path)
    dataset = from_pandas(df, source=path)

    mlflow.set_tracking_uri(mlflow_client.tracking_uri)

    with mlflow.start_run() as run:
        mlflow.log_input(dataset, "train", {"foo": "baz"})

    dataset_inputs = mlflow_client.get_run(run.info.run_id).inputs.dataset_inputs

    assert len(dataset_inputs) == 1
    assert dataset_inputs[0].dataset.name == "dataset"
    assert dataset_inputs[0].dataset.digest == "f0f3e026"
    assert dataset_inputs[0].dataset.source_type == "local"
    assert json.loads(dataset_inputs[0].dataset.source) == {"uri": str(path)}
    assert json.loads(dataset_inputs[0].dataset.schema) == {
        "mlflow_colspec": [
            {"name": "a", "type": "long", "required": True},
            {"name": "b", "type": "long", "required": True},
            {"name": "c", "type": "long", "required": True},
        ]
    }
    assert json.loads(dataset_inputs[0].dataset.profile) == {
        "num_rows": 2,
        "num_elements": 6,
    }

    assert len(dataset_inputs[0].tags) == 2
    assert dataset_inputs[0].tags[0].key == "foo"
    assert dataset_inputs[0].tags[0].value == "baz"
    assert dataset_inputs[0].tags[1].key == mlflow_tags.MLFLOW_DATASET_CONTEXT
    assert dataset_inputs[0].tags[1].value == "train"


def test_create_model_version_model_id(mlflow_client):
    name = "model"
    mlflow_client.create_registered_model(name)
    exp_id = mlflow_client.create_experiment("test")
    model = mlflow_client.create_logged_model(experiment_id=exp_id)
    mlflow_client.create_model_version(
        name=name,
        source=model.artifact_location,
        model_id=model.model_id,
    )
    model = mlflow_client.get_logged_model(model.model_id)
    assert model.tags["mlflow.modelVersions"] == '[{"name": "model", "version": 1}]'
    mlflow_client.create_model_version(
        name=name,
        source=model.artifact_location,
        model_id=model.model_id,
    )
    model = mlflow_client.get_logged_model(model.model_id)
    assert (
        model.tags["mlflow.modelVersions"]
        == '[{"name": "model", "version": 1}, {"name": "model", "version": 2}]'
    )


def test_log_inputs(mlflow_client):
    experiment_id = mlflow_client.create_experiment("log inputs test")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    dataset1 = Dataset(
        name="name1",
        digest="digest1",
        source_type="source_type1",
        source="source1",
    )
    dataset_inputs1 = [DatasetInput(dataset=dataset1, tags=[InputTag(key="tag1", value="value1")])]

    mlflow_client.log_inputs(run_id, dataset_inputs1)
    run = mlflow_client.get_run(run_id)
    assert len(run.inputs.dataset_inputs) == 1

    assert isinstance(run.inputs, RunInputs)
    assert isinstance(run.inputs.dataset_inputs[0], DatasetInput)
    assert isinstance(run.inputs.dataset_inputs[0].dataset, Dataset)
    assert run.inputs.dataset_inputs[0].dataset.name == "name1"
    assert run.inputs.dataset_inputs[0].dataset.digest == "digest1"
    assert run.inputs.dataset_inputs[0].dataset.source_type == "source_type1"
    assert run.inputs.dataset_inputs[0].dataset.source == "source1"
    assert len(run.inputs.dataset_inputs[0].tags) == 1
    assert run.inputs.dataset_inputs[0].tags[0].key == "tag1"
    assert run.inputs.dataset_inputs[0].tags[0].value == "value1"


def test_log_inputs_validation(mlflow_client):
    def assert_bad_request(payload, expected_error_message):
        response = _send_rest_tracking_post_request(
            mlflow_client.tracking_uri,
            "/api/2.0/mlflow/runs/log-inputs",
            payload,
        )
        assert response.status_code == 400
        assert expected_error_message in response.text

    dataset = Dataset(
        name="name1",
        digest="digest1",
        source_type="source_type1",
        source="source1",
    )
    tags = [InputTag(key="tag1", value="value1")]
    dataset_inputs = [
        json.loads(message_to_json(DatasetInput(dataset=dataset, tags=tags).to_proto()))
    ]
    assert_bad_request(
        {
            "datasets": dataset_inputs,
        },
        "Missing value for required parameter 'run_id'",
    )


def test_log_inputs_model(mlflow_client):
    experiment_id = mlflow_client.create_experiment("log inputs test")
    run = mlflow_client.create_run(experiment_id)
    model = mlflow_client.create_logged_model(experiment_id=experiment_id)
    dataset = Dataset(
        name="name1",
        digest="digest1",
        source_type="source_type1",
        source="source1",
    )
    dataset_inputs = [
        DatasetInput(
            dataset=dataset,
            tags=[InputTag(key=MLFLOW_DATASET_CONTEXT, value="training")],
        )
    ]
    mlflow_client.log_inputs(
        run.info.run_id,
        models=[LoggedModelInput(model_id=model.model_id)],
        datasets=dataset_inputs,
    )
    run = mlflow_client.get_run(run.info.run_id)
    assert len(run.inputs.model_inputs) == 1


def test_update_run_name_without_changing_status(mlflow_client):
    experiment_id = mlflow_client.create_experiment("update run name")
    created_run = mlflow_client.create_run(experiment_id)
    mlflow_client.set_terminated(created_run.info.run_id, "FINISHED")

    mlflow_client.update_run(created_run.info.run_id, name="name_abc")
    updated_run_info = mlflow_client.get_run(created_run.info.run_id).info
    assert updated_run_info.run_name == "name_abc"
    assert updated_run_info.status == "FINISHED"


def test_create_promptlab_run_handler_rejects_invalid_requests(mlflow_client):
    def assert_response(resp, message_part):
        assert resp.status_code == 400
        response_json = resp.json()
        assert response_json.get("error_code") == "INVALID_PARAMETER_VALUE"
        assert message_part in response_json.get("message", "")

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/runs/create-promptlab-run",
        json={},
    )
    assert_response(
        response,
        "CreatePromptlabRun request must specify experiment_id.",
    )

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/runs/create-promptlab-run",
        json={"experiment_id": "123"},
    )
    assert_response(
        response,
        "CreatePromptlabRun request must specify prompt_template.",
    )

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/runs/create-promptlab-run",
        json={"experiment_id": "123", "prompt_template": "my_prompt_template"},
    )
    assert_response(
        response,
        "CreatePromptlabRun request must specify prompt_parameters.",
    )

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/runs/create-promptlab-run",
        json={
            "experiment_id": "123",
            "prompt_template": "my_prompt_template",
            "prompt_parameters": [{"key": "my_key", "value": "my_value"}],
        },
    )
    assert_response(
        response,
        "CreatePromptlabRun request must specify model_route.",
    )

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/runs/create-promptlab-run",
        json={
            "experiment_id": "123",
            "prompt_template": "my_prompt_template",
            "prompt_parameters": [{"key": "my_key", "value": "my_value"}],
            "model_route": "my_route",
        },
    )
    assert_response(
        response,
        "CreatePromptlabRun request must specify model_input.",
    )

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/runs/create-promptlab-run",
        json={
            "experiment_id": "123",
            "prompt_template": "my_prompt_template",
            "prompt_parameters": [{"key": "my_key", "value": "my_value"}],
            "model_route": "my_route",
            "model_input": "my_input",
        },
    )
    assert_response(
        response,
        "CreatePromptlabRun request must specify mlflow_version.",
    )

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/runs/create-promptlab-run",
        json={
            "experiment_id": "123",
            "prompt_template": "my_prompt_template",
            "prompt_parameters": [{"key": "my_key", "value": "my_value"}],
            "model_route": "my_route",
            "model_input": "my_input",
            "mlflow_version": "1.0.0",
        },
    )


def test_create_promptlab_run_handler_returns_expected_results(mlflow_client):
    experiment_id = mlflow_client.create_experiment("log inputs test")

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/runs/create-promptlab-run",
        json={
            "experiment_id": experiment_id,
            "run_name": "my_run_name",
            "prompt_template": "my_prompt_template",
            "prompt_parameters": [{"key": "my_key", "value": "my_value"}],
            "model_route": "my_route",
            "model_parameters": [{"key": "temperature", "value": "0.1"}],
            "model_input": "my_input",
            "model_output": "my_output",
            "model_output_parameters": [{"key": "latency", "value": "100"}],
            "mlflow_version": "1.0.0",
            "user_id": "username",
            "start_time": 456,
        },
    )
    assert response.status_code == 200
    run_json = response.json()
    assert run_json["run"]["info"]["run_name"] == "my_run_name"
    assert run_json["run"]["info"]["experiment_id"] == experiment_id
    assert run_json["run"]["info"]["user_id"] == "username"
    assert run_json["run"]["info"]["status"] == "FINISHED"
    assert run_json["run"]["info"]["start_time"] == 456

    assert {"key": "model_route", "value": "my_route"} in run_json["run"]["data"]["params"]
    assert {"key": "prompt_template", "value": "my_prompt_template"} in run_json["run"]["data"][
        "params"
    ]
    assert {"key": "temperature", "value": "0.1"} in run_json["run"]["data"]["params"]

    assert {
        "key": "mlflow.loggedArtifacts",
        "value": '[{"path": "eval_results_table.json", "type": "table"}]',
    } in run_json["run"]["data"]["tags"]
    assert {"key": "mlflow.runSourceType", "value": "PROMPT_ENGINEERING"} in run_json["run"][
        "data"
    ]["tags"]


def test_gateway_proxy_handler_rejects_invalid_requests(mlflow_client):
    def assert_response(resp, message_part):
        assert resp.status_code == 400
        response_json = resp.json()
        assert response_json.get("error_code") == "INVALID_PARAMETER_VALUE"
        assert message_part in response_json.get("message", "")

    with _init_server(
        backend_uri=mlflow_client.tracking_uri,
        root_artifact_uri=mlflow_client.tracking_uri,
        extra_env={"MLFLOW_DEPLOYMENTS_TARGET": "http://localhost:5001"},
        server_type="flask",
    ) as url:
        patched_client = MlflowClient(url)

        response = requests.post(
            f"{patched_client.tracking_uri}/ajax-api/2.0/mlflow/gateway-proxy",
            json={},
        )
        assert_response(
            response,
            "Deployments proxy request must specify a gateway_path.",
        )

        response = requests.post(
            f"{patched_client.tracking_uri}/ajax-api/2.0/mlflow/gateway-proxy",
            json={"gateway_path": "foo/bar"},
        )
        assert_response(
            response,
            "Invalid gateway_path: foo/bar for method: POST",
        )

        response = requests.post(
            f"{patched_client.tracking_uri}/ajax-api/2.0/mlflow/gateway-proxy",
            json={"gateway_path": "foo/bar/baz"},
        )
        assert_response(
            response,
            "Invalid gateway_path: foo/bar/baz for method: POST",
        )

        response = requests.get(
            f"{patched_client.tracking_uri}/ajax-api/2.0/mlflow/gateway-proxy",
            params={"gateway_path": "hello/world"},
        )
        assert_response(
            response,
            "Invalid gateway_path: hello/world for method: GET",
        )

        # Unsupported method
        response = requests.delete(
            f"{patched_client.tracking_uri}/ajax-api/2.0/mlflow/gateway-proxy",
        )
        assert response.status_code == 405


def test_upload_artifact_handler_rejects_invalid_requests(mlflow_client):
    def assert_response(resp, message_part):
        assert resp.status_code == 400
        response_json = resp.json()
        assert response_json.get("error_code") == "INVALID_PARAMETER_VALUE"
        assert message_part in response_json.get("message", "")

    experiment_id = mlflow_client.create_experiment("upload_artifacts_test")
    created_run = mlflow_client.create_run(experiment_id)

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/upload-artifact", params={}
    )
    assert_response(response, "Request must specify run_uuid.")

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/upload-artifact",
        params={
            "run_uuid": created_run.info.run_id,
        },
    )
    assert_response(response, "Request must specify path.")

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/upload-artifact",
        params={"run_uuid": created_run.info.run_id, "path": ""},
    )
    assert_response(response, "Request must specify path.")

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/upload-artifact",
        params={"run_uuid": created_run.info.run_id, "path": "../test.txt"},
    )
    assert_response(response, "Invalid path")

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/upload-artifact",
        params={
            "run_uuid": created_run.info.run_id,
            "path": "test.txt",
        },
    )
    assert_response(response, "Request must specify data.")


def test_upload_artifact_handler(mlflow_client):
    experiment_id = mlflow_client.create_experiment("upload_artifacts_test")
    created_run = mlflow_client.create_run(experiment_id)

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/upload-artifact",
        params={
            "run_uuid": created_run.info.run_id,
            "path": "test.txt",
        },
        data="hello world",
    )
    assert response.status_code == 200

    response = requests.get(
        f"{mlflow_client.tracking_uri}/get-artifact",
        params={
            "run_uuid": created_run.info.run_id,
            "path": "test.txt",
        },
    )
    assert response.status_code == 200
    assert response.text == "hello world"


def test_graphql_handler(mlflow_client):
    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": 'query testQuery {test(inputString: "abc") { output }}',
            "operationName": "testQuery",
        },
        headers={"content-type": "application/json; charset=utf-8"},
    )
    assert response.status_code == 200


def test_graphql_handler_batching_raise_error(mlflow_client):
    # Test max root fields limit
    batch_query = (
        "query testQuery {"
        + " ".join(
            [
                f"key_{i}: " + 'test(inputString: "abc") { output }'
                for i in range(int(MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS.get()) + 2)
            ]
        )
        + "}"
    )
    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": batch_query,
            "operationName": "testQuery",
        },
        headers={"content-type": "application/json; charset=utf-8"},
    )
    assert response.status_code == 200
    assert (
        f"GraphQL queries should have at most {MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS.get()}"
        in response.json()["errors"][0]
    )

    # Test max aliases limit
    batch_query = (
        'query testQuery {mlflowGetExperiment(input: {experimentId: "123"}) {'
        + " ".join(
            f"experiment_{i}: " + "experiment { name }"
            for i in range(int(MLFLOW_SERVER_GRAPHQL_MAX_ALIASES.get()) + 2)
        )
        + "}}"
    )
    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": batch_query,
            "operationName": "testQuery",
        },
    )
    assert response.status_code == 200
    assert (
        f"queries should have at most {MLFLOW_SERVER_GRAPHQL_MAX_ALIASES.get()} aliases"
        in response.json()["errors"][0]
    )

    # Test max depth limit
    inner = "name"
    for _ in range(12):
        inner = f"name {{ {inner} }}"
    deep_query = (
        'query testQuery { mlflowGetExperiment(input: {experimentId: "123"}) { experiment { '
        + inner
        + " } } }"
    )
    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": deep_query,
            "operationName": "testQuery",
        },
    )
    assert response.status_code == 200
    assert "Query exceeds maximum depth of 10" in response.json()["errors"][0]

    # Test max selections limit
    # Exceed the 1000 selection limit
    selections = [f"field_{i} {{ name }}" for i in range(1002)]
    selections_query = (
        'query testQuery { mlflowGetExperiment(input: {experimentId: "123"}) { experiment { '
        + " ".join(selections)
        + " } } }"
    )
    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": selections_query,
            "operationName": "testQuery",
        },
    )
    assert response.status_code == 200
    assert "Query exceeds maximum total selections of 1000" in response.json()["errors"][0]


def test_get_experiment_graphql(mlflow_client):
    experiment_id = mlflow_client.create_experiment("GraphqlTest")
    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": 'query testQuery {mlflowGetExperiment(input: {experimentId: "'
            + experiment_id
            + '"}) { experiment { name } }}',
            "operationName": "testQuery",
        },
        headers={"content-type": "application/json; charset=utf-8"},
    )
    assert response.status_code == 200
    json = response.json()
    assert json["data"]["mlflowGetExperiment"]["experiment"]["name"] == "GraphqlTest"


def test_get_run_and_experiment_graphql(mlflow_client):
    name = "GraphqlTest"
    mlflow_client.create_registered_model(name)
    experiment_id = mlflow_client.create_experiment(name)
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    mlflow_client.create_model_version("GraphqlTest", "runs:/graphql_test/model", run_id)
    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": f"""
                query testQuery @component(name: "Test") {{
                    mlflowGetRun(input: {{runId: "{run_id}"}}) {{
                        run {{
                            info {{
                                status
                            }}
                            experiment {{
                                name
                            }}
                            modelVersions {{
                                name
                            }}
                        }}
                    }}
                }}
            """,
            "operationName": "testQuery",
        },
        headers={"content-type": "application/json; charset=utf-8"},
    )
    assert response.status_code == 200
    json = response.json()
    assert json["errors"] is None
    assert json["data"]["mlflowGetRun"]["run"]["info"]["status"] == created_run.info.status
    assert json["data"]["mlflowGetRun"]["run"]["experiment"]["name"] == name
    assert json["data"]["mlflowGetRun"]["run"]["modelVersions"][0]["name"] == name


def test_legacy_start_and_end_trace_v2(mlflow_client):
    experiment_id = mlflow_client.create_experiment("start end trace")

    # Trace CRUD APIs are not directly exposed as public API of MlflowClient,
    # so we use the underlying tracking client to test them.
    store = mlflow_client._tracing_client.store

    # Helper function to remove auto-added system tags (mlflow.xxx) from testing
    def _exclude_system_tags(tags: dict[str, str]):
        return {k: v for k, v in tags.items() if not k.startswith("mlflow.")}

    trace_info = store.deprecated_start_trace_v2(
        experiment_id=experiment_id,
        timestamp_ms=1000,
        request_metadata={
            "meta1": "apple",
            "meta2": "grape",
        },
        tags={
            "tag1": "football",
            "tag2": "basketball",
        },
    )
    assert trace_info.request_id is not None
    assert trace_info.experiment_id == experiment_id
    assert trace_info.timestamp_ms == 1000
    assert trace_info.execution_time_ms == 0
    assert trace_info.status == TraceStatus.IN_PROGRESS
    assert trace_info.request_metadata == {
        "meta1": "apple",
        "meta2": "grape",
    }
    assert _exclude_system_tags(trace_info.tags) == {
        "tag1": "football",
        "tag2": "basketball",
    }

    trace_info = store.deprecated_end_trace_v2(
        request_id=trace_info.request_id,
        timestamp_ms=3000,
        status=TraceStatus.OK,
        request_metadata={
            "meta1": "orange",
            "meta3": "banana",
        },
        tags={
            "tag1": "soccer",
            "tag3": "tennis",
        },
    )
    assert trace_info.request_id is not None
    assert trace_info.experiment_id == experiment_id
    assert trace_info.timestamp_ms == 1000
    assert trace_info.execution_time_ms == 2000
    assert trace_info.status == TraceStatus.OK
    assert trace_info.request_metadata == {
        "meta1": "orange",
        "meta2": "grape",
        "meta3": "banana",
    }
    assert _exclude_system_tags(trace_info.tags) == {
        "tag1": "soccer",
        "tag2": "basketball",
        "tag3": "tennis",
    }


def test_start_trace(mlflow_client):
    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    experiment_id = mlflow.set_experiment("start end trace").experiment_id

    # Helper function to remove auto-added system tags (mlflow.xxx) from testing
    def _exclude_system_keys(d: dict[str, str]):
        return {k: v for k, v in d.items() if not k.startswith("mlflow.")}

    with mock.patch("mlflow.tracing.export.mlflow_v3._logger.warning") as mock_warning:
        with mlflow.start_span(name="test") as span:
            mlflow.update_current_trace(
                tags={
                    "tag1": "football",
                    "tag2": "basketball",
                },
                metadata={
                    "meta1": "apple",
                    "meta2": "grape",
                },
            )

    trace = mlflow_client.get_trace(span.trace_id)
    assert trace.info.trace_id == span.trace_id
    assert trace.info.experiment_id == experiment_id
    assert trace.info.request_time > 0
    assert trace.info.execution_duration is not None
    assert trace.info.state == TraceState.OK
    assert _exclude_system_keys(trace.info.trace_metadata) == {
        "meta1": "apple",
        "meta2": "grape",
    }
    assert trace.info.trace_metadata[TRACE_SCHEMA_VERSION_KEY] == "3"
    assert _exclude_system_keys(trace.info.tags) == {
        "tag1": "football",
        "tag2": "basketball",
    }

    # No "Failed to log span to MLflow backend" warning should be issued
    for call in mock_warning.call_args_list:
        assert "Failed to log span to MLflow backend" not in str(call)


def test_get_trace(mlflow_client):
    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    experiment_id = mlflow_client.create_experiment("get trace")
    span = mlflow_client.start_trace(name="test", experiment_id=experiment_id)
    mlflow_client.end_trace(request_id=span.request_id, status=TraceStatus.OK)
    trace = mlflow_client.get_trace(span.request_id)
    assert trace is not None
    assert trace.info.request_id == span.request_id
    assert trace.info.experiment_id == experiment_id
    assert trace.info.state == TraceState.OK
    assert len(trace.data.spans) == 1
    assert trace.data.spans[0].name == "test"
    assert trace.data.spans[0].status.status_code == SpanStatusCode.OK
    assert trace.data.spans[0].status.description == ""


def test_search_traces(mlflow_client):
    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    experiment_id = mlflow_client.create_experiment("search traces")

    # Create test traces
    def _create_trace(name, status):
        span = mlflow_client.start_trace(name=name, experiment_id=experiment_id)
        mlflow_client.end_trace(request_id=span.request_id, status=status)
        return span.request_id

    request_id_1 = _create_trace(name="trace1", status=TraceStatus.OK)
    request_id_2 = _create_trace(name="trace2", status=TraceStatus.OK)
    request_id_3 = _create_trace(name="trace3", status=TraceStatus.ERROR)

    def _get_request_ids(traces):
        return [t.info.request_id for t in traces]

    # Validate search
    traces = mlflow_client.search_traces(experiment_ids=[experiment_id])
    assert _get_request_ids(traces) == [request_id_3, request_id_2, request_id_1]
    assert traces.token is None

    traces = mlflow_client.search_traces(
        experiment_ids=[experiment_id],
        filter_string="status = 'OK'",
        order_by=["timestamp ASC"],
    )
    assert _get_request_ids(traces) == [request_id_1, request_id_2]
    assert traces.token is None

    traces = mlflow_client.search_traces(
        experiment_ids=[experiment_id],
        max_results=2,
    )
    assert _get_request_ids(traces) == [request_id_3, request_id_2]
    assert traces.token is not None
    traces = mlflow_client.search_traces(
        experiment_ids=[experiment_id],
        page_token=traces.token,
    )
    assert _get_request_ids(traces) == [request_id_1]
    assert traces.token is None


def test_search_traces_parameter_validation(mlflow_client):
    with pytest.raises(
        MlflowException,
        match="Locations must be a list of experiment IDs",
    ):
        mlflow_client.search_traces(locations=["catalog.schema"])


def test_search_traces_match_text(mlflow_client, store_type):
    if store_type == "file":
        pytest.skip("File store doesn't support full text search")

    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    experiment_id = mlflow_client.create_experiment("search traces full text")

    # Create test traces
    def _create_trace(name, attributes):
        span = mlflow_client.start_trace(name=name, experiment_id=experiment_id)
        span.set_attributes(attributes)
        mlflow_client.end_trace(request_id=span.trace_id, status=TraceStatus.OK)
        return span.trace_id

    trace_id_1 = _create_trace(name="trace1", attributes={"test": "value1"})
    trace_id_2 = _create_trace(name="trace2", attributes={"test": "value2"})
    trace_id_3 = _create_trace(name="trace3", attributes={"test3": "I like it"})

    traces = mlflow_client.search_traces(experiment_ids=[experiment_id])
    assert len([t.info.trace_id for t in traces]) == 3
    assert traces.token is None

    traces = mlflow_client.search_traces(
        experiment_ids=[experiment_id], filter_string="trace.text LIKE '%trace%'"
    )
    assert len([t.info.trace_id for t in traces]) == 3
    assert traces.token is None

    traces = mlflow_client.search_traces(
        experiment_ids=[experiment_id], filter_string="trace.text LIKE '%value%'"
    )
    assert {t.info.trace_id for t in traces} == {trace_id_1, trace_id_2}

    traces = mlflow_client.search_traces(
        experiment_ids=[experiment_id], filter_string="trace.text LIKE '%I like it%'"
    )
    assert [t.info.trace_id for t in traces] == [trace_id_3]


def test_delete_traces(mlflow_client):
    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    experiment_id = mlflow_client.create_experiment("delete traces")

    def _create_trace(name, status):
        span = mlflow_client.start_trace(name=name, experiment_id=experiment_id)
        mlflow_client.end_trace(request_id=span.request_id, status=status)
        return span.request_id

    def _is_trace_exists(request_id):
        try:
            trace_info = mlflow_client._tracing_client.get_trace_info(request_id)
            return trace_info is not None
        except RestException as e:
            if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
                return False
            raise

    # Case 1: Delete all traces under experiment ID
    request_id_1 = _create_trace(name="trace1", status=TraceStatus.OK)
    request_id_2 = _create_trace(name="trace2", status=TraceStatus.OK)
    assert _is_trace_exists(request_id_1)
    assert _is_trace_exists(request_id_2)

    deleted_count = mlflow_client.delete_traces(experiment_id, max_timestamp_millis=int(1e15))
    assert deleted_count == 2
    assert not _is_trace_exists(request_id_1)
    assert not _is_trace_exists(request_id_2)

    # Case 2: Delete with max_traces limit
    request_id_1 = _create_trace(name="trace1", status=TraceStatus.OK)
    time.sleep(0.1)  # Add some time gap to avoid timestamp collision
    request_id_2 = _create_trace(name="trace2", status=TraceStatus.OK)

    deleted_count = mlflow_client.delete_traces(
        experiment_id, max_traces=1, max_timestamp_millis=int(1e15)
    )
    assert deleted_count == 1
    # TODO: Currently the deletion order in the file store is random (based on
    # the order of the trace files in the directory), so we don't validate which
    # one is deleted. Uncomment the following lines once the deletion order is fixed.
    # assert not _is_trace_exists(request_id_1)  # Old created trace should be deleted
    # assert _is_trace_exists(request_id_2)

    # Case 3: Delete with explicit request ID
    request_id_1 = _create_trace(name="trace1", status=TraceStatus.OK)
    request_id_2 = _create_trace(name="trace2", status=TraceStatus.OK)

    deleted_count = mlflow_client.delete_traces(experiment_id, trace_ids=[request_id_1])
    assert deleted_count == 1
    assert not _is_trace_exists(request_id_1)
    assert _is_trace_exists(request_id_2)


def test_calculate_trace_filter_correlation(mlflow_client, store_type):
    if store_type == "file":
        pytest.skip("File store doesn't support calculate_trace_filter_correlation")

    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    experiment_id = mlflow_client.create_experiment("correlation test")

    def _create_trace(name, tags):
        span = mlflow_client.start_trace(name=name, experiment_id=experiment_id, tags=tags)
        mlflow_client.end_trace(request_id=span.request_id, status=TraceStatus.OK)
        return span.request_id

    for i in range(6):
        _create_trace(f"trace-prod-tool-{i}", {"env": "prod", "span_type": "TOOL"})

    for i in range(4):
        _create_trace(f"trace-dev-{i}", {"env": "dev", "span_type": "LLM" if i >= 1 else "TOOL"})

    client = TracingClient(tracking_uri=mlflow_client.tracking_uri)

    result = client.calculate_trace_filter_correlation(
        experiment_ids=[experiment_id],
        filter_string1="tags.env = 'prod'",
        filter_string2="tags.span_type = 'TOOL'",
    )

    assert isinstance(result, TraceFilterCorrelationResult)
    assert result.total_count == 10
    assert result.filter1_count == 6
    assert result.filter2_count == 7
    assert result.joint_count == 6
    assert 0.6 < result.npmi < 0.8
    assert result.npmi_smoothed is not None

    result2 = client.calculate_trace_filter_correlation(
        experiment_ids=[experiment_id],
        filter_string1="tags.env = 'dev'",
        filter_string2="tags.span_type = 'LLM'",
    )

    assert result2.total_count == 10
    assert result2.filter1_count == 4
    assert result2.filter2_count == 3
    assert result2.joint_count == 3
    assert result2.npmi > 0.5

    result3 = client.calculate_trace_filter_correlation(
        experiment_ids=[experiment_id],
        filter_string1="tags.env = 'staging'",
        filter_string2="tags.span_type = 'TOOL'",
    )

    assert result3.total_count == 10
    assert result3.filter1_count == 0
    assert result3.filter2_count == 7
    assert result3.joint_count == 0
    assert math.isnan(result3.npmi)

    with pytest.raises(MlflowException, match="Invalid"):
        client.calculate_trace_filter_correlation(
            experiment_ids=[experiment_id],
            filter_string1="invalid.filter = 'test'",
            filter_string2="tags.span_type = 'TOOL'",
        )


def test_set_and_delete_trace_tag(mlflow_client):
    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    experiment_id = mlflow_client.create_experiment("set delete tag")

    # Create test trace
    trace_info = mlflow_client._tracing_client.start_trace(
        TraceInfo(
            trace_id="tr-1234",
            trace_location=TraceLocation.from_experiment_id(experiment_id),
            request_time=1000,
            execution_duration=2000,
            state=TraceState.OK,
            tags={
                "tag1": "red",
                "tag2": "blue",
            },
        )
    )

    # Validate set tag
    mlflow_client.set_trace_tag(trace_info.request_id, "tag1", "green")
    trace_info = mlflow_client._tracing_client.get_trace_info(trace_info.request_id)
    assert trace_info.tags["tag1"] == "green"

    # Validate delete tag
    mlflow_client.delete_trace_tag(trace_info.request_id, "tag2")
    trace_info = mlflow_client._tracing_client.get_trace_info(trace_info.request_id)
    assert "tag2" not in trace_info.tags


def test_query_trace_metrics(mlflow_client, store_type):
    if store_type == "file":
        pytest.skip("File store doesn't support query trace metrics")

    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    experiment_id = mlflow_client.create_experiment("query trace metrics")

    # Create test traces
    def _create_trace(name, status):
        span = mlflow_client.start_trace(name=name, experiment_id=experiment_id)
        mlflow_client.end_trace(request_id=span.request_id, status=status)
        return span.request_id

    _create_trace(name="trace1", status=TraceStatus.OK)
    _create_trace(name="trace2", status=TraceStatus.OK)
    _create_trace(name="trace3", status=TraceStatus.ERROR)

    metrics = mlflow_client._tracing_client.store.query_trace_metrics(
        experiment_ids=[experiment_id],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        dimensions=[TraceMetricDimensionKey.TRACE_STATUS],
    )
    assert len(metrics) == 2
    assert asdict(metrics[0]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {TraceMetricDimensionKey.TRACE_STATUS: "ERROR"},
        "values": {"COUNT": 1},
    }

    assert asdict(metrics[1]) == {
        "metric_name": TraceMetricKey.TRACE_COUNT,
        "dimensions": {TraceMetricDimensionKey.TRACE_STATUS: "OK"},
        "values": {"COUNT": 2},
    }


@pytest.mark.parametrize("allow_partial", [True, False])
def test_get_trace_handler(mlflow_client, allow_partial: bool, store_type):
    if store_type == "file":
        pytest.skip("File store doesn't support get trace handler")

    mlflow.set_tracking_uri(mlflow_client.tracking_uri)

    with mlflow.start_span(name="test") as span:
        span.set_attributes({"fruit": "apple"})

    response = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/3.0/mlflow/traces/get",
        params={"trace_id": span.trace_id, "allow_partial": allow_partial},
    )

    assert response.status_code == 200

    trace = response.json()["trace"]
    assert trace["trace_info"]["trace_id"] == span.trace_id
    assert len(trace["spans"]) == 1
    assert trace["spans"][0]["name"] == "test"
    attributes = trace["spans"][0]["attributes"]
    assert {"key": "fruit", "value": {"string_value": "apple"}} in attributes


def test_get_trace_artifact_handler(mlflow_client):
    mlflow.set_tracking_uri(mlflow_client.tracking_uri)

    with mlflow.start_span(name="test") as span:
        span.set_attributes({"fruit": "apple"})
        span.add_event(SpanEvent("test_event", timestamp=99999, attributes={"foo": "bar"}))

    response = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/get-trace-artifact",
        params={"request_id": span.trace_id},
    )
    assert response.status_code == 200
    assert response.headers["Content-Disposition"] == "attachment; filename=traces.json"

    # Validate content
    trace_data = TraceData.from_dict(json.loads(response.text))
    assert trace_data.spans[0].to_dict() == span.to_dict()


def test_link_traces_to_run_and_search_traces(mlflow_client, store_type):
    # Skip file store because it doesn't support linking traces to runs
    if store_type == "file":
        pytest.skip("File store doesn't support linking traces to runs")

    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    experiment_id = mlflow.set_experiment("link traces to run test").experiment_id

    run = mlflow_client.create_run(experiment_id)
    run_id = run.info.run_id

    # 1. Trace created under a run
    with mlflow.start_run(run_id=run_id):
        with mlflow.start_span(name="trace1") as span1:
            span1.set_attributes({"test": "value1"})
        trace_id_1 = span1.trace_id

    # 2. Trace associated with a run
    with mlflow.start_span(name="trace2") as span2:
        span2.set_attributes({"test": "value2"})
    trace_id_2 = span2.trace_id
    mlflow_client.link_traces_to_run(trace_ids=[trace_id_2], run_id=run_id)

    # 3. Trace not associated with a run
    with mlflow.start_span(name="trace3") as span3:
        span3.set_attributes({"test": "value3"})
    trace_id_3 = span3.trace_id

    # Search traces without run_id filter - should return all traces in experiment
    all_traces = mlflow_client.search_traces(experiment_ids=[experiment_id])
    assert {t.info.trace_id for t in all_traces} == {trace_id_1, trace_id_2, trace_id_3}

    # Search traces with run_id filter - should return only linked traces
    linked_traces = mlflow_client.search_traces(
        experiment_ids=[experiment_id], filter_string=f"attribute.run_id = '{run_id}'"
    )
    linked_trace_ids = [t.info.trace_id for t in linked_traces]
    assert len(linked_trace_ids) == 2
    assert set(linked_trace_ids) == {trace_id_1, trace_id_2}


def test_get_metric_history_bulk_interval_graphql(mlflow_client):
    name = "GraphqlTest"
    mlflow_client.create_registered_model(name)
    experiment_id = mlflow_client.create_experiment(name)
    created_run = mlflow_client.create_run(experiment_id)

    metric_name = "metric_0"
    for i in range(10):
        mlflow_client.log_metric(created_run.info.run_id, metric_name, i, step=i)

    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": f"""
                query testQuery {{
                    mlflowGetMetricHistoryBulkInterval(input: {{
                        runIds: ["{created_run.info.run_id}"],
                        metricKey: "{metric_name}",
                    }}) {{
                        metrics {{
                            key
                            timestamp
                            value
                        }}
                    }}
                }}
            """,
            "operationName": "testQuery",
        },
        headers={"content-type": "application/json; charset=utf-8"},
    )

    assert response.status_code == 200
    json = response.json()
    expected = [{"key": metric_name, "timestamp": mock.ANY, "value": i} for i in range(10)]
    assert json["data"]["mlflowGetMetricHistoryBulkInterval"]["metrics"] == expected


def test_search_runs_graphql(mlflow_client):
    name = "GraphqlTest"
    mlflow_client.create_registered_model(name)
    experiment_id = mlflow_client.create_experiment(name)
    created_run_1 = mlflow_client.create_run(experiment_id)
    created_run_2 = mlflow_client.create_run(experiment_id)

    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": f"""
                mutation testMutation {{
                    mlflowSearchRuns(input: {{ experimentIds: ["{experiment_id}"] }}) {{
                        runs {{
                            info {{
                                runId
                            }}
                        }}
                    }}
                }}
            """,
            "operationName": "testMutation",
        },
        headers={"content-type": "application/json; charset=utf-8"},
    )

    assert response.status_code == 200
    json = response.json()
    expected = [
        {"info": {"runId": created_run_2.info.run_id}},
        {"info": {"runId": created_run_1.info.run_id}},
    ]
    assert json["data"]["mlflowSearchRuns"]["runs"] == expected


def test_list_artifacts_graphql(mlflow_client, tmp_path):
    name = "GraphqlTest"
    experiment_id = mlflow_client.create_experiment(name)
    created_run_id = mlflow_client.create_run(experiment_id).info.run_id
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello world")
    mlflow_client.log_artifact(created_run_id, file_path.absolute().as_posix())
    mlflow_client.log_artifact(created_run_id, file_path.absolute().as_posix(), "testDir")

    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": f"""
                query testQuery {{
                    files: mlflowListArtifacts(input: {{
                        runId: "{created_run_id}",
                    }}) {{
                            files {{
                            path
                            isDir
                            fileSize
                        }}
                    }}
                }}
            """,
            "operationName": "testQuery",
        },
        headers={"content-type": "application/json; charset=utf-8"},
    )

    assert response.status_code == 200
    json = response.json()
    file_expected = [
        {"path": "test.txt", "isDir": False, "fileSize": "11"},
        {"path": "testDir", "isDir": True, "fileSize": "0"},
    ]
    assert json["data"]["files"]["files"] == file_expected

    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": f"""
                query testQuery {{
                    subdir: mlflowListArtifacts(input: {{
                        runId: "{created_run_id}",
                        path: "testDir",
                    }}) {{
                            files {{
                            path
                            isDir
                            fileSize
                        }}
                    }}
                }}
            """,
            "operationName": "testQuery",
        },
        headers={"content-type": "application/json; charset=utf-8"},
    )

    assert response.status_code == 200
    json = response.json()
    subdir_expected = [
        {"path": "testDir/test.txt", "isDir": False, "fileSize": "11"},
    ]
    assert json["data"]["subdir"]["files"] == subdir_expected


def test_search_datasets_graphql(mlflow_client):
    name = "GraphqlTest"
    experiment_id = mlflow_client.create_experiment(name)
    created_run_id = mlflow_client.create_run(experiment_id).info.run_id
    dataset1 = Dataset(
        name="test-dataset-1",
        digest="12345",
        source_type="script",
        source="test",
    )
    dataset_input1 = DatasetInput(dataset=dataset1, tags=[])
    dataset2 = Dataset(
        name="test-dataset-2",
        digest="12346",
        source_type="script",
        source="test",
    )
    dataset_input2 = DatasetInput(
        dataset=dataset2, tags=[InputTag(key=MLFLOW_DATASET_CONTEXT, value="training")]
    )
    mlflow_client.log_inputs(created_run_id, [dataset_input1, dataset_input2])

    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": f"""
                mutation testMutation {{
                    mlflowSearchDatasets(input:{{experimentIds: ["{experiment_id}"]}}) {{
                        datasetSummaries {{
                            experimentId
                            name
                            digest
                            context
                        }}
                    }}
                }}
            """,
            "operationName": "testMutation",
        },
        headers={"content-type": "application/json; charset=utf-8"},
    )

    assert response.status_code == 200
    json = response.json()

    def sort_dataset_summaries(l1):
        return sorted(l1, key=lambda x: x["digest"])

    expected = sort_dataset_summaries(
        [
            {
                "experimentId": experiment_id,
                "name": "test-dataset-2",
                "digest": "12346",
                "context": "training",
            },
            {
                "experimentId": experiment_id,
                "name": "test-dataset-1",
                "digest": "12345",
                "context": "",
            },
        ]
    )
    assert (
        sort_dataset_summaries(json["data"]["mlflowSearchDatasets"]["datasetSummaries"]) == expected
    )


def test_create_logged_model(mlflow_client: MlflowClient):
    exp_id = mlflow_client.create_experiment("create_logged_model")
    model = mlflow_client.create_logged_model(exp_id)
    loaded_model = mlflow_client.get_logged_model(model.model_id)
    assert model.model_id == loaded_model.model_id

    model = mlflow_client.create_logged_model(exp_id, name="my_model")
    loaded_model = mlflow_client.get_logged_model(model.model_id)
    assert model.name == "my_model"

    model = mlflow_client.create_logged_model(exp_id, model_type="LLM")
    loaded_model = mlflow_client.get_logged_model(model.model_id)
    assert model.model_type == "LLM"

    model = mlflow_client.create_logged_model(exp_id, source_run_id="123")
    loaded_model = mlflow_client.get_logged_model(model.model_id)
    assert model.source_run_id == "123"

    model = mlflow_client.create_logged_model(exp_id, params={"param": "value"})
    loaded_model = mlflow_client.get_logged_model(model.model_id)
    assert model.params == {"param": "value"}

    model = mlflow_client.create_logged_model(exp_id, tags={"tag": "value"})
    loaded_model = mlflow_client.get_logged_model(model.model_id)
    assert model.tags == {"tag": "value"}


def test_log_logged_model_params(mlflow_client: MlflowClient):
    exp_id = mlflow_client.create_experiment("create_logged_model")
    model = mlflow_client.create_logged_model(exp_id)
    mlflow_client.log_model_params(model.model_id, {"param": "value"})
    loaded_model = mlflow_client.get_logged_model(model.model_id)
    assert loaded_model.params == {"param": "value"}


def test_finalize_logged_model(mlflow_client: MlflowClient):
    exp_id = mlflow_client.create_experiment("create_logged_model")
    model = mlflow_client.create_logged_model(exp_id)
    finalized_model = mlflow_client.finalize_logged_model(model.model_id, LoggedModelStatus.READY)
    assert finalized_model.status == LoggedModelStatus.READY

    finalized_model = mlflow_client.finalize_logged_model(model.model_id, LoggedModelStatus.FAILED)
    assert finalized_model.status == LoggedModelStatus.FAILED


def test_delete_logged_model(mlflow_client: MlflowClient):
    exp_id = mlflow_client.create_experiment("delete_logged_model")
    model = mlflow_client.create_logged_model(experiment_id=exp_id)
    mlflow_client.delete_logged_model(model.model_id)
    with pytest.raises(MlflowException, match="not found"):
        mlflow_client.get_logged_model(model.model_id)

    models = mlflow_client.search_logged_models(experiment_ids=[exp_id])
    assert len(models) == 0


def test_set_logged_model_tags(mlflow_client: MlflowClient):
    exp_id = mlflow_client.create_experiment("create_logged_model")
    model = mlflow_client.create_logged_model(exp_id)
    mlflow_client.set_logged_model_tags(model.model_id, {"tag1": "value1", "tag2": "value2"})
    loaded_model = mlflow_client.get_logged_model(model.model_id)
    assert loaded_model.tags == {"tag1": "value1", "tag2": "value2"}

    mlflow_client.set_logged_model_tags(model.model_id, {"tag1": "value3"})
    loaded_model = mlflow_client.get_logged_model(model.model_id)
    assert loaded_model.tags == {"tag1": "value3", "tag2": "value2"}


def test_delete_logged_model_tag(mlflow_client: MlflowClient):
    exp_id = mlflow_client.create_experiment("create_logged_model")
    model = mlflow_client.create_logged_model(exp_id)
    mlflow_client.set_logged_model_tags(model.model_id, {"tag1": "value1", "tag2": "value2"})
    mlflow_client.delete_logged_model_tag(model.model_id, "tag1")
    loaded_model = mlflow_client.get_logged_model(model.model_id)
    assert loaded_model.tags == {"tag2": "value2"}

    with pytest.raises(MlflowException, match="No tag with key"):
        mlflow_client.delete_logged_model_tag(model.model_id, "tag1")


def test_search_logged_models(mlflow_client: MlflowClient):
    exp_id = mlflow_client.create_experiment("create_logged_model")
    model_1 = mlflow_client.create_logged_model(exp_id)
    time.sleep(0.001)  # to ensure different created time
    models = mlflow_client.search_logged_models(experiment_ids=[exp_id])
    assert [m.name for m in models] == [model_1.name]

    # max_results
    model_2 = mlflow_client.create_logged_model(exp_id)
    page_1 = mlflow_client.search_logged_models(experiment_ids=[exp_id], max_results=1)
    assert [m.name for m in page_1] == [model_2.name]
    assert page_1.token is not None

    # pagination
    page_2 = mlflow_client.search_logged_models(
        experiment_ids=[exp_id], max_results=1, page_token=page_1.token
    )
    assert [m.name for m in page_2] == [model_1.name]
    assert page_2.token is None

    # filter_string
    models = mlflow_client.search_logged_models(
        experiment_ids=[exp_id], filter_string=f"name = {model_1.name!r}"
    )
    assert [m.name for m in models] == [model_1.name]

    # datasets
    run_1 = mlflow_client.create_run(exp_id)
    mlflow_client.log_metric(
        run_1.info.run_id,
        key="metric",
        value=1,
        dataset_name="dataset",
        dataset_digest="123",
        model_id=model_1.model_id,
    )
    models = mlflow_client.search_logged_models(
        experiment_ids=[exp_id],
        datasets=[{"dataset_name": "dataset", "dataset_digest": "123"}],
    )

    assert [m.name for m in models] == [model_1.name]

    # order_by
    models = mlflow_client.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[{"field_name": "creation_timestamp", "ascending": False}],
    )
    assert [m.name for m in models] == [model_2.name, model_1.name]


def test_log_outputs(mlflow_client: MlflowClient):
    exp_id = mlflow_client.create_experiment("log_outputs")
    run = mlflow_client.create_run(experiment_id=exp_id)
    model = mlflow_client.create_logged_model(experiment_id=exp_id)
    model_outputs = [LoggedModelOutput(model.model_id, 1)]
    mlflow_client.log_outputs(run.info.run_id, model_outputs)
    run = mlflow_client.get_run(run.info.run_id)
    assert run.outputs.model_outputs == model_outputs


def test_list_logged_model_artifacts(mlflow_client: MlflowClient):
    class Model(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            return model_input

    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    model_info = mlflow.pyfunc.log_model(name="model", python_model=Model())
    resp = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/logged-models/{model_info.model_id}/artifacts/directories"
    )
    assert resp.status_code == 200
    data = resp.json()
    paths = [f["path"] for f in data["files"]]
    assert "MLmodel" in paths


def test_get_logged_model_artifact(mlflow_client: MlflowClient):
    class Model(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            return model_input

    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    model_info = mlflow.pyfunc.log_model(name="model", python_model=Model())
    resp = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/logged-models/{model_info.model_id}/artifacts/files",
        params={"artifact_file_path": "MLmodel"},
    )
    assert resp.status_code == 200
    assert model_info.model_id in resp.text


def test_suppress_url_printing(mlflow_client: MlflowClient, monkeypatch):
    monkeypatch.setenv(MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT.name, "true")
    exp_id = mlflow_client.create_experiment("test_suppress_url_printing")
    run = mlflow_client.create_run(experiment_id=exp_id)
    captured_output = StringIO()
    monkeypatch.setattr(sys, "stdout", captured_output)
    mlflow_client._tracking_client._log_url(run.info.run_id)
    assert captured_output.getvalue() == ""


def test_assessments_end_to_end(mlflow_client):
    mlflow.set_tracking_uri(mlflow_client.tracking_uri)

    # Set up experiment and trace
    experiment_id = mlflow_client.create_experiment("assessment_crud_test")
    trace_info = mlflow_client.start_trace(name="test_trace", experiment_id=experiment_id)
    mlflow_client.end_trace(request_id=trace_info.request_id)

    # CREATE initial feedback assessment
    feedback_payload = {
        "assessment": {
            "assessment_name": "quality_score",
            "feedback": {"value": {"rating": 4, "comments": "Good response"}},
            "source": {"source_type": "HUMAN", "source_id": "evaluator@company.com"},
            "rationale": "Response was accurate and helpful",
            "metadata": {"model": "gpt-4", "version": "1.0"},
        }
    }

    # CREATE assessment
    create_response = requests.post(
        f"{mlflow_client.tracking_uri}/api/3.0/mlflow/traces/{trace_info.request_id}/assessments",
        json=feedback_payload,
    )
    assert create_response.status_code == 200
    assessment = create_response.json()["assessment"]
    assessment_id = assessment["assessment_id"]

    # Verify creation
    assert assessment["assessment_name"] == "quality_score"
    assert assessment["feedback"]["value"]["rating"] == 4
    assert assessment["source"]["source_type"] == "HUMAN"
    assert assessment["valid"] is True

    # GET assessment
    get_response = requests.get(
        f"{mlflow_client.tracking_uri}/api/3.0/mlflow/traces/{trace_info.request_id}/assessments/{assessment_id}"
    )
    assert get_response.status_code == 200
    retrieved = get_response.json()["assessment"]
    assert retrieved["assessment_id"] == assessment_id
    assert retrieved["feedback"]["value"]["rating"] == 4

    # UPDATE assessment
    update_payload = {
        "assessment": {
            "assessment_id": assessment_id,
            "trace_id": trace_info.request_id,
            "assessment_name": "updated_quality_score",
            "feedback": {"value": {"rating": 5, "comments": "Excellent response"}},
            "rationale": "Actually, the response was excellent",
            "metadata": {"model": "gpt-4", "version": "2.0"},
        },
        "update_mask": "assessmentName,feedback,rationale,metadata",
    }

    update_response = requests.patch(
        f"{mlflow_client.tracking_uri}/api/3.0/mlflow/traces/{trace_info.request_id}/assessments/{assessment_id}",
        json=update_payload,
    )
    assert update_response.status_code == 200
    updated = update_response.json()["assessment"]
    assert updated["assessment_name"] == "updated_quality_score"
    assert updated["feedback"]["value"]["rating"] == 5
    assert updated["rationale"] == "Actually, the response was excellent"

    # CREATE override assessment
    override_payload = {
        "assessment": {
            "assessment_name": "corrected_quality_score",
            "feedback": {"value": {"rating": 3, "comments": "Actually needs improvement"}},
            "source": {"source_type": "HUMAN", "source_id": "senior_evaluator@company.com"},
            "overrides": assessment_id,
        }
    }

    override_response = requests.post(
        f"{mlflow_client.tracking_uri}/api/3.0/mlflow/traces/{trace_info.request_id}/assessments",
        json=override_payload,
    )
    assert override_response.status_code == 200
    override_assessment = override_response.json()["assessment"]
    override_id = override_assessment["assessment_id"]

    # Verify original is now invalid
    get_original = requests.get(
        f"{mlflow_client.tracking_uri}/api/3.0/mlflow/traces/{trace_info.request_id}/assessments/{assessment_id}"
    )
    assert get_original.status_code == 200
    assert get_original.json()["assessment"]["valid"] is False

    # Verify override is valid
    get_override = requests.get(
        f"{mlflow_client.tracking_uri}/api/3.0/mlflow/traces/{trace_info.request_id}/assessments/{override_id}"
    )
    assert get_override.status_code == 200
    assert get_override.json()["assessment"]["valid"] is True
    assert get_override.json()["assessment"]["overrides"] == assessment_id

    # DELETE override assessment (should restore original)
    delete_response = requests.delete(
        f"{mlflow_client.tracking_uri}/api/3.0/mlflow/traces/{trace_info.request_id}/assessments/{override_id}"
    )
    assert delete_response.status_code == 200

    # Verify override is deleted
    get_deleted = requests.get(
        f"{mlflow_client.tracking_uri}/api/3.0/mlflow/traces/{trace_info.request_id}/assessments/{override_id}"
    )
    assert get_deleted.status_code == 404

    # Verify original is restored to valid
    get_restored = requests.get(
        f"{mlflow_client.tracking_uri}/api/3.0/mlflow/traces/{trace_info.request_id}/assessments/{assessment_id}"
    )
    assert get_restored.status_code == 200
    assert get_restored.json()["assessment"]["valid"] is True

    # CREATE expectation assessment to test different type
    expectation_payload = {
        "assessment": {
            "assessment_name": "response_time_check",
            "expectation": {"value": {"threshold_ms": 1000, "actual_ms": 750, "passed": True}},
            "source": {"source_type": "CODE", "source_id": "automated_test"},
        }
    }

    expectation_response = requests.post(
        f"{mlflow_client.tracking_uri}/api/3.0/mlflow/traces/{trace_info.request_id}/assessments",
        json=expectation_payload,
    )
    assert expectation_response.status_code == 200
    expectation = expectation_response.json()["assessment"]
    expectation_id = expectation["assessment_id"]

    # Verify expectation was created correctly
    expectation_value = json.loads(expectation["expectation"]["serialized_value"]["value"])
    assert expectation_value["passed"] is True
    assert expectation_value["threshold_ms"] == 1000
    assert expectation_value["actual_ms"] == 750
    assert expectation["source"]["source_type"] == "CODE"

    # Clean up - delete remaining assessments
    for aid in [assessment_id, expectation_id]:
        delete_resp = requests.delete(
            f"{mlflow_client.tracking_uri}/api/3.0/mlflow/traces/{trace_info.request_id}/assessments/{aid}"
        )
        assert delete_resp.status_code == 200


def test_graphql_nan_metric_handling(mlflow_client):
    experiment_id = mlflow_client.create_experiment("test_graphql_nan_metrics")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    # Log a normal metric and a NaN metric
    mlflow_client.log_metric(run_id, key="normal_metric", value=123, timestamp=1, step=1)
    mlflow_client.log_metric(run_id, key="nan_metric", value=math.nan, timestamp=2, step=2)

    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": f"""
                query testQuery {{
                    mlflowGetRun(input: {{runId: "{run_id}"}}) {{
                        run {{
                            data {{
                                metrics {{
                                    key
                                    value
                                    timestamp
                                    step
                                }}
                            }}
                        }}
                    }}
                }}
            """,
            "operationName": "testQuery",
        },
        headers={"content-type": "application/json; charset=utf-8"},
    )

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["errors"] is None

    metrics = json_response["data"]["mlflowGetRun"]["run"]["data"]["metrics"]

    # Find the normal metric and nan metric
    normal_metric = None
    nan_metric = None
    for metric in metrics:
        if metric["key"] == "normal_metric":
            normal_metric = metric
        elif metric["key"] == "nan_metric":
            nan_metric = metric

    # Verify normal metric has a numeric value
    assert normal_metric is not None
    assert normal_metric["key"] == "normal_metric"
    assert normal_metric["value"] == 123
    assert normal_metric["timestamp"] == "1"
    assert normal_metric["step"] == "1"

    # Verify NaN metric has null value
    assert nan_metric is not None
    assert nan_metric["key"] == "nan_metric"
    assert nan_metric["value"] is None
    assert nan_metric["timestamp"] == "2"
    assert nan_metric["step"] == "2"


def test_create_and_get_evaluation_dataset(mlflow_client, store_type):
    if store_type == "file":
        pytest.skip("Evaluation datasets not supported for FileStore")

    experiment_id = mlflow_client.create_experiment("eval_dataset_test")

    dataset = mlflow_client.create_dataset(
        name="test_eval_dataset",
        experiment_id=experiment_id,
        tags={"environment": "test", "version": "1.0"},
    )

    assert dataset.name == "test_eval_dataset"
    assert dataset.experiment_ids == [experiment_id]
    assert dataset.tags["environment"] == "test"
    assert dataset.tags["version"] == "1.0"
    assert dataset.dataset_id is not None

    retrieved = mlflow_client.get_dataset(dataset.dataset_id)
    assert retrieved.name == dataset.name
    assert retrieved.dataset_id == dataset.dataset_id
    assert retrieved.tags == dataset.tags


def test_search_evaluation_datasets(mlflow_client, store_type):
    if store_type == "file":
        pytest.skip("Evaluation datasets not supported for FileStore")

    exp1 = mlflow_client.create_experiment("eval_search_exp1")
    exp2 = mlflow_client.create_experiment("eval_search_exp2")

    mlflow_client.create_dataset(
        name="search_dataset_1", experiment_id=exp1, tags={"team": "ml", "status": "active"}
    )

    mlflow_client.create_dataset(
        name="search_dataset_2",
        experiment_id=[exp1, exp2],
        tags={"team": "data", "status": "active"},
    )

    mlflow_client.create_dataset(
        name="search_dataset_3", experiment_id=exp2, tags={"team": "ml", "status": "archived"}
    )

    all_datasets = mlflow_client.search_datasets()
    assert len(all_datasets) >= 3

    exp1_datasets = mlflow_client.search_datasets(experiment_ids=exp1)
    dataset_names = [d.name for d in exp1_datasets]
    assert "search_dataset_1" in dataset_names
    assert "search_dataset_2" in dataset_names

    ml_datasets = mlflow_client.search_datasets(filter_string="tags.team = 'ml'")
    ml_names = [d.name for d in ml_datasets]
    assert "search_dataset_1" in ml_names
    assert "search_dataset_3" in ml_names
    assert "search_dataset_2" not in ml_names

    ordered_datasets = mlflow_client.search_datasets(order_by=["name ASC"])
    names = [d.name for d in ordered_datasets]
    assert names == sorted(names)


def test_evaluation_dataset_tag_operations(mlflow_client, store_type):
    if store_type == "file":
        pytest.skip("Evaluation datasets not supported for FileStore")

    experiment_id = mlflow_client.create_experiment("eval_tags_test")

    dataset = mlflow_client.create_dataset(
        name="tag_test_dataset",
        experiment_id=experiment_id,
        tags={"initial": "value", "env": "dev"},
    )

    mlflow_client.set_dataset_tags(dataset.dataset_id, {"env": "staging", "new_tag": "new_value"})

    updated = mlflow_client.get_dataset(dataset.dataset_id)
    assert updated.tags["initial"] == "value"  # Original tag preserved
    assert updated.tags["env"] == "staging"  # Updated tag
    assert updated.tags["new_tag"] == "new_value"  # New tag added

    mlflow_client.delete_dataset_tag(dataset.dataset_id, "new_tag")

    final = mlflow_client.get_dataset(dataset.dataset_id)
    assert "new_tag" not in final.tags
    assert final.tags["env"] == "staging"  # Other tags preserved


def test_evaluation_dataset_delete(mlflow_client, store_type):
    if store_type == "file":
        pytest.skip("Evaluation datasets not supported for FileStore")

    experiment_id = mlflow_client.create_experiment("eval_delete_test")

    dataset = mlflow_client.create_dataset(
        name="delete_test_dataset", experiment_id=experiment_id, tags={"to_delete": "yes"}
    )

    retrieved = mlflow_client.get_dataset(dataset.dataset_id)
    assert retrieved.name == "delete_test_dataset"

    mlflow_client.delete_dataset(dataset.dataset_id)

    with pytest.raises(MlflowException, match="not found"):
        mlflow_client.get_dataset(dataset.dataset_id)


def test_evaluation_dataset_upsert_records(mlflow_client, store_type):
    if store_type == "file":
        pytest.skip("Evaluation datasets not supported for FileStore")

    experiment_id = mlflow_client.create_experiment("upsert_records_test")

    dataset = mlflow_client.create_dataset(
        name="test_upsert_dataset",
        experiment_id=experiment_id,
        tags={"test": "upsert"},
    )

    initial_records = [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"answer": "MLflow is an ML platform"},
            "tags": {"difficulty": "easy"},
        },
        {
            "inputs": {"question": "What is Python?"},
            "expectations": {"answer": "Python is a programming language"},
            "tags": {"difficulty": "easy"},
        },
    ]

    # NB: MlflowClient doesn't have upsert_dataset_records method - merge_records() calls
    # the store directly. We make HTTP requests here to test the REST API handler end-to-end.
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/3.0/mlflow/datasets/{dataset.dataset_id}/records",
        json={"records": json.dumps(initial_records)},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["inserted_count"] == 2
    assert result["updated_count"] == 0

    update_records = [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"answer": "MLflow is an open-source ML platform"},
            "tags": {"difficulty": "easy", "updated": "true"},
        },
        {
            "inputs": {"question": "What is Docker?"},
            "expectations": {"answer": "Docker is a containerization platform"},
            "tags": {"difficulty": "medium"},
        },
    ]

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/3.0/mlflow/datasets/{dataset.dataset_id}/records",
        json={"records": json.dumps(update_records)},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["inserted_count"] == 1
    assert result["updated_count"] == 1

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/3.0/mlflow/datasets/invalid-id/records",
        json={"records": json.dumps(initial_records)},
    )
    assert response.status_code != 200


def test_add_dataset_to_experiments_rest_tracking(mlflow_client, store_type):
    if store_type == "file":
        pytest.skip("File store doesn't support dataset operations")
    exp1 = mlflow_client.create_experiment("dataset_exp_1")
    exp2 = mlflow_client.create_experiment("dataset_exp_2")
    exp3 = mlflow_client.create_experiment("dataset_exp_3")

    dataset = create_dataset(
        name="test_multi_exp_dataset",
        experiment_id=[exp1],
        tags={"test": "multi_exp"},
    )

    assert len(dataset.experiment_ids) == 1
    assert exp1 in dataset.experiment_ids

    updated_dataset = add_dataset_to_experiments(
        dataset_id=dataset.dataset_id,
        experiment_ids=[exp2, exp3],
    )

    assert len(updated_dataset.experiment_ids) == 3
    assert exp1 in updated_dataset.experiment_ids
    assert exp2 in updated_dataset.experiment_ids
    assert exp3 in updated_dataset.experiment_ids

    retrieved = mlflow_client.get_dataset(dataset.dataset_id)
    assert len(retrieved.experiment_ids) == 3
    assert exp1 in retrieved.experiment_ids
    assert exp2 in retrieved.experiment_ids
    assert exp3 in retrieved.experiment_ids


def test_remove_dataset_from_experiments_rest_tracking(mlflow_client, store_type):
    if store_type == "file":
        pytest.skip("File store doesn't support dataset operations")
    exp1 = mlflow_client.create_experiment("dataset_remove_exp_1")
    exp2 = mlflow_client.create_experiment("dataset_remove_exp_2")
    exp3 = mlflow_client.create_experiment("dataset_remove_exp_3")

    dataset = create_dataset(
        name="test_remove_exp_dataset",
        experiment_id=[exp1, exp2, exp3],
        tags={"test": "remove_exp"},
    )

    assert len(dataset.experiment_ids) == 3

    updated_dataset = remove_dataset_from_experiments(
        dataset_id=dataset.dataset_id,
        experiment_ids=[exp2],
    )

    assert len(updated_dataset.experiment_ids) == 2
    assert exp1 in updated_dataset.experiment_ids
    assert exp2 not in updated_dataset.experiment_ids
    assert exp3 in updated_dataset.experiment_ids

    retrieved = mlflow_client.get_dataset(dataset.dataset_id)
    assert len(retrieved.experiment_ids) == 2

    updated_dataset = remove_dataset_from_experiments(
        dataset_id=dataset.dataset_id,
        experiment_ids=[exp1, exp3],
    )

    assert len(updated_dataset.experiment_ids) == 0

    retrieved = mlflow_client.get_dataset(dataset.dataset_id)
    assert len(retrieved.experiment_ids) == 0


def test_add_multiple_experiments_at_once_rest_tracking(mlflow_client, store_type):
    if store_type == "file":
        pytest.skip("File store doesn't support dataset operations")
    exps = [mlflow_client.create_experiment(f"bulk_add_exp_{i}") for i in range(5)]

    dataset = create_dataset(
        name="test_bulk_add_dataset",
        experiment_id=[exps[0]],
        tags={"test": "bulk_add"},
    )

    updated_dataset = add_dataset_to_experiments(
        dataset_id=dataset.dataset_id,
        experiment_ids=exps[1:],
    )

    assert len(updated_dataset.experiment_ids) == 5
    for exp in exps:
        assert exp in updated_dataset.experiment_ids


def test_dataset_experiment_association_error_cases_rest_tracking(mlflow_client, store_type):
    if store_type == "file":
        pytest.skip("File store doesn't support dataset operations")
    exp1 = mlflow_client.create_experiment("error_test_exp")

    with pytest.raises(MlflowException, match="not found"):
        add_dataset_to_experiments(
            dataset_id="d-nonexistent1234567890abcdef1234",
            experiment_ids=[exp1],
        )

    with pytest.raises(MlflowException, match="not found"):
        remove_dataset_from_experiments(
            dataset_id="d-nonexistent1234567890abcdef1234",
            experiment_ids=[exp1],
        )


def test_idempotent_add_experiments_rest_tracking(mlflow_client, store_type):
    if store_type == "file":
        pytest.skip("File store doesn't support dataset operations")
    exp1 = mlflow_client.create_experiment("idempotent_test_exp_1")
    exp2 = mlflow_client.create_experiment("idempotent_test_exp_2")

    dataset = create_dataset(
        name="test_idempotent_dataset",
        experiment_id=[exp1, exp2],
        tags={"test": "idempotent"},
    )

    assert len(dataset.experiment_ids) == 2

    updated_dataset = add_dataset_to_experiments(
        dataset_id=dataset.dataset_id,
        experiment_ids=[exp1],
    )

    assert len(updated_dataset.experiment_ids) == 2
    assert exp1 in updated_dataset.experiment_ids
    assert exp2 in updated_dataset.experiment_ids


def test_idempotent_remove_experiments_rest_tracking(mlflow_client, store_type):
    if store_type == "file":
        pytest.skip("File store doesn't support dataset operations")
    exp1 = mlflow_client.create_experiment("remove_idempotent_test_exp_1")
    exp2 = mlflow_client.create_experiment("remove_idempotent_test_exp_2")

    dataset = create_dataset(
        name="test_remove_idempotent_dataset",
        experiment_id=[exp1],
        tags={"test": "remove_idempotent"},
    )

    assert len(dataset.experiment_ids) == 1

    updated_dataset = remove_dataset_from_experiments(
        dataset_id=dataset.dataset_id,
        experiment_ids=[exp2],
    )

    assert len(updated_dataset.experiment_ids) == 1
    assert exp1 in updated_dataset.experiment_ids


def test_client_api_add_remove_experiments_rest_tracking(mlflow_client, store_type):
    if store_type == "file":
        pytest.skip("File store doesn't support dataset operations")
    exp1 = mlflow_client.create_experiment("client_api_exp_1")
    exp2 = mlflow_client.create_experiment("client_api_exp_2")
    exp3 = mlflow_client.create_experiment("client_api_exp_3")

    dataset = mlflow_client.create_dataset(
        name="test_client_api_dataset",
        experiment_id=[exp1],
        tags={"test": "client_api"},
    )

    updated_dataset = mlflow_client.add_dataset_to_experiments(
        dataset_id=dataset.dataset_id,
        experiment_ids=[exp2, exp3],
    )

    assert len(updated_dataset.experiment_ids) == 3

    updated_dataset = mlflow_client.remove_dataset_from_experiments(
        dataset_id=dataset.dataset_id,
        experiment_ids=[exp2],
    )

    assert len(updated_dataset.experiment_ids) == 2
    assert exp1 in updated_dataset.experiment_ids
    assert exp2 not in updated_dataset.experiment_ids
    assert exp3 in updated_dataset.experiment_ids


def test_scorer_CRUD(mlflow_client, store_type):
    if store_type == "file":
        pytest.skip("File store doesn't support scorer CRUD operations")
    experiment_id = mlflow_client.create_experiment("test_scorer_api_experiment")

    # Get the RestStore object directly
    store = mlflow_client._tracking_client.store

    # Test register scorer
    scorer_data = {"name": "test_scorer", "call_source": "test", "original_func_name": "test_func"}
    serialized_scorer = json.dumps(scorer_data)

    version = store.register_scorer(experiment_id, "test_scorer", serialized_scorer)
    assert version.scorer_version == 1

    # Test list scorers
    scorers = store.list_scorers(experiment_id)
    assert len(scorers) == 1
    assert scorers[0].scorer_name == "test_scorer"
    assert scorers[0].scorer_version == 1

    # Test list scorer versions
    versions = store.list_scorer_versions(str(experiment_id), "test_scorer")
    assert len(versions) == 1
    assert versions[0].scorer_name == "test_scorer"
    assert versions[0].scorer_version == 1

    # Test get scorer (latest version)
    scorer = store.get_scorer(str(experiment_id), "test_scorer")
    assert scorer.scorer_name == "test_scorer"
    assert scorer.scorer_version == 1

    # Test get scorer (specific version)
    scorer_v1 = store.get_scorer(str(experiment_id), "test_scorer", version=1)
    assert scorer_v1.scorer_name == "test_scorer"
    assert scorer_v1.scorer_version == 1

    # Test register second version
    scorer_data_v2 = {
        "name": "test_scorer_v2",
        "call_source": "test",
        "original_func_name": "test_func_v2",
    }
    serialized_scorer_v2 = json.dumps(scorer_data_v2)

    version_v2 = store.register_scorer(str(experiment_id), "test_scorer", serialized_scorer_v2)
    assert version_v2.scorer_version == 2

    # Verify list scorers returns latest version
    scorers_after_v2 = store.list_scorers(str(experiment_id))
    assert len(scorers_after_v2) == 1
    assert scorers_after_v2[0].scorer_version == 2

    # Verify list versions returns both versions
    versions_after_v2 = store.list_scorer_versions(str(experiment_id), "test_scorer")
    assert len(versions_after_v2) == 2

    # Test delete specific version
    store.delete_scorer(str(experiment_id), "test_scorer", version=1)

    # Verify version 1 is deleted
    versions_after_delete = store.list_scorer_versions(str(experiment_id), "test_scorer")
    assert len(versions_after_delete) == 1
    assert versions_after_delete[0].scorer_version == 2

    # Test delete all versions
    store.delete_scorer(str(experiment_id), "test_scorer")

    # Verify all versions are deleted
    scorers_after_delete_all = store.list_scorers(str(experiment_id))
    assert len(scorers_after_delete_all) == 0

    # Clean up
    mlflow_client.delete_experiment(experiment_id)


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_rest_store_logs_spans_via_otel_endpoint(mlflow_client, store_type, use_async):
    """
    End-to-end test that verifies RestStore can log spans to a running server via OTLP endpoint.

    This test:
    1. Creates spans using MLflow's span entities
    2. Uses RestStore.log_spans or log_spans_async to send them via OTLP protocol
    3. Verifies the spans were stored and can be retrieved
    """
    if store_type == "file":
        pytest.skip("FileStore does not support OTLP span logging")

    experiment_id = mlflow_client.create_experiment(f"rest_store_otel_test_{use_async}")
    root_span = mlflow_client.start_trace(
        f"rest_store_otel_trace_{use_async}", experiment_id=experiment_id
    )
    otel_span = OTelReadableSpan(
        name=f"test-rest-store-span-{use_async}",
        context=build_otel_context(
            trace_id=int(root_span.trace_id[3:], 16),  # Remove 'tr-' prefix and convert to int
            span_id=0x1234567890ABCDEF,
        ),
        parent=None,
        start_time=1000000000,
        end_time=2000000000,
        attributes={
            SpanAttributeKey.REQUEST_ID: root_span.trace_id,
            "test.attribute": json.dumps(f"test-value-{use_async}"),  # JSON-encoded string value
        },
        resource=None,
    )
    mlflow_span_to_log = Span(otel_span)
    # Call either sync or async version based on parametrization
    if use_async:
        # Use await to execute the async method
        result_spans = await mlflow_client._tracking_client.store.log_spans_async(
            location=experiment_id, spans=[mlflow_span_to_log]
        )
    else:
        result_spans = mlflow_client._tracking_client.store.log_spans(
            location=experiment_id, spans=[mlflow_span_to_log]
        )

    # Verify the spans were returned (indicates successful logging)
    assert len(result_spans) == 1
    assert result_spans[0].name == f"test-rest-store-span-{use_async}"


# =============================================================================
# Secrets and Endpoints E2E Tests
# =============================================================================


def test_create_and_get_secret(mlflow_client_with_secrets):
    store = mlflow_client_with_secrets._tracking_client.store

    secret = store.create_gateway_secret(
        secret_name="test-api-key",
        secret_value={"api_key": "sk-test-12345"},
        provider="openai",
    )

    assert secret.secret_name == "test-api-key"
    assert secret.provider == "openai"
    assert secret.secret_id is not None

    fetched = store.get_secret_info(secret.secret_id)
    assert fetched.secret_name == "test-api-key"
    assert fetched.provider == "openai"
    assert fetched.secret_id == secret.secret_id


def test_update_secret(mlflow_client_with_secrets):
    store = mlflow_client_with_secrets._tracking_client.store

    secret = store.create_gateway_secret(
        secret_name="test-key",
        secret_value={"api_key": "initial-value"},
        provider="anthropic",
    )

    updated = store.update_gateway_secret(
        secret_id=secret.secret_id,
        secret_value={"api_key": "updated-value"},
    )

    assert updated.secret_id == secret.secret_id
    assert updated.secret_name == "test-key"


def test_list_secret_infos(mlflow_client_with_secrets):
    store = mlflow_client_with_secrets._tracking_client.store

    secret1 = store.create_gateway_secret(
        secret_name="openai-key",
        secret_value={"api_key": "sk-openai"},
        provider="openai",
    )
    store.create_gateway_secret(
        secret_name="anthropic-key",
        secret_value={"api_key": "sk-ant"},
        provider="anthropic",
    )

    all_secrets = store.list_secret_infos()
    assert len(all_secrets) >= 2

    openai_secrets = store.list_secret_infos(provider="openai")
    assert len(openai_secrets) >= 1
    assert any(s.secret_id == secret1.secret_id for s in openai_secrets)


def test_delete_secret(mlflow_client_with_secrets):
    store = mlflow_client_with_secrets._tracking_client.store

    secret = store.create_gateway_secret(
        secret_name="temp-key",
        secret_value={"api_key": "temp-value"},
    )

    store.delete_gateway_secret(secret.secret_id)

    all_secrets = store.list_secret_infos()
    assert not any(s.secret_id == secret.secret_id for s in all_secrets)


def test_create_secret_with_dict_value(mlflow_client_with_secrets):
    store = mlflow_client_with_secrets._tracking_client.store

    secret = store.create_gateway_secret(
        secret_name="aws-creds",
        secret_value={"aws_access_key_id": "AKIATEST1234", "aws_secret_access_key": "secret123abc"},
        provider="bedrock",
    )

    assert secret.secret_name == "aws-creds"
    assert secret.provider == "bedrock"
    assert secret.secret_id is not None
    assert isinstance(secret.masked_values, dict)
    assert secret.masked_values == {
        "aws_access_key_id": "AKI...1234",
        "aws_secret_access_key": "sec...3abc",
    }


def test_update_secret_with_dict_value(mlflow_client_with_secrets):
    store = mlflow_client_with_secrets._tracking_client.store

    secret = store.create_gateway_secret(
        secret_name="aws-creds-update",
        secret_value={"api_key": "initial-value-1234"},
        provider="bedrock",
    )

    assert isinstance(secret.masked_values, dict)
    assert secret.masked_values == {"api_key": "ini...1234"}

    updated = store.update_gateway_secret(
        secret_id=secret.secret_id,
        secret_value={
            "aws_access_key_id": "NEWKEY123456",
            "aws_secret_access_key": "newsecret1234",
        },
    )

    assert updated.secret_id == secret.secret_id
    assert updated.secret_name == "aws-creds-update"
    assert isinstance(updated.masked_values, dict)
    assert updated.masked_values == {
        "aws_access_key_id": "NEW...3456",
        "aws_secret_access_key": "new...1234",
    }


def test_create_and_update_compound_secret_via_rest(mlflow_client_with_secrets):
    store = mlflow_client_with_secrets._tracking_client.store

    secret = store.create_gateway_secret(
        secret_name="bedrock-aws-creds",
        secret_value={
            "aws_access_key_id": "AKIAORIGINAL1234",
            "aws_secret_access_key": "original-secret-key-1234",
        },
        provider="bedrock",
        auth_config={"auth_mode": "access_keys", "aws_region_name": "us-east-1"},
    )

    assert secret.secret_name == "bedrock-aws-creds"
    assert secret.provider == "bedrock"
    assert isinstance(secret.masked_values, dict)
    assert secret.masked_values == {
        "aws_access_key_id": "AKI...1234",
        "aws_secret_access_key": "ori...1234",
    }

    fetched = store.get_secret_info(secret_id=secret.secret_id)
    assert fetched.secret_id == secret.secret_id
    assert isinstance(fetched.masked_values, dict)
    assert fetched.masked_values == secret.masked_values

    updated = store.update_gateway_secret(
        secret_id=secret.secret_id,
        secret_value={
            "aws_access_key_id": "AKIAROTATED5678",
            "aws_secret_access_key": "rotated-secret-key-5678",
        },
    )

    assert updated.secret_id == secret.secret_id
    assert updated.last_updated_at > secret.created_at
    assert isinstance(updated.masked_values, dict)
    assert updated.masked_values == {
        "aws_access_key_id": "AKI...5678",
        "aws_secret_access_key": "rot...5678",
    }


def test_create_and_get_endpoint(mlflow_client_with_secrets):
    store = mlflow_client_with_secrets._tracking_client.store

    secret = store.create_gateway_secret(
        secret_name="test-api-key",
        secret_value={"api_key": "sk-test-12345"},
        provider="openai",
    )

    model_def = store.create_gateway_model_definition(
        name="test-model-def",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )

    endpoint = store.create_gateway_endpoint(
        name="test-endpoint",
        model_definition_ids=[model_def.model_definition_id],
    )

    assert endpoint.name == "test-endpoint"
    assert endpoint.endpoint_id is not None
    assert len(endpoint.model_mappings) == 1
    assert endpoint.model_mappings[0].model_definition.model_name == "gpt-4"

    fetched = store.get_gateway_endpoint(endpoint.endpoint_id)
    assert fetched.name == "test-endpoint"
    assert fetched.endpoint_id == endpoint.endpoint_id
    assert len(fetched.model_mappings) == 1


def test_update_endpoint(mlflow_client_with_secrets):
    store = mlflow_client_with_secrets._tracking_client.store

    secret = store.create_gateway_secret(
        secret_name="test-api-key-2",
        secret_value={"api_key": "sk-test-67890"},
        provider="anthropic",
    )

    model_def = store.create_gateway_model_definition(
        name="test-model-def-2",
        secret_id=secret.secret_id,
        provider="anthropic",
        model_name="claude-3-5-sonnet",
    )

    endpoint = store.create_gateway_endpoint(
        name="initial-name",
        model_definition_ids=[model_def.model_definition_id],
    )

    updated = store.update_gateway_endpoint(
        endpoint_id=endpoint.endpoint_id,
        name="updated-name",
    )

    assert updated.endpoint_id == endpoint.endpoint_id
    assert updated.name == "updated-name"


def test_list_endpoints(mlflow_client_with_secrets):
    store = mlflow_client_with_secrets._tracking_client.store

    secret1 = store.create_gateway_secret(
        secret_name="test-api-key-3",
        secret_value={"api_key": "sk-test-11111"},
        provider="openai",
    )
    secret2 = store.create_gateway_secret(
        secret_name="test-api-key-4",
        secret_value={"api_key": "sk-test-22222"},
        provider="openai",
    )

    model_def1 = store.create_gateway_model_definition(
        name="test-model-def-3",
        secret_id=secret1.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    model_def2 = store.create_gateway_model_definition(
        name="test-model-def-4",
        secret_id=secret2.secret_id,
        provider="openai",
        model_name="gpt-3.5-turbo",
    )

    endpoint1 = store.create_gateway_endpoint(
        name="endpoint-1",
        model_definition_ids=[model_def1.model_definition_id],
    )
    endpoint2 = store.create_gateway_endpoint(
        name="endpoint-2",
        model_definition_ids=[model_def2.model_definition_id],
    )

    all_endpoints = store.list_gateway_endpoints()
    assert len(all_endpoints) >= 2
    endpoint_ids = {e.endpoint_id for e in all_endpoints}
    assert endpoint1.endpoint_id in endpoint_ids
    assert endpoint2.endpoint_id in endpoint_ids


def test_delete_endpoint(mlflow_client_with_secrets):
    store = mlflow_client_with_secrets._tracking_client.store

    secret = store.create_gateway_secret(
        secret_name="test-api-key-5",
        secret_value={"api_key": "sk-test-33333"},
        provider="openai",
    )

    model_def = store.create_gateway_model_definition(
        name="test-model-def-5",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )

    endpoint = store.create_gateway_endpoint(
        name="temp-endpoint",
        model_definition_ids=[model_def.model_definition_id],
    )

    store.delete_gateway_endpoint(endpoint.endpoint_id)

    all_endpoints = store.list_gateway_endpoints()
    assert not any(e.endpoint_id == endpoint.endpoint_id for e in all_endpoints)


def test_model_definitions(mlflow_client_with_secrets):
    store = mlflow_client_with_secrets._tracking_client.store

    secret = store.create_gateway_secret(
        secret_name="model-secret",
        secret_value={"api_key": "sk-test"},
        provider="openai",
    )

    model_def = store.create_gateway_model_definition(
        name="test-model-def",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )

    assert model_def.name == "test-model-def"
    assert model_def.secret_id == secret.secret_id
    assert model_def.provider == "openai"
    assert model_def.model_name == "gpt-4"
    assert model_def.model_definition_id is not None

    fetched = store.get_gateway_model_definition(model_def.model_definition_id)
    assert fetched.model_definition_id == model_def.model_definition_id
    assert fetched.name == "test-model-def"

    updated = store.update_gateway_model_definition(
        model_definition_id=model_def.model_definition_id,
        model_name="gpt-4-turbo",
    )
    assert updated.model_definition_id == model_def.model_definition_id
    assert updated.model_name == "gpt-4-turbo"

    all_defs = store.list_gateway_model_definitions()
    assert any(d.model_definition_id == model_def.model_definition_id for d in all_defs)

    store.delete_gateway_model_definition(model_def.model_definition_id)

    all_defs_after = store.list_gateway_model_definitions()
    assert not any(d.model_definition_id == model_def.model_definition_id for d in all_defs_after)


def test_attach_detach_model_to_endpoint(mlflow_client_with_secrets):
    store = mlflow_client_with_secrets._tracking_client.store

    secret = store.create_gateway_secret(
        secret_name="attach-detach-secret",
        secret_value={"api_key": "sk-test-attach"},
        provider="openai",
    )

    model_def1 = store.create_gateway_model_definition(
        name="attach-model-def-1",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )

    model_def2 = store.create_gateway_model_definition(
        name="attach-model-def-2",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-3.5-turbo",
    )

    endpoint = store.create_gateway_endpoint(
        name="attach-test-endpoint",
        model_definition_ids=[model_def1.model_definition_id],
    )

    assert len(endpoint.model_mappings) == 1
    assert endpoint.model_mappings[0].model_definition.model_name == "gpt-4"

    mapping = store.attach_model_to_endpoint(
        endpoint_id=endpoint.endpoint_id,
        model_definition_id=model_def2.model_definition_id,
    )

    assert mapping.endpoint_id == endpoint.endpoint_id
    assert mapping.model_definition_id == model_def2.model_definition_id

    fetched_endpoint = store.get_gateway_endpoint(endpoint.endpoint_id)
    assert len(fetched_endpoint.model_mappings) == 2

    store.detach_model_from_endpoint(
        endpoint_id=endpoint.endpoint_id,
        model_definition_id=model_def2.model_definition_id,
    )

    fetched_endpoint_after = store.get_gateway_endpoint(endpoint.endpoint_id)
    assert len(fetched_endpoint_after.model_mappings) == 1


def test_endpoint_bindings(mlflow_client_with_secrets):
    store = mlflow_client_with_secrets._tracking_client.store

    secret = store.create_gateway_secret(
        secret_name="binding-secret",
        secret_value={"api_key": "sk-test-44444"},
        provider="openai",
    )

    model_def1 = store.create_gateway_model_definition(
        name="binding-model-def-1",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )

    model_def2 = store.create_gateway_model_definition(
        name="binding-model-def-2",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-3.5-turbo",
    )

    endpoint1 = store.create_gateway_endpoint(
        name="binding-test-endpoint-1",
        model_definition_ids=[model_def1.model_definition_id],
    )

    endpoint2 = store.create_gateway_endpoint(
        name="binding-test-endpoint-2",
        model_definition_ids=[model_def2.model_definition_id],
    )

    binding1 = store.create_endpoint_binding(
        endpoint_id=endpoint1.endpoint_id,
        resource_type=GatewayResourceType.SCORER_JOB,
        resource_id="job-123",
    )

    binding2 = store.create_endpoint_binding(
        endpoint_id=endpoint1.endpoint_id,
        resource_type=GatewayResourceType.SCORER_JOB,
        resource_id="job-456",
    )

    binding3 = store.create_endpoint_binding(
        endpoint_id=endpoint2.endpoint_id,
        resource_type=GatewayResourceType.SCORER_JOB,
        resource_id="job-789",
    )

    assert binding1.endpoint_id == endpoint1.endpoint_id
    assert binding1.resource_type == GatewayResourceType.SCORER_JOB
    assert binding1.resource_id == "job-123"

    bindings_endpoint1 = store.list_endpoint_bindings(endpoint_id=endpoint1.endpoint_id)
    assert len(bindings_endpoint1) == 2
    resource_ids = {b.resource_id for b in bindings_endpoint1}
    assert binding1.resource_id in resource_ids
    assert binding2.resource_id in resource_ids
    assert binding3.resource_id not in resource_ids

    bindings_by_type = store.list_endpoint_bindings(resource_type=GatewayResourceType.SCORER_JOB)
    assert len(bindings_by_type) >= 3

    bindings_by_resource = store.list_endpoint_bindings(resource_id="job-123")
    assert len(bindings_by_resource) == 1
    assert bindings_by_resource[0].resource_id == binding1.resource_id

    bindings_multi = store.list_endpoint_bindings(
        endpoint_id=endpoint1.endpoint_id,
        resource_type=GatewayResourceType.SCORER_JOB,
    )
    assert len(bindings_multi) == 2

    store.delete_endpoint_binding(
        endpoint_id=binding1.endpoint_id,
        resource_type=binding1.resource_type.value,
        resource_id=binding1.resource_id,
    )

    bindings_after = store.list_endpoint_bindings(endpoint_id=endpoint1.endpoint_id)
    assert len(bindings_after) == 1
    assert not any(b.resource_id == binding1.resource_id for b in bindings_after)


def test_secrets_and_endpoints_integration(mlflow_client_with_secrets):
    store = mlflow_client_with_secrets._tracking_client.store

    secret = store.create_gateway_secret(
        secret_name="integration-test-key",
        secret_value={"api_key": "sk-integration-test"},
        provider="openai",
    )

    model_def1 = store.create_gateway_model_definition(
        name="integration-model-def-1",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-3.5-turbo",
    )

    model_def2 = store.create_gateway_model_definition(
        name="integration-model-def-2",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )

    endpoint = store.create_gateway_endpoint(
        name="integration-endpoint",
        model_definition_ids=[model_def1.model_definition_id],
    )

    mapping = store.attach_model_to_endpoint(
        endpoint_id=endpoint.endpoint_id,
        model_definition_id=model_def2.model_definition_id,
    )

    binding = store.create_endpoint_binding(
        endpoint_id=endpoint.endpoint_id,
        resource_type=GatewayResourceType.SCORER_JOB,
        resource_id="integration-job",
    )

    fetched_endpoint = store.get_gateway_endpoint(endpoint.endpoint_id)
    assert len(fetched_endpoint.model_mappings) == 2
    mapping_ids = {m.mapping_id for m in fetched_endpoint.model_mappings}
    assert mapping.mapping_id in mapping_ids

    bindings = store.list_endpoint_bindings(resource_id="integration-job")
    assert len(bindings) == 1
    assert bindings[0].resource_id == binding.resource_id

    store.delete_endpoint_binding(
        endpoint_id=binding.endpoint_id,
        resource_type=binding.resource_type.value,
        resource_id=binding.resource_id,
    )
    store.detach_model_from_endpoint(
        endpoint_id=endpoint.endpoint_id,
        model_definition_id=model_def2.model_definition_id,
    )
    store.delete_gateway_endpoint(endpoint.endpoint_id)
    store.delete_gateway_model_definition(model_def1.model_definition_id)
    store.delete_gateway_model_definition(model_def2.model_definition_id)
    store.delete_gateway_secret(secret.secret_id)


@pytest.mark.skipif(
    not _PROVIDER_BACKEND_AVAILABLE, reason="litellm is required for LiteLLM endpoint tests"
)
def test_list_providers(mlflow_client_with_secrets):
    import requests

    base_url = mlflow_client_with_secrets._tracking_client.tracking_uri
    response = requests.get(f"{base_url}/ajax-api/3.0/mlflow/gateway/supported-providers")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    assert isinstance(data["providers"], list)
    assert len(data["providers"]) > 0
    assert "openai" in data["providers"]


@pytest.mark.skipif(
    not _PROVIDER_BACKEND_AVAILABLE, reason="litellm is required for LiteLLM endpoint tests"
)
def test_list_models(mlflow_client_with_secrets):
    import requests

    base_url = mlflow_client_with_secrets._tracking_client.tracking_uri
    response = requests.get(f"{base_url}/ajax-api/3.0/mlflow/gateway/supported-models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0

    model = data["models"][0]
    assert "model" in model
    assert "provider" in model
    assert "mode" in model

    response = requests.get(
        f"{base_url}/ajax-api/3.0/mlflow/gateway/supported-models", params={"provider": "openai"}
    )
    assert response.status_code == 200
    filtered_data = response.json()
    assert all(m["provider"] == "openai" for m in filtered_data["models"])


@pytest.mark.skipif(
    not _PROVIDER_BACKEND_AVAILABLE, reason="litellm is required for LiteLLM endpoint tests"
)
def test_get_provider_config(mlflow_client_with_secrets):
    import requests

    base_url = mlflow_client_with_secrets._tracking_client.tracking_uri

    # Test simple provider (openai) - should have single api_key auth mode
    response = requests.get(
        f"{base_url}/ajax-api/3.0/mlflow/gateway/provider-config",
        params={"provider": "openai"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "auth_modes" in data
    assert "default_mode" in data
    assert data["default_mode"] == "api_key"
    assert len(data["auth_modes"]) >= 1
    api_key_mode = data["auth_modes"][0]
    assert api_key_mode["mode"] == "api_key"

    # Test multi-mode provider (bedrock) - should have multiple auth modes
    response = requests.get(
        f"{base_url}/ajax-api/3.0/mlflow/gateway/provider-config",
        params={"provider": "bedrock"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "auth_modes" in data
    assert data["default_mode"] == "access_keys"
    assert len(data["auth_modes"]) >= 2  # access_keys, iam_role, session_token

    # Check access_keys mode structure
    access_keys_mode = next(m for m in data["auth_modes"] if m["mode"] == "access_keys")
    assert len(access_keys_mode["secret_fields"]) == 2  # access_key_id, secret_access_key
    assert any(f["name"] == "aws_secret_access_key" for f in access_keys_mode["secret_fields"])
    assert any(f["name"] == "aws_region_name" for f in access_keys_mode["config_fields"])

    # Check iam_role mode exists
    iam_role_mode = next(m for m in data["auth_modes"] if m["mode"] == "iam_role")
    assert any(f["name"] == "aws_role_name" for f in iam_role_mode["config_fields"])

    # Unknown providers get a generic fallback
    response = requests.get(
        f"{base_url}/ajax-api/3.0/mlflow/gateway/provider-config",
        params={"provider": "unknown_provider"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["default_mode"] == "api_key"
    assert data["auth_modes"][0]["mode"] == "api_key"

    # Missing provider parameter returns 400
    response = requests.get(f"{base_url}/ajax-api/3.0/mlflow/gateway/provider-config")
    assert response.status_code == 400


def test_get_secrets_config_with_custom_passphrase(mlflow_client_with_secrets):
    base_url = mlflow_client_with_secrets._tracking_client.tracking_uri

    response = requests.get(f"{base_url}/ajax-api/3.0/mlflow/gateway/secrets/config")
    assert response.status_code == 200
    data = response.json()
    assert data["secrets_available"] is True
    assert data["using_default_passphrase"] is False


def test_get_secrets_config_with_default_passphrase(tmp_path: Path, monkeypatch):
    from tests.tracking.integration_test_utils import ServerThread, get_safe_port

    monkeypatch.delenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", raising=False)

    backend_uri = f"sqlite:///{tmp_path}/mlflow.db"
    artifact_uri = (tmp_path / "artifacts").as_uri()

    store = SqlAlchemyStore(backend_uri, artifact_uri)
    store.engine.dispose()

    handlers._tracking_store = None
    handlers._model_registry_store = None
    initialize_backend_stores(backend_uri, default_artifact_root=artifact_uri)

    with ServerThread(app, get_safe_port()) as url:
        response = requests.get(f"{url}/ajax-api/3.0/mlflow/gateway/secrets/config")
        assert response.status_code == 200
        data = response.json()
        assert data["secrets_available"] is True
        assert data["using_default_passphrase"] is True


def test_endpoint_with_orphaned_model_definition(mlflow_client_with_secrets):
    store = mlflow_client_with_secrets._tracking_client.store

    secret = store.create_gateway_secret(
        secret_name="orphan-test-key",
        secret_value={"api_key": "sk-orphan-test"},
        provider="openai",
    )

    model_def = store.create_gateway_model_definition(
        name="orphan-model-def",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )

    endpoint = store.create_gateway_endpoint(
        name="orphan-test-endpoint",
        model_definition_ids=[model_def.model_definition_id],
    )

    assert len(endpoint.model_mappings) == 1
    assert endpoint.model_mappings[0].model_definition.secret_id == secret.secret_id
    assert endpoint.model_mappings[0].model_definition.secret_name == "orphan-test-key"

    store.delete_gateway_secret(secret.secret_id)

    fetched_endpoint = store.get_gateway_endpoint(endpoint.endpoint_id)
    assert len(fetched_endpoint.model_mappings) == 1
    assert fetched_endpoint.model_mappings[0].model_definition.secret_id is None
    assert fetched_endpoint.model_mappings[0].model_definition.secret_name is None


def test_update_model_definition_provider(mlflow_client_with_secrets):
    store = mlflow_client_with_secrets._tracking_client.store

    secret = store.create_gateway_secret(
        secret_name="provider-update-secret",
        secret_value={"api_key": "sk-provider-test"},
        provider="openai",
    )

    model_def = store.create_gateway_model_definition(
        name="provider-update-model-def",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )

    assert model_def.provider == "openai"
    assert model_def.model_name == "gpt-4"

    updated = store.update_gateway_model_definition(
        model_definition_id=model_def.model_definition_id,
        provider="anthropic",
        model_name="claude-3-5-haiku-latest",
    )

    assert updated.provider == "anthropic"
    assert updated.model_name == "claude-3-5-haiku-latest"

    fetched = store.get_gateway_model_definition(model_def.model_definition_id)
    assert fetched.provider == "anthropic"
    assert fetched.model_name == "claude-3-5-haiku-latest"

    store.delete_gateway_model_definition(model_def.model_definition_id)
    store.delete_gateway_secret(secret.secret_id)

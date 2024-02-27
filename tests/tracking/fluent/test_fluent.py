import json
import os
import random
import time
import uuid
from collections import defaultdict
from importlib import reload
from itertools import zip_longest
from unittest import mock

import pandas as pd
import pytest

import mlflow
import mlflow.tracking.context.registry
import mlflow.tracking.fluent
from mlflow import MlflowClient
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities import (
    LifecycleStage,
    Metric,
    Param,
    Run,
    RunData,
    RunInfo,
    RunStatus,
    RunTag,
    SourceType,
    ViewType,
)
from mlflow.environment_variables import (
    MLFLOW_EXPERIMENT_ID,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTRY_URI,
    MLFLOW_RUN_ID,
)
from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry import (
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
)
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.tracking.fluent import (
    _get_experiment_id,
    _get_experiment_id_from_env,
    get_run,
    search_runs,
    set_experiment,
    start_run,
)
from mlflow.utils import get_results_from_paginated_fn, mlflow_tags
from mlflow.utils.time import get_current_time_millis

from tests.helper_functions import multi_context


def create_run(
    run_id="",
    exp_id="",
    uid="",
    start=0,
    end=0,
    metrics=None,
    params=None,
    tags=None,
    status=RunStatus.FINISHED,
    a_uri=None,
):
    return Run(
        RunInfo(
            run_uuid=run_id,
            run_id=run_id,
            experiment_id=exp_id,
            user_id=uid,
            status=status,
            start_time=start,
            end_time=end,
            lifecycle_stage=LifecycleStage.ACTIVE,
            artifact_uri=a_uri,
        ),
        RunData(metrics=metrics, params=params, tags=tags),
    )


def create_test_runs_and_expected_data(experiment_id=None):
    """
    Create a pair of runs and a corresponding data to expect when runs are searched
    for the same experiment.

    Returns:
        A tuple of a list and a dictionary
    """
    start_times = [get_current_time_millis(), get_current_time_millis()]
    end_times = [get_current_time_millis(), get_current_time_millis()]
    exp_id = experiment_id or "123"
    runs = [
        create_run(
            status=RunStatus.FINISHED,
            a_uri="dbfs:/test",
            run_id="abc",
            exp_id=exp_id,
            start=start_times[0],
            end=end_times[0],
            metrics=[Metric("mse", 0.2, 0, 0)],
            params=[Param("param", "value")],
            tags=[RunTag("tag", "value")],
        ),
        create_run(
            status=RunStatus.SCHEDULED,
            a_uri="dbfs:/test2",
            run_id="def",
            exp_id=exp_id,
            start=start_times[1],
            end=end_times[1],
            metrics=[Metric("mse", 0.6, 0, 0), Metric("loss", 1.2, 0, 5)],
            params=[Param("param2", "val"), Param("k", "v")],
            tags=[RunTag("tag2", "v2")],
        ),
    ]
    data = {
        "status": [RunStatus.FINISHED, RunStatus.SCHEDULED],
        "artifact_uri": ["dbfs:/test", "dbfs:/test2"],
        "run_id": ["abc", "def"],
        "experiment_id": [exp_id] * 2,
        "start_time": start_times,
        "end_time": end_times,
        "metrics.mse": [0.2, 0.6],
        "metrics.loss": [None, 1.2],
        "params.param": ["value", None],
        "params.param2": [None, "val"],
        "params.k": [None, "v"],
        "tags.tag": ["value", None],
        "tags.tag2": [None, "v2"],
    }
    return runs, data


def create_experiment(
    experiment_id=uuid.uuid4().hex,
    name="Test Experiment",
    artifact_location="/tmp",
    lifecycle_stage=LifecycleStage.ACTIVE,
    tags=None,
):
    return mlflow.entities.Experiment(experiment_id, name, artifact_location, lifecycle_stage, tags)


@pytest.fixture(autouse=True)
def reset_experiment_id():
    """
    This fixture resets the active experiment id *after* the execution of the test case in which
    its included
    """
    yield
    mlflow.tracking.fluent._active_experiment_id = None


@pytest.fixture(autouse=True)
def reload_context_registry():
    """Reload the context registry module to clear caches."""
    reload(mlflow.tracking.context.registry)


@pytest.fixture(params=["list", "pandas"])
def search_runs_output_format(request):
    if "MLFLOW_SKINNY" in os.environ and request.param == "pandas":
        pytest.skip("pandas output_format is not supported with skinny client")
    return request.param


def test_get_experiment_id_from_env(monkeypatch):
    # When no env variables are set
    assert not MLFLOW_EXPERIMENT_NAME.defined
    assert not MLFLOW_EXPERIMENT_ID.defined
    assert _get_experiment_id_from_env() is None

    # set only ID
    name = f"random experiment {random.randint(1, 1e6)}"
    exp_id = mlflow.create_experiment(name)
    assert exp_id is not None
    monkeypatch.delenv(MLFLOW_EXPERIMENT_NAME.name, raising=False)
    monkeypatch.setenv(MLFLOW_EXPERIMENT_ID.name, exp_id)
    assert _get_experiment_id_from_env() == exp_id

    # set only name
    name = f"random experiment {random.randint(1, 1e6)}"
    exp_id = mlflow.create_experiment(name)
    assert exp_id is not None
    monkeypatch.delenv(MLFLOW_EXPERIMENT_ID.name, raising=False)
    monkeypatch.setenv(MLFLOW_EXPERIMENT_NAME.name, name)
    assert _get_experiment_id_from_env() == exp_id

    # create experiment from env name
    name = f"random experiment {random.randint(1, 1e6)}"
    monkeypatch.delenv(MLFLOW_EXPERIMENT_ID.name, raising=False)
    monkeypatch.setenv(MLFLOW_EXPERIMENT_NAME.name, name)
    assert MlflowClient().get_experiment_by_name(name) is None
    assert _get_experiment_id_from_env() is not None

    # assert experiment creation from encapsulating function
    name = f"random experiment {random.randint(1, 1e6)}"
    monkeypatch.delenv(MLFLOW_EXPERIMENT_ID.name, raising=False)
    monkeypatch.setenv(MLFLOW_EXPERIMENT_NAME.name, name)
    assert MlflowClient().get_experiment_by_name(name) is None
    assert _get_experiment_id() is not None

    # assert raises from conflicting experiment_ids
    name = f"random experiment {random.randint(1, 1e6)}"
    exp_id = mlflow.create_experiment(name)
    random_id = random.randint(100, 1e6)
    assert exp_id != random_id
    monkeypatch.delenv(MLFLOW_EXPERIMENT_NAME.name, raising=False)
    monkeypatch.setenv(MLFLOW_EXPERIMENT_ID.name, random_id)
    with pytest.raises(
        MlflowException,
        match=(
            f"The provided {MLFLOW_EXPERIMENT_ID} environment variable value "
            f"`{random_id}` does not exist in the tracking server"
        ),
    ):
        _get_experiment_id_from_env()

    # assert raises from name to id mismatch
    name = f"random experiment {random.randint(1, 1e6)}"
    exp_id = mlflow.create_experiment(name)
    random_id = random.randint(100, 1e6)
    assert exp_id != random_id
    monkeypatch.setenvs({MLFLOW_EXPERIMENT_ID.name: random_id, MLFLOW_EXPERIMENT_NAME.name: name})
    with pytest.raises(
        MlflowException,
        match=(
            f"The provided {MLFLOW_EXPERIMENT_ID} environment variable value "
            f"`{random_id}` does not match the experiment id"
        ),
    ):
        _get_experiment_id_from_env()

    # assert does not raise if active experiment is set with invalid env variables
    invalid_name = "invalid experiment"
    name = f"random experiment {random.randint(1, 1e6)}"
    exp_id = mlflow.create_experiment(name)
    assert exp_id is not None
    random_id = random.randint(100, 1e6)
    monkeypatch.setenvs(
        {MLFLOW_EXPERIMENT_ID.name: random_id, MLFLOW_EXPERIMENT_NAME.name: invalid_name}
    )
    mlflow.set_experiment(experiment_id=exp_id)
    assert _get_experiment_id() == exp_id


def test_get_experiment_id_with_active_experiment_returns_active_experiment_id():
    # Create a new experiment and set that as active experiment
    name = f"Random experiment {random.randint(1, 1e6)}"
    exp_id = mlflow.create_experiment(name)
    assert exp_id is not None
    mlflow.set_experiment(name)
    assert _get_experiment_id() == exp_id


def test_get_experiment_id_with_no_active_experiments_returns_zero():
    assert _get_experiment_id() == "0"


def test_get_experiment_id_in_databricks_detects_notebook_id_by_default():
    notebook_id = 768

    with mock.patch(
        "mlflow.tracking.fluent.default_experiment_registry.get_experiment_id",
        return_value=notebook_id,
    ):
        assert _get_experiment_id() == notebook_id


def test_get_experiment_id_in_databricks_with_active_experiment_returns_active_experiment_id():
    exp_name = f"random experiment {random.randint(1, 1e6)}"
    exp_id = mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)
    notebook_id = str(int(exp_id) + 73)

    with mock.patch(
        "mlflow.tracking.fluent.default_experiment_registry.get_experiment_id",
        return_value=notebook_id,
    ):
        assert _get_experiment_id() != notebook_id
        assert _get_experiment_id() == exp_id


def test_get_experiment_id_in_databricks_with_experiment_defined_in_env_returns_env_experiment_id(
    monkeypatch,
):
    exp_name = f"random experiment {random.randint(1, 1e6)}"
    exp_id = mlflow.create_experiment(exp_name)
    notebook_id = str(int(exp_id) + 73)
    monkeypatch.delenv(MLFLOW_EXPERIMENT_NAME.name, raising=False)
    monkeypatch.setenv(MLFLOW_EXPERIMENT_ID.name, exp_id)

    with mock.patch(
        "mlflow.tracking.fluent.default_experiment_registry.get_experiment_id",
        return_value=notebook_id,
    ):
        assert _get_experiment_id() != notebook_id
        assert _get_experiment_id() == exp_id


def test_get_experiment_by_id():
    name = f"Random experiment {random.randint(1, 1e6)}"
    exp_id = mlflow.create_experiment(name)

    experiment = mlflow.get_experiment(exp_id)
    assert experiment.experiment_id == exp_id


def test_get_experiment_by_id_with_is_in_databricks_job():
    job_exp_id = 123
    with mock.patch(
        "mlflow.tracking.fluent.default_experiment_registry.get_experiment_id",
        return_value=job_exp_id,
    ):
        assert _get_experiment_id() == job_exp_id


def test_get_experiment_by_name():
    name = f"Random experiment {random.randint(1, 1e6)}"
    exp_id = mlflow.create_experiment(name)

    experiment = mlflow.get_experiment_by_name(name)
    assert experiment.experiment_id == exp_id


def test_search_experiments(tmp_path):
    sqlite_uri = "sqlite:///{}".format(tmp_path.joinpath("test.db"))
    mlflow.set_tracking_uri(sqlite_uri)
    # Why do we need this line? If we didn't have this line, the first `mlflow.create_experiment`
    # call in the loop below would create two experiments, the default experiment (when the sqlite
    # database is initialized) and another one with the specified name. They might have the same
    # creation time, which makes the search order non-deterministic and this test flaky.
    mlflow.search_experiments()

    num_all_experiments = SEARCH_MAX_RESULTS_DEFAULT + 1  # +1 for the default experiment
    num_active_experiments = SEARCH_MAX_RESULTS_DEFAULT // 2
    num_deleted_experiments = SEARCH_MAX_RESULTS_DEFAULT - num_active_experiments

    active_experiment_names = [f"active_{i}" for i in range(num_active_experiments)]
    tag_values = ["x", "x", "y"]
    for tag, active_experiment_name in zip_longest(tag_values, active_experiment_names):
        # Sleep to ensure that each experiment has a different creation time
        time.sleep(0.001)
        mlflow.create_experiment(active_experiment_name, tags={"tag": tag} if tag else None)

    deleted_experiment_names = [f"deleted_{i}" for i in range(num_deleted_experiments)]
    for deleted_experiment_name in deleted_experiment_names:
        time.sleep(0.001)
        exp_id = mlflow.create_experiment(deleted_experiment_name)
        mlflow.delete_experiment(exp_id)

    # max_results is unspecified
    experiments = mlflow.search_experiments(view_type=ViewType.ALL)
    assert len(experiments) == num_all_experiments
    # max_results is larger than the number of experiments in the database
    experiments = mlflow.search_experiments(
        view_type=ViewType.ALL, max_results=num_all_experiments + 1
    )
    assert len(experiments) == num_all_experiments
    # max_results is equal to the number of experiments in the database
    experiments = mlflow.search_experiments(view_type=ViewType.ALL, max_results=num_all_experiments)
    assert len(experiments) == num_all_experiments
    # max_results is smaller than the number of experiments in the database
    experiments = mlflow.search_experiments(
        view_type=ViewType.ALL, max_results=num_all_experiments - 1
    )
    assert len(experiments) == num_all_experiments - 1

    # Filter by view_type
    experiments = mlflow.search_experiments(view_type=ViewType.ACTIVE_ONLY)
    assert [e.name for e in experiments] == active_experiment_names[::-1] + ["Default"]
    experiments = mlflow.search_experiments(view_type=ViewType.DELETED_ONLY)
    assert [e.name for e in experiments] == deleted_experiment_names[::-1]
    experiments = mlflow.search_experiments(view_type=ViewType.ALL)
    assert [e.name for e in experiments] == (
        deleted_experiment_names[::-1] + active_experiment_names[::-1] + ["Default"]
    )
    # Filter by name
    experiments = mlflow.search_experiments(filter_string="name = 'active_1'")
    assert [e.name for e in experiments] == ["active_1"]
    experiments = mlflow.search_experiments(filter_string="name ILIKE 'active_%'")
    assert [e.name for e in experiments] == active_experiment_names[::-1]

    # Filter by tags
    experiments = mlflow.search_experiments(filter_string="tags.tag = 'x'")
    assert [e.name for e in experiments] == active_experiment_names[:2][::-1]
    experiments = mlflow.search_experiments(filter_string="tags.tag = 'y'")
    assert [e.experiment_id for e in experiments] == ["3"]

    # Order by name
    experiments = mlflow.search_experiments(order_by=["name DESC"], max_results=3)
    assert [e.name for e in experiments] == sorted(active_experiment_names, reverse=True)[:3]


def test_search_registered_models(tmp_path):
    sqlite_uri = "sqlite:///{}".format(tmp_path.joinpath("test.db"))
    mlflow.set_tracking_uri(sqlite_uri)

    num_all_models = SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT + 1
    num_a_models = num_all_models // 4
    num_b_models = num_all_models - num_a_models

    a_model_names = [f"AModel_{i}" for i in range(num_a_models)]
    b_model_names = [f"BModel_{i}" for i in range(num_b_models)]
    model_names = b_model_names + a_model_names

    tag_values = ["x", "x", "y"]
    for tag, model_name in zip_longest(tag_values, model_names):
        MlflowClient().create_registered_model(model_name, tags={"tag": tag} if tag else None)

    # max_results is unspecified
    models = mlflow.search_registered_models()
    assert len(models) == num_all_models

    # max_results is larger than the number of models in the database
    models = mlflow.search_registered_models(max_results=num_all_models + 1)
    assert len(models) == num_all_models

    # max_results is equal to the number of models in the database
    models = mlflow.search_registered_models(max_results=num_all_models)
    assert len(models) == num_all_models
    # max_results is smaller than the number of models in the database
    models = mlflow.search_registered_models(max_results=num_all_models - 1)
    assert len(models) == num_all_models - 1

    # Filter by name
    models = mlflow.search_registered_models(filter_string="name = 'AModel_1'")
    assert [m.name for m in models] == ["AModel_1"]
    models = mlflow.search_registered_models(filter_string="name ILIKE 'bmodel_%'")
    assert len(models) == num_b_models

    # Filter by tags
    models = mlflow.search_registered_models(filter_string="tags.tag = 'x'")
    assert [m.name for m in models] == model_names[:2]
    models = mlflow.search_registered_models(filter_string="tags.tag = 'y'")
    assert [m.name for m in models] == [model_names[2]]

    # Order by name
    models = mlflow.search_registered_models(order_by=["name DESC"], max_results=3)
    assert [m.name for m in models] == sorted(model_names, reverse=True)[:3]


def test_search_model_versions(tmp_path):
    sqlite_uri = "sqlite:///{}".format(tmp_path.joinpath("test.db"))
    mlflow.set_tracking_uri(sqlite_uri)
    max_results_default = 100
    with mock.patch(
        "mlflow.store.model_registry.SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT",
        max_results_default,
    ):
        num_all_model_versions = max_results_default + 1
        num_a_model_versions = num_all_model_versions // 4
        num_b_model_versions = num_all_model_versions - num_a_model_versions

        a_model_version_names = ["AModel" for i in range(num_a_model_versions)]
        b_model_version_names = ["BModel" for i in range(num_b_model_versions)]
        model_version_names = b_model_version_names + a_model_version_names

        MlflowClient().create_registered_model(name="AModel")
        MlflowClient().create_registered_model(name="BModel")

        tag_values = ["x", "x", "y"]
        for tag, model_name in zip_longest(tag_values, model_version_names):
            MlflowClient().create_model_version(
                name=model_name, source="foo/bar", tags={"tag": tag} if tag else None
            )

        # max_results is unspecified
        model_versions = mlflow.search_model_versions()
        assert len(model_versions) == num_all_model_versions

        # max_results is larger than the number of model versions in the database
        model_versions = mlflow.search_model_versions(max_results=num_all_model_versions + 1)
        assert len(model_versions) == num_all_model_versions

        # max_results is equal to the number of model versions in the database
        model_versions = mlflow.search_model_versions(max_results=num_all_model_versions)
        assert len(model_versions) == num_all_model_versions
        # max_results is smaller than the number of models in the database
        model_versions = mlflow.search_model_versions(max_results=num_all_model_versions - 1)
        assert len(model_versions) == num_all_model_versions - 1

        # Filter by name
        model_versions = mlflow.search_model_versions(filter_string="name = 'AModel'")
        assert [m.name for m in model_versions] == a_model_version_names
        model_versions = mlflow.search_model_versions(filter_string="name ILIKE 'bmodel'")
        assert len(model_versions) == num_b_model_versions

        # Filter by tags
        model_versions = mlflow.search_model_versions(filter_string="tags.tag = 'x'")
        assert [m.name for m in model_versions] == model_version_names[:2]
        model_versions = mlflow.search_model_versions(filter_string="tags.tag = 'y'")
        assert [m.name for m in model_versions] == [model_version_names[2]]

        # Order by version_number
        model_versions = mlflow.search_model_versions(
            order_by=["version_number ASC"], max_results=5
        )
        assert [m.version for m in model_versions] == [1, 1, 2, 2, 3]


@pytest.fixture
def empty_active_run_stack():
    with mock.patch("mlflow.tracking.fluent._active_run_stack", []):
        yield


def is_from_run(active_run, run):
    return active_run.info == run.info and active_run.data == run.data


def test_start_run_defaults(empty_active_run_stack):
    mlflow.disable_system_metrics_logging()
    mock_experiment_id = mock.Mock()
    experiment_id_patch = mock.patch(
        "mlflow.tracking.fluent._get_experiment_id", return_value=mock_experiment_id
    )
    mock_user = mock.Mock()
    user_patch = mock.patch(
        "mlflow.tracking.context.default_context._get_user", return_value=mock_user
    )
    mock_source_name = mock.Mock()
    source_name_patch = mock.patch(
        "mlflow.tracking.context.default_context._get_source_name", return_value=mock_source_name
    )
    source_type_patch = mock.patch(
        "mlflow.tracking.context.default_context._get_source_type", return_value=SourceType.NOTEBOOK
    )
    mock_source_version = mock.Mock()
    source_version_patch = mock.patch(
        "mlflow.tracking.context.git_context._get_source_version", return_value=mock_source_version
    )
    run_name = "my name"

    expected_tags = {
        mlflow_tags.MLFLOW_USER: mock_user,
        mlflow_tags.MLFLOW_SOURCE_NAME: mock_source_name,
        mlflow_tags.MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
        mlflow_tags.MLFLOW_GIT_COMMIT: mock_source_version,
        mlflow_tags.MLFLOW_RUN_NAME: run_name,
    }

    create_run_patch = mock.patch.object(MlflowClient, "create_run")

    with multi_context(
        experiment_id_patch,
        user_patch,
        source_name_patch,
        source_type_patch,
        source_version_patch,
        create_run_patch,
    ):
        active_run = start_run(run_name=run_name)
        MlflowClient.create_run.assert_called_once_with(
            experiment_id=mock_experiment_id, tags=expected_tags, run_name="my name"
        )
        assert is_from_run(active_run, MlflowClient.create_run.return_value)


def test_start_run_defaults_databricks_notebook(
    empty_active_run_stack,
):
    mock_experiment_id = mock.Mock()
    experiment_id_patch = mock.patch(
        "mlflow.tracking.fluent._get_experiment_id", return_value=mock_experiment_id
    )
    databricks_notebook_patch = mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_notebook", return_value=True
    )
    mock_user = mock.Mock()
    user_patch = mock.patch(
        "mlflow.tracking.context.default_context._get_user", return_value=mock_user
    )
    mock_source_version = mock.Mock()
    source_version_patch = mock.patch(
        "mlflow.tracking.context.git_context._get_source_version", return_value=mock_source_version
    )
    mock_notebook_id = mock.Mock()
    notebook_id_patch = mock.patch(
        "mlflow.utils.databricks_utils.get_notebook_id", return_value=mock_notebook_id
    )
    mock_notebook_path = mock.Mock()
    notebook_path_patch = mock.patch(
        "mlflow.utils.databricks_utils.get_notebook_path", return_value=mock_notebook_path
    )
    mock_webapp_url = mock.Mock()
    webapp_url_patch = mock.patch(
        "mlflow.utils.databricks_utils.get_webapp_url", return_value=mock_webapp_url
    )
    mock_workspace_url = mock.Mock()
    workspace_url_patch = mock.patch(
        "mlflow.utils.databricks_utils.get_workspace_url", return_value=mock_workspace_url
    )
    mock_workspace_id = mock.Mock()
    workspace_info_patch = mock.patch(
        "mlflow.utils.databricks_utils.get_workspace_info_from_dbutils",
        return_value=(mock_webapp_url, mock_workspace_id),
    )

    expected_tags = {
        mlflow_tags.MLFLOW_USER: mock_user,
        mlflow_tags.MLFLOW_SOURCE_NAME: mock_notebook_path,
        mlflow_tags.MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
        mlflow_tags.MLFLOW_GIT_COMMIT: mock_source_version,
        mlflow_tags.MLFLOW_DATABRICKS_NOTEBOOK_ID: mock_notebook_id,
        mlflow_tags.MLFLOW_DATABRICKS_NOTEBOOK_PATH: mock_notebook_path,
        mlflow_tags.MLFLOW_DATABRICKS_WEBAPP_URL: mock_webapp_url,
        mlflow_tags.MLFLOW_DATABRICKS_WORKSPACE_URL: mock_workspace_url,
        mlflow_tags.MLFLOW_DATABRICKS_WORKSPACE_ID: mock_workspace_id,
    }

    create_run_patch = mock.patch.object(MlflowClient, "create_run")

    with multi_context(
        experiment_id_patch,
        databricks_notebook_patch,
        user_patch,
        source_version_patch,
        notebook_id_patch,
        notebook_path_patch,
        webapp_url_patch,
        workspace_url_patch,
        workspace_info_patch,
        create_run_patch,
    ):
        active_run = start_run()
        MlflowClient.create_run.assert_called_once_with(
            experiment_id=mock_experiment_id, tags=expected_tags, run_name=None
        )
        assert is_from_run(active_run, MlflowClient.create_run.return_value)


@pytest.mark.parametrize(
    "experiment_id", [("a", "b"), {"a", "b"}, ["a", "b"], {"a": 1}, [], (), {}]
)
def test_start_run_raises_invalid_experiment_id(experiment_id):
    with pytest.raises(MlflowException, match="Invalid experiment id: "):
        start_run(experiment_id=experiment_id)


@pytest.mark.usefixtures(empty_active_run_stack.__name__)
def test_start_run_creates_new_run_with_user_specified_tags():
    mock_experiment_id = mock.Mock()
    experiment_id_patch = mock.patch(
        "mlflow.tracking.fluent._get_experiment_id", return_value=mock_experiment_id
    )
    mock_user = mock.Mock()
    user_patch = mock.patch(
        "mlflow.tracking.context.default_context._get_user", return_value=mock_user
    )
    mock_source_name = mock.Mock()
    source_name_patch = mock.patch(
        "mlflow.tracking.context.default_context._get_source_name", return_value=mock_source_name
    )
    source_type_patch = mock.patch(
        "mlflow.tracking.context.default_context._get_source_type", return_value=SourceType.NOTEBOOK
    )
    mock_source_version = mock.Mock()
    source_version_patch = mock.patch(
        "mlflow.tracking.context.git_context._get_source_version", return_value=mock_source_version
    )
    user_specified_tags = {
        "ml_task": "regression",
        "num_layers": 7,
        mlflow_tags.MLFLOW_USER: "user_override",
    }
    expected_tags = {
        mlflow_tags.MLFLOW_SOURCE_NAME: mock_source_name,
        mlflow_tags.MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
        mlflow_tags.MLFLOW_GIT_COMMIT: mock_source_version,
        mlflow_tags.MLFLOW_USER: "user_override",
        "ml_task": "regression",
        "num_layers": 7,
    }

    create_run_patch = mock.patch.object(MlflowClient, "create_run")

    with multi_context(
        experiment_id_patch,
        user_patch,
        source_name_patch,
        source_type_patch,
        source_version_patch,
        create_run_patch,
    ):
        active_run = start_run(tags=user_specified_tags)
        MlflowClient.create_run.assert_called_once_with(
            experiment_id=mock_experiment_id, tags=expected_tags, run_name=None
        )
        assert is_from_run(active_run, MlflowClient.create_run.return_value)


@pytest.mark.usefixtures(empty_active_run_stack.__name__)
def test_start_run_resumes_existing_run_and_sets_user_specified_tags():
    tags_to_set = {
        "A": "B",
        "C": "D",
    }
    run_id = mlflow.start_run().info.run_id
    mlflow.end_run()
    restarted_run = mlflow.start_run(run_id, tags=tags_to_set)
    assert tags_to_set.items() <= restarted_run.data.tags.items()


def test_start_run_with_parent():
    parent_run = mock.Mock()
    mock_experiment_id = "123456"
    mock_source_name = mock.Mock()

    active_run_stack_patch = mock.patch("mlflow.tracking.fluent._active_run_stack", [parent_run])

    mock_user = mock.Mock()
    user_patch = mock.patch(
        "mlflow.tracking.context.default_context._get_user", return_value=mock_user
    )
    source_name_patch = mock.patch(
        "mlflow.tracking.context.default_context._get_source_name", return_value=mock_source_name
    )

    expected_tags = {
        mlflow_tags.MLFLOW_USER: mock_user,
        mlflow_tags.MLFLOW_SOURCE_NAME: mock_source_name,
        mlflow_tags.MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.LOCAL),
        mlflow_tags.MLFLOW_PARENT_RUN_ID: parent_run.info.run_id,
    }

    create_run_patch = mock.patch.object(MlflowClient, "create_run")

    with multi_context(
        active_run_stack_patch,
        create_run_patch,
        user_patch,
        source_name_patch,
    ):
        active_run = start_run(experiment_id=mock_experiment_id, nested=True)
        MlflowClient.create_run.assert_called_once_with(
            experiment_id=mock_experiment_id, tags=expected_tags, run_name=None
        )
        assert is_from_run(active_run, MlflowClient.create_run.return_value)


def test_start_run_with_parent_non_nested():
    with mock.patch("mlflow.tracking.fluent._active_run_stack", [mock.Mock()]):
        with pytest.raises(Exception, match=r"Run with UUID .+ is already active"):
            start_run()


def test_start_run_existing_run(empty_active_run_stack):
    mock_run = mock.Mock()
    mock_run.info.lifecycle_stage = LifecycleStage.ACTIVE

    run_id = uuid.uuid4().hex
    mock_get_store = mock.patch("mlflow.tracking.fluent._get_store")

    with mock_get_store, mock.patch.object(MlflowClient, "get_run", return_value=mock_run):
        active_run = start_run(run_id)

        assert is_from_run(active_run, mock_run)
        MlflowClient.get_run.assert_called_with(run_id)


def test_start_run_existing_run_from_environment(empty_active_run_stack, monkeypatch):
    mock_run = mock.Mock()
    mock_run.info.lifecycle_stage = LifecycleStage.ACTIVE

    run_id = uuid.uuid4().hex
    monkeypatch.setenv(MLFLOW_RUN_ID.name, run_id)
    mock_get_store = mock.patch("mlflow.tracking.fluent._get_store")

    with mock_get_store, mock.patch.object(MlflowClient, "get_run", return_value=mock_run):
        active_run = start_run()

        assert is_from_run(active_run, mock_run)
        MlflowClient.get_run.assert_called_with(run_id)


def test_start_run_existing_run_from_environment_with_set_environment(
    empty_active_run_stack, monkeypatch
):
    mock_run = mock.Mock()
    mock_run.info.lifecycle_stage = LifecycleStage.ACTIVE

    run_id = uuid.uuid4().hex
    monkeypatch.setenv(MLFLOW_RUN_ID.name, run_id)
    with mock.patch.object(MlflowClient, "get_run", return_value=mock_run):
        set_experiment("test-run")
        with pytest.raises(
            MlflowException, match="active run ID does not match environment run ID"
        ):
            start_run()


def test_start_run_existing_run_deleted(empty_active_run_stack):
    mock_run = mock.Mock()
    mock_run.info.lifecycle_stage = LifecycleStage.DELETED

    run_id = uuid.uuid4().hex

    match = f"Cannot start run with ID {run_id} because it is in the deleted state"
    with mock.patch.object(MlflowClient, "get_run", return_value=mock_run):
        with pytest.raises(MlflowException, match=match):
            start_run(run_id)


def test_start_existing_run_status(empty_active_run_stack):
    run_id = mlflow.start_run().info.run_id
    mlflow.end_run()
    assert MlflowClient().get_run(run_id).info.status == RunStatus.to_string(RunStatus.FINISHED)
    restarted_run = mlflow.start_run(run_id)
    assert restarted_run.info.status == RunStatus.to_string(RunStatus.RUNNING)


def test_start_existing_run_end_time(empty_active_run_stack):
    run_id = mlflow.start_run().info.run_id
    mlflow.end_run()
    run_obj_info = MlflowClient().get_run(run_id).info
    old_end = run_obj_info.end_time
    assert run_obj_info.status == RunStatus.to_string(RunStatus.FINISHED)
    mlflow.start_run(run_id)
    mlflow.end_run()
    run_obj_info = MlflowClient().get_run(run_id).info
    assert run_obj_info.end_time > old_end


def test_start_run_with_description(empty_active_run_stack):
    mock_experiment_id = mock.Mock()
    experiment_id_patch = mock.patch(
        "mlflow.tracking.fluent._get_experiment_id", return_value=mock_experiment_id
    )
    mock_user = mock.Mock()
    user_patch = mock.patch(
        "mlflow.tracking.context.default_context._get_user", return_value=mock_user
    )
    mock_source_name = mock.Mock()
    source_name_patch = mock.patch(
        "mlflow.tracking.context.default_context._get_source_name", return_value=mock_source_name
    )
    source_type_patch = mock.patch(
        "mlflow.tracking.context.default_context._get_source_type", return_value=SourceType.NOTEBOOK
    )
    mock_source_version = mock.Mock()
    source_version_patch = mock.patch(
        "mlflow.tracking.context.git_context._get_source_version", return_value=mock_source_version
    )

    description = "Test description"

    expected_tags = {
        mlflow_tags.MLFLOW_SOURCE_NAME: mock_source_name,
        mlflow_tags.MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
        mlflow_tags.MLFLOW_GIT_COMMIT: mock_source_version,
        mlflow_tags.MLFLOW_USER: mock_user,
        mlflow_tags.MLFLOW_RUN_NOTE: description,
    }

    create_run_patch = mock.patch.object(MlflowClient, "create_run")

    with multi_context(
        experiment_id_patch,
        user_patch,
        source_name_patch,
        source_type_patch,
        source_version_patch,
        create_run_patch,
    ):
        active_run = start_run(description=description)
        MlflowClient.create_run.assert_called_once_with(
            experiment_id=mock_experiment_id, tags=expected_tags, run_name=None
        )
        assert is_from_run(active_run, MlflowClient.create_run.return_value)


def test_start_run_conflicting_description():
    description = "Test description"
    invalid_tags = {mlflow_tags.MLFLOW_RUN_NOTE: "Another description"}
    match = (
        f"Description is already set via the tag {mlflow_tags.MLFLOW_RUN_NOTE} in tags."
        f"Remove the key {mlflow_tags.MLFLOW_RUN_NOTE} from the tags or omit the description."
    )

    with pytest.raises(MlflowException, match=match):
        start_run(tags=invalid_tags, description=description)


@pytest.mark.usefixtures(empty_active_run_stack.__name__)
def test_start_run_resumes_existing_run_and_sets_description():
    description = "Description"
    run_id = mlflow.start_run().info.run_id
    mlflow.end_run()
    restarted_run = mlflow.start_run(run_id, description=description)
    assert mlflow_tags.MLFLOW_RUN_NOTE in restarted_run.data.tags


@pytest.mark.usefixtures(empty_active_run_stack.__name__)
def test_start_run_resumes_existing_run_and_sets_description_twice():
    description = "Description"
    invalid_tags = {mlflow_tags.MLFLOW_RUN_NOTE: "Another description"}
    match = (
        f"Description is already set via the tag {mlflow_tags.MLFLOW_RUN_NOTE} in tags."
        f"Remove the key {mlflow_tags.MLFLOW_RUN_NOTE} from the tags or omit the description."
    )

    run_id = mlflow.start_run().info.run_id
    mlflow.end_run()
    with pytest.raises(MlflowException, match=match):
        mlflow.start_run(run_id, tags=invalid_tags, description=description)


def test_get_run():
    run_id = uuid.uuid4().hex
    mock_run = mock.Mock()
    mock_run.info.user_id = "my_user_id"
    with mock.patch.object(MlflowClient, "get_run", return_value=mock_run):
        run = get_run(run_id)
        assert run.info.user_id == "my_user_id"


def validate_search_runs(results, data, output_format):
    if output_format == "list":
        keys = ["status", "artifact_uri", "experiment_id", "run_id", "start_time", "end_time"]
        result_data = defaultdict(list)
        for run in results:
            result_data["status"].append(run.info.status)
            result_data["artifact_uri"].append(run.info.artifact_uri)
            result_data["experiment_id"].append(run.info.experiment_id)
            result_data["run_id"].append(run.info.run_id)
            result_data["start_time"].append(run.info.start_time)
            result_data["end_time"].append(run.info.end_time)

        data_subset = {k: data[k] for k in keys if k in keys}
        assert result_data == data_subset
    elif output_format == "pandas":
        expected_df = pd.DataFrame(data)
        expected_df["start_time"] = pd.to_datetime(expected_df["start_time"], unit="ms", utc=True)
        expected_df["end_time"] = pd.to_datetime(expected_df["end_time"], unit="ms", utc=True)
        pd.testing.assert_frame_equal(results, expected_df, check_like=True, check_frame_type=False)
    else:
        raise Exception(f"Invalid output format {output_format}")


def test_search_runs_attributes(search_runs_output_format):
    runs, data = create_test_runs_and_expected_data(search_runs_output_format)
    with mock.patch("mlflow.tracking.fluent.get_results_from_paginated_fn", return_value=runs):
        pdf = search_runs(output_format=search_runs_output_format)
        validate_search_runs(pdf, data, search_runs_output_format)


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny client does not support the np or pandas dependencies",
)
def test_search_runs_data():
    runs, data = create_test_runs_and_expected_data("pandas")
    with mock.patch("mlflow.tracking.fluent.get_results_from_paginated_fn", return_value=runs):
        pdf = search_runs()
        validate_search_runs(pdf, data, "pandas")


def test_search_runs_no_arguments(search_runs_output_format):
    """
    When no experiment ID is specified, it should try to get the implicit one.
    """
    mock_experiment_id = mock.Mock()
    experiment_id_patch = mock.patch(
        "mlflow.tracking.fluent._get_experiment_id", return_value=mock_experiment_id
    )
    get_paginated_runs_patch = mock.patch(
        "mlflow.tracking.fluent.get_results_from_paginated_fn", return_value=[]
    )
    with experiment_id_patch, get_paginated_runs_patch:
        search_runs(output_format=search_runs_output_format)
        mlflow.tracking.fluent.get_results_from_paginated_fn.assert_called_once()
        mlflow.tracking.fluent._get_experiment_id.assert_called_once()


def test_search_runs_all_experiments(search_runs_output_format):
    """
    When no experiment ID is specified but flag is passed, it should search all experiments.
    """
    from mlflow.entities import Experiment

    mock_experiment_id = mock.Mock()
    mock_experiment = mock.Mock(Experiment)
    experiment_id_patch = mock.patch(
        "mlflow.tracking.fluent._get_experiment_id", return_value=mock_experiment_id
    )
    experiment_list_patch = mock.patch(
        "mlflow.tracking.fluent.search_experiments", return_value=[mock_experiment]
    )
    get_paginated_runs_patch = mock.patch(
        "mlflow.tracking.fluent.get_results_from_paginated_fn", return_value=[]
    )
    with experiment_id_patch, experiment_list_patch, get_paginated_runs_patch:
        search_runs(output_format=search_runs_output_format, search_all_experiments=True)
        mlflow.tracking.fluent.search_experiments.assert_called_once()
        mlflow.tracking.fluent._get_experiment_id.assert_not_called()


def test_search_runs_by_experiment_name():
    name = f"Random experiment {random.randint(1, 1e6)}"
    exp_id = uuid.uuid4().hex
    experiment = create_experiment(experiment_id=exp_id, name=name)
    runs, data = create_test_runs_and_expected_data(exp_id)

    get_experiment_patch = mock.patch(
        "mlflow.tracking.fluent.get_experiment_by_name", return_value=experiment
    )
    get_paginated_runs_patch = mock.patch(
        "mlflow.tracking.fluent.get_results_from_paginated_fn", return_value=runs
    )

    with get_experiment_patch, get_paginated_runs_patch:
        result = search_runs(experiment_names=[name])
        validate_search_runs(result, data, "pandas")


def test_search_runs_by_non_existing_experiment_name():
    """When invalid experiment names are used (including None), it should return an empty
    collection.
    """
    for name in [None, f"Random {random.randint(1, 1e6)}"]:
        assert search_runs(experiment_names=[name], output_format="list") == []


def test_search_runs_by_experiment_id_and_name():
    """When both experiment_ids and experiment_names are used, it should throw an exception"""
    err_msg = "Only experiment_ids or experiment_names can be used, but not both"
    with pytest.raises(MlflowException, match=err_msg):
        search_runs(experiment_ids=["id"], experiment_names=["name"])


def test_paginate_lt_maxresults_onepage():
    """
    Number of runs is less than max_results and fits on one page,
    so we only need to fetch one page.
    """
    runs = [create_run() for _ in range(5)]
    tokenized_runs = PagedList(runs, "")
    max_results = 50
    max_per_page = 10
    mocked_lambda = mock.Mock(return_value=tokenized_runs)

    paginated_runs = get_results_from_paginated_fn(mocked_lambda, max_per_page, max_results)
    mocked_lambda.assert_called_once()
    assert len(paginated_runs) == 5


def test_paginate_lt_maxresults_multipage():
    """
    Number of runs is less than max_results, but multiple pages are necessary to get all runs
    """
    tokenized_runs = PagedList([create_run() for _ in range(10)], "token")
    no_token_runs = PagedList([create_run()], "")
    max_results = 50
    max_per_page = 10
    mocked_lambda = mock.Mock(side_effect=[tokenized_runs, tokenized_runs, no_token_runs])
    TOTAL_RUNS = 21

    paginated_runs = get_results_from_paginated_fn(mocked_lambda, max_per_page, max_results)
    assert len(paginated_runs) == TOTAL_RUNS


def test_paginate_lt_maxresults_onepage_nonetoken():
    """
    Number of runs is less than max_results and fits on one page.
    The token passed back on the last page is None, not the emptystring
    """
    runs = [create_run() for _ in range(5)]
    tokenized_runs = PagedList(runs, None)
    max_results = 50
    max_per_page = 10
    mocked_lambda = mock.Mock(return_value=tokenized_runs)

    paginated_runs = get_results_from_paginated_fn(mocked_lambda, max_per_page, max_results)
    mocked_lambda.assert_called_once()
    assert len(paginated_runs) == 5


def test_paginate_eq_maxresults_blanktoken():
    """
    Runs returned are equal to max_results which are equal to a full number of pages.
    The server might send a token back, or they might not (depending on if they know if
    more runs exist). In this example, no token is sent back.
    Expected behavior is to NOT query for more pages.
    """
    # runs returned equal to max_results, blank token
    runs = [create_run() for _ in range(10)]
    tokenized_runs = PagedList(runs, "")
    no_token_runs = PagedList([], "")
    max_results = 10
    max_per_page = 10
    mocked_lambda = mock.Mock(side_effect=[tokenized_runs, no_token_runs])

    paginated_runs = get_results_from_paginated_fn(mocked_lambda, max_per_page, max_results)
    mocked_lambda.assert_called_once()
    assert len(paginated_runs) == 10


def test_paginate_eq_maxresults_token():
    """
    Runs returned are equal to max_results which are equal to a full number of pages.
    The server might send a token back, or they might not (depending on if they know if
    more runs exist). In this example, a token IS sent back.
    Expected behavior is to NOT query for more pages.
    """
    runs = [create_run() for _ in range(10)]
    tokenized_runs = PagedList(runs, "abc")
    blank_runs = PagedList([], "")
    max_results = 10
    max_per_page = 10
    mocked_lambda = mock.Mock(side_effect=[tokenized_runs, blank_runs])

    paginated_runs = get_results_from_paginated_fn(mocked_lambda, max_per_page, max_results)
    mocked_lambda.assert_called_once()
    assert len(paginated_runs) == 10


def test_paginate_gt_maxresults_multipage():
    """
    Number of runs that fit search criteria is greater than max_results. Multiple pages expected.
    Expected to only get max_results number of results back.
    """
    # should ask for and return the correct number of max_results
    full_page_runs = PagedList([create_run() for _ in range(8)], "abc")
    partial_page = PagedList([create_run() for _ in range(4)], "def")
    max_results = 20
    max_per_page = 8
    mocked_lambda = mock.Mock(side_effect=[full_page_runs, full_page_runs, partial_page])

    paginated_runs = get_results_from_paginated_fn(mocked_lambda, max_per_page, max_results)
    calls = [mock.call(8, None), mock.call(8, "abc"), mock.call(20 % 8, "abc")]
    mocked_lambda.assert_has_calls(calls)
    assert len(paginated_runs) == 20


def test_paginate_gt_maxresults_onepage():
    """
    Number of runs that fit search criteria is greater than max_results. Only one page expected.
    Expected to only get max_results number of results back.
    """
    runs = [create_run() for _ in range(10)]
    tokenized_runs = PagedList(runs, "abc")
    max_results = 10
    max_per_page = 20
    mocked_lambda = mock.Mock(return_value=tokenized_runs)

    paginated_runs = get_results_from_paginated_fn(mocked_lambda, max_per_page, max_results)
    mocked_lambda.assert_called_once_with(max_results, None)
    assert len(paginated_runs) == 10


def test_delete_tag():
    """
    Confirm that fluent API delete tags actually works.
    """
    mlflow.set_tag("a", "b")
    run = MlflowClient().get_run(mlflow.active_run().info.run_id)
    assert "a" in run.data.tags
    mlflow.delete_tag("a")
    run = MlflowClient().get_run(mlflow.active_run().info.run_id)
    assert "a" not in run.data.tags
    with pytest.raises(MlflowException, match="No tag with name"):
        mlflow.delete_tag("a")
    with pytest.raises(MlflowException, match="No tag with name"):
        mlflow.delete_tag("b")
    mlflow.end_run()


def test_last_active_run_returns_currently_active_run():
    run_id = mlflow.start_run().info.run_id
    last_active_run_id = mlflow.last_active_run().info.run_id
    mlflow.end_run()
    assert run_id == last_active_run_id


def test_last_active_run_returns_most_recently_ended_active_run():
    run_id = mlflow.start_run().info.run_id
    mlflow.log_metric("a", 1.0)
    mlflow.log_param("b", 2)
    mlflow.end_run()
    last_active_run = mlflow.last_active_run()
    assert last_active_run.info.run_id == run_id
    assert last_active_run.data.metrics == {"a": 1.0}
    assert last_active_run.data.params == {"b": "2"}


def test_set_experiment_tag():
    test_tags = {"new_test_tag_1": "abc", "new_test_tag_2": 5, "new/nested/tag": "cbd"}
    tag_counter = 0
    with start_run() as active_run:
        test_experiment = active_run.info.experiment_id
        current_experiment = mlflow.tracking.MlflowClient().get_experiment(test_experiment)
        assert len(current_experiment.tags) == 0
        for tag_key, tag_value in test_tags.items():
            mlflow.set_experiment_tag(tag_key, tag_value)
            tag_counter += 1
            current_experiment = mlflow.tracking.MlflowClient().get_experiment(test_experiment)
            assert tag_counter == len(current_experiment.tags)
        finished_experiment = mlflow.tracking.MlflowClient().get_experiment(test_experiment)
        assert len(finished_experiment.tags) == len(test_tags)
        for tag_key, tag_value in test_tags.items():
            assert str(test_tags[tag_key] == tag_value)


def test_set_experiment_tags():
    exact_expected_tags = {"name_1": "c", "name_2": "b", "nested/nested/name": 5}
    with start_run() as active_run:
        test_experiment = active_run.info.experiment_id
        current_experiment = mlflow.tracking.MlflowClient().get_experiment(test_experiment)
        assert len(current_experiment.tags) == 0
        mlflow.set_experiment_tags(exact_expected_tags)
    finished_experiment = mlflow.tracking.MlflowClient().get_experiment(test_experiment)
    # Validate tags
    assert len(finished_experiment.tags) == len(exact_expected_tags)
    for tag_key, tag_value in finished_experiment.tags.items():
        assert str(exact_expected_tags[tag_key]) == tag_value


def test_log_input(tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    path = tmp_path / "temp.csv"
    df.to_csv(path)
    dataset = from_pandas(df, source=path)
    with start_run() as run:
        mlflow.log_input(dataset, "train", {"foo": "baz"})
    dataset_inputs = MlflowClient().get_run(run.info.run_id).inputs.dataset_inputs

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
    assert json.loads(dataset_inputs[0].dataset.profile) == {"num_rows": 2, "num_elements": 6}

    assert len(dataset_inputs[0].tags) == 2
    assert dataset_inputs[0].tags[0].key == "foo"
    assert dataset_inputs[0].tags[0].value == "baz"
    assert dataset_inputs[0].tags[1].key == mlflow_tags.MLFLOW_DATASET_CONTEXT
    assert dataset_inputs[0].tags[1].value == "train"

    # ensure log_input also works without tags
    with start_run() as run:
        mlflow.log_input(dataset, "train")
    dataset_inputs = MlflowClient().get_run(run.info.run_id).inputs.dataset_inputs

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
    assert json.loads(dataset_inputs[0].dataset.profile) == {"num_rows": 2, "num_elements": 6}

    assert len(dataset_inputs[0].tags) == 1
    assert dataset_inputs[0].tags[0].key == mlflow_tags.MLFLOW_DATASET_CONTEXT
    assert dataset_inputs[0].tags[0].value == "train"


def test_log_input_metadata_only():
    source_uri = "test:/my/test/uri"
    source = HTTPDatasetSource(url=source_uri)
    dataset = mlflow.data.meta_dataset.MetaDataset(source=source)

    with start_run() as run:
        mlflow.log_input(dataset, "train")
    dataset_inputs = MlflowClient().get_run(run.info.run_id).inputs.dataset_inputs
    assert len(dataset_inputs) == 1
    assert dataset_inputs[0].dataset.name == "dataset"
    assert dataset_inputs[0].dataset.digest is not None
    assert dataset_inputs[0].dataset.source_type == "http"
    assert json.loads(dataset_inputs[0].dataset.source) == {"url": source_uri}


def test_get_parent_run():
    with mlflow.start_run() as parent:
        mlflow.log_param("a", 1)
        mlflow.log_metric("b", 2.0)
        with mlflow.start_run(nested=True) as child_run:
            child_run_id = child_run.info.run_id

    with mlflow.start_run() as run:
        run_id = run.info.run_id

    parent_run = mlflow.get_parent_run(child_run_id)
    assert parent_run.info.run_id == parent.info.run_id
    assert parent_run.data.metrics == {"b": 2.0}
    assert parent_run.data.params == {"a": "1"}

    assert mlflow.get_parent_run(run_id) is None


def test_log_metric_async():
    run_operations = []

    with mlflow.start_run() as parent:
        for num in range(100):
            run_operations.append(
                mlflow.log_metric("async single metric", step=num, value=num, synchronous=False)
            )
        metrics = {f"async batch metric {num}": num for num in range(100)}
        run_operations.append(mlflow.log_metrics(metrics=metrics, step=1, synchronous=False))

    for run_operation in run_operations:
        run_operation.wait()
    parent_run = mlflow.get_run(parent.info.run_id)
    assert parent_run.info.run_id == parent.info.run_id
    assert parent_run.data.metrics["async single metric"] == 99
    for num in range(100):
        assert parent_run.data.metrics[f"async batch metric {num}"] == num


def test_log_metric_async_throws():
    with mlflow.start_run():
        with pytest.raises(MlflowException, match="Please specify value as a valid double"):
            mlflow.log_metric(
                "async single metric", step=1, value="single metric value", synchronous=False
            ).wait()

        with pytest.raises(MlflowException, match="Please specify value as a valid double"):
            mlflow.log_metrics(
                metrics={f"async batch metric {num}": "batch metric value" for num in range(2)},
                step=1,
                synchronous=False,
            ).wait()


def test_log_param_async():
    run_operations = []

    with mlflow.start_run() as parent:
        run_operations.append(mlflow.log_param("async single param", value="1", synchronous=False))
        params = {f"async batch param {num}": num for num in range(100)}
        run_operations.append(mlflow.log_params(params=params, synchronous=False))

    for run_operation in run_operations:
        run_operation.wait()
    parent_run = mlflow.get_run(parent.info.run_id)
    assert parent_run.info.run_id == parent.info.run_id
    assert parent_run.data.params["async single param"] == "1"
    for num in range(100):
        assert parent_run.data.params[f"async batch param {num}"] == str(num)


def test_log_param_async_throws():
    with mlflow.start_run():
        mlflow.log_param("async single param", value="1", synchronous=False)
        with pytest.raises(MlflowException, match="Changing param values is not allowed"):
            mlflow.log_param("async single param", value="2", synchronous=False).wait()

        mlflow.log_params({"async batch param": "2"}, synchronous=False)
        with pytest.raises(MlflowException, match="Changing param values is not allowed"):
            mlflow.log_params({"async batch param": "3"}, synchronous=False).wait()


def test_flush_async_logging():
    with mlflow.start_run() as run:
        for i in range(100):
            mlflow.log_metric("dummy", i, step=i, synchronous=False)

        mlflow.flush_async_logging()

        metric_history = mlflow.MlflowClient().get_metric_history(run.info.run_id, "dummy")
        assert len(metric_history) == 100


def test_enable_async_logging():
    mlflow.config.enable_async_logging(True)
    with mock.patch(
        "mlflow.utils.async_logging.async_logging_queue.AsyncLoggingQueue.log_batch_async"
    ) as mock_log_batch_async:
        with mlflow.start_run():
            mlflow.log_metric("dummy", 1)
            mlflow.log_param("dummy", 1)
            mlflow.set_tag("dummy", 1)
            mlflow.log_metrics({"dummy": 1})
            mlflow.log_params({"dummy": 1})
            mlflow.set_tags({"dummy": 1})

    assert mock_log_batch_async.call_count == 6

    mlflow.config.enable_async_logging(False)
    with mock.patch(
        "mlflow.utils.async_logging.async_logging_queue.AsyncLoggingQueue.log_batch_async"
    ) as mock_log_batch_async:
        with mlflow.start_run():
            mlflow.log_metric("dummy", 1)
            mlflow.log_param("dummy", 1)
            mlflow.set_tag("dummy", 1)
            mlflow.log_metrics({"dummy": 1})
            mlflow.log_params({"dummy": 1})
            mlflow.set_tags({"dummy": 1})

    mock_log_batch_async.assert_not_called()


def test_set_tag_async():
    run_operations = []

    with mlflow.start_run() as parent:
        run_operations.append(mlflow.set_tag("async single tag", value="1", synchronous=False))
        tags = {f"async batch tag {num}": num for num in range(100)}
        run_operations.append(mlflow.set_tags(tags=tags, synchronous=False))

    for run_operation in run_operations:
        run_operation.wait()
    parent_run = mlflow.get_run(parent.info.run_id)
    assert parent_run.info.run_id == parent.info.run_id
    assert parent_run.data.tags["async single tag"] == "1"
    for num in range(100):
        assert parent_run.data.tags[f"async batch tag {num}"] == str(num)


@pytest.fixture
def spark_session_with_registry_uri(request):
    with mock.patch(
        "mlflow.tracking._model_registry.utils._get_active_spark_session"
    ) as spark_session_getter:
        spark = mock.MagicMock()
        spark_session_getter.return_value = spark
        spark.conf.get.side_effect = lambda key, _: "http://custom.uri"
        yield spark


def test_registry_uri_from_spark_conf(spark_session_with_registry_uri):
    assert mlflow.get_registry_uri() == "http://custom.uri"
    # The MLFLOW_REGISTRY_URI environment variable should still take precedence over the
    # spark conf if present
    with mock.patch.dict(os.environ, {MLFLOW_REGISTRY_URI.name: "something-else"}):
        assert mlflow.get_registry_uri() == "something-else"

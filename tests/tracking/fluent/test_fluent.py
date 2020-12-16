from importlib import reload
import os
import random
import uuid
import inspect

import numpy as np
import pandas as pd
import pytest
from unittest import mock

import mlflow
import mlflow.tracking.context.registry
import mlflow.tracking.fluent
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
)
from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import (
    _EXPERIMENT_ID_ENV_VAR,
    _EXPERIMENT_NAME_ENV_VAR,
    _RUN_ID_ENV_VAR,
    _get_experiment_id,
    _get_experiment_id_from_env,
    _paginate,
    search_runs,
    set_experiment,
    start_run,
    get_run,
)
from mlflow.utils import mlflow_tags
from mlflow.utils.file_utils import TempDir


class HelperEnv:
    def __init__(self):
        pass

    @classmethod
    def assert_values(cls, exp_id, name):
        assert os.environ.get(_EXPERIMENT_NAME_ENV_VAR) == name
        assert os.environ.get(_EXPERIMENT_ID_ENV_VAR) == exp_id

    @classmethod
    def set_values(cls, experiment_id=None, name=None):
        if experiment_id:
            os.environ[_EXPERIMENT_ID_ENV_VAR] = str(experiment_id)
        elif os.environ.get(_EXPERIMENT_ID_ENV_VAR):
            del os.environ[_EXPERIMENT_ID_ENV_VAR]

        if name:
            os.environ[_EXPERIMENT_NAME_ENV_VAR] = str(name)
        elif os.environ.get(_EXPERIMENT_NAME_ENV_VAR):
            del os.environ[_EXPERIMENT_NAME_ENV_VAR]


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


@pytest.fixture(autouse=True)
def reset_experiment_id():
    """
    This fixture resets the active experiment id *after* the execution of the test case in which
    its included
    """
    yield
    HelperEnv.set_values()
    mlflow.tracking.fluent._active_experiment_id = None


@pytest.fixture(autouse=True)
def reload_context_registry():
    """Reload the context registry module to clear caches."""
    reload(mlflow.tracking.context.registry)


def test_all_fluent_apis_are_included_in_dunder_all():
    def _is_function_or_class(obj):
        return callable(obj) or inspect.isclass(obj)

    apis = [
        a
        for a in dir(mlflow)
        if _is_function_or_class(getattr(mlflow, a)) and not a.startswith("_")
    ]
    assert set(apis).issubset(set(mlflow.__all__))


def test_get_experiment_id_from_env():
    # When no env variables are set
    HelperEnv.assert_values(None, None)
    assert _get_experiment_id_from_env() is None

    # set only ID
    random_id = random.randint(1, 1e6)
    HelperEnv.set_values(experiment_id=random_id)
    HelperEnv.assert_values(str(random_id), None)
    assert _get_experiment_id_from_env() == str(random_id)

    # set only name
    with TempDir(chdr=True):
        name = "random experiment %d" % random.randint(1, 1e6)
        exp_id = mlflow.create_experiment(name)
        assert exp_id is not None
        HelperEnv.set_values(name=name)
        HelperEnv.assert_values(None, name)
        assert _get_experiment_id_from_env() == exp_id

    # set both: assert that name variable takes precedence
    with TempDir(chdr=True):
        name = "random experiment %d" % random.randint(1, 1e6)
        exp_id = mlflow.create_experiment(name)
        assert exp_id is not None
        random_id = random.randint(1, 1e6)
        HelperEnv.set_values(name=name, experiment_id=random_id)
        HelperEnv.assert_values(str(random_id), name)
        assert _get_experiment_id_from_env() == exp_id


def test_get_experiment_id_with_active_experiment_returns_active_experiment_id():
    # Create a new experiment and set that as active experiment
    with TempDir(chdr=True):
        name = "Random experiment %d" % random.randint(1, 1e6)
        exp_id = mlflow.create_experiment(name)
        assert exp_id is not None
        mlflow.set_experiment(name)
        assert _get_experiment_id() == exp_id


def test_get_experiment_id_with_no_active_experiments_returns_zero():
    assert _get_experiment_id() == "0"


def test_get_experiment_id_in_databricks_detects_notebook_id_by_default():
    notebook_id = 768

    with mock.patch(
        "mlflow.tracking.fluent.is_in_databricks_notebook"
    ) as notebook_detection_mock, mock.patch(
        "mlflow.tracking.fluent.get_notebook_id"
    ) as notebook_id_mock:
        notebook_detection_mock.return_value = True
        notebook_id_mock.return_value = notebook_id
        assert _get_experiment_id() == notebook_id


def test_get_experiment_id_in_databricks_with_active_experiment_returns_active_experiment_id():
    with TempDir(chdr=True):
        exp_name = "random experiment %d" % random.randint(1, 1e6)
        exp_id = mlflow.create_experiment(exp_name)
        mlflow.set_experiment(exp_name)
        notebook_id = str(int(exp_id) + 73)

    with mock.patch(
        "mlflow.tracking.fluent.is_in_databricks_notebook"
    ) as notebook_detection_mock, mock.patch(
        "mlflow.tracking.fluent.get_notebook_id"
    ) as notebook_id_mock:
        notebook_detection_mock.return_value = True
        notebook_id_mock.return_value = notebook_id

        assert _get_experiment_id() != notebook_id
        assert _get_experiment_id() == exp_id


def test_get_experiment_id_in_databricks_with_experiment_defined_in_env_returns_env_experiment_id():
    with TempDir(chdr=True):
        exp_name = "random experiment %d" % random.randint(1, 1e6)
        exp_id = mlflow.create_experiment(exp_name)
        notebook_id = str(int(exp_id) + 73)
        HelperEnv.set_values(experiment_id=exp_id)

    with mock.patch(
        "mlflow.tracking.fluent.is_in_databricks_notebook"
    ) as notebook_detection_mock, mock.patch(
        "mlflow.tracking.fluent.get_notebook_id"
    ) as notebook_id_mock:
        notebook_detection_mock.side_effect = lambda *args, **kwargs: True
        notebook_id_mock.side_effect = lambda *args, **kwargs: notebook_id

        assert _get_experiment_id() != notebook_id
        assert _get_experiment_id() == exp_id


def test_get_experiment_by_id():
    with TempDir(chdr=True):
        name = "Random experiment %d" % random.randint(1, 1e6)
        exp_id = mlflow.create_experiment(name)

        experiment = mlflow.get_experiment(exp_id)
        print(experiment)
        assert experiment.experiment_id == exp_id


def test_get_experiment_by_name():
    with TempDir(chdr=True):
        name = "Random experiment %d" % random.randint(1, 1e6)
        exp_id = mlflow.create_experiment(name)

        experiment = mlflow.get_experiment_by_name(name)
        assert experiment.experiment_id == exp_id


@pytest.fixture
def empty_active_run_stack():
    with mock.patch("mlflow.tracking.fluent._active_run_stack", []):
        yield


def is_from_run(active_run, run):
    return active_run.info == run.info and active_run.data == run.data


def test_start_run_defaults(empty_active_run_stack):  # pylint: disable=unused-argument

    mock_experiment_id = mock.Mock()
    experiment_id_patch = mock.patch(
        "mlflow.tracking.fluent._get_experiment_id", return_value=mock_experiment_id
    )
    databricks_notebook_patch = mock.patch(
        "mlflow.tracking.fluent.is_in_databricks_notebook", return_value=False
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

    expected_tags = {
        mlflow_tags.MLFLOW_USER: mock_user,
        mlflow_tags.MLFLOW_SOURCE_NAME: mock_source_name,
        mlflow_tags.MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
        mlflow_tags.MLFLOW_GIT_COMMIT: mock_source_version,
    }

    create_run_patch = mock.patch.object(MlflowClient, "create_run")

    with experiment_id_patch, databricks_notebook_patch, user_patch, source_name_patch, source_type_patch, source_version_patch, create_run_patch:  # noqa
        active_run = start_run()
        MlflowClient.create_run.assert_called_once_with(
            experiment_id=mock_experiment_id, tags=expected_tags
        )
        assert is_from_run(active_run, MlflowClient.create_run.return_value)


def test_start_run_defaults_databricks_notebook(
    empty_active_run_stack,
):  # pylint: disable=unused-argument

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

    expected_tags = {
        mlflow_tags.MLFLOW_USER: mock_user,
        mlflow_tags.MLFLOW_SOURCE_NAME: mock_notebook_path,
        mlflow_tags.MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
        mlflow_tags.MLFLOW_GIT_COMMIT: mock_source_version,
        mlflow_tags.MLFLOW_DATABRICKS_NOTEBOOK_ID: mock_notebook_id,
        mlflow_tags.MLFLOW_DATABRICKS_NOTEBOOK_PATH: mock_notebook_path,
        mlflow_tags.MLFLOW_DATABRICKS_WEBAPP_URL: mock_webapp_url,
    }

    create_run_patch = mock.patch.object(MlflowClient, "create_run")

    with experiment_id_patch, databricks_notebook_patch, user_patch, source_version_patch, notebook_id_patch, notebook_path_patch, webapp_url_patch, create_run_patch:  # noqa
        active_run = start_run()
        MlflowClient.create_run.assert_called_once_with(
            experiment_id=mock_experiment_id, tags=expected_tags
        )
        assert is_from_run(active_run, MlflowClient.create_run.return_value)


@pytest.mark.usefixtures(empty_active_run_stack.__name__)
def test_start_run_with_user_specified_tags():

    mock_experiment_id = mock.Mock()
    experiment_id_patch = mock.patch(
        "mlflow.tracking.fluent._get_experiment_id", return_value=mock_experiment_id
    )
    databricks_notebook_patch = mock.patch(
        "mlflow.tracking.fluent.is_in_databricks_notebook", return_value=False
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

    with experiment_id_patch, databricks_notebook_patch, user_patch, source_name_patch, source_type_patch, source_version_patch, create_run_patch:  # noqa
        active_run = start_run(tags=user_specified_tags)
        MlflowClient.create_run.assert_called_once_with(
            experiment_id=mock_experiment_id, tags=expected_tags
        )
        assert is_from_run(active_run, MlflowClient.create_run.return_value)


def test_start_run_with_parent():

    parent_run = mock.Mock()
    mock_experiment_id = mock.Mock()
    mock_source_name = mock.Mock()

    active_run_stack_patch = mock.patch("mlflow.tracking.fluent._active_run_stack", [parent_run])

    databricks_notebook_patch = mock.patch(
        "mlflow.tracking.fluent.is_in_databricks_notebook", return_value=False
    )
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

    with databricks_notebook_patch, active_run_stack_patch, create_run_patch, user_patch, source_name_patch:  # noqa
        active_run = start_run(experiment_id=mock_experiment_id, nested=True)
        MlflowClient.create_run.assert_called_once_with(
            experiment_id=mock_experiment_id, tags=expected_tags
        )
        assert is_from_run(active_run, MlflowClient.create_run.return_value)


def test_start_run_with_parent_non_nested():
    with mock.patch("mlflow.tracking.fluent._active_run_stack", [mock.Mock()]):
        with pytest.raises(Exception):
            start_run()


def test_start_run_existing_run(empty_active_run_stack):  # pylint: disable=unused-argument
    mock_run = mock.Mock()
    mock_run.info.lifecycle_stage = LifecycleStage.ACTIVE

    run_id = uuid.uuid4().hex
    mock_get_store = mock.patch("mlflow.tracking.fluent._get_store")

    with mock_get_store, mock.patch.object(MlflowClient, "get_run", return_value=mock_run):
        active_run = start_run(run_id)

        assert is_from_run(active_run, mock_run)
        MlflowClient.get_run.assert_called_with(run_id)


def test_start_run_existing_run_from_environment(
    empty_active_run_stack,
):  # pylint: disable=unused-argument
    mock_run = mock.Mock()
    mock_run.info.lifecycle_stage = LifecycleStage.ACTIVE

    run_id = uuid.uuid4().hex
    env_patch = mock.patch.dict("os.environ", {_RUN_ID_ENV_VAR: run_id})
    mock_get_store = mock.patch("mlflow.tracking.fluent._get_store")

    with env_patch, mock_get_store, mock.patch.object(
        MlflowClient, "get_run", return_value=mock_run
    ):
        active_run = start_run()

        assert is_from_run(active_run, mock_run)
        MlflowClient.get_run.assert_called_with(run_id)


def test_start_run_existing_run_from_environment_with_set_environment(
    empty_active_run_stack,
):  # pylint: disable=unused-argument
    mock_run = mock.Mock()
    mock_run.info.lifecycle_stage = LifecycleStage.ACTIVE

    run_id = uuid.uuid4().hex
    env_patch = mock.patch.dict("os.environ", {_RUN_ID_ENV_VAR: run_id})

    with env_patch, mock.patch.object(MlflowClient, "get_run", return_value=mock_run):
        with pytest.raises(MlflowException):
            set_experiment("test-run")
            start_run()


def test_start_run_existing_run_deleted(empty_active_run_stack):  # pylint: disable=unused-argument
    mock_run = mock.Mock()
    mock_run.info.lifecycle_stage = LifecycleStage.DELETED

    run_id = uuid.uuid4().hex

    with mock.patch.object(MlflowClient, "get_run", return_value=mock_run):
        with pytest.raises(MlflowException):
            start_run(run_id)


def test_start_existing_run_status(empty_active_run_stack):  # pylint: disable=unused-argument
    run_id = mlflow.start_run().info.run_id
    mlflow.end_run()
    assert MlflowClient().get_run(run_id).info.status == RunStatus.to_string(RunStatus.FINISHED)
    restarted_run = mlflow.start_run(run_id)
    assert restarted_run.info.status == RunStatus.to_string(RunStatus.RUNNING)


def test_start_existing_run_end_time(empty_active_run_stack):  # pylint: disable=unused-argument
    run_id = mlflow.start_run().info.run_id
    mlflow.end_run()
    run_obj_info = MlflowClient().get_run(run_id).info
    old_end = run_obj_info.end_time
    assert run_obj_info.status == RunStatus.to_string(RunStatus.FINISHED)
    mlflow.start_run(run_id)
    mlflow.end_run()
    run_obj_info = MlflowClient().get_run(run_id).info
    assert run_obj_info.end_time > old_end


def test_get_run():
    run_id = uuid.uuid4().hex
    mock_run = mock.Mock()
    mock_run.info.user_id = "my_user_id"
    with mock.patch.object(MlflowClient, "get_run", return_value=mock_run):
        run = get_run(run_id)
        assert run.info.user_id == "my_user_id"


def test_search_runs_attributes():
    runs = [
        create_run(status=RunStatus.FINISHED, a_uri="dbfs:/test", run_id="abc", exp_id="123"),
        create_run(status=RunStatus.SCHEDULED, a_uri="dbfs:/test2", run_id="def", exp_id="321"),
    ]
    with mock.patch("mlflow.tracking.fluent._paginate", return_value=runs):
        pdf = search_runs()
        data = {
            "status": [RunStatus.FINISHED, RunStatus.SCHEDULED],
            "artifact_uri": ["dbfs:/test", "dbfs:/test2"],
            "run_id": ["abc", "def"],
            "experiment_id": ["123", "321"],
            "start_time": [pd.to_datetime(0, utc=True), pd.to_datetime(0, utc=True)],
            "end_time": [pd.to_datetime(0, utc=True), pd.to_datetime(0, utc=True)],
        }
        expected_df = pd.DataFrame(data)
        pd.testing.assert_frame_equal(pdf, expected_df, check_like=True, check_frame_type=False)


def test_search_runs_data():
    runs = [
        create_run(
            metrics=[Metric("mse", 0.2, 0, 0)],
            params=[Param("param", "value")],
            tags=[RunTag("tag", "value")],
            start=1564675200000,
            end=1564683035000,
        ),
        create_run(
            metrics=[Metric("mse", 0.6, 0, 0), Metric("loss", 1.2, 0, 5)],
            params=[Param("param2", "val"), Param("k", "v")],
            tags=[RunTag("tag2", "v2")],
            start=1564765200000,
            end=1564783200000,
        ),
    ]
    with mock.patch("mlflow.tracking.fluent._paginate", return_value=runs):
        pdf = search_runs()
        data = {
            "status": [RunStatus.FINISHED] * 2,
            "artifact_uri": [None] * 2,
            "run_id": [""] * 2,
            "experiment_id": [""] * 2,
            "metrics.mse": [0.2, 0.6],
            "metrics.loss": [np.nan, 1.2],
            "params.param": ["value", None],
            "params.param2": [None, "val"],
            "params.k": [None, "v"],
            "tags.tag": ["value", None],
            "tags.tag2": [None, "v2"],
            "start_time": [
                pd.to_datetime(1564675200000, unit="ms", utc=True),
                pd.to_datetime(1564765200000, unit="ms", utc=True),
            ],
            "end_time": [
                pd.to_datetime(1564683035000, unit="ms", utc=True),
                pd.to_datetime(1564783200000, unit="ms", utc=True),
            ],
        }
        expected_df = pd.DataFrame(data)
        pd.testing.assert_frame_equal(pdf, expected_df, check_like=True, check_frame_type=False)


def test_search_runs_no_arguments():
    """
    When no experiment ID is specified, it should try to get the implicit one.
    """
    mock_experiment_id = mock.Mock()
    experiment_id_patch = mock.patch(
        "mlflow.tracking.fluent._get_experiment_id", return_value=mock_experiment_id
    )
    get_paginated_runs_patch = mock.patch("mlflow.tracking.fluent._paginate", return_value=[])
    with experiment_id_patch, get_paginated_runs_patch:
        search_runs()
        mlflow.tracking.fluent._paginate.assert_called_once()
        mlflow.tracking.fluent._get_experiment_id.assert_called_once()


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

    paginated_runs = _paginate(mocked_lambda, max_per_page, max_results)
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

    paginated_runs = _paginate(mocked_lambda, max_per_page, max_results)
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

    paginated_runs = _paginate(mocked_lambda, max_per_page, max_results)
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

    paginated_runs = _paginate(mocked_lambda, max_per_page, max_results)
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

    paginated_runs = _paginate(mocked_lambda, max_per_page, max_results)
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

    paginated_runs = _paginate(mocked_lambda, max_per_page, max_results)
    calls = [mock.call(8, None), mock.call(8, "abc"), mock.call(20 % 8, "abc")]
    mocked_lambda.assert_has_calls(calls)
    assert len(paginated_runs) == 20


def test_paginate_gt_maxresults_onepage():
    """"
    Number of runs that fit search criteria is greater than max_results. Only one page expected.
    Expected to only get max_results number of results back.
    """
    runs = [create_run() for _ in range(10)]
    tokenized_runs = PagedList(runs, "abc")
    max_results = 10
    max_per_page = 20
    mocked_lambda = mock.Mock(return_value=tokenized_runs)

    paginated_runs = _paginate(mocked_lambda, max_per_page, max_results)
    mocked_lambda.assert_called_once_with(max_results, None)
    assert len(paginated_runs) == 10


def test_delete_tag():
    """
    Confirm that fluent API delete tags actually works
    :return:
    """
    mlflow.set_tag("a", "b")
    run = MlflowClient().get_run(mlflow.active_run().info.run_id)
    print(run.info.run_id)
    assert "a" in run.data.tags
    mlflow.delete_tag("a")
    run = MlflowClient().get_run(mlflow.active_run().info.run_id)
    assert "a" not in run.data.tags
    with pytest.raises(MlflowException):
        mlflow.delete_tag("a")
    with pytest.raises(MlflowException):
        mlflow.delete_tag("b")
    mlflow.end_run()

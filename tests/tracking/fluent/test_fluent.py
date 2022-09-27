from collections import defaultdict
from importlib import reload
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT

import os
import random
import uuid
import inspect

import pytest
from unittest import mock

import mlflow
from mlflow import MlflowClient
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
    ViewType,
)
from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking.dbmodels.models import SqlExperiment
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
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
from mlflow.utils.time_utils import get_current_time_millis

from tests.tracking.integration_test_utils import _init_server
from tests.helper_functions import multi_context


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


def create_test_runs_and_expected_data(experiment_id=None):
    """Create a pair of runs and a corresponding data to expect when runs are searched
    for the same experiment

    :return: (list, dict)
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
    HelperEnv.set_values()
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
        "mlflow.tracking.fluent.default_experiment_registry.get_experiment_id"
    ) as notebook_id_mock:
        notebook_id_mock.return_value = notebook_id
        assert _get_experiment_id() == notebook_id


def test_get_experiment_id_in_databricks_with_active_experiment_returns_active_experiment_id():
    with TempDir(chdr=True):
        exp_name = "random experiment %d" % random.randint(1, 1e6)
        exp_id = mlflow.create_experiment(exp_name)
        mlflow.set_experiment(exp_name)
        notebook_id = str(int(exp_id) + 73)

    with mock.patch(
        "mlflow.tracking.fluent.default_experiment_registry.get_experiment_id"
    ) as notebook_id_mock:
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
        "mlflow.tracking.fluent.default_experiment_registry.get_experiment_id"
    ) as notebook_id_mock:
        notebook_id_mock.return_value = notebook_id

        assert _get_experiment_id() != notebook_id
        assert _get_experiment_id() == exp_id


def test_get_experiment_by_id():
    with TempDir(chdr=True):
        name = "Random experiment %d" % random.randint(1, 1e6)
        exp_id = mlflow.create_experiment(name)

        experiment = mlflow.get_experiment(exp_id)
        assert experiment.experiment_id == exp_id


def test_get_experiment_by_id_with_is_in_databricks_job():
    job_exp_id = 123
    with mock.patch(
        "mlflow.tracking.fluent.default_experiment_registry.get_experiment_id"
    ) as job_id_mock:
        job_id_mock.return_value = job_exp_id
        assert _get_experiment_id() == job_exp_id


def test_get_experiment_by_name():
    with TempDir(chdr=True):
        name = "Random experiment %d" % random.randint(1, 1e6)
        exp_id = mlflow.create_experiment(name)

        experiment = mlflow.get_experiment_by_name(name)
        assert experiment.experiment_id == exp_id


@pytest.mark.parametrize("view_type", [ViewType.ACTIVE_ONLY, ViewType.DELETED_ONLY, ViewType.ALL])
def test_list_experiments(view_type, tmpdir):
    sqlite_uri = "sqlite:///" + os.path.join(tmpdir.strpath, "test.db")
    store = SqlAlchemyStore(sqlite_uri, default_artifact_root=tmpdir.strpath)

    num_experiments = SEARCH_MAX_RESULTS_DEFAULT + 1

    if view_type == ViewType.DELETED_ONLY:
        # Delete the default experiment
        MlflowClient(sqlite_uri).delete_experiment("0")

    # This is a bit hacky but much faster than creating experiments one by one with
    # `mlflow.create_experiment`
    with store.ManagedSessionMaker() as session:
        lifecycle_stages = LifecycleStage.view_type_to_stages(view_type)
        experiments = [
            SqlExperiment(
                name=f"exp_{i + 1}",
                lifecycle_stage=random.choice(lifecycle_stages),
                artifact_location=tmpdir.strpath,
            )
            for i in range(num_experiments - 1)
        ]
        session.add_all(experiments)

    url, process = _init_server(sqlite_uri, root_artifact_uri=tmpdir.strpath)
    try:
        mlflow.set_tracking_uri(url)
        # `max_results` is unspecified
        assert len(mlflow.list_experiments(view_type)) == num_experiments
        # `max_results` is larger than the number of experiments in the database
        assert len(mlflow.list_experiments(view_type, num_experiments + 1)) == num_experiments
        # `max_results` is equal to the number of experiments in the database
        assert len(mlflow.list_experiments(view_type, num_experiments)) == num_experiments
        # `max_results` is smaller than the number of experiments in the database
        assert len(mlflow.list_experiments(view_type, num_experiments - 1)) == num_experiments - 1
    finally:
        process.terminate()


def test_search_experiments(tmp_path):
    sqlite_uri = "sqlite:///{}".format(tmp_path.joinpath("test.db"))
    mlflow.set_tracking_uri(sqlite_uri)

    num_all_experiments = SEARCH_MAX_RESULTS_DEFAULT + 1  # +1 for the default experiment
    num_active_experiments = SEARCH_MAX_RESULTS_DEFAULT // 2
    num_deleted_experiments = SEARCH_MAX_RESULTS_DEFAULT - num_active_experiments

    active_experiment_names = [f"active_{i}" for i in range(num_active_experiments)]
    tag_values = ["x", "x", "y"]
    for (idx, active_experiment_name) in enumerate(active_experiment_names):
        if idx < 3:
            tags = {"tag": tag_values[idx]}
        else:
            tags = None
        mlflow.create_experiment(active_experiment_name, tags=tags)

    deleted_experiment_names = [f"deleted_{i}" for i in range(num_deleted_experiments)]
    for deleted_experiment_name in deleted_experiment_names:
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

    with multi_context(
        experiment_id_patch,
        user_patch,
        source_name_patch,
        source_type_patch,
        source_version_patch,
        create_run_patch,
    ):
        active_run = start_run(run_name="my name")
        MlflowClient.create_run.assert_called_once_with(
            experiment_id=mock_experiment_id, tags=expected_tags, run_name="my name"
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
        set_experiment("test-run")
        with pytest.raises(
            MlflowException, match="active run ID does not match environment run ID"
        ):
            start_run()


def test_start_run_existing_run_deleted(empty_active_run_stack):  # pylint: disable=unused-argument
    mock_run = mock.Mock()
    mock_run.info.lifecycle_stage = LifecycleStage.DELETED

    run_id = uuid.uuid4().hex

    match = f"Cannot start run with ID {run_id} because it is in the deleted state"
    with mock.patch.object(MlflowClient, "get_run", return_value=mock_run):
        with pytest.raises(MlflowException, match=match):
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


def test_start_run_with_description(empty_active_run_stack):  # pylint: disable=unused-argument
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
        import pandas as pd

        expected_df = pd.DataFrame(data)
        expected_df["start_time"] = pd.to_datetime(expected_df["start_time"], unit="ms", utc=True)
        expected_df["end_time"] = pd.to_datetime(expected_df["end_time"], unit="ms", utc=True)
        pd.testing.assert_frame_equal(results, expected_df, check_like=True, check_frame_type=False)
    else:
        raise Exception("Invalid output format %s" % output_format)


def test_search_runs_attributes(search_runs_output_format):
    runs, data = create_test_runs_and_expected_data()
    with mock.patch("mlflow.tracking.fluent._paginate", return_value=runs):
        pdf = search_runs(output_format=search_runs_output_format)
        validate_search_runs(pdf, data, search_runs_output_format)


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny client does not support the np or pandas dependencies",
)
def test_search_runs_data():
    runs, data = create_test_runs_and_expected_data("pandas")
    with mock.patch("mlflow.tracking.fluent._paginate", return_value=runs):
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
    get_paginated_runs_patch = mock.patch("mlflow.tracking.fluent._paginate", return_value=[])
    with experiment_id_patch, get_paginated_runs_patch:
        search_runs(output_format=search_runs_output_format)
        mlflow.tracking.fluent._paginate.assert_called_once()
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
        "mlflow.tracking.fluent.list_experiments", return_value=[mock_experiment]
    )
    get_paginated_runs_patch = mock.patch("mlflow.tracking.fluent._paginate", return_value=[])
    with experiment_id_patch, experiment_list_patch, get_paginated_runs_patch:
        search_runs(output_format=search_runs_output_format, search_all_experiments=True)
        mlflow.tracking.fluent.list_experiments.assert_called_once()
        mlflow.tracking.fluent._get_experiment_id.assert_not_called()


def test_search_runs_by_experiment_name():
    name = f"Random experiment {random.randint(1, 1e6)}"
    exp_id = uuid.uuid4().hex
    experiment = create_experiment(experiment_id=exp_id, name=name)
    runs, data = create_test_runs_and_expected_data(exp_id)

    get_experiment_patch = mock.patch(
        "mlflow.tracking.fluent.get_experiment_by_name", return_value=experiment
    )
    get_paginated_runs_patch = mock.patch("mlflow.tracking.fluent._paginate", return_value=runs)

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
    """
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

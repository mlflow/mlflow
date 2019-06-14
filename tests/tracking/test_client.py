import pytest
import mock

from mlflow.entities import RunTag, SourceType, ViewType
from mlflow.store import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_USER, MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE, \
    MLFLOW_PARENT_RUN_ID, MLFLOW_GIT_COMMIT, MLFLOW_PROJECT_ENTRY_POINT


@pytest.fixture
def mock_store():
    with mock.patch("mlflow.tracking.utils._get_store") as mock_get_store:
        yield mock_get_store.return_value


@pytest.fixture
def mock_time():
    time = 1552319350.244724
    with mock.patch("time.time", return_value=time):
        yield time


def test_client_create_run(mock_store, mock_time):

    experiment_id = mock.Mock()

    MlflowClient().create_run(experiment_id)

    mock_store.create_run.assert_called_once_with(
        experiment_id=experiment_id,
        user_id="unknown",
        start_time=int(mock_time * 1000),
        tags=[]
    )


def test_client_create_run_overrides(mock_store):

    experiment_id = mock.Mock()
    user = mock.Mock()
    start_time = mock.Mock()
    tags = {
        MLFLOW_USER: user,
        MLFLOW_PARENT_RUN_ID: mock.Mock(),
        MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.JOB),
        MLFLOW_SOURCE_NAME: mock.Mock(),
        MLFLOW_PROJECT_ENTRY_POINT: mock.Mock(),
        MLFLOW_GIT_COMMIT: mock.Mock(),
        "other-key": "other-value"
    }

    MlflowClient().create_run(experiment_id, start_time, tags)

    mock_store.create_run.assert_called_once_with(
        experiment_id=experiment_id,
        user_id=user,
        start_time=start_time,
        tags=[RunTag(key, value) for key, value in tags.items()],
    )
    mock_store.reset_mock()
    parent_run_id = "mock-parent-run-id"
    MlflowClient().create_run(experiment_id, start_time, tags)
    mock_store.create_run.assert_called_once_with(
        experiment_id=experiment_id,
        user_id=user,
        start_time=start_time,
        tags=[RunTag(key, value) for key, value in tags.items()]
    )


def test_client_search_runs_defaults(mock_store):
    MlflowClient().search_runs([1, 2, 3])
    mock_store.search_runs.assert_called_once_with(experiment_ids=[1, 2, 3],
                                                   filter_string="",
                                                   run_view_type=ViewType.ACTIVE_ONLY,
                                                   max_results=SEARCH_MAX_RESULTS_DEFAULT,
                                                   order_by=None)


def test_client_search_runs_filter(mock_store):
    MlflowClient().search_runs(["a", "b", "c"], "my filter")
    mock_store.search_runs.assert_called_once_with(experiment_ids=["a", "b", "c"],
                                                   filter_string="my filter",
                                                   run_view_type=ViewType.ACTIVE_ONLY,
                                                   max_results=SEARCH_MAX_RESULTS_DEFAULT,
                                                   order_by=None)


def test_client_search_runs_view_type(mock_store):
    MlflowClient().search_runs(["a", "b", "c"], "my filter", ViewType.DELETED_ONLY)
    mock_store.search_runs.assert_called_once_with(experiment_ids=["a", "b", "c"],
                                                   filter_string="my filter",
                                                   run_view_type=ViewType.DELETED_ONLY,
                                                   max_results=SEARCH_MAX_RESULTS_DEFAULT,
                                                   order_by=None)


def test_client_search_runs_max_results(mock_store):
    MlflowClient().search_runs([5], "my filter", ViewType.ALL, 2876)
    mock_store.search_runs.assert_called_once_with(experiment_ids=[5],
                                                   filter_string="my filter",
                                                   run_view_type=ViewType.ALL,
                                                   max_results=2876,
                                                   order_by=None)


def test_client_search_runs_int_experiment_id(mock_store):
    MlflowClient().search_runs(123)
    mock_store.search_runs.assert_called_once_with(experiment_ids=[123],
                                                   filter_string="",
                                                   run_view_type=ViewType.ACTIVE_ONLY,
                                                   max_results=SEARCH_MAX_RESULTS_DEFAULT,
                                                   order_by=None)


def test_client_search_runs_string_experiment_id(mock_store):
    MlflowClient().search_runs("abc")
    mock_store.search_runs.assert_called_once_with(experiment_ids=["abc"],
                                                   filter_string="",
                                                   run_view_type=ViewType.ACTIVE_ONLY,
                                                   max_results=SEARCH_MAX_RESULTS_DEFAULT,
                                                   order_by=None)


def test_client_search_runs_order_by(mock_store):
    MlflowClient().search_runs([5], order_by=["a", "b"])
    mock_store.search_runs.assert_called_once_with(experiment_ids=[5],
                                                   filter_string="",
                                                   run_view_type=ViewType.ACTIVE_ONLY,
                                                   max_results=SEARCH_MAX_RESULTS_DEFAULT,
                                                   order_by=["a", "b"])

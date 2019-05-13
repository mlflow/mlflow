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


@pytest.fixture
def mock_search_filter():
    with mock.patch("mlflow.tracking.client.SearchFilter") as mock_search_filter:
        yield mock_search_filter.return_value


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


def test_client_search_runs(mock_store, mock_search_filter):
    experiment_ids = [mock.Mock() for _ in range(5)]

    # Test defaults for view type and max results
    MlflowClient().search_runs(experiment_ids, "metrics.acc > 0.93")
    mock_store.search_runs.assert_called_once_with(experiment_ids=experiment_ids,
                                                   search_filter=mock_search_filter,
                                                   run_view_type=ViewType.ACTIVE_ONLY,
                                                   max_results=SEARCH_MAX_RESULTS_DEFAULT)

    # Test alternate view type
    mock_store.reset_mock()
    MlflowClient().search_runs(experiment_ids, "dummy filter", ViewType.DELETED_ONLY)
    mock_store.search_runs.assert_called_once_with(experiment_ids=experiment_ids,
                                                   search_filter=mock_search_filter,
                                                   run_view_type=ViewType.DELETED_ONLY,
                                                   max_results=SEARCH_MAX_RESULTS_DEFAULT)

    # Test with non-default max_results value
    mock_store.reset_mock()
    MlflowClient().search_runs(experiment_ids, "dummy filter", ViewType.ALL, 2876)
    mock_store.search_runs.assert_called_once_with(experiment_ids=experiment_ids,
                                                   search_filter=mock_search_filter,
                                                   run_view_type=ViewType.ALL,
                                                   max_results=2876)

import pytest
import mock

from mlflow.entities import RunTag, SourceType, ViewType
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE, MLFLOW_PARENT_RUN_ID, \
    MLFLOW_GIT_COMMIT, MLFLOW_PROJECT_ENTRY_POINT


@pytest.fixture
def mock_store():
    with mock.patch("mlflow.tracking.utils._get_store") as mock_get_store:
        yield mock_get_store.return_value


@pytest.fixture
def mock_user_id():
    with mock.patch("mlflow.tracking.client._get_user_id") as mock_get_user_id:
        yield mock_get_user_id.return_value


@pytest.fixture
def mock_time():
    time = 1552319350.244724
    with mock.patch("time.time", return_value=time):
        yield time


@pytest.fixture
def mock_search_filter():
    with mock.patch("mlflow.tracking.client.SearchFilter") as mock_search_filter:
        yield mock_search_filter.return_value


def test_client_create_run(mock_store, mock_user_id, mock_time):

    experiment_id = mock.Mock()

    MlflowClient().create_run(experiment_id)

    mock_store.create_run.assert_called_once_with(
        experiment_id=experiment_id,
        user_id=mock_user_id,
        run_name=None,
        start_time=int(mock_time * 1000),
        tags=[],
        parent_run_id=None,
        source_type=SourceType.LOCAL,
        source_name="Python Application",
        entry_point_name=None,
        source_version=None
    )


def test_client_create_run_overrides(mock_store):

    experiment_id = mock.Mock()
    user_id = mock.Mock()
    run_name = mock.Mock()
    start_time = mock.Mock()
    tags = {
        MLFLOW_PARENT_RUN_ID: mock.Mock(),
        MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.JOB),
        MLFLOW_SOURCE_NAME: mock.Mock(),
        MLFLOW_PROJECT_ENTRY_POINT: mock.Mock(),
        MLFLOW_GIT_COMMIT: mock.Mock(),
        "other-key": "other-value"
    }

    MlflowClient().create_run(experiment_id, user_id, run_name, start_time, tags)

    mock_store.create_run.assert_called_once_with(
        experiment_id=experiment_id,
        user_id=user_id,
        run_name=run_name,
        start_time=start_time,
        tags=[RunTag(key, value) for key, value in tags.items()],
        parent_run_id=tags[MLFLOW_PARENT_RUN_ID],
        source_type=SourceType.JOB,
        source_name=tags[MLFLOW_SOURCE_NAME],
        entry_point_name=tags[MLFLOW_PROJECT_ENTRY_POINT],
        source_version=tags[MLFLOW_GIT_COMMIT]
    )


def test_client_search_runs(mock_store, mock_search_filter):
    experiment_ids = [mock.Mock() for _ in range(5)]

    MlflowClient().search_runs(experiment_ids, "metrics.acc > 0.93")

    mock_store.search_runs.assert_called_once_with(experiment_ids=experiment_ids,
                                                   search_filter=mock_search_filter,
                                                   run_view_type=ViewType.ACTIVE_ONLY)

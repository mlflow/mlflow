import mock
import pytest

from mlflow.entities import SourceType, ViewType, RunTag
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode, FEATURE_DISABLED
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.tracking import set_registry_uri, MlflowClient
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_USER, MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE, \
    MLFLOW_PARENT_RUN_ID, MLFLOW_GIT_COMMIT, MLFLOW_PROJECT_ENTRY_POINT


@pytest.fixture
def mock_store():
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store") as mock_get_store:
        yield mock_get_store.return_value


@pytest.fixture
def mock_registry_store():
    with mock.patch("mlflow.tracking._model_registry.utils._get_store") as mock_get_store:
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
                                                   order_by=None,
                                                   page_token=None)


def test_client_search_runs_filter(mock_store):
    MlflowClient().search_runs(["a", "b", "c"], "my filter")
    mock_store.search_runs.assert_called_once_with(experiment_ids=["a", "b", "c"],
                                                   filter_string="my filter",
                                                   run_view_type=ViewType.ACTIVE_ONLY,
                                                   max_results=SEARCH_MAX_RESULTS_DEFAULT,
                                                   order_by=None,
                                                   page_token=None)


def test_client_search_runs_view_type(mock_store):
    MlflowClient().search_runs(["a", "b", "c"], "my filter", ViewType.DELETED_ONLY)
    mock_store.search_runs.assert_called_once_with(experiment_ids=["a", "b", "c"],
                                                   filter_string="my filter",
                                                   run_view_type=ViewType.DELETED_ONLY,
                                                   max_results=SEARCH_MAX_RESULTS_DEFAULT,
                                                   order_by=None,
                                                   page_token=None)


def test_client_search_runs_max_results(mock_store):
    MlflowClient().search_runs([5], "my filter", ViewType.ALL, 2876)
    mock_store.search_runs.assert_called_once_with(experiment_ids=[5],
                                                   filter_string="my filter",
                                                   run_view_type=ViewType.ALL,
                                                   max_results=2876,
                                                   order_by=None,
                                                   page_token=None)


def test_client_search_runs_int_experiment_id(mock_store):
    MlflowClient().search_runs(123)
    mock_store.search_runs.assert_called_once_with(experiment_ids=[123],
                                                   filter_string="",
                                                   run_view_type=ViewType.ACTIVE_ONLY,
                                                   max_results=SEARCH_MAX_RESULTS_DEFAULT,
                                                   order_by=None,
                                                   page_token=None)


def test_client_search_runs_string_experiment_id(mock_store):
    MlflowClient().search_runs("abc")
    mock_store.search_runs.assert_called_once_with(experiment_ids=["abc"],
                                                   filter_string="",
                                                   run_view_type=ViewType.ACTIVE_ONLY,
                                                   max_results=SEARCH_MAX_RESULTS_DEFAULT,
                                                   order_by=None,
                                                   page_token=None)


def test_client_search_runs_order_by(mock_store):
    MlflowClient().search_runs([5], order_by=["a", "b"])
    mock_store.search_runs.assert_called_once_with(experiment_ids=[5],
                                                   filter_string="",
                                                   run_view_type=ViewType.ACTIVE_ONLY,
                                                   max_results=SEARCH_MAX_RESULTS_DEFAULT,
                                                   order_by=["a", "b"],
                                                   page_token=None)


def test_client_search_runs_page_token(mock_store):
    MlflowClient().search_runs([5], page_token="blah")
    mock_store.search_runs.assert_called_once_with(experiment_ids=[5],
                                                   filter_string="",
                                                   run_view_type=ViewType.ACTIVE_ONLY,
                                                   max_results=SEARCH_MAX_RESULTS_DEFAULT,
                                                   order_by=None,
                                                   page_token="blah")


def test_client_registry_operations_raise_exception_with_unsupported_registry_store():
    """
    This test case ensures that Model Registry operations invoked on the `MlflowClient`
    fail with an informative error message when the registry store URI refers to a
    store that does not support Model Registry features (e.g., FileStore).
    """
    with TempDir() as tmp:
        client = MlflowClient(registry_uri=tmp.path())
        expected_failure_functions = [
            client._get_registry_client,
            lambda: client.create_registered_model("test"),
            lambda: client.get_registered_model("test"),
            lambda: client.create_model_version("test", "source", "run_id"),
            lambda: client.get_model_version("test", 1),
        ]
        for func in expected_failure_functions:
            with pytest.raises(MlflowException) as exc:
                func()
            assert exc.value.error_code == ErrorCode.Name(FEATURE_DISABLED)


def test_update_registered_model(mock_registry_store):
    """
    Update registered model no longer supports name change.
    """
    expected_return_value = "some expected return value."
    mock_registry_store.rename_registered_model.return_value = expected_return_value
    expected_return_value_2 = "other expected return value."
    mock_registry_store.update_registered_model.return_value = expected_return_value_2
    res = MlflowClient(registry_uri="sqlite:///somedb.db").update_registered_model(
        name="orig name", description="new description")
    assert expected_return_value_2 == res
    mock_registry_store.update_registered_model.assert_called_once_with(
        name="orig name", description="new description")
    mock_registry_store.rename_registered_model.assert_not_called()


def test_update_model_version(mock_registry_store):
    """
    Update registered model no longer support state changes.
    """
    expected_return_value = "some expected return value."
    mock_registry_store.update_model_version.return_value = expected_return_value
    res = MlflowClient(registry_uri="sqlite:///somedb.db").update_model_version(
        name="orig name", version="1", description="desc")
    assert expected_return_value == res
    mock_registry_store.update_model_version.assert_called_once_with(
        name="orig name", version="1", description="desc")
    mock_registry_store.transition_model_version_stage.assert_not_called()


def test_transition_model_version_stage(mock_registry_store):
    name = "Model 1"
    version = "12"
    stage = "Production"
    expected_result = ModelVersion(name, version, creation_timestamp=123, current_stage=stage)
    mock_registry_store.transition_model_version_stage.return_value = expected_result
    actual_result = (
        MlflowClient(registry_uri="sqlite:///somedb.db")
        .transition_model_version_stage(name, version, stage)
    )
    mock_registry_store.transition_model_version_stage.assert_called_once_with(
        name=name, version=version, stage=stage, archive_existing_versions=False)
    assert expected_result == actual_result


def test_registry_uri_set_as_param():
    uri = "sqlite:///somedb.db"
    client = MlflowClient(tracking_uri="databricks://tracking", registry_uri=uri)
    assert client._registry_uri == uri


def test_registry_uri_from_set_registry_uri():
    uri = "sqlite:///somedb.db"
    set_registry_uri(uri)
    client = MlflowClient(tracking_uri="databricks://tracking")
    assert client._registry_uri == uri
    set_registry_uri(None)


def test_registry_uri_from_tracking_uri_param():
    with mock.patch("mlflow.tracking._tracking_service.utils.get_tracking_uri") \
            as get_tracking_uri_mock:
        get_tracking_uri_mock.return_value = "databricks://default_tracking"
        tracking_uri = "databricks://tracking_vhawoierj"
        client = MlflowClient(tracking_uri=tracking_uri)
        assert client._registry_uri == tracking_uri


def test_registry_uri_from_implicit_tracking_uri():
    with mock.patch("mlflow.tracking._tracking_service.utils.get_tracking_uri")\
            as get_tracking_uri_mock:
        tracking_uri = "databricks://tracking_wierojasdf"
        get_tracking_uri_mock.return_value = tracking_uri
        client = MlflowClient()
        assert client._registry_uri == tracking_uri

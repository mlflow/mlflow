import filecmp
import json
import mock
import os
import shutil

from databricks_cli.configure.provider import DatabricksConfig
import databricks_cli
import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.projects.databricks import DatabricksJobRunner
from mlflow.entities import RunStatus
from mlflow.projects import databricks, ExecutionException
from mlflow.tracking import MlflowClient
from mlflow.utils import file_utils
from mlflow.store.tracking.file_store import FileStore
from mlflow.utils.mlflow_tags import MLFLOW_DATABRICKS_RUN_URL, \
    MLFLOW_DATABRICKS_SHELL_JOB_RUN_ID, \
    MLFLOW_DATABRICKS_WEBAPP_URL
from mlflow.utils.rest_utils import _DEFAULT_HEADERS


from tests.projects.utils import validate_exit_status, TEST_PROJECT_DIR
from tests.projects.utils import tracking_uri_mock  # pylint: disable=unused-import


@pytest.fixture()
def runs_cancel_mock():
    """Mocks the Jobs Runs Cancel API request"""
    with mock.patch("mlflow.projects.databricks.DatabricksJobRunner.jobs_runs_cancel")\
            as runs_cancel_mock:
        runs_cancel_mock.return_value = None
        yield runs_cancel_mock


@pytest.fixture()
def runs_submit_mock():
    """Mocks the Jobs Runs Submit API request"""
    with mock.patch("mlflow.projects.databricks.DatabricksJobRunner._jobs_runs_submit")\
            as runs_submit_mock:
        runs_submit_mock.return_value = {"run_id": "-1"}
        yield runs_submit_mock


@pytest.fixture()
def runs_get_mock():
    """Mocks the Jobs Runs Get API request"""
    with mock.patch("mlflow.projects.databricks.DatabricksJobRunner.jobs_runs_get")\
            as runs_get_mock:
        yield runs_get_mock


@pytest.fixture()
def cluster_spec_mock(tmpdir):
    cluster_spec_handle = tmpdir.join("cluster_spec.json")
    cluster_spec_handle.write(json.dumps(dict()))
    yield str(cluster_spec_handle)


@pytest.fixture()
def dbfs_root_mock(tmpdir):
    yield str(tmpdir.join("dbfs-root"))


@pytest.fixture()
def upload_to_dbfs_mock(dbfs_root_mock):
    def upload_mock_fn(_, src_path, dbfs_uri):
        mock_dbfs_dst = os.path.join(dbfs_root_mock, dbfs_uri.split("/dbfs/")[1])
        os.makedirs(os.path.dirname(mock_dbfs_dst))
        shutil.copy(src_path, mock_dbfs_dst)

    with mock.patch.object(
            mlflow.projects.databricks.DatabricksJobRunner, "_upload_to_dbfs",
            new=upload_mock_fn) as upload_mock:
        yield upload_mock


@pytest.fixture()
def dbfs_path_exists_mock(dbfs_root_mock):  # pylint: disable=unused-argument
    with mock.patch("mlflow.projects.databricks.DatabricksJobRunner._dbfs_path_exists")\
            as path_exists_mock:
        yield path_exists_mock


@pytest.fixture()
def dbfs_mocks(dbfs_path_exists_mock, upload_to_dbfs_mock):  # pylint: disable=unused-argument
    yield


@pytest.fixture()
def before_run_validations_mock():  # pylint: disable=unused-argument
    with mock.patch("mlflow.projects.databricks.before_run_validations"):
        yield


@pytest.fixture()
def set_tag_mock():
    with mock.patch("mlflow.projects.databricks.tracking.MlflowClient") as m:
        mlflow_service_mock = mock.Mock(wraps=MlflowClient())
        m.return_value = mlflow_service_mock
        yield mlflow_service_mock.set_tag


def _get_mock_run_state(succeeded):
    if succeeded is None:
        return {"life_cycle_state": "RUNNING", "state_message": ""}
    if succeeded:
        run_result_state = "SUCCESS"
    else:
        run_result_state = "FAILED"
    return {"life_cycle_state": "TERMINATED", "state_message": "", "result_state": run_result_state}


def mock_runs_get_result(succeeded):
    run_state = _get_mock_run_state(succeeded)
    return {"state": run_state, "run_page_url": "test_url"}


def run_databricks_project(cluster_spec, **kwargs):
    return mlflow.projects.run(
        uri=TEST_PROJECT_DIR, backend="databricks", backend_config=cluster_spec,
        parameters={"alpha": "0.4"}, **kwargs)


def test_upload_project_to_dbfs(
        dbfs_root_mock, tmpdir, dbfs_path_exists_mock,
        upload_to_dbfs_mock):  # pylint: disable=unused-argument
    # Upload project to a mock directory
    dbfs_path_exists_mock.return_value = False
    runner = DatabricksJobRunner(databricks_profile="DEFAULT")
    dbfs_uri = runner._upload_project_to_dbfs(
        project_dir=TEST_PROJECT_DIR, experiment_id=FileStore.DEFAULT_EXPERIMENT_ID)
    # Get expected tar
    local_tar_path = os.path.join(dbfs_root_mock, dbfs_uri.split("/dbfs/")[1])
    expected_tar_path = str(tmpdir.join("expected.tar.gz"))
    file_utils.make_tarfile(
        output_filename=expected_tar_path, source_dir=TEST_PROJECT_DIR,
        archive_name=databricks.DB_TARFILE_ARCHIVE_NAME)
    # Extract the tarred project, verify its contents
    assert filecmp.cmp(local_tar_path, expected_tar_path, shallow=False)


def test_upload_existing_project_to_dbfs(dbfs_path_exists_mock):  # pylint: disable=unused-argument
    # Check that we don't upload the project if it already exists on DBFS
    with mock.patch("mlflow.projects.databricks.DatabricksJobRunner._upload_to_dbfs")\
            as upload_to_dbfs_mock:
        dbfs_path_exists_mock.return_value = True
        runner = DatabricksJobRunner(databricks_profile="DEFAULT")
        runner._upload_project_to_dbfs(
            project_dir=TEST_PROJECT_DIR, experiment_id=FileStore.DEFAULT_EXPERIMENT_ID)
        assert upload_to_dbfs_mock.call_count == 0


def test_run_databricks_validations(
        tmpdir, cluster_spec_mock,  # pylint: disable=unused-argument
        tracking_uri_mock, dbfs_mocks, set_tag_mock):  # pylint: disable=unused-argument
    """
    Tests that running on Databricks fails before making any API requests if validations fail.
    """
    with mock.patch.dict(os.environ, {'DATABRICKS_HOST': 'test-host', 'DATABRICKS_TOKEN': 'foo'}),\
        mock.patch("mlflow.projects.databricks.DatabricksJobRunner._databricks_api_request")\
            as db_api_req_mock:
        # Test bad tracking URI
        tracking_uri_mock.return_value = tmpdir.strpath
        with pytest.raises(ExecutionException):
            run_databricks_project(cluster_spec_mock, synchronous=True)
        assert db_api_req_mock.call_count == 0
        db_api_req_mock.reset_mock()
        mlflow_service = mlflow.tracking.MlflowClient()
        assert (len(mlflow_service.list_run_infos(experiment_id=FileStore.DEFAULT_EXPERIMENT_ID))
                == 0)
        tracking_uri_mock.return_value = "http://"
        # Test misspecified parameters
        with pytest.raises(ExecutionException):
            mlflow.projects.run(
                TEST_PROJECT_DIR, backend="databricks", entry_point="greeter",
                backend_config=cluster_spec_mock)
        assert db_api_req_mock.call_count == 0
        db_api_req_mock.reset_mock()
        # Test bad cluster spec
        with pytest.raises(ExecutionException):
            mlflow.projects.run(TEST_PROJECT_DIR, backend="databricks", synchronous=True,
                                backend_config=None)
        assert db_api_req_mock.call_count == 0
        db_api_req_mock.reset_mock()
        # Test that validations pass with good tracking URIs
        databricks.before_run_validations("http://", cluster_spec_mock)
        databricks.before_run_validations("databricks", cluster_spec_mock)


def test_run_databricks(
        before_run_validations_mock,  # pylint: disable=unused-argument
        tracking_uri_mock, runs_cancel_mock, dbfs_mocks,  # pylint: disable=unused-argument
        runs_submit_mock, runs_get_mock, cluster_spec_mock, set_tag_mock):
    """Test running on Databricks with mocks."""
    with mock.patch.dict(os.environ, {'DATABRICKS_HOST': 'test-host', 'DATABRICKS_TOKEN': 'foo'}):
        # Test that MLflow gets the correct run status when performing a Databricks run
        for run_succeeded, expect_status in [(True, RunStatus.FINISHED), (False, RunStatus.FAILED)]:
            runs_get_mock.return_value = mock_runs_get_result(succeeded=run_succeeded)
            submitted_run = run_databricks_project(cluster_spec_mock, synchronous=False)
            assert submitted_run.wait() == run_succeeded
            assert submitted_run.run_id is not None
            assert runs_submit_mock.call_count == 1
            tags = {}
            for call_args, _ in set_tag_mock.call_args_list:
                tags[call_args[1]] = call_args[2]
            assert tags[MLFLOW_DATABRICKS_RUN_URL] == 'test_url'
            assert tags[MLFLOW_DATABRICKS_SHELL_JOB_RUN_ID] == '-1'
            assert tags[MLFLOW_DATABRICKS_WEBAPP_URL] == 'test-host'
            set_tag_mock.reset_mock()
            runs_submit_mock.reset_mock()
            validate_exit_status(submitted_run.get_status(), expect_status)


def test_run_databricks_cluster_spec_json(
        before_run_validations_mock,  # pylint: disable=unused-argument
        tracking_uri_mock, runs_cancel_mock, dbfs_mocks,  # pylint: disable=unused-argument
        runs_submit_mock, runs_get_mock,
        cluster_spec_mock, set_tag_mock):  # pylint: disable=unused-argument
    with mock.patch.dict(os.environ, {'DATABRICKS_HOST': 'test-host', 'DATABRICKS_TOKEN': 'foo'}):
        runs_get_mock.return_value = mock_runs_get_result(succeeded=True)
        cluster_spec = {
            "spark_version": "5.0.x-scala2.11",
            "num_workers": 2,
            "node_type_id": "i3.xlarge",
        }
        # Run project synchronously, verify that it succeeds (doesn't throw)
        run_databricks_project(cluster_spec=cluster_spec, synchronous=True)
        assert runs_submit_mock.call_count == 1
        runs_submit_args, _ = runs_submit_mock.call_args_list[0]
        req_body = runs_submit_args[0]
        assert req_body["new_cluster"] == cluster_spec


def test_run_databricks_cancel(
        before_run_validations_mock, tracking_uri_mock,  # pylint: disable=unused-argument
        runs_submit_mock, dbfs_mocks, set_tag_mock,  # pylint: disable=unused-argument
        runs_cancel_mock, runs_get_mock, cluster_spec_mock):
    # Test that MLflow properly handles Databricks run cancellation. We mock the result of
    # the runs-get API to indicate run failure so that cancel() exits instead of blocking while
    # waiting for run status.
    with mock.patch.dict(os.environ, {'DATABRICKS_HOST': 'test-host', 'DATABRICKS_TOKEN': 'foo'}):
        runs_get_mock.return_value = mock_runs_get_result(succeeded=False)
        submitted_run = run_databricks_project(cluster_spec_mock, synchronous=False)
        submitted_run.cancel()
        validate_exit_status(submitted_run.get_status(), RunStatus.FAILED)
        assert runs_cancel_mock.call_count == 1
        # Test that we raise an exception when a blocking Databricks run fails
        runs_get_mock.return_value = mock_runs_get_result(succeeded=False)
        with pytest.raises(mlflow.projects.ExecutionException):
            run_databricks_project(cluster_spec_mock, synchronous=True)


def test_get_tracking_uri_for_run():
    mlflow.set_tracking_uri("http://some-uri")
    assert databricks._get_tracking_uri_for_run() == "http://some-uri"
    mlflow.set_tracking_uri("databricks://profile")
    assert databricks._get_tracking_uri_for_run() == "databricks"
    mlflow.set_tracking_uri(None)
    with mock.patch.dict(os.environ, {mlflow.tracking._TRACKING_URI_ENV_VAR: "http://some-uri"}):
        assert mlflow.tracking._tracking_service.utils.get_tracking_uri() == "http://some-uri"


class MockProfileConfigProvider:
    def __init__(self, profile):
        assert profile == "my-profile"

    def get_config(self):
        return DatabricksConfig("host", "user", "pass", None, insecure=False)


@mock.patch('requests.request')
@mock.patch('databricks_cli.configure.provider.get_config')
@mock.patch.object(databricks_cli.configure.provider, 'ProfileConfigProvider',
                   MockProfileConfigProvider)
def test_databricks_http_request_integration(get_config, request):
    """Confirms that the databricks http request params can in fact be used as an HTTP request"""
    def confirm_request_params(**kwargs):
        headers = dict(_DEFAULT_HEADERS)
        headers['Authorization'] = 'Basic dXNlcjpwYXNz'
        assert kwargs == {
            'method': 'PUT',
            'url': 'host/clusters/list',
            'headers': headers,
            'verify': True,
            'json': {'a': 'b'}
        }
        http_response = mock.MagicMock()
        http_response.status_code = 200
        http_response.text = '{"OK": "woo"}'
        return http_response
    request.side_effect = confirm_request_params
    get_config.return_value = \
        DatabricksConfig("host", "user", "pass", None, insecure=False)

    response = DatabricksJobRunner(databricks_profile=None)._databricks_api_request(
        '/clusters/list', 'PUT', json={'a': 'b'})
    assert json.loads(response.text) == {'OK': 'woo'}
    get_config.reset_mock()
    response = DatabricksJobRunner(databricks_profile="my-profile")._databricks_api_request(
        '/clusters/list', 'PUT', json={'a': 'b'})
    assert json.loads(response.text) == {'OK': 'woo'}
    assert get_config.call_count == 0


@mock.patch("mlflow.utils.databricks_utils.get_databricks_host_creds")
def test_run_databricks_failed(_):
    with mock.patch('mlflow.utils.rest_utils.http_request') as m:
        text = '{"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "Node type not supported"}'
        m.return_value = mock.Mock(text=text, status_code=400)
        runner = DatabricksJobRunner('profile')
        with pytest.raises(MlflowException):
            runner._run_shell_command_job('/project', 'command', {}, {})

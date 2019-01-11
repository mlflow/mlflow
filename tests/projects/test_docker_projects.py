import os
import git

import mock
import pytest

from databricks_cli.configure.provider import DatabricksConfig

import mlflow
from mlflow.entities import ViewType
from mlflow.store import file_store

from tests.projects.utils import TEST_DOCKER_PROJECT_DIR
from tests.projects.utils import tracking_uri_mock  # pylint: disable=unused-import


def _build_uri(base_uri, subdirectory):
    if subdirectory != "":
        return "%s#%s" % (base_uri, subdirectory)
    return base_uri


def _get_version_local_git_repo(local_git_repo):
    repo = git.Repo(local_git_repo, search_parent_directories=True)
    return repo.git.rev_parse("HEAD")


@pytest.mark.parametrize("use_start_run", map(str, [0, 1]))
def test_docker_project_execution(
        use_start_run, tmpdir, tracking_uri_mock):  # pylint: disable=unused-argument
    expected_params = {"use_start_run": use_start_run}
    submitted_run = mlflow.projects.run(
        TEST_DOCKER_PROJECT_DIR, experiment_id=0, parameters=expected_params,
        entry_point="test_tracking")
    # Validate run contents in the FileStore
    run_uuid = submitted_run.run_id
    mlflow_service = mlflow.tracking.MlflowClient()
    run_infos = mlflow_service.list_run_infos(experiment_id=0, run_view_type=ViewType.ACTIVE_ONLY)
    print(run_infos)
    assert "file:" in run_infos[0].source_name
    assert len(run_infos) == 1
    store_run_uuid = run_infos[0].run_uuid
    assert run_uuid == store_run_uuid
    run = mlflow_service.get_run(run_uuid)
    assert len(run.data.params) == len(expected_params)
    for param in run.data.params:
        assert param.value == expected_params[param.key]
    expected_metrics = {"some_key": 3}
    assert len(run.data.metrics) == len(expected_metrics)
    for metric in run.data.metrics:
        assert metric.value == expected_metrics[metric.key]
    print(run.data.tags)
    exact_expected_tags = {
        "mlflow.gitRepoURL": "https://github.com/mlflow/mlflow", "mlflow.docker": "true"}
    approx_expected_tags = {
        "mlflow.docker.image.name": "mlflow-docker-example",
        "mlflow.docker.image.id": "sha256:",
    }
    assert len(run.data.tags) == len(exact_expected_tags) + len(approx_expected_tags)
    for tag in run.data.tags:
        if tag.key in exact_expected_tags:
            assert tag.value == exact_expected_tags[tag.key]
        else:
            assert tag.value.startswith(approx_expected_tags[tag.key])


@pytest.mark.parametrize("tracking_uri, expected_command_segment", [
    (None, "-e MLFLOW_TRACKING_URI=/mlflow/tmp/mlruns"),
    ("http://some-tracking-uri", "-e MLFLOW_TRACKING_URI=http://some-tracking-uri"),
    ("databricks://some-profile", "-e MLFLOW_TRACKING_URI=databricks ")
])
@mock.patch('databricks_cli.configure.provider.ProfileConfigProvider')
def test_docker_project_tracking_uri_propagation(
        ProfileConfigProvider, tmpdir, tracking_uri,
        expected_command_segment):  # pylint: disable=unused-argument
    mock_provider = mock.MagicMock()
    mock_provider.get_config.return_value = \
        DatabricksConfig("host", "user", "pass", None, insecure=True)
    ProfileConfigProvider.return_value = mock_provider
    # Create and mock local tracking directory
    local_tracking_dir = os.path.join(tmpdir.strpath, "mlruns")
    if tracking_uri is None:
        tracking_uri = local_tracking_dir
    old_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(tracking_uri)
        with mock.patch("mlflow.tracking.utils._get_store") as _get_store_mock:
            _get_store_mock.return_value = file_store.FileStore(local_tracking_dir)
            mlflow.projects.run(TEST_DOCKER_PROJECT_DIR, experiment_id=0)
    finally:
        mlflow.set_tracking_uri(old_uri)

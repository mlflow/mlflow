import pytest
from unittest import mock
from unittest.mock import Mock

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository


@pytest.mark.parametrize(
    "uri, expected_run_id, expected_artifact_path",
    [
        ("runs:/1234abcdf1394asdfwer33/path/to/model", "1234abcdf1394asdfwer33", "path/to/model"),
        ("runs:/1234abcdf1394asdfwer33/path/to/model/", "1234abcdf1394asdfwer33", "path/to/model/"),
        ("runs://profile@databricks/1234abcdf1394asdfwer33/path", "1234abcdf1394asdfwer33", "path"),
        ("runs:/1234abcdf1394asdfwer33", "1234abcdf1394asdfwer33", None),
        ("runs:/1234abcdf1394asdfwer33/", "1234abcdf1394asdfwer33", None),
        ("runs:///1234abcdf1394asdfwer33/", "1234abcdf1394asdfwer33", None),
        ("runs://profile@databricks/1234abcdf1394asdfwer33/", "1234abcdf1394asdfwer33", None),
    ],
)
def test_parse_runs_uri_valid_input(uri, expected_run_id, expected_artifact_path):
    (run_id, artifact_path) = RunsArtifactRepository.parse_runs_uri(uri)
    assert run_id == expected_run_id
    assert artifact_path == expected_artifact_path


@pytest.mark.parametrize(
    "uri",
    [
        "notruns:/1234abcdf1394asdfwer33/",  # wrong scheme
        "runs:/",  # no run id
        "runs:1234abcdf1394asdfwer33/",  # missing slash
        "runs://1234abcdf1394asdfwer33/",  # hostnames are not yet supported
    ],
)
def test_parse_runs_uri_invalid_input(uri):
    with pytest.raises(MlflowException, match="Not a proper runs"):
        RunsArtifactRepository.parse_runs_uri(uri)


@pytest.mark.parametrize(
    "uri, expected_tracking_uri, mock_uri, expected_result_uri",
    [
        ("runs:/1234abcdf1394asdfwer33/path/model", None, "s3:/some/path", "s3:/some/path"),
        ("runs:/1234abcdf1394asdfwer33/path/model", None, "dbfs:/some/path", "dbfs:/some/path"),
        (
            "runs://profile@databricks/1234abcdf1394asdfwer33/path/model",
            "databricks://profile",
            "s3:/some/path",
            "s3:/some/path",
        ),
        (
            "runs://profile@databricks/1234abcdf1394asdfwer33/path/model",
            "databricks://profile",
            "dbfs:/some/path",
            "dbfs://profile@databricks/some/path",
        ),
        (
            "runs://scope:key@databricks/1234abcdf1394asdfwer33/path/model",
            "databricks://scope:key",
            "dbfs:/some/path",
            "dbfs://scope:key@databricks/some/path",
        ),
    ],
)
def test_get_artifact_uri(uri, expected_tracking_uri, mock_uri, expected_result_uri):
    with mock.patch(
        "mlflow.tracking.artifact_utils.get_artifact_uri", return_value=mock_uri
    ) as get_artifact_uri_mock:
        result_uri = RunsArtifactRepository.get_underlying_uri(uri)
        get_artifact_uri_mock.assert_called_once_with(
            "1234abcdf1394asdfwer33", "path/model", expected_tracking_uri
        )
        assert result_uri == expected_result_uri


def test_runs_artifact_repo_init_with_real_run():
    artifact_location = "s3://blah_bucket/"
    experiment_id = mlflow.create_experiment("expr_abc", artifact_location)
    with mlflow.start_run(experiment_id=experiment_id):
        run_id = mlflow.active_run().info.run_id
    runs_uri = "runs:/%s/path/to/model" % run_id
    runs_repo = RunsArtifactRepository(runs_uri)

    assert runs_repo.artifact_uri == runs_uri
    assert isinstance(runs_repo.repo, S3ArtifactRepository)
    expected_absolute_uri = "%s%s/artifacts/path/to/model" % (artifact_location, run_id)
    assert runs_repo.repo.artifact_uri == expected_absolute_uri


def test_runs_artifact_repo_uses_repo_download_artifacts():
    """
    The RunsArtifactRepo should delegate `download_artifacts` to it's self.repo.download_artifacts
    function
    """
    artifact_location = "s3://blah_bucket/"
    experiment_id = mlflow.create_experiment("expr_abcd", artifact_location)
    with mlflow.start_run(experiment_id=experiment_id):
        run_id = mlflow.active_run().info.run_id
    runs_repo = RunsArtifactRepository("runs:/{}".format(run_id))
    runs_repo.repo = Mock()
    runs_repo.download_artifacts("artifact_path", "dst_path")
    runs_repo.repo.download_artifacts.assert_called_once()

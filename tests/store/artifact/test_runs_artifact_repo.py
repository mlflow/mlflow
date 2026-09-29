from unittest import mock
from unittest.mock import Mock

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import PERMISSION_DENIED, RESOURCE_DOES_NOT_EXIST
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.store.entities.paged_list import PagedList


@pytest.mark.parametrize(
    ("uri", "expected_run_id", "expected_artifact_path"),
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
    ("uri", "expected_tracking_uri", "mock_uri", "expected_result_uri"),
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
            run_id="1234abcdf1394asdfwer33",
            artifact_path="path/model",
            tracking_uri=expected_tracking_uri,
        )
        assert result_uri == expected_result_uri


def test_runs_artifact_repo_init_with_real_run():
    artifact_location = "s3://blah_bucket/"
    experiment_id = mlflow.create_experiment("expr_abc", artifact_location)
    with mlflow.start_run(experiment_id=experiment_id):
        run_id = mlflow.active_run().info.run_id
    runs_uri = f"runs:/{run_id}/path/to/model"
    runs_repo = RunsArtifactRepository(runs_uri)

    assert runs_repo.artifact_uri == runs_uri
    assert isinstance(runs_repo.repo, S3ArtifactRepository)
    expected_absolute_uri = f"{artifact_location}{run_id}/artifacts/path/to/model"
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
    runs_repo = RunsArtifactRepository(f"runs:/{run_id}")
    runs_repo.repo = Mock()
    runs_repo.download_artifacts("artifact_path", "dst_path")
    runs_repo.repo.download_artifacts.assert_called_once()


@pytest.fixture
def runs_artifact_repo():
    with (
        mock.patch.object(
            RunsArtifactRepository, "get_underlying_uri", return_value="file:///unused"
        ),
        mock.patch(
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            return_value=Mock(),
        ),
    ):
        return RunsArtifactRepository("runs:/run-id")


def test_download_artifacts_reports_run_backend_failure(runs_artifact_repo, tmp_path):
    run_error = ConnectionError("Artifact store unavailable")
    runs_artifact_repo.repo.download_artifacts.side_effect = run_error

    with (
        mock.patch.object(runs_artifact_repo, "_get_logged_model_artifact_repo", return_value=None),
        pytest.raises(MlflowException, match="backend error") as exc_info,
    ):
        runs_artifact_repo.download_artifacts("model", str(tmp_path))

    assert exc_info.value.error_code == "INTERNAL_ERROR"
    assert exc_info.value.__cause__ is run_error


def test_download_artifacts_reports_logged_model_backend_failure(runs_artifact_repo, tmp_path):
    run_error = MlflowException("No such artifact", error_code=RESOURCE_DOES_NOT_EXIST)
    model_error = MlflowException("Permission denied", error_code=PERMISSION_DENIED)
    runs_artifact_repo.repo.download_artifacts.side_effect = run_error
    model_repo = Mock()
    model_repo.download_artifacts.side_effect = model_error

    with (
        mock.patch.object(
            runs_artifact_repo, "_get_logged_model_artifact_repo", return_value=model_repo
        ),
        pytest.raises(MlflowException, match="backend error") as exc_info,
    ):
        runs_artifact_repo.download_artifacts("model/MLmodel", str(tmp_path))

    model_repo.download_artifacts.assert_called_once_with(
        artifact_path="MLmodel", dst_path=str(tmp_path / "model")
    )
    assert exc_info.value.error_code == "PERMISSION_DENIED"
    assert exc_info.value.__cause__ is model_error


@pytest.mark.parametrize(
    "run_error",
    [
        MlflowException("No such artifact", error_code=RESOURCE_DOES_NOT_EXIST),
        FileNotFoundError("No such artifact"),
    ],
)
def test_download_artifacts_reports_missing_path_after_both_repos(
    runs_artifact_repo, tmp_path, run_error
):
    model_error = MlflowException("No such model artifact", error_code=RESOURCE_DOES_NOT_EXIST)
    runs_artifact_repo.repo.download_artifacts.side_effect = run_error
    model_repo = Mock()
    model_repo.download_artifacts.side_effect = model_error

    with (
        mock.patch.object(
            runs_artifact_repo, "_get_logged_model_artifact_repo", return_value=model_repo
        ),
        pytest.raises(MlflowException, match="please ensure that the path is correct") as exc_info,
    ):
        runs_artifact_repo.download_artifacts("model/MLmodel", str(tmp_path))

    assert exc_info.value.error_code == "RESOURCE_DOES_NOT_EXIST"
    assert exc_info.value.__cause__ is run_error


def test_download_artifacts_uses_logged_model_after_run_backend_failure(
    runs_artifact_repo, tmp_path
):
    runs_artifact_repo.repo.download_artifacts.side_effect = ConnectionError(
        "Run store unavailable"
    )
    model_repo = Mock()
    model_repo.download_artifacts.return_value = str(tmp_path / "model" / "MLmodel")

    with mock.patch.object(
        runs_artifact_repo, "_get_logged_model_artifact_repo", return_value=model_repo
    ):
        path = runs_artifact_repo.download_artifacts("model/MLmodel", str(tmp_path))

    assert path == str(tmp_path / "model" / "MLmodel")


def test_download_artifacts_uses_run_after_logged_model_backend_failure(
    runs_artifact_repo, tmp_path
):
    run_path = str(tmp_path / "artifact")
    runs_artifact_repo.repo.download_artifacts.return_value = run_path
    with mock.patch.object(
        runs_artifact_repo,
        "_download_model_artifacts",
        side_effect=ConnectionError("Model store unavailable"),
    ):
        assert runs_artifact_repo.download_artifacts("model", str(tmp_path)) == run_path


def test_runs_artifact_repo_tracking_uri_passed_as_keyword():
    """
    Test that tracking_uri is passed as keyword argument to get_artifact_repository.
    This verifies the fix for issue #16873 where tracking_uri was incorrectly passed
    as a positional argument, causing it to be interpreted as access_key_id in S3.
    """
    with mock.patch(
        "mlflow.tracking.artifact_utils.get_artifact_uri",
        return_value="s3://test-bucket/some-run-id/artifacts/path/to/model",
    ) as mock_get_artifact_uri:
        runs_repo = RunsArtifactRepository(
            artifact_uri="runs:/some-run-id/path/to/model",
            tracking_uri="http://test-tracking-server:5000",
        )
        assert isinstance(runs_repo.repo, S3ArtifactRepository)
        mock_get_artifact_uri.assert_called_once()


def test_get_logged_model_artifact_repo_uses_models_uri():
    with (
        mock.patch(
            "mlflow.store.artifact.runs_artifact_repo.RunsArtifactRepository.get_underlying_uri",
            return_value="mlflow-artifacts:/1/some-run-id/artifacts/model",
        ),
        mock.patch(
            "mlflow.store.artifact.runs_artifact_repo.mlflow.tracking.MlflowClient"
        ) as mock_mlflow_client,
        mock.patch(
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository"
        ) as mock_get_artifact_repo,
    ):
        runs_repo = RunsArtifactRepository(
            artifact_uri="runs:/some-run-id/model",
            tracking_uri="http://test-tracking-server:5000",
            registry_uri="sqlite:///mlflow.db",
        )
        mock_get_artifact_repo.reset_mock()

        run = Mock()
        run.info.experiment_id = "123"
        mock_mlflow_client.return_value.get_run.return_value = run

        matched_model = Mock()
        matched_model.source_run_id = "some-run-id"
        matched_model.model_id = "m-123456"
        mock_mlflow_client.return_value.search_logged_models.return_value = PagedList(
            [matched_model], token=None
        )

        repo = runs_repo._get_logged_model_artifact_repo(run_id="some-run-id", name="model")

        assert repo == mock_get_artifact_repo.return_value
        mock_get_artifact_repo.assert_called_once_with(
            "models:/m-123456",
            tracking_uri="http://test-tracking-server:5000",
            registry_uri="sqlite:///mlflow.db",
        )

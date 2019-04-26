import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.store.runs_artifact_repo import RunsArtifactRepository
from mlflow.store.s3_artifact_repo import S3ArtifactRepository

from tests.projects.utils import tracking_uri_mock  # pylint: disable=unused-import


@pytest.mark.parametrize("uri, expected_run_id, expected_artifact_path", [
    ('runs:/1234abcdf1394asdfwer33/path/to/model', '1234abcdf1394asdfwer33', 'path/to/model'),
    ('runs:/1234abcdf1394asdfwer33/path/to/model/', '1234abcdf1394asdfwer33', 'path/to/model/'),
    ('runs:/1234abcdf1394asdfwer33', '1234abcdf1394asdfwer33', None),
    ('runs:/1234abcdf1394asdfwer33/', '1234abcdf1394asdfwer33', None),
    ('runs:///1234abcdf1394asdfwer33/', '1234abcdf1394asdfwer33', None),
])
def test_parse_runs_uri_valid_input(uri, expected_run_id, expected_artifact_path):
    (run_id, artifact_path) = RunsArtifactRepository.parse_runs_uri(uri)
    assert run_id == expected_run_id
    assert artifact_path == expected_artifact_path


@pytest.mark.parametrize("uri", [
    'notruns:/1234abcdf1394asdfwer33/',  # wrong scheme
    'runs:/',                            # no run id
    'runs:1234abcdf1394asdfwer33/',      # missing slash
    'runs://1234abcdf1394asdfwer33/',    # hostnames are not yet supported
])
def test_parse_runs_uri_invalid_input(uri):
    with pytest.raises(MlflowException):
        RunsArtifactRepository.parse_runs_uri(uri)


@pytest.mark.usefixtures("tracking_uri_mock")
def test_runs_artifact_repo_init():
    artifact_location = "s3://blah_bucket/"
    experiment_id = mlflow.create_experiment("expr_abc", artifact_location)
    with mlflow.start_run(experiment_id=experiment_id):
        run_id = mlflow.active_run().info.run_uuid
    runs_uri = "runs:/%s/path/to/model" % run_id
    runs_repo = RunsArtifactRepository(runs_uri)

    assert runs_repo.artifact_uri == runs_uri
    assert isinstance(runs_repo.repo, S3ArtifactRepository)
    expected_absolute_uri = "%s%s/artifacts/path/to/model" % (artifact_location, run_id)
    assert runs_repo.repo.artifact_uri == expected_absolute_uri

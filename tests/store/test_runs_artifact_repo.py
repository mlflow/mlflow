import mock
import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.runs_artifact_repo import RunsArtifactRepository


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


# We should add an integration test for this code path.
# This test at least makes sure there are no silly mistakes and the code run in testing
def test_runs_artifact_repo_gets_correct_internal_repo():
    runs_uri = 'runs:/1234abcdf1394asdfwer33/path/to/model'
    absolute_uri = 's3://blah_bucket/cool'

    with mock.patch('mlflow.store.artifact_repository_registry.get_artifact_repository') \
        as get_artifact_repo_mock, \
            mock.patch('mlflow.tracking.artifact_utils.get_artifact_uri') as get_artifact_uri_mock:
        get_artifact_uri_mock.return_value = absolute_uri

        runs_repo = RunsArtifactRepository(runs_uri)

        assert runs_repo.artifact_uri == runs_uri
        assert get_artifact_repo_mock.call_count == 1
        assert get_artifact_repo_mock.call_args_list[0][0][0] == absolute_uri

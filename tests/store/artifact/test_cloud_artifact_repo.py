import pytest
from unittest import mock
from mlflow.environment_variables import _MLFLOW_MPD_NUM_RETRIES, _MLFLOW_MPD_RETRY_INTERVAL_SECONDS
from concurrent.futures import Future
from mlflow.store.artifact.cloud_artifact_repo import CloudArtifactRepository

from mlflow.exceptions import MlflowException
from mlflow.store.artifact.cloud_artifact_repo import _readable_size, _validate_chunk_size_aws
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo, ArtifactCredentialType



@pytest.mark.parametrize(
    ("size", "size_str"), [(5 * 1024**2, "5.00 MB"), (712.345 * 1024**2, "712.35 MB")]
)

def test_readable_size(size, size_str):
    assert _readable_size(size) == size_str


def test_chunk_size_validation_failure():
    with pytest.raises(MlflowException, match="Multipart chunk size"):
        _validate_chunk_size_aws(5 * 1024**2 - 1)
    with pytest.raises(MlflowException, match="Multipart chunk size"):
        _validate_chunk_size_aws(5 * 1024**3 + 1)

@pytest.mark.parametrize("future_result, expected_call_count", [
    (None, 2),  # Simulate where creds are expired, but successfully download after refresh
    (Exception("fake_exception"), _MLFLOW_MPD_NUM_RETRIES.get() + 1)  # Simulate where there is a download failure and retries are exhausted
])
def test__parallelized_download_from_cloud(monkeypatch, future_result, expected_call_count):
    print("test__parallelized_download_from_cloud")

    # Mock environment variables
    monkeypatch.setattr(_MLFLOW_MPD_NUM_RETRIES, 'get', lambda: 3)
    monkeypatch.setattr(_MLFLOW_MPD_RETRY_INTERVAL_SECONDS, 'get', lambda: 5)

    with mock.patch("mlflow.store.artifact.cloud_artifact_repo.CloudArtifactRepository") as cloud_artifact_mock:
        cloud_artifact_instance = cloud_artifact_mock.return_value

        # Mock all methods except 'method_to_test'
        cloud_artifact_instance._parallelized_download_from_cloud.side_effect = (
            lambda *args, **kwargs: CloudArtifactRepository._parallelized_download_from_cloud(
                cloud_artifact_instance, *args, **kwargs
            )
        )

        # Mock the chunk object
        class FakeChunk:
            def __init__(self, index, start, end, path):
                self.index = index
                self.start = start
                self.end = end
                self.path = path

        fake_chunk_1 = FakeChunk(index=1, start=0, end=100, path="fake_path_1")
        mock_failed_downloads = {fake_chunk_1: "fake_chunk_1"}

        with mock.patch("mlflow.store.artifact.cloud_artifact_repo.parallelized_download_file_using_http_uri") as mock_download:
            mock_download.return_value = mock_failed_downloads

            # Create a new ArtifactCredentialInfo object
            fake_credential = ArtifactCredentialInfo()

            # Set the values for the ArtifactCredentialInfo object
            fake_credential.signed_uri = "fake_signed_uri"
            fake_credential.type = ArtifactCredentialType.AWS_PRESIGNED_URL
            fake_credential.headers.extend([ArtifactCredentialInfo.HttpHeader(name="fake_header_name", value="fake_header_value")])

            # Set the return value of _get_read_credential_infos to the fake_credential object
            cloud_artifact_instance._get_read_credential_infos.return_value = [fake_credential]

            # Wrap fake_chunk_1 in a Future
            future = Future()
            if future_result is None:
                future.set_result(fake_chunk_1)
            else:
                future.set_exception(future_result)

            futures = {future: fake_chunk_1}

            with mock.patch("mlflow.store.artifact.cloud_artifact_repo.as_completed", return_value=futures):
                cloud_artifact_instance.chunk_thread_pool.submit.return_value = future

                # Call the method to test
                cloud_artifact_instance._parallelized_download_from_cloud(1, "fake_remote_path", "fake_local_path")

                # Assert that _get_read_credential_infos is called the expected number of times
                assert cloud_artifact_instance._get_read_credential_infos.call_count == expected_call_count
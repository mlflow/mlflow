from concurrent.futures import Future
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo, ArtifactCredentialType
from mlflow.store.artifact.cloud_artifact_repo import (
    CloudArtifactRepository,
    _readable_size,
    _validate_chunk_size_aws,
)


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


@pytest.mark.parametrize(
    ("future_result", "expected_call_count"),
    [
        (None, 2),  # Simulate where creds are expired, but successfully download after refresh
        (Exception("fake_exception"), 4),
        # Simulate where there is a download failure and retries are exhausted
    ],
)
def test__parallelized_download_from_cloud(
    monkeypatch, future_result, expected_call_count, tmp_path
):
    # Mock environment variables
    monkeypatch.setenv("_MLFLOW_MPD_NUM_RETRIES", "3")
    monkeypatch.setenv("_MLFLOW_MPD_RETRY_INTERVAL_SECONDS", "0")

    with mock.patch(
        "mlflow.store.artifact.cloud_artifact_repo.CloudArtifactRepository"
    ) as cloud_artifact_mock:
        cloud_artifact_instance = cloud_artifact_mock.return_value

        # Mock all methods except '_parallelized_download_from_cloud'
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

        # Wrap fake_chunk_1 in a Future
        future = Future()
        if future_result is None:
            future.set_result(fake_chunk_1)
        else:
            future.set_exception(future_result)

        futures = {future: fake_chunk_1}

        # Create a new ArtifactCredentialInfo object
        fake_credential = ArtifactCredentialInfo(
            signed_uri="fake_signed_uri",
            type=ArtifactCredentialType.AWS_PRESIGNED_URL,
        )
        fake_credential.headers.extend(
            [ArtifactCredentialInfo.HttpHeader(name="fake_header_name", value="fake_header_value")]
        )

        # Set the return value of _get_read_credential_infos to the fake_credential object
        cloud_artifact_instance._get_read_credential_infos.return_value = [fake_credential]

        # Set return value for mocks
        cloud_artifact_instance._get_read_credential_infos.return_value = [fake_credential]
        cloud_artifact_instance._get_uri_for_path.return_value = "fake_uri_path"

        cloud_artifact_instance.chunk_thread_pool.submit.return_value = future

        # Create a fake local path using tmp_path
        fake_local_path = tmp_path / "downloaded_file"

        with mock.patch(
            "mlflow.store.artifact.cloud_artifact_repo.parallelized_download_file_using_http_uri",
            return_value=mock_failed_downloads,
        ), mock.patch(
            "mlflow.store.artifact.cloud_artifact_repo.as_completed",
            return_value=futures,
        ):
            if future_result:
                with pytest.raises(
                    MlflowException, match="All retries have been exhausted. Download has failed."
                ):
                    cloud_artifact_instance._parallelized_download_from_cloud(
                        1, "fake_remote_path", str(fake_local_path)
                    )
            else:
                cloud_artifact_instance._parallelized_download_from_cloud(
                    1, "fake_remote_path", str(fake_local_path)
                )

            assert (
                cloud_artifact_instance._get_read_credential_infos.call_count == expected_call_count
            )
            assert (
                cloud_artifact_instance._get_read_credential_infos.call_count == expected_call_count
            )

            for call in cloud_artifact_instance.chunk_thread_pool.submit.call_args_list:
                assert call == mock.call(
                    mock.ANY,
                    range_start=fake_chunk_1.start,
                    range_end=fake_chunk_1.end,
                    headers=mock.ANY,
                    download_path=str(fake_local_path),
                    http_uri="fake_signed_uri",
                )

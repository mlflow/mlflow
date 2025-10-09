from concurrent.futures import Future
from contextlib import contextmanager
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo, ArtifactCredentialType
from mlflow.store.artifact.cloud_artifact_repo import (
    CloudArtifactRepository,
    _readable_size,
    _validate_chunk_size_aws,
)


@contextmanager
def large_file(path, size_mb):
    file_path = path
    file_path.write_bytes(b"x" * (size_mb * 1024 * 1024))
    try:
        yield file_path
    finally:
        if file_path.exists():
            file_path.unlink()


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

        with (
            mock.patch(
                "mlflow.store.artifact.cloud_artifact_repo.parallelized_download_file_using_http_uri",
                return_value=mock_failed_downloads,
            ),
            mock.patch(
                "mlflow.store.artifact.cloud_artifact_repo.as_completed",
                return_value=futures,
            ),
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


def test_log_artifacts_partitions_by_file_size(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE", str(100 * 1024 * 1024))

    small_file = tmp_path / "small.txt"
    small_file.write_bytes(b"small" * 1000)

    with large_file(tmp_path / "large.bin", 101):

        class TestCloudRepo(CloudArtifactRepository):
            def __init__(self):
                self.artifact_uri = "test://bucket/path"
                self.thread_pool = mock.MagicMock()
                self.uploaded_files = []

            def _get_write_credential_infos(self, remote_file_paths):
                return [
                    ArtifactCredentialInfo(
                        signed_uri=f"https://example.com/{path}",
                        type=ArtifactCredentialType.AWS_PRESIGNED_URL,
                    )
                    for path in remote_file_paths
                ]

            def _upload_to_cloud(self, cloud_credential_info, src_file_path, artifact_file_path):
                self.uploaded_files.append(
                    {
                        "path": src_file_path,
                        "artifact_path": artifact_file_path,
                        "has_credentials": cloud_credential_info is not None,
                    }
                )

            def _get_read_credential_infos(self, remote_file_paths):
                return []

            def _download_from_cloud(self, remote_file_path, local_path):
                pass

        repo = TestCloudRepo()
        repo.thread_pool.submit.side_effect = lambda fn, **kwargs: mock.MagicMock(
            result=lambda: fn(**kwargs)
        )

        repo.log_artifacts(str(tmp_path))

        small_uploads = [u for u in repo.uploaded_files if "small.txt" in u["path"]]
        large_uploads = [u for u in repo.uploaded_files if "large.bin" in u["path"]]

        assert len(small_uploads) == 1
        assert small_uploads[0]["has_credentials"] is True

        assert len(large_uploads) == 1
        assert large_uploads[0]["has_credentials"] is False


def test_log_artifacts_mixed_sizes_requests_credentials_only_for_small_files(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE", str(50 * 1024 * 1024))

    files_dir = tmp_path / "files"
    files_dir.mkdir()

    (files_dir / "small1.txt").write_bytes(b"a" * 1000)
    (files_dir / "small2.txt").write_bytes(b"b" * 2000)

    with large_file(files_dir / "large1.bin", 60), large_file(files_dir / "large2.bin", 70):
        credential_requests = []

        class TestCloudRepo(CloudArtifactRepository):
            def __init__(self):
                self.artifact_uri = "test://bucket/path"
                self.thread_pool = mock.MagicMock()

            def _get_write_credential_infos(self, remote_file_paths):
                credential_requests.append(remote_file_paths)
                return [
                    ArtifactCredentialInfo(
                        signed_uri=f"https://example.com/{path}",
                        type=ArtifactCredentialType.AWS_PRESIGNED_URL,
                    )
                    for path in remote_file_paths
                ]

            def _upload_to_cloud(self, cloud_credential_info, src_file_path, artifact_file_path):
                pass

            def _get_read_credential_infos(self, remote_file_paths):
                return []

            def _download_from_cloud(self, remote_file_path, local_path):
                pass

        repo = TestCloudRepo()
        repo.thread_pool.submit.side_effect = lambda fn, **kwargs: mock.MagicMock(
            result=lambda: fn(**kwargs)
        )

        repo.log_artifacts(str(files_dir))

        assert len(credential_requests) == 1
        requested_files = credential_requests[0]
        assert all("small" in f for f in requested_files)
        assert not any("large" in f for f in requested_files)
        assert len(requested_files) == 2


def test_databricks_upload_to_cloud_handles_none_credential_for_large_files(tmp_path):
    from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository

    with large_file(tmp_path / "large.bin", 600) as large_file_path:
        with mock.patch.object(DatabricksArtifactRepository, "_multipart_upload") as mock_multipart:
            with mock.patch.object(
                DatabricksArtifactRepository, "__init__", lambda self, *args, **kwargs: None
            ):
                repo = DatabricksArtifactRepository()
                repo.resource = mock.MagicMock()
                repo.resource.relative_path = "path"

                mock_multipart.side_effect = lambda *args, **kwargs: setattr(
                    mock, "multipart_upload_called", True
                )

                repo._upload_to_cloud(
                    cloud_credential_info=None,
                    src_file_path=str(large_file_path),
                    artifact_file_path="artifacts/large.bin",
                )

                assert mock_multipart.called

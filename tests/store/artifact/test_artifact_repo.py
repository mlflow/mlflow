import os
import posixpath
import pytest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from mlflow.entities import FileInfo
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import TempDir


class ArtifactRepositoryImpl(ArtifactRepository):
    def log_artifact(self, local_file, artifact_path=None):
        raise NotImplementedError()

    def log_artifacts(self, local_dir, artifact_path=None):
        raise NotImplementedError()

    def list_artifacts(self, path):
        raise NotImplementedError()

    def _download_file(self, remote_file_path, local_path):
        print("download_file called with '%s'" % remote_file_path)
        assert remote_file_path.endswith("modelfile")


@pytest.mark.parametrize(
    "base_uri, download_arg, list_return_val",
    [
        ("12345/model", "", ["modelfile"]),
        ("12345/model", "", [".", "modelfile"]),
        ("12345", "model", ["model/modelfile"]),
        ("12345", "model", ["model", "model/modelfile"]),
        ("", "12345/model", ["12345/model/modelfile"]),
        ("", "12345/model", ["12345/model", "12345/model/modelfile"]),
    ],
)
def test_download_artifacts_does_not_infinitely_loop(base_uri, download_arg, list_return_val):
    def list_artifacts(path):
        fullpath = posixpath.join(base_uri, path)
        if fullpath.endswith("model") or fullpath.endswith("model/"):
            return [FileInfo(item, False, 123) for item in list_return_val]
        elif fullpath.endswith("12345") or fullpath.endswith("12345/"):
            return [FileInfo(posixpath.join(path, "model"), True, 0)]
        else:
            return []

    with mock.patch.object(ArtifactRepositoryImpl, "list_artifacts") as list_artifacts_mock:
        list_artifacts_mock.side_effect = list_artifacts
        repo = ArtifactRepositoryImpl(base_uri)
        repo.download_artifacts(download_arg)


@pytest.mark.parametrize(
    "base_uri, download_arg, list_return_val",
    [("", "12345/model", ["12345/model", "12345/model/modelfile", "12345/model/emptydir"])],
)
def test_download_artifacts_handles_empty_dir(base_uri, download_arg, list_return_val):
    def list_artifacts(path):
        if path.endswith("model"):
            return [FileInfo(item, item.endswith("emptydir"), 123) for item in list_return_val]
        elif path.endswith("12345") or path.endswith("12345/"):
            return [FileInfo("12345/model", True, 0)]
        else:
            return []

    with mock.patch.object(ArtifactRepositoryImpl, "list_artifacts") as list_artifacts_mock:
        list_artifacts_mock.side_effect = list_artifacts
        repo = ArtifactRepositoryImpl(base_uri)
        with TempDir() as tmp:
            repo.download_artifacts(download_arg, dst_path=tmp.path())


def test_download_artifacts_supports_asynchronous_file_downloads(tmpdir):
    """
    Verifies that `ArtifactRepository.download_artifacts()` supports `_download_file`
    implementations that return a `concurrent.futures.Future` representing an asynchronous
    file download
    """
    artifact_paths = [
        "f1.txt",
        "f2.json",
        "subdir1/subdir2/f3.txt",
        "subdir1/f4.txt",
        "subdir3/f5.txt",
    ]

    def list_artifacts(path):
        if path in ["f1.txt", "f2.json"]:
            return [FileInfo(path, False, 123)]
        elif path == "":
            return [
                FileInfo("f1.txt", False, 123),
                FileInfo("f2.json", False, 123),
                FileInfo("subdir1", True, 0),
                FileInfo("subdir3", True, 0),
            ]
        elif path.startswith("subdir1/subdir2"):
            return [
                FileInfo("subdir1/subdir2/f3.txt", False, 123),
            ]
        elif path.startswith("subdir1"):
            return [
                FileInfo("subdir1/f4.txt", False, 123),
                FileInfo("subdir1/subdir2", True, 0),
            ]
        elif path.startswith("subdir3"):
            return [
                FileInfo("subdir3/f5.txt", False, 123),
            ]
        else:
            return []

    thread_pool = ThreadPoolExecutor()

    def _download_file(remote_file_path, local_path):
        def perform_download():
            import time

            assert remote_file_path in artifact_paths, "path does not exist"
            # Sleep for a second to simulate a remote file download, ensuring that the artifact
            # repository implementation must wait for future completion prior to checking for the
            # existence of the downloaded file in order to pass the test
            time.sleep(1)

            with open(local_path, "w") as f:
                f.write("test file")

        return thread_pool.submit(perform_download)

    with mock.patch.object(
        ArtifactRepositoryImpl, "list_artifacts"
    ) as list_artifacts_mock, mock.patch.object(
        ArtifactRepositoryImpl, "_download_file"
    ) as download_file_mock:
        list_artifacts_mock.side_effect = list_artifacts
        download_file_mock.side_effect = _download_file
        repo = ArtifactRepositoryImpl("")

        destination1 = os.path.join(str(tmpdir), "dest1")
        os.makedirs(destination1)

        repo.download_artifacts("f1.txt", destination1)
        assert os.path.exists(os.path.join(destination1, "f1.txt"))

        destination2 = os.path.join(str(tmpdir), "dest2")
        os.makedirs(destination2)
        repo.download_artifacts("", destination2)

        for path in artifact_paths:
            assert os.path.exists(os.path.join(destination2, path))

        with pytest.raises(AssertionError, match="path does not exist"):
            repo.download_artifacts("nonexistent", destination1)

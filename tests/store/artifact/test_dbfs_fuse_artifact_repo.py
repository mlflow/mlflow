import os

import pytest
from unittest import mock
from unittest.mock import PropertyMock

from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

TEST_FILE_1_CONTENT = "Hello 🍆🍔".encode("utf-8")
TEST_FILE_2_CONTENT = "World 🍆🍔🍆".encode("utf-8")
TEST_FILE_3_CONTENT = "¡🍆🍆🍔🍆🍆!".encode("utf-8")


@pytest.fixture()
def artifact_dir(tmpdir):
    return tmpdir.join("artifacts-to-log").strpath


@pytest.fixture()
def force_dbfs_fuse_repo(artifact_dir):
    in_databricks_mock_path = "mlflow.utils.databricks_utils.is_dbfs_fuse_available"
    local_artifact_repo_package = "mlflow.store.artifact.local_artifact_repo"
    artifact_dir_mock_path = local_artifact_repo_package + ".LocalArtifactRepository.artifact_dir"
    with mock.patch(in_databricks_mock_path) as is_dbfs_fuse_available, mock.patch(
        artifact_dir_mock_path, new_callable=PropertyMock
    ) as artifact_dir_mock:
        is_dbfs_fuse_available.return_value = True
        artifact_dir_mock.return_value = artifact_dir
        yield


@pytest.fixture()
def dbfs_fuse_artifact_repo(force_dbfs_fuse_repo):  # pylint: disable=unused-argument
    return get_artifact_repository("dbfs:/unused/path/replaced/by/mock")


@pytest.fixture()
def files_dir(tmpdir):
    return tmpdir.mkdir("files")


@pytest.fixture()
def test_file(files_dir):
    p = files_dir.join("test.txt")
    with open(p.strpath, "wb") as f:
        f.write(TEST_FILE_1_CONTENT)
    return p


@pytest.fixture()
def test_dir(files_dir):
    with open(files_dir.mkdir("subdir").join("test.txt").strpath, "wb") as f:
        f.write(TEST_FILE_2_CONTENT)
    with open(files_dir.join("test.txt").strpath, "wb") as f:
        f.write(TEST_FILE_3_CONTENT)
    with open(files_dir.join("empty-file").strpath, "wb"):
        pass
    return files_dir


class TestDbfsFuseArtifactRepository:
    @pytest.mark.parametrize("artifact_path", [None, "output", ""])
    def test_log_artifact(self, dbfs_fuse_artifact_repo, test_file, artifact_path, artifact_dir):
        dbfs_fuse_artifact_repo.log_artifact(test_file.strpath, artifact_path)
        print(os.listdir(artifact_dir))
        expected_file_path = os.path.join(
            artifact_dir,
            artifact_path if artifact_path else "",
            os.path.basename(test_file.strpath),
        )
        with open(expected_file_path, "rb") as handle:
            data = handle.read()
        assert data == TEST_FILE_1_CONTENT

    def test_log_artifact_empty_file(self, dbfs_fuse_artifact_repo, test_dir, artifact_dir):
        dbfs_fuse_artifact_repo.log_artifact(os.path.join(test_dir.strpath, "empty-file"))
        expected_file_path = os.path.join(artifact_dir, "empty-file")
        with open(expected_file_path, "rb") as handle:
            data = handle.read()
        assert data == "".encode("utf-8")

    @pytest.mark.parametrize(
        "artifact_path",
        [
            None,
            "",  # should behave like '/' and exclude base name of logged_dir
            "abc",
            # We should add '.',
        ],
    )
    def test_log_artifacts(self, dbfs_fuse_artifact_repo, test_dir, artifact_path, artifact_dir):
        dbfs_fuse_artifact_repo.log_artifacts(test_dir.strpath, artifact_path)
        artifact_dst_path = os.path.join(artifact_dir, artifact_path if artifact_path else "")
        assert os.path.exists(artifact_dst_path)
        expected_contents = {
            "subdir/test.txt": TEST_FILE_2_CONTENT,
            "test.txt": TEST_FILE_3_CONTENT,
            "empty-file": "".encode("utf-8"),
        }
        for filename, contents in expected_contents.items():
            with open(os.path.join(artifact_dst_path, filename), "rb") as handle:
                assert handle.read() == contents

    def test_list_artifacts(self, dbfs_fuse_artifact_repo, test_dir):
        assert len(dbfs_fuse_artifact_repo.list_artifacts()) == 0
        dbfs_fuse_artifact_repo.log_artifacts(test_dir.strpath)
        artifacts = dbfs_fuse_artifact_repo.list_artifacts()
        assert len(artifacts) == 3
        assert artifacts[0].path == "empty-file"
        assert artifacts[0].is_dir is False
        assert artifacts[0].file_size == 0
        assert artifacts[1].path == "subdir"
        assert artifacts[1].is_dir is True
        assert artifacts[1].file_size is None
        assert artifacts[2].path == "test.txt"
        assert artifacts[2].is_dir is False
        assert artifacts[2].file_size == 23

    def test_download_artifacts(self, dbfs_fuse_artifact_repo, test_dir):
        dbfs_fuse_artifact_repo.log_artifacts(test_dir.strpath)
        local_download_dir = dbfs_fuse_artifact_repo.download_artifacts("")
        expected_contents = {
            "subdir/test.txt": TEST_FILE_2_CONTENT,
            "test.txt": TEST_FILE_3_CONTENT,
            "empty-file": "".encode("utf-8"),
        }
        for filename, contents in expected_contents.items():
            with open(os.path.join(local_download_dir, filename), "rb") as handle:
                assert handle.read() == contents

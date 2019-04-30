import os
import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.local_artifact_repo import LocalArtifactRepository
from mlflow.utils.file_utils import TempDir


@pytest.fixture
def local_artifact_root(tmpdir):
    return str(tmpdir)


@pytest.fixture
def local_artifact_repo(local_artifact_root):
    return LocalArtifactRepository(artifact_uri=local_artifact_root)


def test_list_artifacts(local_artifact_repo, local_artifact_root):
    assert len(local_artifact_repo.list_artifacts()) == 0

    artifact_rel_path = "artifact"
    artifact_path = os.path.join(local_artifact_root, artifact_rel_path)
    with open(artifact_path, "w") as f:
        f.write("artifact")
    artifacts_list = local_artifact_repo.list_artifacts()
    assert len(artifacts_list) == 1
    assert artifacts_list[0].path == artifact_rel_path


def test_log_artifacts(local_artifact_repo, local_artifact_root):
    artifact_rel_path = "test.txt"
    artifact_text = "hello world!"
    with TempDir() as src_dir:
        artifact_src_path = src_dir.path(artifact_rel_path)
        with open(artifact_src_path, "w") as f:
            f.write(artifact_text)
        local_artifact_repo.log_artifact(artifact_src_path)

    artifacts_list = local_artifact_repo.list_artifacts()
    assert len(artifacts_list) == 1
    assert artifacts_list[0].path == artifact_rel_path

    artifact_dst_path = os.path.join(local_artifact_root, artifact_rel_path)
    assert os.path.exists(artifact_dst_path)
    assert artifact_dst_path != artifact_src_path
    assert open(artifact_dst_path).read() == artifact_text


def test_download_artifacts(local_artifact_repo):
    artifact_rel_path = "test.txt"
    artifact_text = "hello world!"
    with TempDir(chdr=True) as local_dir:
        artifact_src_path = local_dir.path(artifact_rel_path)
        with open(artifact_src_path, "w") as f:
            f.write(artifact_text)
        local_artifact_repo.log_artifact(artifact_src_path)
        dst_path = local_artifact_repo.download_artifacts(artifact_path=artifact_rel_path)
        assert open(dst_path).read() == artifact_text


def test_download_artifacts_returns_absolute_paths(local_artifact_repo):
    artifact_rel_path = "test.txt"
    artifact_text = "hello world!"
    with TempDir(chdr=True) as local_dir:
        artifact_src_path = local_dir.path(artifact_rel_path)
        with open(artifact_src_path, "w") as f:
            f.write(artifact_text)
        local_artifact_repo.log_artifact(artifact_src_path)

        for dst_dir in ["dst1", local_dir.path("dst2"), None]:
            if dst_dir is not None:
                os.makedirs(dst_dir)
            dst_path = local_artifact_repo.download_artifacts(
                artifact_path=artifact_rel_path,
                dst_path=dst_dir)
            assert dst_path == os.path.abspath(dst_path)


@pytest.mark.parametrize("repo_subdir_path", [
    "aaa",
    "aaa/bbb",
    "aaa/bbb/ccc/ddd",
])
def test_artifacts_are_logged_to_and_downloaded_from_repo_subdirectory_successfully(
        local_artifact_repo, repo_subdir_path):
    artifact_rel_path = "test.txt"
    artifact_text = "hello world!"
    with TempDir(chdr=True) as local_dir:
        artifact_src_path = local_dir.path(artifact_rel_path)
        with open(artifact_src_path, "w") as f:
            f.write(artifact_text)
        local_artifact_repo.log_artifact(artifact_src_path, artifact_path=repo_subdir_path)

    downloaded_subdir = local_artifact_repo.download_artifacts(repo_subdir_path)
    assert os.path.isdir(downloaded_subdir)
    subdir_contents = os.listdir(downloaded_subdir)
    assert len(subdir_contents) == 1
    assert artifact_rel_path in subdir_contents
    assert open(os.path.join(downloaded_subdir, artifact_rel_path)).read() == artifact_text

    downloaded_file = local_artifact_repo.download_artifacts(
        os.path.join(repo_subdir_path, artifact_rel_path))
    assert open(downloaded_file).read() == artifact_text


def test_log_artifact_throws_exception_for_invalid_artifact_paths(local_artifact_repo):
    with TempDir() as local_dir:
        for bad_artifact_path in ["/", "//", "/tmp", "/bad_path", ".", "../terrible_path"]:
            with pytest.raises(MlflowException) as exc_info:
                local_artifact_repo.log_artifact(local_dir.path(), bad_artifact_path)
            assert "Invalid artifact path" in str(exc_info)


def test_logging_directory_of_artifacts_produces_expected_repo_contents(local_artifact_repo):
    with TempDir() as local_dir:
        os.mkdir(local_dir.path("subdir"))
        os.mkdir(local_dir.path("subdir", "nested"))
        with open(local_dir.path("subdir", "a.txt"), "w") as f:
            f.write("A")
        with open(local_dir.path("subdir", "b.txt"), "w") as f:
            f.write("B")
        with open(local_dir.path("subdir", "nested", "c.txt"), "w") as f:
            f.write("C")
        local_artifact_repo.log_artifacts(local_dir.path("subdir"))
        assert open(local_artifact_repo.download_artifacts("a.txt")).read() == "A"
        assert open(local_artifact_repo.download_artifacts("b.txt")).read() == "B"
        assert open(local_artifact_repo.download_artifacts("nested/c.txt")).read() == "C"


def test_hidden_files_are_logged_correctly(local_artifact_repo):
    with TempDir() as local_dir:
        hidden_file = local_dir.path(".mystery")
        with open(hidden_file, "w") as f:
            f.write("42")
        local_artifact_repo.log_artifact(hidden_file)
        assert open(local_artifact_repo.download_artifacts(hidden_file)).read() == "42"

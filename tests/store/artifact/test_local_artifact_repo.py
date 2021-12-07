import os
import pytest
import posixpath

from mlflow.exceptions import MlflowException
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.utils.file_utils import TempDir


@pytest.fixture
def local_artifact_root(tmpdir):
    return str(tmpdir)


@pytest.fixture
def local_artifact_repo(local_artifact_root):
    from mlflow.utils.file_utils import path_to_local_file_uri

    return LocalArtifactRepository(artifact_uri=path_to_local_file_uri(local_artifact_root))


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


@pytest.mark.parametrize("dst_path", [None, "dest"])
def test_download_artifacts(local_artifact_repo, dst_path):
    artifact_rel_path = "test.txt"
    artifact_text = "hello world!"
    empty_dir_path = "empty_dir"
    with TempDir(chdr=True) as local_dir:
        if dst_path:
            os.mkdir(dst_path)
        artifact_src_path = local_dir.path(artifact_rel_path)
        os.mkdir(local_dir.path(empty_dir_path))
        with open(artifact_src_path, "w") as f:
            f.write(artifact_text)
        local_artifact_repo.log_artifacts(local_dir.path())
        result = local_artifact_repo.download_artifacts(
            artifact_path=artifact_rel_path, dst_path=dst_path
        )
        assert open(result).read() == artifact_text
        result = local_artifact_repo.download_artifacts(artifact_path="", dst_path=dst_path)
        empty_dir_dst_path = os.path.join(result, empty_dir_path)
        assert os.path.isdir(empty_dir_dst_path)
        assert len(os.listdir(empty_dir_dst_path)) == 0


def test_download_artifacts_does_not_copy(local_artifact_repo):
    """
    The LocalArtifactRepository.download_artifact function should not copy the artifact if
    the ``dst_path`` argument is None.
    """
    artifact_rel_path = "test.txt"
    artifact_text = "hello world!"
    with TempDir(chdr=True) as local_dir:
        artifact_src_path = local_dir.path(artifact_rel_path)
        with open(artifact_src_path, "w") as f:
            f.write(artifact_text)
        local_artifact_repo.log_artifact(artifact_src_path)
        dst_path = local_artifact_repo.download_artifacts(artifact_path=artifact_rel_path)
        assert open(dst_path).read() == artifact_text
        assert dst_path.startswith(
            local_artifact_repo.artifact_dir
        ), "downloaded artifact is not in local_artifact_repo.artifact_dir root"


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
                artifact_path=artifact_rel_path, dst_path=dst_dir
            )
            if dst_dir is not None:
                # If dst_dir isn't none, assert we're actually downloading to dst_dir.
                assert dst_path.startswith(os.path.abspath(dst_dir))
            assert dst_path == os.path.abspath(dst_path)


@pytest.mark.parametrize("repo_subdir_path", ["aaa", "aaa/bbb", "aaa/bbb/ccc/ddd"])
def test_artifacts_are_logged_to_and_downloaded_from_repo_subdirectory_successfully(
    local_artifact_repo, repo_subdir_path
):
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
        posixpath.join(repo_subdir_path, artifact_rel_path)
    )
    assert open(downloaded_file).read() == artifact_text


def test_log_artifact_throws_exception_for_invalid_artifact_paths(local_artifact_repo):
    with TempDir() as local_dir:
        for bad_artifact_path in ["/", "//", "/tmp", "/bad_path", ".", "../terrible_path"]:
            with pytest.raises(MlflowException, match="Invalid artifact path"):
                local_artifact_repo.log_artifact(local_dir.path(), bad_artifact_path)


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
        assert open(local_artifact_repo.download_artifacts(".mystery")).read() == "42"


def test_delete_artifacts(local_artifact_repo):
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
        assert os.path.exists(os.path.join(local_artifact_repo._artifact_dir, "nested"))
        assert os.path.exists(os.path.join(local_artifact_repo._artifact_dir, "a.txt"))
        assert os.path.exists(os.path.join(local_artifact_repo._artifact_dir, "b.txt"))
        local_artifact_repo.delete_artifacts()
        assert not os.path.exists(os.path.join(local_artifact_repo._artifact_dir))

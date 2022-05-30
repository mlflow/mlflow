import pytest
from tempfile import NamedTemporaryFile
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.sftp_artifact_repo import SFTPArtifactRepository
from mlflow.utils.file_utils import TempDir
import os
import mlflow
import posixpath


pytestmark = pytest.mark.requires_ssh


def test_artifact_uri_factory(tmp_path):
    assert isinstance(get_artifact_repository(f"sftp://{tmp_path}"), SFTPArtifactRepository)


def test_list_artifacts_empty(tmp_path):
    repo = SFTPArtifactRepository(f"sftp://{tmp_path}")
    assert repo.list_artifacts() == []


@pytest.mark.parametrize("artifact_path", [None, "sub_dir", "very/nested/sub/dir"])
def test_list_artifacts(tmp_path, artifact_path):
    file_path = "file"
    dir_path = "model"
    tmp_path.joinpath(artifact_path or "").mkdir(parents=True, exist_ok=True)
    tmp_path.joinpath(artifact_path or "", file_path).write_text("test")
    tmp_path.joinpath(artifact_path or "", dir_path).mkdir()

    repo = SFTPArtifactRepository(f"sftp://{tmp_path}")
    artifacts = repo.list_artifacts(path=artifact_path)
    assert len(artifacts) == 2
    assert artifacts[0].path == posixpath.join(artifact_path or "", file_path)
    assert artifacts[0].is_dir is False
    assert artifacts[0].file_size == 4
    assert artifacts[1].path == posixpath.join(artifact_path or "", dir_path)
    assert artifacts[1].is_dir is True
    assert artifacts[1].file_size is None


@pytest.mark.parametrize("artifact_path", [None, "sub_dir", "very/nested/sub/dir"])
def test_log_artifact(artifact_path):
    file_content = "A simple test artifact\nThe artifact is located in: " + str(artifact_path)
    with NamedTemporaryFile(mode="w") as local, TempDir() as remote:
        local.write(file_content)
        local.flush()

        sftp_path = "sftp://" + remote.path()
        store = SFTPArtifactRepository(sftp_path)
        store.log_artifact(local.name, artifact_path)

        remote_file = posixpath.join(
            remote.path(),
            "." if artifact_path is None else artifact_path,
            os.path.basename(local.name),
        )
        assert posixpath.isfile(remote_file)

        with open(remote_file, "r") as remote_content:
            assert remote_content.read() == file_content


@pytest.mark.parametrize("artifact_path", [None, "sub_dir", "very/nested/sub/dir"])
def test_log_artifacts(artifact_path):
    file_content_1 = "A simple test artifact\nThe artifact is located in: " + str(artifact_path)
    file_content_2 = os.urandom(300)

    file1 = "meta.yaml"
    directory = "saved_model"
    file2 = "sk_model.pickle"
    with TempDir() as local, TempDir() as remote:
        with open(os.path.join(local.path(), file1), "w") as f:
            f.write(file_content_1)
        os.mkdir(os.path.join(local.path(), directory))
        with open(os.path.join(local.path(), directory, file2), "wb") as f:
            f.write(file_content_2)

        sftp_path = "sftp://" + remote.path()
        store = SFTPArtifactRepository(sftp_path)
        store.log_artifacts(local.path(), artifact_path)

        remote_dir = posixpath.join(remote.path(), "." if artifact_path is None else artifact_path)
        assert posixpath.isdir(remote_dir)
        assert posixpath.isdir(posixpath.join(remote_dir, directory))
        assert posixpath.isfile(posixpath.join(remote_dir, file1))
        assert posixpath.isfile(posixpath.join(remote_dir, directory, file2))

        with open(posixpath.join(remote_dir, file1), "r") as remote_content:
            assert remote_content.read() == file_content_1

        with open(posixpath.join(remote_dir, directory, file2), "rb") as remote_content:
            assert remote_content.read() == file_content_2


@pytest.mark.parametrize("artifact_path", [None, "sub_dir", "very/nested/sub/dir"])
def test_delete_artifact(artifact_path):
    file_content = f"A simple test artifact\nThe artifact is located in: {artifact_path}"
    with NamedTemporaryFile(mode="w") as local, TempDir() as remote:
        local.write(file_content)
        local.flush()

        sftp_path = f"sftp://{remote.path()}"
        store = SFTPArtifactRepository(sftp_path)
        store.log_artifact(local.name, artifact_path)

        remote_file = posixpath.join(
            remote.path(),
            "." if artifact_path is None else artifact_path,
            os.path.basename(local.name),
        )
        assert posixpath.isfile(remote_file)

        with open(remote_file, "r") as remote_content:
            assert remote_content.read() == file_content

        store.delete_artifacts(remote.path())

        assert not posixpath.exists(remote_file)
        assert not posixpath.exists(remote.path())


@pytest.mark.parametrize("artifact_path", [None, "sub_dir", "very/nested/sub/dir"])
def test_delete_artifacts(artifact_path):
    file_content_1 = f"A simple test artifact\nThe artifact is located in: {artifact_path}"
    file_content_2 = os.urandom(300)

    file1 = "meta.yaml"
    directory = "saved_model"
    file2 = "sk_model.pickle"
    with TempDir() as local, TempDir() as remote:
        with open(os.path.join(local.path(), file1), "w", encoding="utf8") as f:
            f.write(file_content_1)
        os.mkdir(os.path.join(local.path(), directory))
        with open(os.path.join(local.path(), directory, file2), "wb") as f:
            f.write(file_content_2)

        sftp_path = f"sftp://{remote.path()}"
        store = SFTPArtifactRepository(sftp_path)
        store.log_artifacts(local.path(), artifact_path)

        remote_dir = posixpath.join(remote.path(), "." if artifact_path is None else artifact_path)
        assert posixpath.isdir(remote_dir)
        assert posixpath.isdir(posixpath.join(remote_dir, directory))
        assert posixpath.isfile(posixpath.join(remote_dir, file1))
        assert posixpath.isfile(posixpath.join(remote_dir, directory, file2))

        with open(posixpath.join(remote_dir, file1), "r", encoding="utf8") as remote_content:
            assert remote_content.read() == file_content_1

        with open(posixpath.join(remote_dir, directory, file2), "rb") as remote_content:
            assert remote_content.read() == file_content_2

        store.delete_artifacts(remote.path())

        assert not posixpath.exists(posixpath.join(remote_dir, directory))
        assert not posixpath.exists(posixpath.join(remote_dir, file1))
        assert not posixpath.exists(posixpath.join(remote_dir, directory, file2))
        assert not posixpath.exists(remote_dir)
        assert not posixpath.exists(remote.path())


@pytest.mark.parametrize("artifact_path", [None, "sub_dir", "very/nested/sub/dir"])
def test_delete_selective_artifacts(artifact_path):
    file_content_1 = f"A simple test artifact\nThe artifact is located in: {artifact_path}"
    file_content_2 = os.urandom(300)

    file1 = "meta.yaml"
    directory = "saved_model"
    file2 = "sk_model.pickle"
    with TempDir() as local, TempDir() as remote:
        with open(os.path.join(local.path(), file1), "w", encoding="utf8") as f:
            f.write(file_content_1)
        os.mkdir(os.path.join(local.path(), directory))
        with open(os.path.join(local.path(), directory, file2), "wb") as f:
            f.write(file_content_2)

        sftp_path = f"sftp://{remote.path()}"
        store = SFTPArtifactRepository(sftp_path)
        store.log_artifacts(local.path(), artifact_path)

        remote_dir = posixpath.join(remote.path(), "." if artifact_path is None else artifact_path)
        assert posixpath.isdir(remote_dir)
        assert posixpath.isdir(posixpath.join(remote_dir, directory))
        assert posixpath.isfile(posixpath.join(remote_dir, file1))
        assert posixpath.isfile(posixpath.join(remote_dir, directory, file2))

        with open(posixpath.join(remote_dir, file1), "r", encoding="utf8") as remote_content:
            assert remote_content.read() == file_content_1

        with open(posixpath.join(remote_dir, directory, file2), "rb") as remote_content:
            assert remote_content.read() == file_content_2

        store.delete_artifacts(posixpath.join(remote_dir, file1))

        assert posixpath.isdir(posixpath.join(remote_dir, directory))
        assert not posixpath.exists(posixpath.join(remote_dir, file1))
        assert posixpath.isfile(posixpath.join(remote_dir, directory, file2))
        assert posixpath.isdir(remote_dir)


def test_log_and_download_sklearn_model(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    from numpy.testing import assert_allclose

    X, y = load_iris(return_X_y=True)
    original = LogisticRegression().fit(X, y)

    experiment_id = mlflow.create_experiment(
        name="sklearn-model-experiment",
        artifact_location=f"sftp://{tmp_path}",
    )
    with mlflow.start_run(experiment_id=experiment_id):
        model_uri = mlflow.sklearn.log_model(original, "model").model_uri
        downloaded = mlflow.sklearn.load_model(model_uri)

    assert_allclose(original.predict(X), downloaded.predict(X))

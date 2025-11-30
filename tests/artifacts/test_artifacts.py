import pathlib
import uuid
from typing import NamedTuple
from unittest import mock

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.utils.file_utils import mkdir, path_to_local_file_uri
from mlflow.utils.os import is_windows


class Artifact(NamedTuple):
    uri: str
    content: str


@pytest.fixture
def run_with_artifact(tmp_path):
    artifact_path = "test"
    artifact_content = "content"
    local_path = tmp_path.joinpath("file.txt")
    local_path.write_text(artifact_content)
    with mlflow.start_run() as run:
        mlflow.log_artifact(local_path, artifact_path)

    return (run, artifact_path, artifact_content)


@pytest.fixture
def run_with_artifacts(tmp_path):
    artifact_path = "test"
    artifact_content = "content"

    local_dir = tmp_path / "dir"
    local_dir.mkdir()
    local_dir.joinpath("file.txt").write_text(artifact_content)
    local_dir.joinpath("subdir").mkdir()
    local_dir.joinpath("subdir").joinpath("text.txt").write_text(artifact_content)

    with mlflow.start_run() as run:
        mlflow.log_artifact(local_dir, artifact_path)

    return (run, artifact_path)


def test_download_artifacts_with_uri(run_with_artifact):
    run, artifact_path, artifact_content = run_with_artifact
    run_uri = f"runs:/{run.info.run_id}/{artifact_path}"
    actual_uri = str(pathlib.PurePosixPath(run.info.artifact_uri) / artifact_path)
    for uri in (run_uri, actual_uri):
        download_output_path = mlflow.artifacts.download_artifacts(artifact_uri=uri)
        downloaded_artifact_path = next(pathlib.Path(download_output_path).iterdir())
        assert downloaded_artifact_path.read_text() == artifact_content


def test_download_artifacts_with_run_id_and_path(run_with_artifact):
    run, artifact_path, artifact_content = run_with_artifact
    download_output_path = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path=artifact_path
    )
    downloaded_artifact_path = next(pathlib.Path(download_output_path).iterdir())
    assert downloaded_artifact_path.read_text() == artifact_content


def test_download_artifacts_with_run_id_no_path(run_with_artifact):
    run, artifact_path, _ = run_with_artifact
    artifact_relative_path_top_level_dir = pathlib.PurePosixPath(artifact_path).parts[0]
    downloaded_output_path = mlflow.artifacts.download_artifacts(run_id=run.info.run_id)
    downloaded_artifact_directory_name = next(pathlib.Path(downloaded_output_path).iterdir()).name
    assert downloaded_artifact_directory_name == artifact_relative_path_top_level_dir


@pytest.mark.parametrize("dst_subdir_path", [None, "doesnt_exist_yet/subdiir"])
def test_download_artifacts_with_dst_path(run_with_artifact, tmp_path, dst_subdir_path):
    run, artifact_path, _ = run_with_artifact
    dst_path = tmp_path / dst_subdir_path if dst_subdir_path else tmp_path

    download_output_path = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path=artifact_path, dst_path=dst_path
    )
    assert pathlib.Path(download_output_path).samefile(dst_path / artifact_path)


def test_download_artifacts_throws_for_invalid_arguments():
    with pytest.raises(MlflowException, match="Exactly one of"):
        mlflow.artifacts.download_artifacts(
            run_id="run_id", artifact_path="path", artifact_uri="uri"
        )

    with pytest.raises(MlflowException, match="Exactly one of"):
        mlflow.artifacts.download_artifacts()

    with pytest.raises(MlflowException, match="`artifact_path` cannot be specified"):
        mlflow.artifacts.download_artifacts(artifact_path="path", artifact_uri="uri")


@pytest.fixture
def run_with_text_artifact():
    artifact_path = "test/file.txt"
    artifact_content = "This is a sentence"
    with mlflow.start_run() as run:
        mlflow.log_text(artifact_content, artifact_path)

    artifact_uri = str(pathlib.PurePosixPath(run.info.artifact_uri) / artifact_path)
    return Artifact(artifact_uri, artifact_content)


@pytest.fixture
def run_with_json_artifact():
    artifact_path = "test/config.json"
    artifact_content = {"mlflow-version": "0.28", "n_cores": "10"}
    with mlflow.start_run() as run:
        mlflow.log_dict(artifact_content, artifact_path)

    artifact_uri = str(pathlib.PurePosixPath(run.info.artifact_uri) / artifact_path)
    return Artifact(artifact_uri, artifact_content)


@pytest.fixture
def run_with_image_artifact():
    from PIL import Image

    artifact_path = "test/image.png"
    image = Image.new("RGB", (100, 100))
    with mlflow.start_run() as run:
        mlflow.log_image(image, artifact_path)

    artifact_uri = str(pathlib.PurePosixPath(run.info.artifact_uri) / artifact_path)
    return Artifact(artifact_uri, image)


def test_load_text(run_with_text_artifact):
    artifact = run_with_text_artifact
    assert mlflow.artifacts.load_text(artifact.uri) == artifact.content


def test_load_dict(run_with_json_artifact):
    artifact = run_with_json_artifact
    assert mlflow.artifacts.load_dict(artifact.uri) == artifact.content


def test_load_json_invalid_json(run_with_text_artifact):
    artifact = run_with_text_artifact
    with pytest.raises(mlflow.exceptions.MlflowException, match="Unable to form a JSON object"):
        mlflow.artifacts.load_dict(artifact.uri)


def test_load_image(run_with_image_artifact):
    from PIL import Image

    artifact = run_with_image_artifact
    assert isinstance(mlflow.artifacts.load_image(artifact.uri), Image.Image)


def test_load_image_invalid_image(run_with_text_artifact):
    artifact = run_with_text_artifact
    with pytest.raises(
        mlflow.exceptions.MlflowException, match="Unable to form a PIL Image object"
    ):
        mlflow.artifacts.load_image(artifact.uri)


class ArtifactReturnType(NamedTuple):
    tmp_path: pathlib.Path
    artifact_path: pathlib.Path
    artifact_name: str


@pytest.fixture
def text_artifact(tmp_path):
    artifact_name = "test.txt"
    artifacts_root_tmp = mkdir(tmp_path.joinpath(str(uuid.uuid4())))
    test_artifact_path = artifacts_root_tmp.joinpath(artifact_name)
    test_artifact_path.write_text("test")
    return ArtifactReturnType(artifacts_root_tmp, test_artifact_path, artifact_name)


def _assert_artifact_uri(expected_artifact_path: pathlib.Path, test_artifact):
    mlflow.log_artifact(test_artifact.artifact_path)
    assert expected_artifact_path.exists()


def test_default_relative_artifact_uri_resolves(text_artifact, tmp_path, monkeypatch):
    tracking_uri = path_to_local_file_uri(text_artifact.tmp_path.joinpath("mlruns"))
    mlflow.set_tracking_uri(tracking_uri)
    monkeypatch.chdir(tmp_path)
    experiment_id = mlflow.create_experiment("test_exp_a", "test_artifacts_root")
    with mlflow.start_run(experiment_id=experiment_id) as run:
        _assert_artifact_uri(
            tmp_path.joinpath(
                "test_artifacts_root",
                run.info.run_id,
                "artifacts",
                text_artifact.artifact_name,
            ),
            text_artifact,
        )


def test_custom_relative_artifact_uri_resolves(text_artifact):
    tracking_uri = path_to_local_file_uri(text_artifact.tmp_path.joinpath("tracking"))
    artifacts_root_path = text_artifact.tmp_path.joinpath("test_artifacts")
    artifacts_root_uri = path_to_local_file_uri(artifacts_root_path)
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment("test_exp_b", artifacts_root_uri)
    with mlflow.start_run(experiment_id=experiment_id) as run:
        _assert_artifact_uri(
            artifacts_root_path.joinpath(run.info.run_id, "artifacts", text_artifact.artifact_name),
            text_artifact,
        )


def test_artifact_logging_resolution_works_with_non_root_working_directory(tmp_path, monkeypatch):
    text_file = tmp_path.joinpath("test.txt")
    text_file.write_text("test")
    cwd = tmp_path.joinpath("cwd")
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    experiment_id = mlflow.create_experiment("test_exp_c", "some_path")
    not_cwd = tmp_path.joinpath("not_cwd")
    not_cwd.mkdir()
    monkeypatch.chdir(not_cwd)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        _assert_artifact_uri(
            cwd.joinpath("some_path", run.info.run_id, "artifacts", text_file.name),
            ArtifactReturnType(tmp_path, text_file, text_file.name),
        )


@pytest.mark.skipif(not is_windows(), reason="This test only passes on Windows")
def test_log_artifact_windows_path_with_hostname(text_artifact):
    experiment_test_1_artifact_location = r"\\my_server\my_path\my_sub_path\1"
    experiment_test_1_id = mlflow.create_experiment(
        "test_exp_d", experiment_test_1_artifact_location
    )
    with mlflow.start_run(experiment_id=experiment_test_1_id) as run:
        with (
            mock.patch("shutil.copy2") as copyfile_mock,
            mock.patch("os.path.exists", return_value=True) as exists_mock,
        ):
            mlflow.log_artifact(text_artifact.artifact_path)
            exists_mock.assert_called_once()
            copyfile_mock.assert_called_once_with(
                text_artifact.artifact_path,
                (
                    rf"{experiment_test_1_artifact_location}\{run.info.run_id}"
                    rf"\artifacts\{text_artifact.artifact_name}"
                ),
            )


def test_list_artifacts_with_artifact_uri(run_with_artifacts):
    run, artifact_path = run_with_artifacts
    run_uri = f"runs:/{run.info.run_id}/{artifact_path}"
    actual_uri = str(pathlib.PurePosixPath(run.info.artifact_uri) / artifact_path)
    for uri in (run_uri, actual_uri):
        artifacts = mlflow.artifacts.list_artifacts(artifact_uri=uri)
        assert len(artifacts) == 1
        assert artifacts[0].path == f"{artifact_path}/dir"

        artifacts = mlflow.artifacts.list_artifacts(artifact_uri=f"{uri}/dir")
        assert len(artifacts) == 2
        assert artifacts[0].path == "dir/file.txt"
        assert artifacts[1].path == "dir/subdir"

        artifacts = mlflow.artifacts.list_artifacts(artifact_uri=f"{uri}/dir/subdir")
        assert len(artifacts) == 1
        assert artifacts[0].path == "subdir/text.txt"

        artifacts = mlflow.artifacts.list_artifacts(artifact_uri=f"{uri}/non-exist-path")
        assert len(artifacts) == 0


def test_list_artifacts_with_run_id(run_with_artifacts):
    run, artifact_path = run_with_artifacts
    artifacts = mlflow.artifacts.list_artifacts(run_id=run.info.run_id)
    assert len(artifacts) == 1
    assert artifacts[0].path == artifact_path

    artifacts = mlflow.artifacts.list_artifacts(run_id=run.info.run_id, artifact_path=artifact_path)
    assert len(artifacts) == 1
    assert artifacts[0].path == f"{artifact_path}/dir"


def test_list_artifacts_throws_for_invalid_arguments():
    with pytest.raises(MlflowException, match="Exactly one of"):
        mlflow.artifacts.list_artifacts(
            artifact_uri="uri",
            run_id="run_id",
            artifact_path="path",
        )

    with pytest.raises(MlflowException, match="Exactly one of"):
        mlflow.artifacts.list_artifacts()

    with pytest.raises(MlflowException, match="`artifact_path` cannot be specified"):
        mlflow.artifacts.list_artifacts(artifact_uri="uri", artifact_path="path")


def test_download_artifacts_with_run_id_and_artifact_path(tmp_path):
    class DummyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input: list[str]) -> list[str]:
            return model_input

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=DummyModel())
    mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path="model", dst_path=tmp_path
    )


def test_list_artifacts_with_client_and_tracking_uri(tmp_path: pathlib.Path):
    tracking_uri = f"sqlite:///{tmp_path}/mlflow-{uuid.uuid4().hex}.db"
    assert mlflow.get_tracking_uri() != tracking_uri
    client = mlflow.MlflowClient(tracking_uri)
    experiment_id = client.create_experiment("my_experiment")
    run = client.create_run(experiment_id)
    tmp_dir = tmp_path / "subdir"
    tmp_dir.mkdir()
    tmp_file = tmp_dir / "file.txt"
    tmp_file.touch()
    client.log_artifacts(run.info.run_id, tmp_dir, "subdir")

    artifacts = mlflow.artifacts.list_artifacts(
        run_id=run.info.run_id,
        tracking_uri=tracking_uri,
    )
    assert [p.path for p in artifacts] == ["subdir"]

    artifacts = mlflow.artifacts.list_artifacts(
        run_id=run.info.run_id,
        artifact_path="subdir",
        tracking_uri=tracking_uri,
    )
    assert [p.path for p in artifacts] == ["subdir/file.txt"]


def test_download_artifacts_with_client_and_tracking_uri(tmp_path: pathlib.Path):
    tracking_uri = f"sqlite:///{tmp_path}/mlflow-{uuid.uuid4().hex}.db"
    assert mlflow.get_tracking_uri() != tracking_uri
    client = mlflow.MlflowClient(tracking_uri)
    experiment_id = client.create_experiment("my_experiment")
    run = client.create_run(experiment_id)
    tmp_dir = tmp_path / "subdir"
    tmp_dir.mkdir()
    tmp_file = tmp_dir / "file.txt"
    tmp_file.touch()
    client.log_artifacts(run.info.run_id, tmp_dir, "subdir")

    dst_path = tmp_path / "dst"
    mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id,
        artifact_path="subdir",
        tracking_uri=tracking_uri,
        dst_path=dst_path,
    )
    assert tmp_file.name in [p.name for p in dst_path.rglob("*")]


def test_single_run_artifact_download_when_both_run_and_model_artifacts_exist(tmp_path):
    class DummyModel(mlflow.pyfunc.PythonModel):
        def predict(self, model_input: list[str]) -> list[str]:
            return model_input

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(python_model=DummyModel(), name="model")
        mlflow.log_text("test", "model/file.txt")

    out = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path="model/file.txt", dst_path=tmp_path
    )
    assert pathlib.Path(out).read_text() == "test"

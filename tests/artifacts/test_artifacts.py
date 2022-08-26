import json
import pathlib

import pytest

import mlflow
from mlflow.exceptions import MlflowException


@pytest.fixture()
def run_with_artifact(tmp_path):
    artifact_path = "test"
    artifact_content = "content"
    local_path = tmp_path.joinpath("file.txt")
    local_path.write_text(artifact_content)
    with mlflow.start_run() as run:
        mlflow.log_artifact(local_path, artifact_path)

    artifact_uri = str(pathlib.PurePosixPath(run.info.artifact_uri) / artifact_path)
    return (artifact_uri, run, artifact_path, artifact_content)


def test_download_artifacts_with_uri(run_with_artifact):
    actual_uri, run, artifact_path, artifact_content = run_with_artifact
    run_uri = f"runs:/{run.info.run_id}/{artifact_path}"
    for uri in (run_uri, actual_uri):
        download_output_path = mlflow.artifacts.download_artifacts(artifact_uri=uri)
        downloaded_artifact_path = next(pathlib.Path(download_output_path).iterdir())
        assert downloaded_artifact_path.read_text() == artifact_content


def test_download_artifacts_with_run_id_and_path(run_with_artifact):
    _, run, artifact_path, artifact_content = run_with_artifact
    download_output_path = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path=artifact_path
    )
    downloaded_artifact_path = next(pathlib.Path(download_output_path).iterdir())
    assert downloaded_artifact_path.read_text() == artifact_content


def test_download_artifacts_with_run_id_no_path(run_with_artifact):
    _, run, artifact_path, _ = run_with_artifact
    artifact_relative_path_top_level_dir = pathlib.PurePosixPath(artifact_path).parts[0]
    downloaded_output_path = mlflow.artifacts.download_artifacts(run_id=run.info.run_id)
    downloaded_artifact_directory_name = next(pathlib.Path(downloaded_output_path).iterdir()).name
    assert downloaded_artifact_directory_name == artifact_relative_path_top_level_dir


@pytest.mark.parametrize("dst_subdir_path", [None, "doesnt_exist_yet/subdiir"])
def test_download_artifacts_with_dst_path(run_with_artifact, tmp_path, dst_subdir_path):
    _, run, artifact_path, _ = run_with_artifact
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


@pytest.fixture()
def run_with_json_artifact(tmp_path):
    artifact_path = "test"
    artifact_content = {"mlflow-version": "0.28", "n_cores": "10"}
    local_path = tmp_path.joinpath("config.json")
    local_path.write_text(json.dumps(artifact_content))
    with mlflow.start_run() as run:
        mlflow.log_artifact(local_path, artifact_path)

    artifact_uri = str(pathlib.PurePosixPath(run.info.artifact_uri) / artifact_path)
    return (artifact_uri, run, artifact_path, artifact_content)


@pytest.fixture()
def run_with_image_artifact():
    from PIL import Image

    artifact_path = "test"
    image = Image.new("RGB", (100, 100))
    with mlflow.start_run() as run:
        mlflow.log_image(image, artifact_path + "/image.png")

    artifact_uri = str(pathlib.PurePosixPath(run.info.artifact_uri) / artifact_path)
    return (artifact_uri, run, artifact_path, image)


def test_load_text(run_with_artifact):
    artifact_uri, _, _, artifact_content = run_with_artifact
    text_file = artifact_uri + "/file.txt"
    assert mlflow.artifacts.load_text(text_file) == artifact_content


def test_load_dict(run_with_json_artifact):
    artifact_uri, _, _, artifact_content = run_with_json_artifact
    artifact_file = artifact_uri + "/config.json"
    assert mlflow.artifacts.load_dict(artifact_file) == artifact_content


def test_load_json_invalid_json(run_with_artifact):
    artifact_uri, _, _, _ = run_with_artifact
    artifact_file = artifact_uri + "/file.txt"
    with pytest.raises(mlflow.exceptions.MlflowException, match="Unable to form a JSON object"):
        mlflow.artifacts.load_dict(artifact_file)


def test_load_image(run_with_image_artifact):
    from PIL import Image

    artifact_uri, _, _, _ = run_with_image_artifact
    artifact_file = artifact_uri + "/image.png"
    assert isinstance(mlflow.artifacts.load_image(artifact_file), Image.Image)


def test_load_image_invalid_image(run_with_artifact):
    artifact_uri, _, _, _ = run_with_artifact
    artifact_file = artifact_uri + "/file.txt"
    with pytest.raises(
        mlflow.exceptions.MlflowException, match="Unable to form a PIL Image object"
    ):
        mlflow.artifacts.load_image(artifact_file)

import os

import mlflow
from mlflow.tracking.artifact_utils import _download_artifact_from_uri


def test_artifact_can_be_downloaded_from_absolute_uri_successfully(tmpdir):
    artifact_file_name = "artifact.txt"
    artifact_text = "Sample artifact text"
    local_artifact_path = tmpdir.join(artifact_file_name).strpath
    with open(local_artifact_path, "w") as out:
        out.write(artifact_text)

    logged_artifact_path = "artifact"
    with mlflow.start_run():
        mlflow.log_artifact(local_path=local_artifact_path, artifact_path=logged_artifact_path)
        artifact_uri = mlflow.get_artifact_uri(artifact_path=logged_artifact_path)

    downloaded_artifact_path = os.path.join(
        _download_artifact_from_uri(artifact_uri), artifact_file_name)
    assert downloaded_artifact_path != local_artifact_path
    assert downloaded_artifact_path != logged_artifact_path
    with open(downloaded_artifact_path, "r") as f:
        assert f.read() == artifact_text


def test_download_artifact_from_absolute_uri_persists_data_to_specified_output_directory(tmpdir):
    artifact_file_name = "artifact.txt"
    artifact_text = "Sample artifact text"
    local_artifact_path = tmpdir.join(artifact_file_name).strpath
    with open(local_artifact_path, "w") as out:
        out.write(artifact_text)

    logged_artifact_subdir = "logged_artifact"
    with mlflow.start_run():
        mlflow.log_artifact(local_path=local_artifact_path, artifact_path=logged_artifact_subdir)
        artifact_uri = mlflow.get_artifact_uri(artifact_path=logged_artifact_subdir)

    artifact_output_path = tmpdir.join("artifact_output").strpath
    os.makedirs(artifact_output_path)
    _download_artifact_from_uri(artifact_uri=artifact_uri, output_path=artifact_output_path)
    assert logged_artifact_subdir in os.listdir(artifact_output_path)
    assert artifact_file_name in os.listdir(
        os.path.join(artifact_output_path, logged_artifact_subdir))
    with open(os.path.join(
            artifact_output_path, logged_artifact_subdir, artifact_file_name), "r") as f:
        assert f.read() == artifact_text


# TODO(sueann): this might be worth having since we are switching everything over
# def test_download_artifact_from_uri_vs_get_model_log_dir():
#     _download_artifact_from_uri
#
#     def pretrained_model():
#         model_path = "model"
#     with mlflow.start_run():
#         X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
#         y = np.array([0, 0, 1, 1, 1, 0])
#         lr = LogisticRegression()
#         lr.fit(X, y)
#         mlflow.sklearn.log_model(lr, model_path)
#         run_id = mlflow.active_run().info.run_uuid
#         run_uri = "runs:/" + run_id + "/" + model_path
#         return TrainedModel(model_path, run_id, run_uri)

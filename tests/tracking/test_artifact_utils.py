import os

from unittest import mock
from unittest.mock import ANY

import mlflow
from mlflow.tracking.artifact_utils import (
    _download_artifact_from_uri,
    _upload_artifacts_to_databricks,
)


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
        _download_artifact_from_uri(artifact_uri), artifact_file_name
    )
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
        os.path.join(artifact_output_path, logged_artifact_subdir)
    )
    with open(
        os.path.join(artifact_output_path, logged_artifact_subdir, artifact_file_name), "r"
    ) as f:
        assert f.read() == artifact_text


def test_download_artifact_with_special_characters_in_file_name_and_path(tmpdir):
    artifact_file_name = " artifact_ with! special  characters.txt"
    artifact_sub_dir = " path with ! special  characters"
    artifact_text = "Sample artifact text"
    local_sub_path = os.path.join(tmpdir, artifact_sub_dir)
    os.makedirs(local_sub_path)

    local_artifact_path = os.path.join(local_sub_path, artifact_file_name)
    with open(local_artifact_path, "w") as out:
        out.write(artifact_text)

    logged_artifact_subdir = "logged_artifact"
    with mlflow.start_run():
        mlflow.log_artifact(local_path=local_artifact_path, artifact_path=logged_artifact_subdir)
        artifact_uri = mlflow.get_artifact_uri(artifact_path=logged_artifact_subdir)

    artifact_output_path = os.path.join(tmpdir, "artifact output path!")
    os.makedirs(artifact_output_path)
    _download_artifact_from_uri(artifact_uri=artifact_uri, output_path=artifact_output_path)
    assert logged_artifact_subdir in os.listdir(artifact_output_path)
    assert artifact_file_name in os.listdir(
        os.path.join(artifact_output_path, logged_artifact_subdir)
    )
    with open(
        os.path.join(artifact_output_path, logged_artifact_subdir, artifact_file_name), "r"
    ) as f:
        assert f.read() == artifact_text


def test_upload_artifacts_to_databricks():
    import_root = "mlflow.tracking.artifact_utils"
    with mock.patch(import_root + "._download_artifact_from_uri") as download_mock, mock.patch(
        import_root + ".DbfsRestArtifactRepository"
    ) as repo_mock:
        new_source = _upload_artifacts_to_databricks(
            "dbfs:/original/sourcedir/",
            "runid12345",
            "databricks://tracking",
            "databricks://registry:ws",
        )
        download_mock.assert_called_once_with("dbfs://tracking@databricks/original/sourcedir/", ANY)
        repo_mock.assert_called_once_with(
            "dbfs://registry:ws@databricks/databricks/mlflow/tmp-external-source/"
        )
        assert new_source == "dbfs:/databricks/mlflow/tmp-external-source/runid12345/sourcedir"


def test_upload_artifacts_to_databricks_no_run_id():
    from uuid import UUID

    import_root = "mlflow.tracking.artifact_utils"
    with mock.patch(import_root + "._download_artifact_from_uri") as download_mock, mock.patch(
        import_root + ".DbfsRestArtifactRepository"
    ) as repo_mock, mock.patch("uuid.uuid4", return_value=UUID("4f746cdcc0374da2808917e81bb53323")):
        new_source = _upload_artifacts_to_databricks(
            "dbfs:/original/sourcedir/", None, "databricks://tracking:ws", "databricks://registry"
        )
        download_mock.assert_called_once_with(
            "dbfs://tracking:ws@databricks/original/sourcedir/", ANY
        )
        repo_mock.assert_called_once_with(
            "dbfs://registry@databricks/databricks/mlflow/tmp-external-source/"
        )
        assert (
            new_source == "dbfs:/databricks/mlflow/tmp-external-source/"
            "4f746cdcc0374da2808917e81bb53323/sourcedir"
        )

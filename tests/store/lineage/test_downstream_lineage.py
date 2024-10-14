import base64
from unittest import mock

import pytest

import mlflow
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    Entity,
    Job,
    LineageHeaderInfo,
    Notebook,
)
from mlflow.store._unity_catalog.lineage.constants import _DATABRICKS_LINEAGE_ID_HEADER
from mlflow.store._unity_catalog.registry.rest_store import UcModelRegistryStore
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.store.artifact.unity_catalog_models_artifact_repo import (
    UnityCatalogModelsArtifactRepository,
)
from mlflow.utils.proto_json_utils import message_to_json


class SimpleModel(mlflow.pyfunc.PythonModel):
    def predict(self, _, model_input):
        return model_input.applymap(lambda x: x * 2)


@pytest.fixture
def store(mock_databricks_uc_host_creds):
    with mock.patch("mlflow.utils.databricks_utils.get_databricks_host_creds"):
        yield UcModelRegistryStore(store_uri="databricks-uc", tracking_uri="databricks")


def lineage_header_info_to_extra_headers(lineage_header_info):
    extra_headers = {}
    if lineage_header_info:
        header_json = message_to_json(lineage_header_info)
        header_base64 = base64.b64encode(header_json.encode())
        extra_headers[_DATABRICKS_LINEAGE_ID_HEADER] = header_base64
    return extra_headers


@pytest.mark.parametrize(
    ("is_in_notebook", "is_in_job", "notebook_id", "job_id"),
    [
        (True, True, None, None),
        (True, True, "1234", None),
        (True, True, None, "5678"),
        (True, True, "1234", "5678"),
        (False, False, "1234", "5678"),
    ],
)
def test_downstream_notebook_job_lineage(
    tmp_path, is_in_notebook, is_in_job, notebook_id, job_id, monkeypatch
):
    monkeypatch.setenvs(
        {
            "DATABRICKS_HOST": "my-host",
            "DATABRICKS_TOKEN": "my-token",
        }
    )
    model_dir = str(tmp_path.joinpath("model"))
    model_name = "mycatalog.myschema.mymodel"
    model_uri = f"models:/{model_name}/1"

    mock_artifact_repo = mock.MagicMock(autospec=S3ArtifactRepository)
    mock_artifact_repo.download_artifacts.return_value = model_dir

    entity_list = []
    if is_in_notebook and notebook_id:
        notebook_entity = Notebook(id=str(notebook_id))
        entity_list.append(Entity(notebook=notebook_entity))

    if is_in_job and job_id:
        job_entity = Job(id=str(job_id))
        entity_list.append(Entity(job=job_entity))

    expected_lineage_header_info = LineageHeaderInfo(entities=entity_list) if entity_list else None

    # Mock out all necessary dependency
    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_notebook",
        return_value=is_in_notebook,
    ), mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_runtime",
        return_value=is_in_notebook or is_in_job,
    ), mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_job",
        return_value=is_in_job,
    ), mock.patch(
        "mlflow.utils.databricks_utils.get_notebook_id",
        return_value=notebook_id,
    ), mock.patch(
        "mlflow.utils.databricks_utils.get_job_id",
        return_value=job_id,
    ), mock.patch("mlflow.get_registry_uri", return_value="databricks-uc"), mock.patch.object(
        UnityCatalogModelsArtifactRepository,
        "_get_blob_storage_path",
        return_value="fake_blob_storage_path",
    ), mock.patch(
        "mlflow.utils._unity_catalog_utils._get_artifact_repo_from_storage_info",
        return_value=mock_artifact_repo,
    ), mock.patch(
        "mlflow.utils.rest_utils.http_request",
        return_value=mock.MagicMock(status_code=200, text="{}"),
    ) as mock_http:
        mlflow.pyfunc.save_model(path=model_dir, python_model=SimpleModel())
        mlflow.pyfunc.load_model(model_uri)
        extra_headers = lineage_header_info_to_extra_headers(expected_lineage_header_info)
        if is_in_notebook or is_in_job:
            mock_http.assert_called_once_with(
                host_creds=mock.ANY,
                endpoint=mock.ANY,
                method=mock.ANY,
                json=mock.ANY,
                extra_headers=extra_headers,
            )

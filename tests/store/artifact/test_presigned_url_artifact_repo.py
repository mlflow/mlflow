import json
from unittest import mock

from mlflow.protos.databricks_filesystem_service_pb2 import ListDirectoryResponse, DirectoryEntry
from mlflow.store.artifact.presigned_url_artifact_repo import PresignedUrlArtifactRepository, DIRECTORIES_ENDPOINT

model_uri = "/Models/catalog/schema/model/1"


def mock_call_endpoint_impl(*args, **kwargs):
    endpoint = kwargs["endpoint"]
    json_body = kwargs["json_body"]

    if (endpoint == f'{DIRECTORIES_ENDPOINT}/Models/catalog/schema/model/1/dir'
            and json_body == json.dumps({"page_token": "some_token"})):
        return ListDirectoryResponse(
            contents=[
                DirectoryEntry(is_directory=False, path=f"{model_uri}/dir/file2", file_size=2)
            ]
        )
    elif endpoint == f'{DIRECTORIES_ENDPOINT}/Models/catalog/schema/model/1/dir':
        return ListDirectoryResponse(
            contents=[DirectoryEntry(is_directory=False, path=f"{model_uri}/dir/file1", file_size=1)],
            next_page_token="some_token"
        )
    elif endpoint == f'{DIRECTORIES_ENDPOINT}/Models/catalog/schema/model/1/':
        return ListDirectoryResponse(
            contents=[DirectoryEntry(is_directory=True, path=f"{model_uri}/dir")],
        )
    else:
        raise ValueError(f"Unexpected endpoint: {endpoint}")


def test_list_artifact_pagination():
    global model_uri
    artifact_repo = PresignedUrlArtifactRepository(model_uri)

    with mock.patch('mlflow.store.artifact.presigned_url_artifact_repo.call_endpoint',
                    side_effect=mock_call_endpoint_impl) as mock_call_endpoint:
        artifact_repo = PresignedUrlArtifactRepository(model_uri)
        resp = artifact_repo.list_artifacts()
        assert len(resp) == 1
        assert resp[0].path == "dir"
        assert resp[0].is_dir is True
        assert resp[0].file_size is None

        resp = artifact_repo.list_artifacts("dir")
        assert len(resp) == 2
        assert {r.path for r in resp} == {"dir/file1", "dir/file2"}
        assert {r.is_dir for r in resp} == {False}
        assert {r.file_size for r in resp} == {1, 2}

        assert mock_call_endpoint.call_count == 3

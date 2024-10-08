from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

from mlflow.entities.file_info import FileInfo
from mlflow.protos.databricks_artifacts_pb2 import (
    DatabricksMlflowArtifactsService,
    GetCredentialsForLoggedModelDownload,
    GetCredentialsForLoggedModelUpload,
    GetCredentialsForRead,
    GetCredentialsForWrite,
)
from mlflow.protos.service_pb2 import (
    GetLoggedModel,
    GetRun,
    ListArtifacts,
    ListLoggedModelArtifacts,
    MlflowService,
)
from mlflow.utils.proto_json_utils import message_to_json


class _CredentialType(Enum):
    READ = 1
    WRITE = 2


@dataclass
class HttpHeader:
    name: str
    value: str


@dataclass
class ArtifactCredentialInfo:
    signed_uri: str
    type: Any
    headers: List[HttpHeader]


@dataclass
class ListArtifactsPage:
    files: List[FileInfo]
    next_page_token: Optional[str] = None

    @classmethod
    def empty(cls):
        return cls(files=[], next_page_token=None)


class _Resource(ABC):
    """
    Represents a resource that `DatabricksArtifactRepository` interacts with.
    """

    def __init__(self, id_: str, call_endpoint: Callable[..., Any]):
        self.id = id_
        self.call_endpoint = call_endpoint
        self.artifact_root = self.get_artifact_root()

    @abstractmethod
    def get_credentials(
        self,
        cred_type: _CredentialType,
        paths: List[str],
        page_token: Optional[str] = None,
    ) -> Tuple[List[ArtifactCredentialInfo], Optional[str]]:
        """
        Fetches read/write credentials for the specified paths.
        """

    @abstractmethod
    def get_artifact_root(self) -> str:
        """
        Get the artifact root URI of this resource.
        """

    @abstractmethod
    def _list_artifacts(
        self, path: Optional[str] = None, page_token: Optional[str] = None
    ) -> ListArtifactsPage:
        """
        List artifacts under the specified path.
        """

    def list_artifacts(self, path: Optional[str] = None) -> List[FileInfo]:
        files: List[FileInfo] = []
        page_token: Optional[str] = None
        while True:
            page = self._list_artifacts(path, page_token)
            files.extend(page.files)
            if len(page.files) == 0 or not page.next_page_token:
                break
            page_token = page.next_page_token

        return files


class _LoggedModel(_Resource):
    def get_credentials(
        self,
        cred_type: _CredentialType,
        paths: List[str],
        page_token: Optional[str] = None,
    ) -> Tuple[List[Any], Optional[str]]:
        api = (
            GetCredentialsForLoggedModelDownload
            if cred_type == _CredentialType.READ
            else GetCredentialsForLoggedModelUpload
        )
        payload = api(paths=paths, page_token=page_token)
        response = self.call_endpoint(
            DatabricksMlflowArtifactsService,
            api,
            message_to_json(payload),
            path_params={"model_id": self.id},
        )
        credential_infos = [
            ArtifactCredentialInfo(
                signed_uri=c.credential_info.signed_uri,
                type=c.credential_info.type,
                headers=[HttpHeader(name=h.name, value=h.value) for h in c.credential_info.headers],
            )
            for c in response.credentials
        ]
        return credential_infos, response.next_page_token

    def get_artifact_root(self) -> str:
        json_body = message_to_json(GetLoggedModel(model_id=self.id))
        response = self.call_endpoint(
            MlflowService, GetLoggedModel, json_body, path_params={"model_id": self.id}
        )
        return response.model.info.artifact_uri

    def _list_artifacts(
        self, path: Optional[str] = None, page_token: Optional[str] = None
    ) -> Tuple[List[FileInfo], Optional[str]]:
        json_body = message_to_json(ListLoggedModelArtifacts(page_token=page_token))
        response = self.call_endpoint(
            MlflowService,
            ListLoggedModelArtifacts,
            json_body,
            path_params={"model_id": self.id, "artifact_directory_path": path},
        )
        files = response.files
        # If `path` is a file, ListArtifacts returns a single list element with the
        # same name as `path`. The list_artifacts API expects us to return an empty list in this
        # case, so we do so here.
        if len(files) == 1 and files[0].path == path and not files[0].is_dir:
            return ListArtifactsPage.empty()

        return ListArtifactsPage(
            files=[FileInfo(f.path, f.is_dir, None if f.is_dir else f.file_size) for f in files],
            next_page_token=response.next_page_token,
        )


class _Run(_Resource):
    def get_credentials(
        self,
        cred_type: _CredentialType,
        paths: List[str],
        page_token: Optional[str] = None,
    ) -> List[ArtifactCredentialInfo]:
        api = GetCredentialsForRead if cred_type == _CredentialType.READ else GetCredentialsForWrite
        json_body = api(run_id=self.id, path=paths, page_token=page_token)
        response = self.call_endpoint(
            DatabricksMlflowArtifactsService, api, message_to_json(json_body)
        )
        credential_infos = [
            ArtifactCredentialInfo(
                signed_uri=c.signed_uri,
                type=c.type,
                headers=[HttpHeader(name=h.name, value=h.value) for h in c.headers],
            )
            for c in response.credential_infos
        ]
        return credential_infos, response.next_page_token

    def get_artifact_root(self) -> str:
        json_body = message_to_json(GetRun(run_id=self.id))
        run_response = self.call_endpoint(MlflowService, GetRun, json_body)
        return run_response.run.info.artifact_uri

    def _list_artifacts(
        self, path: Optional[str] = None, page_token: Optional[str] = None
    ) -> ListArtifactsPage:
        json_body = message_to_json(
            ListArtifacts(run_id=self.resource.id, path=path, page_token=page_token)
        )
        response = self._call_endpoint(MlflowService, ListArtifacts, json_body)
        files = response.files
        # If `path` is a file, ListArtifacts returns a single list element with the
        # same name as `path`. The list_artifacts API expects us to return an empty list in this
        # case, so we do so here.
        if len(files) == 1 and files[0].path == path and not files[0].is_dir:
            return ListArtifactsPage.empty()

        return ListArtifactsPage(
            files=[FileInfo(f.path, f.is_dir, None if f.is_dir else f.file_size) for f in files],
            next_page_token=response.next_page_token,
        )

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

from mlflow.protos.databricks_artifacts_pb2 import (
    DatabricksMlflowArtifactsService,
    GetCredentialsForLoggedModelDownload,
    GetCredentialsForLoggedModelUpload,
    GetCredentialsForRead,
    GetCredentialsForWrite,
)
from mlflow.protos.service_pb2 import GetLoggedModel, GetRun, MlflowService
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
    """
    Represents the credentials needed to access an artifact.
    """

    signed_uri: str
    type: Any
    headers: List[HttpHeader]


class _Resource(ABC):
    """
    Represents a resource that `DatabricksArtifactRepository` interacts with.
    """

    def __init__(self, id: str, call_endpoint: Callable[..., Any]):
        self.id = id
        self.call_endpoint = call_endpoint
        self.artifact_root = self.get_artifact_root()

    @abstractmethod
    def get_credentials(
        self, cred_type: _CredentialType, paths: List[str], page_token: Optional[str] = None
    ) -> Tuple[List[ArtifactCredentialInfo], Optional[str]]:
        """
        Fetches read/write credentials for the specified paths.
        """

    @abstractmethod
    def get_artifact_root(self) -> str:
        """
        Get the artifact root URI of this resource.
        """


class _LoggedModel(_Resource):
    def get_credentials(
        self, cred_type: _CredentialType, paths: List[str], page_token: Optional[str] = None
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


class _Run(_Resource):
    def get_credentials(
        self, cred_type: _CredentialType, paths: List[str], page_token: Optional[str] = None
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

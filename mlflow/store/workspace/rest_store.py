from __future__ import annotations

from urllib.parse import quote

from mlflow.entities import Workspace
from mlflow.entities.workspace import WorkspaceDeletionMode
from mlflow.exceptions import MlflowException, RestException
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import INVALID_STATE, RESOURCE_ALREADY_EXISTS
from mlflow.protos.service_pb2 import (
    CreateWorkspace,
    DeleteWorkspace,
    GetWorkspace,
    ListWorkspaces,
    UpdateWorkspace,
)
from mlflow.store.workspace.abstract_store import AbstractStore, WorkspaceNameValidator
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import call_endpoint
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

WORKSPACES_ENDPOINT = "/api/3.0/mlflow/workspaces"


def _quote_workspace(workspace_name: str) -> str:
    if workspace_name != DEFAULT_WORKSPACE_NAME:
        WorkspaceNameValidator.validate(workspace_name)
    return quote(workspace_name, safe="")


class RestWorkspaceStore(AbstractStore):
    """REST-backed workspace store implementation."""

    def __init__(self, get_host_creds):
        self.get_host_creds = get_host_creds

    def _workspace_from_proto(self, proto) -> Workspace:
        if not proto.HasField("workspace"):
            raise MlflowException(
                "Workspace response payload was missing the 'workspace' field",
                INVALID_STATE,
            )
        return Workspace.from_proto(proto.workspace)

    def list_workspaces(self) -> list[Workspace]:
        proto = call_endpoint(
            host_creds=self.get_host_creds(),
            endpoint=WORKSPACES_ENDPOINT,
            method="GET",
            json_body=None,
            response_proto=ListWorkspaces.Response(),
        )
        return [Workspace.from_proto(ws) for ws in proto.workspaces]

    def get_workspace(self, workspace_name: str) -> Workspace:
        proto = call_endpoint(
            host_creds=self.get_host_creds(),
            endpoint=f"{WORKSPACES_ENDPOINT}/{_quote_workspace(workspace_name)}",
            method="GET",
            json_body=None,
            response_proto=GetWorkspace.Response(),
        )
        return self._workspace_from_proto(proto)

    def create_workspace(self, workspace: Workspace) -> Workspace:
        WorkspaceNameValidator.validate(workspace.name)
        request_message = CreateWorkspace(name=workspace.name)
        if workspace.description is not None:
            request_message.description = workspace.description
        if workspace.default_artifact_root is not None:
            request_message.default_artifact_root = workspace.default_artifact_root
        try:
            proto = call_endpoint(
                host_creds=self.get_host_creds(),
                endpoint=WORKSPACES_ENDPOINT,
                method="POST",
                json_body=message_to_json(request_message),
                response_proto=CreateWorkspace.Response(),
                expected_status=201,
            )
        except RestException as exc:
            if exc.error_code == databricks_pb2.ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
                message = exc.message or f"Workspace '{workspace.name}' already exists."
                raise MlflowException(message, RESOURCE_ALREADY_EXISTS) from exc
            raise

        return self._workspace_from_proto(proto)

    def update_workspace(self, workspace: Workspace) -> Workspace:
        request_message = UpdateWorkspace()
        if workspace.description is not None:
            request_message.description = workspace.description
        if workspace.default_artifact_root is not None:
            request_message.default_artifact_root = workspace.default_artifact_root
        proto = call_endpoint(
            host_creds=self.get_host_creds(),
            endpoint=f"{WORKSPACES_ENDPOINT}/{_quote_workspace(workspace.name)}",
            method="PATCH",
            json_body=message_to_json(request_message),
            response_proto=UpdateWorkspace.Response(),
        )
        return self._workspace_from_proto(proto)

    def delete_workspace(
        self,
        workspace_name: str,
        mode: WorkspaceDeletionMode = WorkspaceDeletionMode.RESTRICT,
    ) -> None:
        endpoint = f"{WORKSPACES_ENDPOINT}/{_quote_workspace(workspace_name)}"
        if mode != WorkspaceDeletionMode.RESTRICT:
            endpoint += f"?mode={mode.value}"
        call_endpoint(
            host_creds=self.get_host_creds(),
            endpoint=endpoint,
            method="DELETE",
            json_body=None,
            response_proto=DeleteWorkspace.Response(),
            expected_status=204,
        )

    def get_default_workspace(self) -> Workspace:
        raise NotImplementedError(
            "REST workspace provider does not expose a default workspace; "
            "please specify a workspace explicitly or omit a workspace to leverage the server's "
            "configured default."
        )

from __future__ import annotations

import json
from urllib.parse import quote

from mlflow.entities import Workspace
from mlflow.exceptions import MlflowException
from mlflow.store.workspace.abstract_store import AbstractStore
from mlflow.utils.rest_utils import http_request, verify_rest_response


class RestWorkspaceStore(AbstractStore):
    """REST-backed workspace store implementation."""

    def __init__(self, get_host_creds):
        self.get_host_creds = get_host_creds

    def list_workspaces(self, request) -> list[Workspace]:
        endpoint = "/api/2.0/mlflow/workspaces"
        response = http_request(
            host_creds=self.get_host_creds(),
            endpoint=endpoint,
            method="GET",
        )
        response = verify_rest_response(response, endpoint)
        payload = json.loads(response.text) if response.text else {}
        return [
            Workspace(name=ws["name"], description=ws.get("description"))
            for ws in payload.get("workspaces", [])
        ]

    def get_workspace(self, workspace_name: str, request) -> Workspace:
        endpoint = f"/api/2.0/mlflow/workspaces/{quote(workspace_name, safe='')}"
        response = http_request(
            host_creds=self.get_host_creds(),
            endpoint=endpoint,
            method="GET",
        )
        response = verify_rest_response(response, endpoint)
        data = json.loads(response.text) if response.text else {}
        return Workspace(name=data["name"], description=data.get("description"))

    def create_workspace(self, workspace: Workspace, request) -> Workspace:
        endpoint = "/api/2.0/mlflow/workspaces"
        payload = {"name": workspace.name}
        if workspace.description is not None:
            payload["description"] = workspace.description
        response = http_request(
            host_creds=self.get_host_creds(),
            endpoint=endpoint,
            method="POST",
            json=payload,
        )
        if response.status_code == 201:
            data = json.loads(response.text) if response.text else {}
            if not data:
                data = {"name": workspace.name, "description": workspace.description}
            return Workspace(name=data["name"], description=data.get("description"))

        response = verify_rest_response(response, endpoint)
        data = json.loads(response.text) if response.text else {}
        return Workspace(name=data["name"], description=data.get("description"))

    def update_workspace(self, workspace: Workspace, request) -> Workspace:
        endpoint = f"/api/2.0/mlflow/workspaces/{quote(workspace.name, safe='')}"
        request_payload = {"description": workspace.description}
        response = http_request(
            host_creds=self.get_host_creds(),
            endpoint=endpoint,
            method="PATCH",
            json=request_payload,
        )
        response = verify_rest_response(response, endpoint)
        data = json.loads(response.text) if response.text else {}
        if not data:
            data = {
                "name": workspace.name,
                "description": workspace.description,
            }
        return Workspace(name=data["name"], description=data.get("description"))

    def delete_workspace(self, workspace_name: str, request) -> None:
        endpoint = f"/api/2.0/mlflow/workspaces/{quote(workspace_name, safe='')}"
        response = http_request(
            host_creds=self.get_host_creds(),
            endpoint=endpoint,
            method="DELETE",
        )
        if response.status_code == 204:
            return
        verify_rest_response(response, endpoint)

    def get_default_workspace(self, request) -> Workspace:
        raise MlflowException.invalid_parameter_value(
            "REST workspace provider does not expose a default workspace; "
            "please specify a workspace explicitly."
        )

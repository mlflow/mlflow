from __future__ import annotations

from mlflow.entities.workspace import Workspace
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking._workspace.registry import get_workspace_store


class WorkspaceProviderClient:
    """
    Client that exposes workspace CRUD operations via the configured provider.

    The provider is resolved based on the workspace URI scheme
    (for example ``sqlite`` or ``mysql``).
    This mirrors the scheme-based resolution used by tracking and model registry stores.
    """

    def __init__(self, tracking_uri: str, workspace_uri: str | None = None):
        self._tracking_uri = tracking_uri
        self._workspace_uri = workspace_uri
        self._provider = None
        # Eagerly validate configuration to surface errors early.
        self.provider

    @property
    def provider(self):
        if self._provider is None:
            if self._workspace_uri is None:
                raise MlflowException(
                    "Workspace URI resolution failed. Please provide a workspace URI.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            self._provider = get_workspace_store(workspace_uri=self._workspace_uri)
        return self._provider

    def list_workspaces(self) -> list[Workspace]:
        return list(self.provider.list_workspaces(request=None))

    def create_workspace(self, name: str, description: str | None = None) -> Workspace:
        return self.provider.create_workspace(
            Workspace(name=name, description=description),
            request=None,
        )

    def get_workspace(self, name: str) -> Workspace:
        return self.provider.get_workspace(name, request=None)

    def update_workspace(self, name: str, description: str | None = None) -> Workspace:
        return self.provider.update_workspace(
            Workspace(name=name, description=description),
            request=None,
        )

    def delete_workspace(self, name: str) -> None:
        self.provider.delete_workspace(name, request=None)

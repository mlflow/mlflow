from __future__ import annotations

from mlflow.entities.workspace import Workspace
from mlflow.tracking._workspace.registry import get_workspace_store


class WorkspaceProviderClient:
    """
    Client that exposes workspace CRUD operations via the configured provider.

    The provider is resolved based on the workspace URI scheme
    (for example ``sqlite`` or ``mysql``).
    This mirrors the scheme-based resolution used by tracking and model registry stores.
    """

    def __init__(self, workspace_uri: str):
        self._workspace_uri = workspace_uri
        self._store = None
        # Eagerly validate configuration to surface errors early.
        self.store

    @property
    def store(self):
        if self._store is None:
            self._store = get_workspace_store(workspace_uri=self._workspace_uri)
        return self._store

    def list_workspaces(self) -> list[Workspace]:
        return list(self.store.list_workspaces())

    def create_workspace(self, name: str, description: str | None = None) -> Workspace:
        return self.store.create_workspace(Workspace(name=name, description=description))

    def get_workspace(self, name: str) -> Workspace:
        return self.store.get_workspace(name)

    def update_workspace(self, name: str, description: str | None = None) -> Workspace:
        return self.store.update_workspace(Workspace(name=name, description=description))

    def delete_workspace(self, name: str) -> None:
        self.store.delete_workspace(name)

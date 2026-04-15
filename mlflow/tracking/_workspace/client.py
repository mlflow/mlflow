from __future__ import annotations

from mlflow.entities.workspace import Workspace, WorkspaceDeletionMode
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

    def create_workspace(
        self,
        name: str,
        description: str | None = None,
        default_artifact_root: str | None = None,
    ) -> Workspace:
        """Create a new workspace.

        Args:
            name: The workspace name (lowercase alphanumeric with optional internal hyphens).
            description: Optional description of the workspace.
            default_artifact_root: Optional artifact root URI; falls back to server default.

        Returns:
            The newly created workspace.
        """
        return self.store.create_workspace(
            Workspace(
                name=name,
                description=description,
                default_artifact_root=default_artifact_root,
            )
        )

    def get_workspace(self, name: str) -> Workspace:
        return self.store.get_workspace(name)

    def update_workspace(
        self,
        name: str,
        description: str | None = None,
        default_artifact_root: str | None = None,
    ) -> Workspace:
        """Update metadata for an existing workspace.

        Args:
            name: The name of the workspace to update.
            description: New description, or ``None`` to leave unchanged.
            default_artifact_root: New artifact root URI, empty string to clear, or ``None``.

        Returns:
            The updated workspace.
        """
        return self.store.update_workspace(
            Workspace(
                name=name,
                description=description,
                default_artifact_root=default_artifact_root,
            )
        )

    def delete_workspace(
        self,
        name: str,
        mode: WorkspaceDeletionMode = WorkspaceDeletionMode.RESTRICT,
    ) -> None:
        self.store.delete_workspace(name, mode=mode)

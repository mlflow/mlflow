from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Iterable

from mlflow.entities import Workspace
from mlflow.entities.workspace import WorkspaceDeletionMode
from mlflow.exceptions import MlflowException


# The workspace store can be backed by something other than the tracking store. For example,
# Kubeflow integrations map MLflow workspaces onto Kubernetes namespaces and rely on a
# workspace store plugin, so we keep this as a separate store rather than extending the
# tracking store.
class AbstractStore(ABC):
    """Interface for resolving and managing workspaces in the tracking server."""

    @abstractmethod
    def list_workspaces(self) -> Iterable[Workspace]:
        """
        Return the workspaces visible to the current request context.

        Implementations may inspect the request (e.g., for authN/Z context) to
        determine which workspaces to expose.
        """

    @abstractmethod
    def get_workspace(self, workspace_name: str) -> Workspace:
        """
        Gets a workspace by name and return its metadata.

        Implementations should raise ``MlflowException`` with
        ``RESOURCE_DOES_NOT_EXIST`` if the workspace cannot be found.
        """

    def create_workspace(self, workspace: Workspace) -> Workspace:
        """Provision a new workspace.

        Raises ``NotImplementedError`` when the active provider is read-only.
        Implementations should raise ``MlflowException`` with
        ``RESOURCE_ALREADY_EXISTS`` when the workspace already exists or
        ``INVALID_PARAMETER_VALUE`` when validation fails.
        """

        raise NotImplementedError

    def update_workspace(self, workspace: Workspace) -> Workspace:
        """Update metadata for an existing workspace."""

        raise NotImplementedError

    def delete_workspace(
        self,
        workspace_name: str,
        mode: WorkspaceDeletionMode = WorkspaceDeletionMode.RESTRICT,
    ) -> None:
        """Delete an existing workspace.

        Args:
            workspace_name: Name of the workspace to delete.
            mode: Controls what happens to resources in the workspace:
                - SET_DEFAULT: Reassign resources to the default workspace.
                - CASCADE: Delete all resources in the workspace.
                - RESTRICT: Refuse if the workspace still contains resources.
        """

        raise NotImplementedError

    def get_default_workspace(self) -> Workspace:
        """
        Return the workspace to select when none is explicitly supplied.

        Implementations that require an explicit workspace should raise an
        ``MlflowException`` with ``INVALID_PARAMETER_VALUE``.
        """

        raise NotImplementedError

    def resolve_artifact_root(
        self, default_artifact_root: str | None, workspace_name: str
    ) -> tuple[str | None, bool]:
        """
        Allow a provider to customize artifact storage roots per workspace.

        Returns:
            A tuple ``(root, append_workspace_prefix)`` where ``root`` is the base artifact
            location to use for the workspace, and ``append_workspace_prefix`` controls whether
            MLflow should append the ``/workspaces/<workspace_name>`` suffix automatically.
        """

        return default_artifact_root, True


class WorkspaceNameValidator:
    """Validator for workspace names based on Kubernetes naming conventions."""

    _PATTERN = r"^(?!.*--)[a-z0-9]([-a-z0-9]*[a-z0-9])?$"
    _MIN_LENGTH = 2
    _MAX_LENGTH = 63
    _RESERVED = {"workspaces", "api", "ajax-api", "static-files"}

    @classmethod
    def pattern(cls) -> str:
        return cls._PATTERN

    @classmethod
    def validate(cls, name: str) -> None:
        if not isinstance(name, str):
            raise MlflowException.invalid_parameter_value(
                f"Workspace name must be a string, got {type(name).__name__!s}."
            )

        if not (cls._MIN_LENGTH <= len(name) <= cls._MAX_LENGTH):
            raise MlflowException.invalid_parameter_value(
                f"Workspace name '{name}' must be between {cls._MIN_LENGTH} and "
                f"{cls._MAX_LENGTH} characters."
            )

        if not re.match(cls._PATTERN, name):
            raise MlflowException.invalid_parameter_value(
                f"Workspace name '{name}' must match the pattern {cls.pattern()} "
                "(lowercase alphanumeric with optional internal hyphens)."
            )

        if name in cls._RESERVED:
            raise MlflowException.invalid_parameter_value(
                f"Workspace name '{name}' is reserved and cannot be used."
            )

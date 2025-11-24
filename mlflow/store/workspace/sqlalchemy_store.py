from __future__ import annotations

import logging
from typing import Iterable

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from mlflow.entities.workspace import Workspace
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.workspace.abstract_store import AbstractStore, WorkspaceNameValidator
from mlflow.store.workspace.dbmodels import SqlWorkspace
from mlflow.utils.uri import extract_db_type_from_uri
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

_logger = logging.getLogger(__name__)


class SqlAlchemyStore(AbstractStore):
    """SQL-backed workspace store implementation."""

    def __init__(self, db_uri: str):
        from mlflow.store.db import utils as db_utils

        self._workspace_uri = db_uri
        self._db_type = extract_db_type_from_uri(db_uri)
        self._engine = db_utils.create_sqlalchemy_engine_with_retry(db_uri)
        db_utils._safe_initialize_tables(self._engine)
        session_factory = sessionmaker(bind=self._engine)
        self.ManagedSessionMaker = db_utils._get_managed_session_maker(
            session_factory, self._db_type
        )

    def list_workspaces(self) -> Iterable[Workspace]:
        with self.ManagedSessionMaker() as session:
            rows = session.query(SqlWorkspace).order_by(SqlWorkspace.name.asc()).all()
            return [row.to_mlflow_entity() for row in rows]

    def get_workspace(self, workspace_name: str) -> Workspace:
        with self.ManagedSessionMaker() as session:
            workspace = self._get_workspace(session, workspace_name)
            return workspace.to_mlflow_entity()

    def create_workspace(self, workspace: Workspace) -> Workspace:
        WorkspaceNameValidator.validate(workspace.name)
        with self.ManagedSessionMaker() as session:
            try:
                entity = SqlWorkspace(name=workspace.name, description=workspace.description)
                session.add(entity)
                session.flush()
                workspace_entity = entity.to_mlflow_entity()
            except IntegrityError as exc:
                raise MlflowException(
                    f"Workspace '{workspace.name}' already exists. Error: {exc}",
                    RESOURCE_ALREADY_EXISTS,
                ) from exc

        _logger.info("Created workspace '%s'", workspace.name)
        return workspace_entity

    def update_workspace(self, workspace: Workspace) -> Workspace:
        with self.ManagedSessionMaker() as session:
            entity = self._get_workspace(session, workspace.name)
            entity.description = workspace.description
            session.flush()

            _logger.info("Updated workspace '%s'", workspace.name)
            return entity.to_mlflow_entity()

    def delete_workspace(self, workspace_name: str) -> None:
        if workspace_name == DEFAULT_WORKSPACE_NAME:
            raise MlflowException(
                f"Cannot delete the reserved '{DEFAULT_WORKSPACE_NAME}' workspace",
                INVALID_STATE,
            )

        with self.ManagedSessionMaker() as session:
            entity = self._get_workspace(session, workspace_name)
            session.delete(entity)
            _logger.info("Deleted workspace '%s'", workspace_name)

    def get_default_workspace(self) -> Workspace:
        return self.get_workspace(DEFAULT_WORKSPACE_NAME)

    def resolve_artifact_root(
        self, default_artifact_root: str, workspace_name: str | None = None
    ) -> tuple[str, bool]:
        return default_artifact_root, True

    def _get_workspace(self, session, workspace_name: str) -> SqlWorkspace:
        workspace = session.get(SqlWorkspace, workspace_name)
        if workspace is None:
            raise MlflowException(
                f"Workspace '{workspace_name}' not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        return workspace

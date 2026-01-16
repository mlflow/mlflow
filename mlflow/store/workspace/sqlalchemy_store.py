from __future__ import annotations

import logging
from threading import Lock
from typing import Iterable

from cachetools import TTLCache
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from mlflow.entities.workspace import Workspace
from mlflow.environment_variables import (
    MLFLOW_WORKSPACE_ARTIFACT_ROOT_CACHE_CAPACITY,
    MLFLOW_WORKSPACE_ARTIFACT_ROOT_CACHE_TTL_SECONDS,
)
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

_CACHE_MISS = object()


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
        # Use a per-process TTL cache to reduce DB lookups; values converge via TTL expiration.
        self._artifact_root_cache: TTLCache[str, str | None] = TTLCache(
            maxsize=MLFLOW_WORKSPACE_ARTIFACT_ROOT_CACHE_CAPACITY.get(),
            ttl=MLFLOW_WORKSPACE_ARTIFACT_ROOT_CACHE_TTL_SECONDS.get(),
        )
        self._artifact_root_cache_lock = Lock()

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
                entity = SqlWorkspace(
                    name=workspace.name,
                    description=workspace.description,
                    default_artifact_root=workspace.default_artifact_root or None,
                )
                session.add(entity)
                session.flush()
                workspace_entity = entity.to_mlflow_entity()
            except IntegrityError as exc:
                raise MlflowException(
                    f"Workspace '{workspace.name}' already exists. Error: {exc}",
                    RESOURCE_ALREADY_EXISTS,
                ) from exc

        # Only update cache after the transaction has successfully committed.
        with self._artifact_root_cache_lock:
            self._artifact_root_cache[workspace.name] = workspace_entity.default_artifact_root
        _logger.info("Created workspace '%s'", workspace.name)
        return workspace_entity

    def update_workspace(self, workspace: Workspace) -> Workspace:
        with self.ManagedSessionMaker() as session:
            entity = self._get_workspace(session, workspace.name)
            if workspace.description is not None:
                entity.description = workspace.description
            if workspace.default_artifact_root is not None:
                # If the default_artifact_root is an empty string, set it to None to "clear" the
                # value
                entity.default_artifact_root = workspace.default_artifact_root or None
            session.flush()

            _logger.info("Updated workspace '%s'", workspace.name)
            workspace_entity = entity.to_mlflow_entity()
        # Only update cache after the transaction has successfully committed.
        with self._artifact_root_cache_lock:
            self._artifact_root_cache[workspace.name] = workspace_entity.default_artifact_root
        return workspace_entity

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
        with self._artifact_root_cache_lock:
            self._artifact_root_cache.pop(workspace_name, None)

    def get_default_workspace(self) -> Workspace:
        return self.get_workspace(DEFAULT_WORKSPACE_NAME)

    def resolve_artifact_root(
        self, default_artifact_root: str | None, workspace_name: str
    ) -> tuple[str | None, bool]:
        with self._artifact_root_cache_lock:
            cached_value = self._artifact_root_cache.get(workspace_name, _CACHE_MISS)
            if cached_value is not _CACHE_MISS:
                if cached_value:
                    return cached_value, False
                return default_artifact_root, True

        with self.ManagedSessionMaker() as session:
            workspace = session.get(SqlWorkspace, workspace_name)
            workspace_root = workspace.default_artifact_root if workspace else None
        with self._artifact_root_cache_lock:
            self._artifact_root_cache[workspace_name] = workspace_root
        if workspace_root:
            return workspace_root, False

        return default_artifact_root, True

    def _get_workspace(self, session, workspace_name: str) -> SqlWorkspace:
        workspace = session.get(SqlWorkspace, workspace_name)
        if workspace is None:
            raise MlflowException(
                f"Workspace '{workspace_name}' not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        return workspace

from __future__ import annotations

import logging
import os

import sqlalchemy
import sqlalchemy.orm
import sqlalchemy.sql.expression as sql
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.future import select

from mlflow.entities import (
    Experiment,
)
from mlflow.entities.entity_type import EntityAssociationType
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_STATE,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.tracking.dbmodels.models import (
    SqlAssessments,
    SqlEvaluationDataset,
    SqlExperiment,
    SqlLoggedModel,
    SqlRun,
    SqlTraceInfo,
)
from mlflow.store.tracking.sqlalchemy_store import (
    SqlAlchemyStore,
)
from mlflow.store.workspace.utils import get_default_workspace_optional
from mlflow.store.workspace_aware_mixin import WorkspaceAwareMixin
from mlflow.tracking._workspace.registry import get_workspace_store
from mlflow.utils import workspace_context, workspace_utils
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import (
    append_to_uri_path,
)
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME, WORKSPACES_DIR_NAME

_logger = logging.getLogger(__name__)


class WorkspaceAwareSqlAlchemyStore(WorkspaceAwareMixin, SqlAlchemyStore):
    """
    Workspace-aware variant of the SQLAlchemy tracking store.
    """

    def __init__(self, db_uri, default_artifact_root):
        self._workspace_provider = None
        super().__init__(db_uri, default_artifact_root)

    def supports_workspaces(self) -> bool:
        return True

    def _get_query(self, session, model):
        query = super()._get_query(session, model)
        workspace = self._get_active_workspace()

        if model is SqlExperiment:
            return query.filter(SqlExperiment.workspace == workspace)

        if model is SqlRun:
            return query.join(
                SqlExperiment, SqlExperiment.experiment_id == SqlRun.experiment_id
            ).filter(SqlExperiment.workspace == workspace)

        if model is SqlTraceInfo:
            return query.join(
                SqlExperiment, SqlTraceInfo.experiment_id == SqlExperiment.experiment_id
            ).filter(SqlExperiment.workspace == workspace)

        if model is SqlLoggedModel:
            workspace_experiment_ids = (
                session.query(SqlExperiment.experiment_id)
                .filter(SqlExperiment.workspace == workspace)
                .subquery()
            )
            return query.filter(
                SqlLoggedModel.experiment_id.in_(select(workspace_experiment_ids.c.experiment_id))
            )

        if model is SqlEvaluationDataset:
            return query.filter(SqlEvaluationDataset.workspace == workspace)

        return query

    def _initialize_store_state(self):
        self._validate_artifact_isolation_constraints()
        self._ensure_default_workspace_experiment()

    def _trace_query(self, session, for_update_or_delete=False):
        if for_update_or_delete:
            workspace = self._get_active_workspace()
            workspace_experiment_ids = (
                session.query(SqlExperiment.experiment_id)
                .filter(SqlExperiment.workspace == workspace)
                .subquery()
            )
            return SqlAlchemyStore._get_query(self, session, SqlTraceInfo).filter(
                SqlTraceInfo.experiment_id.in_(select(workspace_experiment_ids.c.experiment_id))
            )
        return super()._trace_query(session, for_update_or_delete=False)

    def _experiment_where_clauses(self):
        return [SqlExperiment.workspace == self._get_active_workspace()]

    def _filter_experiment_ids(self, session, experiment_ids):
        workspace = self._get_active_workspace()
        rows = (
            session.query(SqlExperiment.experiment_id)
            .filter(
                SqlExperiment.experiment_id.in_(experiment_ids),
                SqlExperiment.workspace == workspace,
            )
            .all()
        )
        return [row[0] for row in rows]

    def _filter_entity_ids(
        self, session, entity_type: EntityAssociationType, entity_ids: list[str]
    ):
        workspace = self._get_active_workspace()
        if not entity_ids:
            return []

        def _rows_to_strings(rows):
            return [str(row[0]) for row in rows]

        if entity_type == EntityAssociationType.EXPERIMENT:
            rows = (
                session.query(SqlExperiment.experiment_id)
                .filter(
                    SqlExperiment.experiment_id.in_(entity_ids),
                    SqlExperiment.workspace == workspace,
                )
                .all()
            )
            return _rows_to_strings(rows)

        if entity_type == EntityAssociationType.RUN:
            rows = (
                session.query(SqlRun.run_uuid)
                .join(SqlExperiment, SqlRun.experiment_id == SqlExperiment.experiment_id)
                .filter(SqlRun.run_uuid.in_(entity_ids), SqlExperiment.workspace == workspace)
                .all()
            )
            return _rows_to_strings(rows)

        if entity_type == EntityAssociationType.TRACE:
            rows = (
                session.query(SqlTraceInfo.request_id)
                .join(SqlExperiment, SqlTraceInfo.experiment_id == SqlExperiment.experiment_id)
                .filter(
                    SqlTraceInfo.request_id.in_(entity_ids),
                    SqlExperiment.workspace == workspace,
                )
                .all()
            )
            return _rows_to_strings(rows)

        if entity_type == EntityAssociationType.EVALUATION_DATASET:
            rows = (
                session.query(SqlEvaluationDataset.dataset_id)
                .filter(
                    SqlEvaluationDataset.dataset_id.in_(entity_ids),
                    SqlEvaluationDataset.workspace == workspace,
                )
                .all()
            )
            return _rows_to_strings(rows)

        return []

    def _filter_association_query(self, session, query, target_type, id_column):
        """Filter entity associations to only include targets in the active workspace."""
        workspace = self._get_active_workspace()

        if target_type == EntityAssociationType.EXPERIMENT:
            # Cast experiment_id to String to match the String type of
            # SqlEntityAssociation.destination_id. PostgreSQL requires explicit type
            # matching for IN comparisons.
            subquery = (
                session.query(
                    sql.cast(SqlExperiment.experiment_id, sqlalchemy.String).label("experiment_id")
                )
                .filter(SqlExperiment.workspace == workspace)
                .subquery()
            )
            id_source = subquery.c.experiment_id
        elif target_type == EntityAssociationType.RUN:
            subquery = (
                session.query(SqlRun.run_uuid)
                .join(SqlExperiment, SqlRun.experiment_id == SqlExperiment.experiment_id)
                .filter(SqlExperiment.workspace == workspace)
                .subquery()
            )
            id_source = subquery.c.run_uuid
        elif target_type == EntityAssociationType.TRACE:
            subquery = (
                session.query(SqlTraceInfo.request_id)
                .join(SqlExperiment, SqlTraceInfo.experiment_id == SqlExperiment.experiment_id)
                .filter(SqlExperiment.workspace == workspace)
                .subquery()
            )
            id_source = subquery.c.request_id
        elif target_type == EntityAssociationType.EVALUATION_DATASET:
            subquery = (
                session.query(SqlEvaluationDataset.dataset_id)
                .filter(SqlEvaluationDataset.workspace == workspace)
                .subquery()
            )
            id_source = subquery.c.dataset_id
        else:
            return query

        return query.filter(id_column.in_(select(id_source)))

    def _validate_run_accessible(self, session, run_id: str) -> None:
        workspace = self._get_active_workspace()
        exists_row = (
            session.query(SqlRun.run_uuid)
            .filter(SqlRun.run_uuid == run_id)
            .filter(
                SqlRun.experiment_id.in_(
                    session.query(SqlExperiment.experiment_id).filter(
                        SqlExperiment.workspace == workspace
                    )
                )
            )
            .first()
        )
        if exists_row is None:
            raise MlflowException(
                f"Run with id={run_id} not found",
                RESOURCE_DOES_NOT_EXIST,
            )

    def _validate_trace_accessible(self, session, trace_id: str) -> None:
        workspace = self._get_active_workspace()
        exists_row = (
            session.query(SqlTraceInfo.request_id)
            .filter(SqlTraceInfo.request_id == trace_id)
            .filter(
                SqlTraceInfo.experiment_id.in_(
                    session.query(SqlExperiment.experiment_id).filter(
                        SqlExperiment.workspace == workspace
                    )
                )
            )
            .first()
        )
        if exists_row is None:
            raise MlflowException(
                f"Trace with ID '{trace_id}' not found.",
                RESOURCE_DOES_NOT_EXIST,
            )

    def _validate_dataset_accessible(self, session, dataset_id: str) -> None:
        workspace = self._get_active_workspace()
        exists_row = (
            session.query(SqlEvaluationDataset.dataset_id)
            .filter(SqlEvaluationDataset.dataset_id == dataset_id)
            .filter(SqlEvaluationDataset.workspace == workspace)
            .first()
        )
        if exists_row is None:
            raise MlflowException(
                f"Dataset '{dataset_id}' not found.",
                RESOURCE_DOES_NOT_EXIST,
            )

    def _get_sql_assessment(self, session, trace_id: str, assessment_id: str) -> SqlAssessments:
        trace_subquery = (
            self._trace_query(session)
            .with_entities(SqlTraceInfo.request_id)
            .filter(SqlTraceInfo.request_id == trace_id)
            .subquery()
        )

        sql_assessment = (
            session.query(SqlAssessments)
            .join(trace_subquery, SqlAssessments.trace_id == trace_subquery.c.request_id)
            .filter(SqlAssessments.assessment_id == assessment_id)
            .one_or_none()
        )
        if sql_assessment is None:
            trace_record = (
                self._trace_query(session).filter(SqlTraceInfo.request_id == trace_id).one_or_none()
            )
            if trace_record is None:
                raise MlflowException(
                    f"Trace with ID '{trace_id}' not found.",
                    RESOURCE_DOES_NOT_EXIST,
                )

            raise MlflowException(
                f"Assessment with ID '{assessment_id}' not found for trace '{trace_id}'",
                RESOURCE_DOES_NOT_EXIST,
            )
        return sql_assessment

    def _with_workspace_field(self, instance):
        if hasattr(instance, "workspace"):
            instance.workspace = self._get_active_workspace()
        return instance

    def _get_workspace_provider_instance(self):
        if self._workspace_provider is None:
            workspace_uri = workspace_utils.resolve_workspace_store_uri(tracking_uri=self.db_uri)
            self._workspace_provider = get_workspace_store(workspace_uri=workspace_uri)
        return self._workspace_provider

    def _validate_artifact_isolation_constraints(self) -> None:
        """Ensure the default artifact root and existing artifacts do not occupy reserved paths."""

        segments = self._artifact_path_segments(self.artifact_root_uri.rstrip("/"))
        if segments and segments[-1] == WORKSPACES_DIR_NAME:
            raise MlflowException(
                "Cannot enable workspace mode because the default artifact root "
                + f"{self.artifact_root_uri} ends with the reserved '{WORKSPACES_DIR_NAME}' "
                + "segment. Choose a different artifact root before enabling workspaces.",
                error_code=INVALID_STATE,
            )
        if len(segments) >= 2 and segments[-2] == WORKSPACES_DIR_NAME:
            raise MlflowException(
                "Cannot enable workspace mode because the default artifact root "
                + f"{self.artifact_root_uri} is already scoped under the reserved "
                + f"'{WORKSPACES_DIR_NAME}/<name>' prefix. Configure a different artifact root "
                + "before enabling workspaces.",
                error_code=INVALID_STATE,
            )

        reserved_prefix = (
            append_to_uri_path(self.artifact_root_uri, WORKSPACES_DIR_NAME).rstrip("/") + "/"
        )
        reserved_prefix_windows = reserved_prefix.replace("/", "\\")
        with self.ManagedSessionMaker() as session:
            has_non_default_workspace = (
                session.query(SqlExperiment.experiment_id)
                .filter(SqlExperiment.workspace != DEFAULT_WORKSPACE_NAME)
                .first()
                is not None
            )
            if has_non_default_workspace:
                # Database already contains workspace data; skip further validation.
                return

            conflict_row = (
                session.query(SqlExperiment.artifact_location)
                .filter(SqlExperiment.name != Experiment.DEFAULT_EXPERIMENT_NAME)
                .filter(SqlExperiment.artifact_location.isnot(None))
                .filter(
                    or_(
                        SqlExperiment.artifact_location.like(f"{reserved_prefix}%"),
                        SqlExperiment.artifact_location.like(f"{reserved_prefix_windows}%"),
                    )
                )
                .order_by(SqlExperiment.experiment_id)
                .first()
            )
            if conflict_row:
                raise MlflowException(
                    f"Cannot enable workspace mode because existing experiment artifact location "
                    f"'{conflict_row[0]}' already resides under the reserved '{reserved_prefix}' "
                    "namespace. Move or rename the artifacts before enabling workspaces.",
                    error_code=INVALID_STATE,
                )

    def _ensure_default_workspace_experiment(self) -> None:
        """
        Ensure the default experiment exists in the provider's default workspace when enabled.
        """

        provider = self._get_workspace_provider_instance()
        default_workspace, supports_default = get_default_workspace_optional(provider)

        if not supports_default:
            provider_name = (
                type(self._workspace_provider).__name__ if self._workspace_provider else "unknown"
            )
            _logger.warning(
                "Workspace provider %s does not expose a default workspace; "
                "skipping default experiment bootstrap.",
                provider_name,
            )
            return

        if default_workspace is None:
            return

        with workspace_context.WorkspaceContext(default_workspace.name):
            if self.get_experiment_by_name(Experiment.DEFAULT_EXPERIMENT_NAME) is None:
                with self.ManagedSessionMaker() as session:
                    self._create_default_experiment(
                        session, workspace_override=default_workspace.name
                    )

    def _create_default_experiment(self, session, workspace_override: str | None = None):
        workspace = workspace_override or self._get_active_workspace()

        if workspace == DEFAULT_WORKSPACE_NAME:
            # Use the context to create the default experiment in the default workspace
            # in case the default workspace was a workspace override. It's important to keep the
            # default workspace experiment ID as 0 to allow a user to disable workspaces later.
            with workspace_context.WorkspaceContext(workspace):
                return super()._create_default_experiment(session)

        creation_time = get_current_time_millis()
        existing = (
            session.query(SqlExperiment)
            .filter(
                SqlExperiment.name == Experiment.DEFAULT_EXPERIMENT_NAME,
                SqlExperiment.workspace == workspace,
            )
            .one_or_none()
        )
        if existing is not None:
            return

        experiment = SqlExperiment(
            name=Experiment.DEFAULT_EXPERIMENT_NAME,
            lifecycle_stage=LifecycleStage.ACTIVE,
            artifact_location=None,
            creation_time=creation_time,
            last_update_time=creation_time,
            workspace=workspace,
        )
        session.add(experiment)
        try:
            session.flush()
        except IntegrityError as exc:
            session.rollback()
            _logger.debug(
                "Default experiment already exists for workspace '%s'; another worker likely "
                "created it. Swallowing IntegrityError: %s",
                workspace,
                exc,
            )
            return

        if not experiment.artifact_location:
            experiment.artifact_location = self._get_artifact_location(
                experiment.experiment_id, workspace
            )
            session.flush()

    def _get_artifact_location(self, experiment_id, workspace: str | None = None):
        workspace = workspace or self._get_active_workspace()
        base_root = self.artifact_root_uri

        serving_artifacts = os.environ.get("_MLFLOW_SERVER_SERVE_ARTIFACTS", "").lower() == "true"
        if serving_artifacts:
            scoped_root = append_to_uri_path(base_root, WORKSPACES_DIR_NAME)
            base_root = append_to_uri_path(scoped_root, workspace)
        else:
            resolved_root = self.artifact_root_uri
            should_append = True
            provider = self._get_workspace_provider_instance()
            if provider and hasattr(provider, "resolve_artifact_root"):
                resolved_root, should_append = provider.resolve_artifact_root(
                    self.artifact_root_uri, workspace
                )
            resolved_root = resolved_root or base_root
            scoped_root = resolved_root
            if should_append:
                scoped_root = append_to_uri_path(scoped_root, WORKSPACES_DIR_NAME)
                scoped_root = append_to_uri_path(scoped_root, workspace)
            base_root = scoped_root

        return append_to_uri_path(base_root, str(experiment_id))

    def create_experiment(self, name, artifact_location=None, tags=None):
        if artifact_location:
            raise MlflowException.invalid_parameter_value(
                "artifact_location cannot be specified when workspaces are enabled"
            )
        return super().create_experiment(name, artifact_location=None, tags=tags)

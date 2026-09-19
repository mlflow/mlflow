from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import Session, aliased

from mlflow.entities.entity_type import EntityAssociationType
from mlflow.entities.gateway_endpoint import GatewayResourceType
from mlflow.store.tracking.dbmodels.models import (
    SqlEntityAssociation,
    SqlEvaluationDataset,
    SqlExperiment,
    SqlGatewayEndpointBinding,
    SqlGatewayGuardrailConfig,
    SqlInput,
    SqlInputTag,
    SqlLoggedModel,
    SqlRun,
    SqlScorer,
    SqlTraceInfo,
)


def delete_entity_associations(session: Session, entity_type: str, entity_ids) -> None:
    session.query(SqlEntityAssociation).filter(
        sa.or_(
            sa.and_(
                SqlEntityAssociation.source_type == entity_type,
                SqlEntityAssociation.source_id.in_(entity_ids),
            ),
            sa.and_(
                SqlEntityAssociation.destination_type == entity_type,
                SqlEntityAssociation.destination_id.in_(entity_ids),
            ),
        )
    ).delete(synchronize_session=False)


def delete_scorer_endpoint_bindings(session: Session, scorer_ids) -> None:
    session.query(SqlGatewayEndpointBinding).filter(
        SqlGatewayEndpointBinding.resource_type == GatewayResourceType.SCORER.value,
        SqlGatewayEndpointBinding.resource_id.in_(scorer_ids),
    ).delete(synchronize_session=False)


def delete_workspace_dependencies(session: Session, workspace_name: str) -> None:
    """Delete tracking-owned soft references before their workspace roots are removed.

    The selectors below depend on the workspace roots still being present. This helper must use the
    caller's transaction and must not flush or commit. Prompt-version association IDs contain only
    ``name/version`` and are not workspace-unique, so those rows are selected through their trace
    source rather than through the prompt destination. Dataset and input UUIDs are also not unique;
    inputs are selected through their run or logged-model owner, and tags are retained while any
    surviving input shares their UUID.
    """
    experiment_ids = sa.select(SqlExperiment.experiment_id).where(
        SqlExperiment.workspace == workspace_name
    )
    experiment_ids_as_strings = sa.select(sa.cast(SqlExperiment.experiment_id, sa.String)).where(
        SqlExperiment.workspace == workspace_name
    )
    run_ids = sa.select(SqlRun.run_uuid).where(SqlRun.experiment_id.in_(experiment_ids))
    trace_ids = sa.select(SqlTraceInfo.request_id).where(
        SqlTraceInfo.experiment_id.in_(experiment_ids)
    )
    evaluation_dataset_ids = sa.select(SqlEvaluationDataset.dataset_id).where(
        SqlEvaluationDataset.workspace == workspace_name
    )
    logged_model_ids = sa.select(SqlLoggedModel.model_id).where(
        SqlLoggedModel.experiment_id.in_(experiment_ids)
    )
    scorer_ids = sa.select(SqlScorer.scorer_id).where(SqlScorer.experiment_id.in_(experiment_ids))

    for entity_type, entity_ids in (
        (EntityAssociationType.EXPERIMENT, experiment_ids_as_strings),
        (EntityAssociationType.RUN, run_ids),
        (EntityAssociationType.TRACE, trace_ids),
        (EntityAssociationType.EVALUATION_DATASET, evaluation_dataset_ids),
    ):
        delete_entity_associations(session, entity_type, entity_ids)

    def workspace_input_filter(input_model):
        return sa.or_(
            sa.and_(
                input_model.destination_type == "RUN",
                input_model.destination_id.in_(run_ids),
            ),
            sa.and_(
                input_model.source_type.in_(["RUN_INPUT", "RUN_OUTPUT"]),
                input_model.source_id.in_(run_ids),
            ),
            sa.and_(
                input_model.destination_type.in_(["MODEL_INPUT", "MODEL_OUTPUT"]),
                input_model.destination_id.in_(logged_model_ids),
            ),
        )

    target_inputs = aliased(SqlInput)
    surviving_inputs = aliased(SqlInput)
    target_input_ids = sa.select(target_inputs.input_uuid).where(
        workspace_input_filter(target_inputs)
    )
    surviving_input_exists = sa.exists(
        sa.select(1).where(
            surviving_inputs.input_uuid == SqlInputTag.input_uuid,
            sa.not_(workspace_input_filter(surviving_inputs)),
        )
    )
    session.query(SqlInputTag).filter(
        SqlInputTag.input_uuid.in_(target_input_ids),
        sa.not_(surviving_input_exists),
    ).delete(synchronize_session=False)
    session.query(SqlInput).filter(workspace_input_filter(SqlInput)).delete(
        synchronize_session=False
    )

    delete_scorer_endpoint_bindings(session, scorer_ids)


def reassign_workspace_dependencies(
    session: Session, source_workspace: str, target_workspace: str
) -> None:
    session.query(SqlGatewayGuardrailConfig).filter(
        SqlGatewayGuardrailConfig.workspace == source_workspace
    ).update(
        {SqlGatewayGuardrailConfig.workspace: target_workspace},
        synchronize_session=False,
    )

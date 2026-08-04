from __future__ import annotations

import uuid
from dataclasses import dataclass

import pytest
import sqlalchemy as sa

from mlflow.entities.entity_type import EntityAssociationType
from mlflow.entities.gateway_endpoint import GatewayResourceType
from mlflow.entities.mcp_server import MCPStatus
from mlflow.entities.workspace import Workspace, WorkspaceDeletionMode
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.store.tracking.dbmodels.models import (
    SqlEntityAssociation,
    SqlExperiment,
    SqlGatewayEndpoint,
    SqlGatewayEndpointBinding,
    SqlGatewayEndpointModelMapping,
    SqlGatewayGuardrail,
    SqlGatewayGuardrailConfig,
    SqlGatewayModelDefinition,
    SqlInput,
    SqlInputTag,
    SqlMCPServer,
    SqlMCPServerVersion,
    SqlMCPServerVersionTag,
    SqlRun,
    SqlScorer,
    SqlScorerVersion,
)
from mlflow.store.workspace.dbmodels.models import SqlWorkspace
from mlflow.store.workspace.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

pytestmark = pytest.mark.notrackingurimock


@dataclass(frozen=True)
class _Resources:
    token: str

    @property
    def workspace(self) -> str:
        return f"ws-{self.token}"

    @property
    def experiment_name(self) -> str:
        return f"exp-{self.token}"

    @property
    def scorer_id(self) -> str:
        return f"sc-{self.token}"

    @property
    def endpoint_id(self) -> str:
        return f"ep-{self.token}"

    @property
    def guardrail_id(self) -> str:
        return f"gr-{self.token}"

    @property
    def model_definition_id(self) -> str:
        return f"md-{self.token}"

    @property
    def mapping_id(self) -> str:
        return f"map-{self.token}"

    @property
    def mcp_name(self) -> str:
        return f"io.mlflow.test/{self.token}"

    @property
    def association_id(self) -> str:
        return f"a-{self.token}"

    @property
    def neighbor_association_id(self) -> str:
        return f"a-neighbor-{self.token}"

    @property
    def neighbor_experiment_name(self) -> str:
        return f"neighbor-exp-{self.token}"

    @property
    def neighbor_scorer_id(self) -> str:
        return f"neighbor-sc-{self.token}"

    @property
    def neighbor_endpoint_id(self) -> str:
        return f"neighbor-ep-{self.token}"

    @property
    def target_run_id(self) -> str:
        return f"target-run-{self.token}"

    @property
    def neighbor_run_id(self) -> str:
        return f"neighbor-run-{self.token}"

    @property
    def shared_input_id(self) -> str:
        return f"shared-input-{self.token}"

    @property
    def target_input_id(self) -> str:
        return f"target-input-{self.token}"


@pytest.fixture
def workspace_store():
    store = SqlAlchemyStore(MLFLOW_TRACKING_URI.get())
    try:
        yield store
    finally:
        store._engine.dispose()


def _new_resources() -> _Resources:
    return _Resources(uuid.uuid4().hex[:12])


def _add_scorer_guardrail_graph(session, resources: _Resources, *, add_config: bool) -> int:
    experiment = SqlExperiment(
        name=resources.experiment_name,
        lifecycle_stage="active",
        workspace=resources.workspace,
    )
    session.add(experiment)
    session.flush()

    session.add_all([
        SqlScorer(
            experiment_id=experiment.experiment_id,
            scorer_name=f"scorer-{resources.token}",
            scorer_id=resources.scorer_id,
        ),
        SqlGatewayEndpoint(
            endpoint_id=resources.endpoint_id,
            name=f"endpoint-{resources.token}",
            created_at=0,
            last_updated_at=0,
            usage_tracking=False,
            workspace=resources.workspace,
        ),
    ])
    session.flush()

    session.add(
        SqlScorerVersion(
            scorer_id=resources.scorer_id,
            scorer_version=1,
            serialized_scorer="{}",
            creation_time=0,
        )
    )
    session.flush()

    session.add(
        SqlGatewayGuardrail(
            guardrail_id=resources.guardrail_id,
            name=f"guardrail-{resources.token}",
            scorer_id=resources.scorer_id,
            scorer_version=1,
            stage="BEFORE",
            action="VALIDATION",
            created_at=0,
            last_updated_at=0,
            workspace=resources.workspace,
        )
    )
    session.flush()

    if add_config:
        session.add(
            SqlGatewayGuardrailConfig(
                endpoint_id=resources.endpoint_id,
                guardrail_id=resources.guardrail_id,
                execution_order=0,
                created_at=0,
                workspace=resources.workspace,
            )
        )
        session.flush()

    return experiment.experiment_id


def _cleanup_exact_rows(store: SqlAlchemyStore, resources: _Resources) -> None:
    with store.ManagedSessionMaker(read_only=False) as session:
        session.execute(
            sa.delete(SqlGatewayGuardrailConfig).where(
                SqlGatewayGuardrailConfig.guardrail_id == resources.guardrail_id
            )
        )
        session.execute(
            sa.delete(SqlEntityAssociation).where(
                SqlEntityAssociation.association_id.in_([
                    resources.association_id,
                    resources.neighbor_association_id,
                ])
            )
        )
        session.execute(
            sa.delete(SqlInputTag).where(
                SqlInputTag.input_uuid.in_([
                    resources.shared_input_id,
                    resources.target_input_id,
                ])
            )
        )
        session.execute(
            sa.delete(SqlInput).where(
                SqlInput.input_uuid.in_([
                    resources.shared_input_id,
                    resources.target_input_id,
                ])
            )
        )
        session.execute(
            sa.delete(SqlGatewayEndpointBinding).where(
                SqlGatewayEndpointBinding.endpoint_id == resources.neighbor_endpoint_id
            )
        )
        session.execute(
            sa.delete(SqlGatewayGuardrail).where(
                SqlGatewayGuardrail.guardrail_id == resources.guardrail_id
            )
        )
        session.execute(
            sa.delete(SqlGatewayEndpointModelMapping).where(
                SqlGatewayEndpointModelMapping.mapping_id == resources.mapping_id
            )
        )
        session.execute(
            sa.delete(SqlGatewayEndpoint).where(
                SqlGatewayEndpoint.endpoint_id.in_([
                    resources.endpoint_id,
                    resources.neighbor_endpoint_id,
                ])
            )
        )
        session.execute(
            sa.delete(SqlGatewayModelDefinition).where(
                SqlGatewayModelDefinition.model_definition_id == resources.model_definition_id
            )
        )
        session.execute(
            sa.delete(SqlScorerVersion).where(
                SqlScorerVersion.scorer_id.in_([
                    resources.scorer_id,
                    resources.neighbor_scorer_id,
                ])
            )
        )
        session.execute(
            sa.delete(SqlScorer).where(
                SqlScorer.scorer_id.in_([
                    resources.scorer_id,
                    resources.neighbor_scorer_id,
                ])
            )
        )
        session.execute(
            sa.delete(SqlMCPServerVersionTag).where(
                SqlMCPServerVersionTag.name == resources.mcp_name
            )
        )
        session.execute(
            sa.delete(SqlMCPServerVersion).where(SqlMCPServerVersion.name == resources.mcp_name)
        )
        session.execute(sa.delete(SqlMCPServer).where(SqlMCPServer.name == resources.mcp_name))
        session.execute(
            sa.delete(SqlRun).where(
                SqlRun.run_uuid.in_([resources.target_run_id, resources.neighbor_run_id])
            )
        )
        session.execute(
            sa.delete(SqlExperiment).where(
                SqlExperiment.name.in_([
                    resources.experiment_name,
                    resources.neighbor_experiment_name,
                ])
            )
        )
        session.execute(sa.delete(SqlWorkspace).where(SqlWorkspace.name == resources.workspace))


def test_set_default_reassigns_explicit_and_composite_fk_dependents(workspace_store):
    resources = _new_resources()
    workspace_store.create_workspace(Workspace(name=resources.workspace))

    try:
        with workspace_store.ManagedSessionMaker(read_only=False) as session:
            experiment_id = _add_scorer_guardrail_graph(session, resources, add_config=True)
            session.add_all([
                SqlMCPServer(
                    workspace=resources.workspace,
                    name=resources.mcp_name,
                    created_at=0,
                    last_updated_at=0,
                ),
                SqlMCPServerVersion(
                    workspace=resources.workspace,
                    name=resources.mcp_name,
                    version="1.0.0",
                    version_major=1,
                    version_minor=0,
                    version_patch=0,
                    version_prerelease_sort_key="",
                    server_json={"name": resources.mcp_name, "version": "1.0.0"},
                    status=MCPStatus.DRAFT.value,
                    created_at=0,
                    last_updated_at=0,
                ),
                SqlMCPServerVersionTag(
                    workspace=resources.workspace,
                    name=resources.mcp_name,
                    version="1.0.0",
                    key="stage",
                    value="test",
                ),
            ])

        workspace_store.delete_workspace(
            resources.workspace,
            mode=WorkspaceDeletionMode.SET_DEFAULT,
        )

        with workspace_store.ManagedSessionMaker() as session:
            assert session.get(SqlWorkspace, resources.workspace) is None
            assert session.get(SqlExperiment, experiment_id).workspace == DEFAULT_WORKSPACE_NAME
            assert (
                session.get(SqlGatewayEndpoint, resources.endpoint_id).workspace
                == DEFAULT_WORKSPACE_NAME
            )
            assert (
                session.get(SqlGatewayGuardrail, resources.guardrail_id).workspace
                == DEFAULT_WORKSPACE_NAME
            )
            assert (
                session.get(
                    SqlGatewayGuardrailConfig,
                    (resources.endpoint_id, resources.guardrail_id),
                ).workspace
                == DEFAULT_WORKSPACE_NAME
            )
            assert (
                session.get(SqlMCPServer, (DEFAULT_WORKSPACE_NAME, resources.mcp_name)) is not None
            )
            assert (
                session.get(
                    SqlMCPServerVersion,
                    (DEFAULT_WORKSPACE_NAME, resources.mcp_name, "1.0.0"),
                )
                is not None
            )
            assert (
                session.get(
                    SqlMCPServerVersionTag,
                    (DEFAULT_WORKSPACE_NAME, resources.mcp_name, "1.0.0", "stage"),
                )
                is not None
            )
    finally:
        _cleanup_exact_rows(workspace_store, resources)


def test_cascade_deletes_hard_fk_graphs_and_owned_soft_edges(workspace_store):
    resources = _new_resources()
    workspace_store.create_workspace(Workspace(name=resources.workspace))

    try:
        with workspace_store.ManagedSessionMaker(read_only=False) as session:
            experiment_id = _add_scorer_guardrail_graph(session, resources, add_config=False)
            neighbor_experiment = SqlExperiment(
                name=resources.neighbor_experiment_name,
                lifecycle_stage="active",
                workspace=DEFAULT_WORKSPACE_NAME,
            )
            session.add(neighbor_experiment)
            session.flush()
            session.add_all([
                SqlGatewayModelDefinition(
                    model_definition_id=resources.model_definition_id,
                    name=f"model-{resources.token}",
                    provider="test",
                    model_name="test-model",
                    created_at=0,
                    last_updated_at=0,
                    workspace=resources.workspace,
                ),
                SqlRun(
                    run_uuid=resources.target_run_id,
                    name=f"target-run-{resources.token}",
                    experiment_id=experiment_id,
                    lifecycle_stage="active",
                    status="FINISHED",
                    source_type="LOCAL",
                    start_time=0,
                    end_time=0,
                ),
                SqlRun(
                    run_uuid=resources.neighbor_run_id,
                    name=f"neighbor-run-{resources.token}",
                    experiment_id=neighbor_experiment.experiment_id,
                    lifecycle_stage="active",
                    status="FINISHED",
                    source_type="LOCAL",
                    start_time=0,
                    end_time=0,
                ),
                SqlScorer(
                    experiment_id=neighbor_experiment.experiment_id,
                    scorer_name=f"neighbor-scorer-{resources.token}",
                    scorer_id=resources.neighbor_scorer_id,
                ),
                SqlGatewayEndpoint(
                    endpoint_id=resources.neighbor_endpoint_id,
                    name=f"neighbor-endpoint-{resources.token}",
                    created_at=0,
                    last_updated_at=0,
                    usage_tracking=False,
                    workspace=DEFAULT_WORKSPACE_NAME,
                ),
                SqlEntityAssociation(
                    association_id=resources.association_id,
                    source_type=EntityAssociationType.EVALUATION_DATASET,
                    source_id=f"dataset-{resources.token}",
                    destination_type=EntityAssociationType.EXPERIMENT,
                    destination_id=str(experiment_id),
                    created_time=0,
                ),
                SqlEntityAssociation(
                    association_id=resources.neighbor_association_id,
                    source_type=EntityAssociationType.EVALUATION_DATASET,
                    source_id=f"neighbor-{resources.token}",
                    destination_type=EntityAssociationType.EXPERIMENT,
                    destination_id=f"neighbor-{resources.token}",
                    created_time=0,
                ),
            ])
            session.flush()
            session.add_all([
                SqlScorerVersion(
                    scorer_id=resources.neighbor_scorer_id,
                    scorer_version=1,
                    serialized_scorer="{}",
                    creation_time=0,
                ),
                SqlGatewayEndpointModelMapping(
                    mapping_id=resources.mapping_id,
                    endpoint_id=resources.endpoint_id,
                    model_definition_id=resources.model_definition_id,
                    weight=1.0,
                    linkage_type="PRIMARY",
                    created_at=0,
                ),
                SqlInput(
                    input_uuid=resources.shared_input_id,
                    source_type="DATASET",
                    source_id=f"shared-dataset-{resources.token}",
                    destination_type="RUN",
                    destination_id=resources.target_run_id,
                ),
                SqlInput(
                    input_uuid=resources.shared_input_id,
                    source_type="DATASET",
                    source_id=f"shared-dataset-{resources.token}",
                    destination_type="RUN",
                    destination_id=resources.neighbor_run_id,
                ),
                SqlInput(
                    input_uuid=resources.target_input_id,
                    source_type="RUN_INPUT",
                    source_id=resources.target_run_id,
                    destination_type="MODEL_INPUT",
                    destination_id=f"model-input-{resources.token}",
                ),
                SqlInputTag(
                    input_uuid=resources.shared_input_id,
                    name="context",
                    value="shared",
                ),
                SqlInputTag(
                    input_uuid=resources.target_input_id,
                    name="context",
                    value="target",
                ),
                SqlGatewayEndpointBinding(
                    endpoint_id=resources.neighbor_endpoint_id,
                    resource_type=GatewayResourceType.SCORER.value,
                    resource_id=resources.scorer_id,
                    created_at=0,
                    last_updated_at=0,
                ),
                SqlGatewayEndpointBinding(
                    endpoint_id=resources.neighbor_endpoint_id,
                    resource_type=GatewayResourceType.SCORER.value,
                    resource_id=resources.neighbor_scorer_id,
                    created_at=0,
                    last_updated_at=0,
                ),
            ])

        workspace_store.delete_workspace(
            resources.workspace,
            mode=WorkspaceDeletionMode.CASCADE,
        )

        with workspace_store.ManagedSessionMaker() as session:
            assert session.get(SqlWorkspace, resources.workspace) is None
            assert session.get(SqlExperiment, experiment_id) is None
            assert session.get(SqlScorer, resources.scorer_id) is None
            assert session.get(SqlScorerVersion, (resources.scorer_id, 1)) is None
            assert session.get(SqlGatewayGuardrail, resources.guardrail_id) is None
            assert session.get(SqlGatewayEndpoint, resources.endpoint_id) is None
            assert session.get(SqlGatewayModelDefinition, resources.model_definition_id) is None
            assert session.get(SqlGatewayEndpointModelMapping, resources.mapping_id) is None
            assert session.get(SqlRun, resources.target_run_id) is None
            assert session.get(SqlRun, resources.neighbor_run_id) is not None
            assert session.get(SqlScorer, resources.neighbor_scorer_id) is not None
            assert session.get(SqlGatewayEndpoint, resources.neighbor_endpoint_id) is not None
            assert (
                session
                .query(SqlEntityAssociation)
                .filter(SqlEntityAssociation.association_id == resources.association_id)
                .one_or_none()
                is None
            )
            assert (
                session
                .query(SqlEntityAssociation)
                .filter(SqlEntityAssociation.association_id == resources.neighbor_association_id)
                .one()
                .destination_id
                == f"neighbor-{resources.token}"
            )
            shared_inputs = (
                session
                .query(SqlInput)
                .filter(SqlInput.input_uuid == resources.shared_input_id)
                .all()
            )
            assert [input_row.destination_id for input_row in shared_inputs] == [
                resources.neighbor_run_id
            ]
            assert (
                session.get(SqlInputTag, (resources.shared_input_id, "context")).value == "shared"
            )
            assert (
                session
                .query(SqlInput)
                .filter(SqlInput.input_uuid == resources.target_input_id)
                .one_or_none()
                is None
            )
            assert session.get(SqlInputTag, (resources.target_input_id, "context")) is None
            bindings = (
                session
                .query(SqlGatewayEndpointBinding)
                .filter(SqlGatewayEndpointBinding.endpoint_id == resources.neighbor_endpoint_id)
                .all()
            )
            assert [binding.resource_id for binding in bindings] == [resources.neighbor_scorer_id]
    finally:
        _cleanup_exact_rows(workspace_store, resources)

from unittest import mock

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from mlflow.entities.entity_type import EntityAssociationType
from mlflow.entities.gateway_endpoint import GatewayResourceType
from mlflow.entities.workspace import Workspace, WorkspaceDeletionMode
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.db import workspace_migration
from mlflow.store.workspace.dbmodels.models import SqlWorkspace
from mlflow.store.workspace.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

from tests.store.tracking.sqlalchemy_store.test_sqlalchemy_store_schema import _insert_row


@pytest.fixture
def workspace_store(db_uri, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "true")

    store = SqlAlchemyStore(db_uri)

    with store.ManagedSessionMaker(read_only=False) as session:
        try:
            session.add(
                SqlWorkspace(
                    name=DEFAULT_WORKSPACE_NAME,
                    description="Default workspace",
                )
            )
            session.commit()
        except IntegrityError:
            session.rollback()

    try:
        yield store
    finally:
        store._engine.dispose()


def _workspace_rows(store):
    with store.ManagedSessionMaker() as session:
        return {
            (row.name, row.description)
            for row in session.query(SqlWorkspace).order_by(SqlWorkspace.name).all()
        }


def _insert_scorer_graph(session, *, workspace, experiment_id, suffix):
    ids = {
        "experiment_id": experiment_id,
        "scorer_id": f"scorer-{suffix}",
        "scorer_version": 1,
    }
    session.execute(
        sa.text(
            "INSERT INTO experiments (experiment_id, name, workspace, lifecycle_stage) "
            "VALUES (:experiment_id, :name, :workspace, 'active')"
        ),
        {
            "experiment_id": experiment_id,
            "name": f"experiment-{suffix}",
            "workspace": workspace,
        },
    )
    session.execute(
        sa.text(
            "INSERT INTO scorers (experiment_id, scorer_name, scorer_id) "
            "VALUES (:experiment_id, :scorer_name, :scorer_id)"
        ),
        {
            "experiment_id": experiment_id,
            "scorer_name": f"scorer-{suffix}",
            "scorer_id": ids["scorer_id"],
        },
    )
    session.execute(
        sa.text(
            "INSERT INTO scorer_versions "
            "(scorer_id, scorer_version, serialized_scorer, creation_time) "
            "VALUES (:scorer_id, :scorer_version, '{}', 0)"
        ),
        ids,
    )
    return ids


def _insert_endpoint(session, *, workspace, suffix):
    endpoint_id = f"endpoint-{suffix}"
    session.execute(
        sa.text(
            "INSERT INTO endpoints "
            "(endpoint_id, name, created_at, last_updated_at, usage_tracking, workspace) "
            "VALUES (:endpoint_id, :name, 0, 0, 0, :workspace)"
        ),
        {
            "endpoint_id": endpoint_id,
            "name": f"endpoint-{suffix}",
            "workspace": workspace,
        },
    )
    return endpoint_id


def _insert_endpoint_model_graph(session, *, workspace, suffix):
    endpoint_id = _insert_endpoint(session, workspace=workspace, suffix=suffix)
    model_definition_id = f"model-definition-{suffix}"
    session.execute(
        sa.text(
            "INSERT INTO model_definitions "
            "(model_definition_id, name, provider, model_name, created_at, last_updated_at, "
            "workspace) VALUES (:model_definition_id, :name, 'openai', 'test-model', 0, 0, "
            ":workspace)"
        ),
        {
            "model_definition_id": model_definition_id,
            "name": f"model-definition-{suffix}",
            "workspace": workspace,
        },
    )
    mapping_id = f"mapping-{suffix}"
    session.execute(
        sa.text(
            "INSERT INTO endpoint_model_mappings "
            "(mapping_id, endpoint_id, model_definition_id, weight, linkage_type, created_at) "
            "VALUES (:mapping_id, :endpoint_id, :model_definition_id, 1.0, 'PRIMARY', 0)"
        ),
        {
            "mapping_id": mapping_id,
            "endpoint_id": endpoint_id,
            "model_definition_id": model_definition_id,
        },
    )
    return endpoint_id, model_definition_id, mapping_id


def _insert_guardrail(
    session,
    *,
    workspace,
    scorer_id,
    scorer_version,
    suffix,
    endpoint_id=None,
):
    guardrail_id = f"guardrail-{suffix}"
    session.execute(
        sa.text(
            "INSERT INTO guardrails "
            "(guardrail_id, name, scorer_id, scorer_version, stage, action, "
            "created_at, last_updated_at, workspace) "
            "VALUES (:guardrail_id, :name, :scorer_id, :scorer_version, "
            "'BEFORE', 'VALIDATION', 0, 0, :workspace)"
        ),
        {
            "guardrail_id": guardrail_id,
            "name": f"guardrail-{suffix}",
            "scorer_id": scorer_id,
            "scorer_version": scorer_version,
            "workspace": workspace,
        },
    )
    if endpoint_id is not None:
        session.execute(
            sa.text(
                "INSERT INTO guardrail_configs "
                "(endpoint_id, guardrail_id, execution_order, created_at, workspace) "
                "VALUES (:endpoint_id, :guardrail_id, 0, 0, :workspace)"
            ),
            {
                "endpoint_id": endpoint_id,
                "guardrail_id": guardrail_id,
                "workspace": workspace,
            },
        )
    return guardrail_id


def _insert_tracking_graph(
    session,
    *,
    workspace,
    experiment_id,
    suffix,
    dataset_uuid=None,
):
    ids = _insert_scorer_graph(
        session,
        workspace=workspace,
        experiment_id=experiment_id,
        suffix=suffix,
    )
    ids.update({
        "run_id": f"run-{suffix}",
        "trace_id": f"trace-{suffix}",
        "dataset_uuid": dataset_uuid or f"dataset-{suffix}",
        "evaluation_dataset_id": f"eval-{suffix}",
        "model_id": f"model-{suffix}",
    })
    session.execute(
        sa.text(
            "INSERT INTO runs "
            "(run_uuid, name, experiment_id, lifecycle_stage, status, source_type, "
            "start_time, end_time) "
            "VALUES (:run_id, :name, :experiment_id, 'active', 'FINISHED', 'LOCAL', 0, 0)"
        ),
        {**ids, "name": f"run-{suffix}"},
    )
    session.execute(
        sa.text(
            "INSERT INTO trace_info "
            "(request_id, experiment_id, timestamp_ms, execution_time_ms, status) "
            "VALUES (:trace_id, :experiment_id, 0, 0, 'OK')"
        ),
        ids,
    )
    session.execute(
        sa.text(
            "INSERT INTO datasets "
            "(dataset_uuid, experiment_id, name, digest, dataset_source_type, dataset_source) "
            "VALUES (:dataset_uuid, :experiment_id, :name, :digest, 'code', '{}')"
        ),
        {**ids, "name": f"dataset-{suffix}", "digest": f"digest-{suffix}"},
    )
    session.execute(
        sa.text(
            "INSERT INTO evaluation_datasets (dataset_id, name, workspace) "
            "VALUES (:evaluation_dataset_id, :name, :workspace)"
        ),
        {**ids, "name": f"evaluation-dataset-{suffix}", "workspace": workspace},
    )
    session.execute(
        sa.text(
            "INSERT INTO logged_models "
            "(model_id, experiment_id, name, artifact_location, creation_timestamp_ms, "
            "last_updated_timestamp_ms, status, lifecycle_stage) "
            "VALUES (:model_id, :experiment_id, :name, '/tmp/model', 0, 0, 2, 'active')"
        ),
        {**ids, "name": f"model-{suffix}"},
    )
    return ids


def _insert_soft_edges(session, ids, *, suffix):
    association_ids = {
        f"association-{suffix}-dataset",
        f"association-{suffix}-run",
        f"association-{suffix}-prompt",
    }
    session.execute(
        sa.text(
            "INSERT INTO entity_associations "
            "(association_id, source_type, source_id, destination_type, destination_id) "
            "VALUES (:association_id, :source_type, :source_id, "
            ":destination_type, :destination_id)"
        ),
        [
            {
                "association_id": f"association-{suffix}-dataset",
                "source_type": EntityAssociationType.EVALUATION_DATASET,
                "source_id": ids["evaluation_dataset_id"],
                "destination_type": EntityAssociationType.EXPERIMENT,
                "destination_id": str(ids["experiment_id"]),
            },
            {
                "association_id": f"association-{suffix}-run",
                "source_type": EntityAssociationType.TRACE,
                "source_id": ids["trace_id"],
                "destination_type": EntityAssociationType.RUN,
                "destination_id": ids["run_id"],
            },
            {
                "association_id": f"association-{suffix}-prompt",
                "source_type": EntityAssociationType.TRACE,
                "source_id": ids["trace_id"],
                "destination_type": EntityAssociationType.PROMPT_VERSION,
                "destination_id": "shared-prompt/1",
            },
        ],
    )

    input_ids = {
        f"input-{suffix}-dataset",
        f"input-{suffix}-model-input",
        f"input-{suffix}-model-output",
    }
    session.execute(
        sa.text(
            "INSERT INTO inputs "
            "(input_uuid, source_type, source_id, destination_type, destination_id) "
            "VALUES (:input_uuid, :source_type, :source_id, "
            ":destination_type, :destination_id)"
        ),
        [
            {
                "input_uuid": f"input-{suffix}-dataset",
                "source_type": "DATASET",
                "source_id": ids["dataset_uuid"],
                "destination_type": "RUN",
                "destination_id": ids["run_id"],
            },
            {
                "input_uuid": f"input-{suffix}-model-input",
                "source_type": "RUN_INPUT",
                "source_id": ids["run_id"],
                "destination_type": "MODEL_INPUT",
                "destination_id": ids["model_id"],
            },
            {
                "input_uuid": f"input-{suffix}-model-output",
                "source_type": "RUN_OUTPUT",
                "source_id": ids["run_id"],
                "destination_type": "MODEL_OUTPUT",
                "destination_id": ids["model_id"],
            },
        ],
    )
    session.execute(
        sa.text(
            "INSERT INTO input_tags (input_uuid, name, value) "
            "VALUES (:input_uuid, 'context', :value)"
        ),
        [{"input_uuid": input_uuid, "value": suffix} for input_uuid in sorted(input_ids)],
    )
    return association_ids, input_ids


def _select_values(store, statement, params=None):
    with store.ManagedSessionMaker() as session:
        return {row[0] for row in session.execute(sa.text(statement), params or {}).all()}


def _insert_all_workspace_table_rows(session, *, workspace, seed):
    conn = session.connection()
    table_order = [
        "experiments",
        "registered_models",
        "model_versions",
        "registered_model_tags",
        "model_version_tags",
        "registered_model_aliases",
        "evaluation_datasets",
        "webhooks",
        "secrets",
        "endpoints",
        "model_definitions",
        "budget_policies",
        "jobs",
        "mcp_servers",
        "mcp_server_versions",
        "mcp_server_tags",
        "mcp_server_version_tags",
        "mcp_server_aliases",
        "mcp_access_endpoints",
        "guardrails",
        "guardrail_configs",
    ]
    assert set(table_order) == set(workspace_migration._WORKSPACE_TABLES)

    _insert_row(conn, "experiments", workspace, seed=seed)
    experiment_id = session.execute(
        sa.text("SELECT experiment_id FROM experiments WHERE name = :name"),
        {"name": f"experiment_{seed}"},
    ).scalar_one()
    session.execute(
        sa.text(
            "INSERT INTO scorers (experiment_id, scorer_name, scorer_id) "
            "VALUES (:experiment_id, :scorer_name, :scorer_id)"
        ),
        {
            "experiment_id": experiment_id,
            "scorer_name": f"scorer_{seed}",
            "scorer_id": f"scorer_{seed}",
        },
    )
    session.execute(
        sa.text(
            "INSERT INTO scorer_versions "
            "(scorer_id, scorer_version, serialized_scorer, creation_time) "
            "VALUES (:scorer_id, :scorer_version, '{}', :creation_time)"
        ),
        {
            "scorer_id": f"scorer_{seed}",
            "scorer_version": seed,
            "creation_time": seed,
        },
    )
    for table_name in table_order[1:]:
        _insert_row(conn, table_name, workspace, seed=seed)


def _workspace_table_counts(session, workspace):
    conn = session.connection()
    inspector = sa.inspect(conn)
    table_names = sorted(
        table_name
        for table_name in inspector.get_table_names()
        if any(column["name"] == "workspace" for column in inspector.get_columns(table_name))
    )
    counts = {}
    for table_name in table_names:
        table = sa.Table(table_name, sa.MetaData(), autoload_with=conn)
        counts[table_name] = session.execute(
            sa.select(sa.func.count()).select_from(table).where(table.c.workspace == workspace)
        ).scalar_one()
    return counts


def test_list_workspaces_returns_all(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="Team A"))
    workspace_store.create_workspace(Workspace(name="team-b", description=None))

    workspaces = workspace_store.list_workspaces()
    rows = {(ws.name, ws.description) for ws in workspaces}
    default_description = next(desc for name, desc in rows if name == DEFAULT_WORKSPACE_NAME)
    assert rows == {
        (DEFAULT_WORKSPACE_NAME, default_description),
        ("team-a", "Team A"),
        ("team-b", None),
    }


def test_get_workspace_success(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="Team A"))

    workspace = workspace_store.get_workspace("team-a")
    assert workspace.name == "team-a"
    assert workspace.description == "Team A"


def test_get_workspace_not_found(workspace_store):
    with pytest.raises(MlflowException, match="Workspace 'unknown' not found") as exc:
        workspace_store.get_workspace("unknown")
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_create_workspace_persists_record(workspace_store):
    created = workspace_store.create_workspace(
        Workspace(
            name="team-a",
            description="Team A",
            default_artifact_root="s3://root/team-a",
            trace_archival_location="s3://archive/team-a",
            trace_archival_retention="30d",
        ),
    )
    assert created.name == "team-a"
    assert created.description == "Team A"
    assert created.default_artifact_root == "s3://root/team-a"
    assert created.trace_archival_location == "s3://archive/team-a"
    assert created.trace_archival_retention == "30d"
    assert ("team-a", "Team A") in _workspace_rows(workspace_store)


def test_create_workspace_duplicate_raises(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    with pytest.raises(
        MlflowException,
        match="Workspace 'team-a' already exists\\.",
    ) as exc:
        workspace_store.create_workspace(Workspace(name="team-a", description=None))
    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"


def test_create_workspace_invalid_name_raises(workspace_store):
    with pytest.raises(
        MlflowException,
        match="Workspace name 'Team-A' must match the pattern",
    ) as exc:
        workspace_store.create_workspace(Workspace(name="Team-A", description=None))
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_create_workspace_invalid_trace_archival_location_raises(workspace_store):
    with pytest.raises(MlflowException, match="proxy-only `mlflow-artifacts:` scheme") as exc:
        workspace_store.create_workspace(
            Workspace(
                name="team-a",
                description=None,
                trace_archival_location="mlflow-artifacts:/archive/team-a",
            )
        )
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_create_workspace_unsupported_trace_archival_location_raises(workspace_store):
    class UnsupportedArchiveRepo(ArtifactRepository):
        def log_artifact(self, local_file, artifact_path=None):
            raise NotImplementedError

        def log_artifacts(self, local_dir, artifact_path=None):
            raise NotImplementedError

        def list_artifacts(self, path=None):
            raise NotImplementedError

    with mock.patch(
        "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
        return_value=UnsupportedArchiveRepo("dbfs:/archive/team-a"),
    ):
        with pytest.raises(
            MlflowException,
            match="does not support deleting archived payloads",
        ) as exc:
            workspace_store.create_workspace(
                Workspace(
                    name="team-a",
                    description=None,
                    trace_archival_location="dbfs:/archive/team-a",
                )
            )
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_create_workspace_invalid_trace_archival_retention_raises(workspace_store):
    with pytest.raises(MlflowException, match="Trace archival retention must") as exc:
        workspace_store.create_workspace(
            Workspace(
                name="team-a",
                description=None,
                trace_archival_retention="thirty-days",
            )
        )
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_update_workspace_changes_description(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="old"))

    updated = workspace_store.update_workspace(
        Workspace(name="team-a", description="new description"),
    )
    assert updated.description == "new description"
    assert ("team-a", "new description") in _workspace_rows(workspace_store)


def test_update_workspace_sets_default_artifact_root(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="old"))

    updated = workspace_store.update_workspace(
        Workspace(name="team-a", default_artifact_root="s3://bucket/team-a"),
    )
    assert updated.default_artifact_root == "s3://bucket/team-a"
    fetched = workspace_store.get_workspace("team-a")
    assert fetched.default_artifact_root == "s3://bucket/team-a"


def test_update_workspace_can_clear_default_artifact_root(workspace_store):
    workspace_store.create_workspace(
        Workspace(name="team-a", description="old", default_artifact_root="s3://bucket/team-a")
    )

    # Empty string signals "clear this field"
    cleared = workspace_store.update_workspace(
        Workspace(name="team-a", default_artifact_root=""),
    )
    assert cleared.default_artifact_root is None
    fetched = workspace_store.get_workspace("team-a")
    assert fetched.default_artifact_root is None


def test_update_workspace_sets_trace_archival_location(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="old"))

    updated = workspace_store.update_workspace(
        Workspace(name="team-a", trace_archival_location="s3://archive/team-a")
    )
    assert updated.trace_archival_location == "s3://archive/team-a"
    fetched = workspace_store.get_workspace("team-a")
    assert fetched.trace_archival_location == "s3://archive/team-a"


def test_update_workspace_invalid_trace_archival_location_raises(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="old"))

    with pytest.raises(MlflowException, match="proxy-only `mlflow-artifacts:` scheme") as exc:
        workspace_store.update_workspace(
            Workspace(name="team-a", trace_archival_location="mlflow-artifacts:/archive/team-a")
        )
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_update_workspace_unsupported_trace_archival_location_raises(workspace_store):
    class UnsupportedArchiveRepo(ArtifactRepository):
        def log_artifact(self, local_file, artifact_path=None):
            raise NotImplementedError

        def log_artifacts(self, local_dir, artifact_path=None):
            raise NotImplementedError

        def list_artifacts(self, path=None):
            raise NotImplementedError

    workspace_store.create_workspace(Workspace(name="team-a", description="old"))

    with mock.patch(
        "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
        return_value=UnsupportedArchiveRepo("dbfs:/archive/team-a"),
    ):
        with pytest.raises(
            MlflowException,
            match="does not support deleting archived payloads",
        ) as exc:
            workspace_store.update_workspace(
                Workspace(name="team-a", trace_archival_location="dbfs:/archive/team-a")
            )
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_update_workspace_can_clear_trace_archival_location(workspace_store):
    workspace_store.create_workspace(
        Workspace(
            name="team-a",
            description="old",
            trace_archival_location="s3://archive/team-a",
        )
    )

    cleared = workspace_store.update_workspace(Workspace(name="team-a", trace_archival_location=""))
    assert cleared.trace_archival_location is None
    fetched = workspace_store.get_workspace("team-a")
    assert fetched.trace_archival_location is None


def test_update_workspace_sets_trace_archival_retention(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="old"))

    updated = workspace_store.update_workspace(
        Workspace(name="team-a", trace_archival_retention="14d")
    )
    assert updated.trace_archival_retention == "14d"
    fetched = workspace_store.get_workspace("team-a")
    assert fetched.trace_archival_retention == "14d"


def test_update_workspace_invalid_trace_archival_retention_raises(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description="old"))

    with pytest.raises(MlflowException, match="Trace archival retention must") as exc:
        workspace_store.update_workspace(
            Workspace(name="team-a", trace_archival_retention="thirty-days")
        )
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_update_workspace_can_clear_trace_archival_retention(workspace_store):
    workspace_store.create_workspace(
        Workspace(
            name="team-a",
            description="old",
            trace_archival_retention="30d",
        )
    )

    cleared = workspace_store.update_workspace(
        Workspace(name="team-a", trace_archival_retention="")
    )
    assert cleared.trace_archival_retention is None
    fetched = workspace_store.get_workspace("team-a")
    assert fetched.trace_archival_retention is None


def test_delete_workspace_removes_empty_workspace(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    workspace_store.delete_workspace("team-a")
    rows = _workspace_rows(workspace_store)
    assert ("team-a", None) not in rows
    default_ws = workspace_store.get_default_workspace()
    assert (DEFAULT_WORKSPACE_NAME, default_ws.description) in rows


def test_delete_default_workspace_rejected(workspace_store):
    with pytest.raises(
        MlflowException,
        match=f"Cannot delete the reserved '{DEFAULT_WORKSPACE_NAME}' workspace",
    ) as exc:
        workspace_store.delete_workspace(DEFAULT_WORKSPACE_NAME)
    assert exc.value.error_code == "INVALID_STATE"


def test_update_workspace_not_found(workspace_store):
    with pytest.raises(
        MlflowException,
        match="Workspace 'unknown' not found",
    ) as exc:
        workspace_store.update_workspace(Workspace(name="unknown", description="new description"))
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_delete_workspace_not_found(workspace_store):
    with pytest.raises(
        MlflowException,
        match="Workspace 'unknown' not found",
    ) as exc:
        workspace_store.delete_workspace("unknown")
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_resolve_artifact_root_returns_default(workspace_store):
    default_root = "/default/path"
    assert workspace_store.resolve_artifact_root(default_root, DEFAULT_WORKSPACE_NAME) == (
        default_root,
        True,
    )
    workspace_store.create_workspace(Workspace(name="team-a", description=None))
    assert workspace_store.resolve_artifact_root(default_root, workspace_name="team-a") == (
        default_root,
        True,
    )


def test_resolve_artifact_root_prefers_workspace_override(workspace_store):
    workspace_store.create_workspace(
        Workspace(
            name="team-a",
            description=None,
            default_artifact_root="s3://team-a-artifacts",
        )
    )

    resolved_root, should_append = workspace_store.resolve_artifact_root(
        "/default/path", workspace_name="team-a"
    )
    assert resolved_root == "s3://team-a-artifacts"
    assert not should_append


def test_resolve_artifact_root_cache_updates_on_override_change(workspace_store):
    default_root = "/default/path"
    workspace_store.create_workspace(Workspace(name="team-cache", description=None))

    assert workspace_store.resolve_artifact_root(default_root, "team-cache") == (
        default_root,
        True,
    )

    workspace_store.update_workspace(
        Workspace(name="team-cache", default_artifact_root="s3://cache/team")
    )

    assert workspace_store.resolve_artifact_root(default_root, "team-cache") == (
        "s3://cache/team",
        False,
    )


def test_resolve_artifact_root_cache_handles_delete_and_recreate(workspace_store):
    default_root = "/default/path"
    workspace_store.create_workspace(
        Workspace(name="team-cache", description=None, default_artifact_root="s3://cache/a")
    )

    assert workspace_store.resolve_artifact_root(default_root, "team-cache") == (
        "s3://cache/a",
        False,
    )

    workspace_store.delete_workspace("team-cache")
    workspace_store.create_workspace(
        Workspace(name="team-cache", description=None, default_artifact_root="s3://cache/b")
    )

    assert workspace_store.resolve_artifact_root(default_root, "team-cache") == (
        "s3://cache/b",
        False,
    )


def test_resolve_artifact_root_cache_clears_when_override_removed(workspace_store):
    default_root = "/default/path"
    workspace_store.create_workspace(
        Workspace(name="team-cache", description=None, default_artifact_root="s3://cache/a")
    )

    assert workspace_store.resolve_artifact_root(default_root, "team-cache") == (
        "s3://cache/a",
        False,
    )

    workspace_store.update_workspace(Workspace(name="team-cache", default_artifact_root=""))

    assert workspace_store.resolve_artifact_root(default_root, "team-cache") == (
        default_root,
        True,
    )


def test_resolve_trace_archival_config_returns_defaults(workspace_store):
    config = workspace_store.resolve_trace_archival_config(
        default_trace_archival_root="s3://archive/default",
        default_retention="30d",
        workspace_name=DEFAULT_WORKSPACE_NAME,
    )
    assert config.config.location == "s3://archive/default"
    assert config.append_workspace_prefix
    assert config.config.retention == "30d"


def test_resolve_trace_archival_config_prefers_workspace_overrides(workspace_store):
    workspace_store.create_workspace(
        Workspace(
            name="team-a",
            description=None,
            trace_archival_location="s3://archive/team-a",
            trace_archival_retention="14d",
        )
    )

    config = workspace_store.resolve_trace_archival_config(
        default_trace_archival_root="s3://archive/default",
        default_retention="30d",
        workspace_name="team-a",
    )
    assert config.config.location == "s3://archive/team-a"
    assert not config.append_workspace_prefix
    assert config.config.retention == "14d"


def test_resolve_trace_archival_config_cache_updates_on_override_change(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-cache", description=None))

    initial = workspace_store.resolve_trace_archival_config(
        default_trace_archival_root="s3://archive/default",
        default_retention="30d",
        workspace_name="team-cache",
    )
    assert initial.config.location == "s3://archive/default"
    assert initial.append_workspace_prefix
    assert initial.config.retention == "30d"

    workspace_store.update_workspace(
        Workspace(
            name="team-cache",
            trace_archival_location="s3://archive/team-cache",
            trace_archival_retention="7d",
        )
    )

    updated = workspace_store.resolve_trace_archival_config(
        default_trace_archival_root="s3://archive/default",
        default_retention="30d",
        workspace_name="team-cache",
    )
    assert updated.config.location == "s3://archive/team-cache"
    assert not updated.append_workspace_prefix
    assert updated.config.retention == "7d"


def test_get_default_workspace_returns_default(workspace_store):
    default_ws = workspace_store.get_default_workspace()
    assert default_ws.name == DEFAULT_WORKSPACE_NAME
    assert default_ws.description is not None


def test_delete_workspace_reassigns_resources_to_default(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        session.execute(
            sa.text(
                "INSERT INTO experiments (name, workspace, lifecycle_stage) "
                "VALUES (:name, :ws, 'active')"
            ),
            {"name": "exp-in-team-a", "ws": "team-a"},
        )

    workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.SET_DEFAULT)

    with workspace_store.ManagedSessionMaker() as session:
        row = session.execute(
            sa.text("SELECT workspace FROM experiments WHERE name = :name"),
            {"name": "exp-in-team-a"},
        ).fetchone()
        assert row[0] == DEFAULT_WORKSPACE_NAME


def test_delete_workspace_set_default_reassigns_guardrail_configs(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        scorer = _insert_scorer_graph(
            session,
            workspace="team-a",
            experiment_id=991,
            suffix="set-default",
        )
        endpoint_id = _insert_endpoint(session, workspace="team-a", suffix="set-default")
        _insert_guardrail(
            session,
            workspace="team-a",
            scorer_id=scorer["scorer_id"],
            scorer_version=scorer["scorer_version"],
            suffix="set-default",
            endpoint_id=endpoint_id,
        )

    workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.SET_DEFAULT)

    for table in ["experiments", "endpoints", "guardrails", "guardrail_configs"]:
        assert _select_values(workspace_store, f"SELECT workspace FROM {table}") == {
            DEFAULT_WORKSPACE_NAME
        }


def test_delete_workspace_set_default_reassigns_all_workspace_tables(workspace_store):
    workspace = "team-a"
    workspace_store.create_workspace(Workspace(name=workspace, description=None))

    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        _insert_all_workspace_table_rows(session, workspace=workspace, seed=17)
        source_before = _workspace_table_counts(session, workspace)
        default_before = _workspace_table_counts(session, DEFAULT_WORKSPACE_NAME)

    assert set(source_before) == set(workspace_migration._WORKSPACE_TABLES)
    assert all(count == 1 for count in source_before.values())

    workspace_store.delete_workspace(workspace, mode=WorkspaceDeletionMode.SET_DEFAULT)

    with workspace_store.ManagedSessionMaker() as session:
        source_after = _workspace_table_counts(session, workspace)
        default_after = _workspace_table_counts(session, DEFAULT_WORKSPACE_NAME)

    assert all(count == 0 for count in source_after.values())
    assert default_after == {
        table_name: default_before[table_name] + source_before[table_name]
        for table_name in source_before
    }


def test_delete_workspace_fails_on_naming_conflict(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        session.execute(
            sa.text(
                "INSERT INTO experiments (name, workspace, lifecycle_stage) "
                "VALUES (:name, :ws, 'active')"
            ),
            {"name": "shared-exp", "ws": "team-a"},
        )
        session.execute(
            sa.text(
                "INSERT INTO experiments (name, workspace, lifecycle_stage) "
                "VALUES (:name, :ws, 'active')"
            ),
            {"name": "shared-exp", "ws": DEFAULT_WORKSPACE_NAME},
        )

    with pytest.raises(MlflowException, match="already exist in the default workspace") as exc:
        workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.SET_DEFAULT)
    assert exc.value.error_code == "INVALID_STATE"

    # Workspace should still exist (transaction rolled back)
    ws = workspace_store.get_workspace("team-a")
    assert ws.name == "team-a"


def test_delete_workspace_cascade_removes_resources(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        session.execute(
            sa.text(
                "INSERT INTO experiments (name, workspace, lifecycle_stage) "
                "VALUES (:name, :ws, 'active')"
            ),
            {"name": "exp-in-team-a", "ws": "team-a"},
        )

    workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.CASCADE)

    with workspace_store.ManagedSessionMaker() as session:
        row = session.execute(
            sa.text("SELECT count(*) FROM experiments WHERE name = :name"),
            {"name": "exp-in-team-a"},
        ).scalar()
        assert row == 0

    with pytest.raises(MlflowException, match="not found"):
        workspace_store.get_workspace("team-a")


def test_delete_workspace_cascade_removes_all_workspace_tables(workspace_store):
    workspace = "team-a"
    workspace_store.create_workspace(Workspace(name=workspace, description=None))

    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        _insert_all_workspace_table_rows(session, workspace=workspace, seed=18)
        source_before = _workspace_table_counts(session, workspace)
        default_before = _workspace_table_counts(session, DEFAULT_WORKSPACE_NAME)

    assert set(source_before) == set(workspace_migration._WORKSPACE_TABLES)
    assert all(count == 1 for count in source_before.values())

    workspace_store.delete_workspace(workspace, mode=WorkspaceDeletionMode.CASCADE)

    with workspace_store.ManagedSessionMaker() as session:
        source_after = _workspace_table_counts(session, workspace)
        default_after = _workspace_table_counts(session, DEFAULT_WORKSPACE_NAME)

    assert all(count == 0 for count in source_after.values())
    assert default_after == default_before


def test_delete_workspace_cascade_removes_experiment_with_runs(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        session.execute(
            sa.text(
                "INSERT INTO experiments (experiment_id, name, workspace, lifecycle_stage) "
                "VALUES (:id, :name, :ws, 'active')"
            ),
            {"id": 999, "name": "exp-with-runs", "ws": "team-a"},
        )
        session.execute(
            sa.text(
                "INSERT INTO runs (run_uuid, name, experiment_id, lifecycle_stage, status, "
                "source_type, start_time, end_time) "
                "VALUES (:run_id, :name, :exp_id, 'active', 'FINISHED', 'LOCAL', 0, 0)"
            ),
            {"run_id": "run-in-team-a", "name": "test-run", "exp_id": 999},
        )

    workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.CASCADE)

    with workspace_store.ManagedSessionMaker() as session:
        exp_count = session.execute(
            sa.text("SELECT count(*) FROM experiments WHERE name = :name"),
            {"name": "exp-with-runs"},
        ).scalar()
        assert exp_count == 0
        run_count = session.execute(
            sa.text("SELECT count(*) FROM runs WHERE run_uuid = :run_id"),
            {"run_id": "run-in-team-a"},
        ).scalar()
        assert run_count == 0


def test_delete_workspace_cascade_removes_guardrail_backed_scorer(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        scorer = _insert_scorer_graph(
            session,
            workspace="team-a",
            experiment_id=991,
            suffix="cascade",
        )
        _insert_guardrail(
            session,
            workspace="team-a",
            scorer_id=scorer["scorer_id"],
            scorer_version=scorer["scorer_version"],
            suffix="cascade",
        )

    workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.CASCADE)

    assert (
        _select_values(
            workspace_store,
            "SELECT guardrail_id FROM guardrails WHERE guardrail_id = 'guardrail-cascade'",
        )
        == set()
    )
    assert (
        _select_values(
            workspace_store,
            "SELECT scorer_id FROM scorer_versions WHERE scorer_id = 'scorer-cascade'",
        )
        == set()
    )
    assert (
        _select_values(
            workspace_store,
            "SELECT scorer_id FROM scorers WHERE scorer_id = 'scorer-cascade'",
        )
        == set()
    )
    assert (
        _select_values(
            workspace_store,
            "SELECT experiment_id FROM experiments WHERE experiment_id = 991",
        )
        == set()
    )


def test_delete_workspace_cascade_removes_endpoint_before_model_definition(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        endpoint_id, model_definition_id, mapping_id = _insert_endpoint_model_graph(
            session,
            workspace="team-a",
            suffix="cascade-order",
        )

    workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.CASCADE)

    assert (
        _select_values(
            workspace_store,
            "SELECT endpoint_id FROM endpoints WHERE endpoint_id = :endpoint_id",
            {"endpoint_id": endpoint_id},
        )
        == set()
    )
    assert (
        _select_values(
            workspace_store,
            "SELECT model_definition_id FROM model_definitions "
            "WHERE model_definition_id = :model_definition_id",
            {"model_definition_id": model_definition_id},
        )
        == set()
    )
    assert (
        _select_values(
            workspace_store,
            "SELECT mapping_id FROM endpoint_model_mappings WHERE mapping_id = :mapping_id",
            {"mapping_id": mapping_id},
        )
        == set()
    )


def test_delete_workspace_cascade_removes_tracking_soft_edges(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        target = _insert_tracking_graph(
            session,
            workspace="team-a",
            experiment_id=991,
            suffix="target",
        )
        neighbor = _insert_tracking_graph(
            session,
            workspace=DEFAULT_WORKSPACE_NAME,
            experiment_id=992,
            suffix="neighbor",
        )
        _insert_soft_edges(session, target, suffix="target")
        neighbor_associations, neighbor_inputs = _insert_soft_edges(
            session, neighbor, suffix="neighbor"
        )
        endpoint_id = _insert_endpoint(
            session,
            workspace=DEFAULT_WORKSPACE_NAME,
            suffix="shared",
        )
        session.execute(
            sa.text(
                "INSERT INTO endpoint_bindings "
                "(endpoint_id, resource_type, resource_id, created_at, last_updated_at) "
                "VALUES (:endpoint_id, :resource_type, :resource_id, 0, 0)"
            ),
            [
                {
                    "endpoint_id": endpoint_id,
                    "resource_type": GatewayResourceType.SCORER.value,
                    "resource_id": target["scorer_id"],
                },
                {
                    "endpoint_id": endpoint_id,
                    "resource_type": GatewayResourceType.SCORER.value,
                    "resource_id": neighbor["scorer_id"],
                },
            ],
        )

    workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.CASCADE)

    assert (
        _select_values(workspace_store, "SELECT association_id FROM entity_associations")
        == neighbor_associations
    )
    assert _select_values(workspace_store, "SELECT input_uuid FROM inputs") == neighbor_inputs
    assert _select_values(workspace_store, "SELECT input_uuid FROM input_tags") == neighbor_inputs
    assert _select_values(workspace_store, "SELECT resource_id FROM endpoint_bindings") == {
        neighbor["scorer_id"]
    }
    assert _select_values(workspace_store, "SELECT endpoint_id FROM endpoints") == {endpoint_id}


def test_delete_workspace_cascade_preserves_reused_soft_edge_ids(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))
    shared_dataset_id = "shared-dataset"
    shared_input_id = "shared-input"

    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        target = _insert_tracking_graph(
            session,
            workspace="team-a",
            experiment_id=991,
            suffix="duplicate-target",
            dataset_uuid=shared_dataset_id,
        )
        neighbor = _insert_tracking_graph(
            session,
            workspace=DEFAULT_WORKSPACE_NAME,
            experiment_id=992,
            suffix="duplicate-neighbor",
            dataset_uuid=shared_dataset_id,
        )
        session.execute(
            sa.text(
                "INSERT INTO inputs "
                "(input_uuid, source_type, source_id, destination_type, destination_id) "
                "VALUES (:input_uuid, 'DATASET', :dataset_id, 'RUN', :run_id)"
            ),
            [
                {
                    "input_uuid": shared_input_id,
                    "dataset_id": shared_dataset_id,
                    "run_id": target["run_id"],
                },
                {
                    "input_uuid": shared_input_id,
                    "dataset_id": shared_dataset_id,
                    "run_id": neighbor["run_id"],
                },
            ],
        )
        session.execute(
            sa.text(
                "INSERT INTO input_tags (input_uuid, name, value) "
                "VALUES (:input_uuid, 'context', 'neighbor')"
            ),
            {"input_uuid": shared_input_id},
        )

    workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.CASCADE)

    assert _select_values(
        workspace_store,
        "SELECT destination_id FROM inputs WHERE input_uuid = :input_uuid",
        {"input_uuid": shared_input_id},
    ) == {neighbor["run_id"]}
    assert _select_values(
        workspace_store,
        "SELECT input_uuid FROM input_tags WHERE input_uuid = :input_uuid",
        {"input_uuid": shared_input_id},
    ) == {shared_input_id}


def test_delete_workspace_cascade_rolls_back_on_external_guardrail_reference(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        target = _insert_tracking_graph(
            session,
            workspace="team-a",
            experiment_id=991,
            suffix="rollback",
        )
        target_associations, target_inputs = _insert_soft_edges(session, target, suffix="rollback")
        endpoint_id = _insert_endpoint(
            session,
            workspace=DEFAULT_WORKSPACE_NAME,
            suffix="rollback",
        )
        session.execute(
            sa.text(
                "INSERT INTO endpoint_bindings "
                "(endpoint_id, resource_type, resource_id, created_at, last_updated_at) "
                "VALUES (:endpoint_id, :resource_type, :resource_id, 0, 0)"
            ),
            {
                "endpoint_id": endpoint_id,
                "resource_type": GatewayResourceType.SCORER.value,
                "resource_id": target["scorer_id"],
            },
        )
        guardrail_id = _insert_guardrail(
            session,
            workspace=DEFAULT_WORKSPACE_NAME,
            scorer_id=target["scorer_id"],
            scorer_version=target["scorer_version"],
            suffix="external",
        )

    with pytest.raises(MlflowException, match="database integrity constraints") as exc:
        workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.CASCADE)
    assert exc.value.error_code == "INVALID_STATE"

    assert ("team-a", None) in _workspace_rows(workspace_store)
    assert (
        _select_values(workspace_store, "SELECT association_id FROM entity_associations")
        == target_associations
    )
    assert _select_values(workspace_store, "SELECT input_uuid FROM inputs") == target_inputs
    assert _select_values(workspace_store, "SELECT input_uuid FROM input_tags") == target_inputs
    assert _select_values(workspace_store, "SELECT resource_id FROM endpoint_bindings") == {
        target["scorer_id"]
    }
    assert _select_values(workspace_store, "SELECT guardrail_id FROM guardrails") == {guardrail_id}
    assert _select_values(workspace_store, "SELECT scorer_id FROM scorer_versions") == {
        target["scorer_id"]
    }


def test_delete_workspace_restrict_blocks_when_resources_exist(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    with workspace_store.ManagedSessionMaker(read_only=False) as session:
        session.execute(
            sa.text(
                "INSERT INTO experiments (name, workspace, lifecycle_stage) "
                "VALUES (:name, :ws, 'active')"
            ),
            {"name": "exp-in-team-a", "ws": "team-a"},
        )

    with pytest.raises(MlflowException, match="still contains") as exc:
        workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.RESTRICT)
    assert exc.value.error_code == "INVALID_STATE"

    # Workspace and resources should still exist
    ws = workspace_store.get_workspace("team-a")
    assert ws.name == "team-a"
    with workspace_store.ManagedSessionMaker() as session:
        row = session.execute(
            sa.text("SELECT workspace FROM experiments WHERE name = :name"),
            {"name": "exp-in-team-a"},
        ).fetchone()
        assert row[0] == "team-a"


def test_delete_workspace_restrict_allows_empty_workspace(workspace_store):
    workspace_store.create_workspace(Workspace(name="team-a", description=None))

    workspace_store.delete_workspace("team-a", mode=WorkspaceDeletionMode.RESTRICT)

    with pytest.raises(MlflowException, match="not found"):
        workspace_store.get_workspace("team-a")

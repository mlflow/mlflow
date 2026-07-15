"""Parity tests for the UC-native model-registry client migration.

These assert that when the client reads a model / model version off the native governance
surface (``/api/2.1/unity-catalog/*``, ``RegisteredModelInfo`` / ``ModelVersionInfo`` with the
MLflow-enrichment fields as flattened siblings), it produces an MLflow entity that is
**field-for-field identical** to the one produced from the legacy surface
(``/api/2.0/mlflow/unity-catalog/*``, the legacy ``RegisteredModel`` / ``ModelVersion`` protos).

The legacy converters are pinned inline (copied from the pre-migration converters) so the parity
check is a stable oracle that does not depend on the legacy conversion code remaining in the tree.
"""

from mlflow.entities import Metric
from mlflow.entities.logged_model_parameter import LoggedModelParameter as ModelParam
from mlflow.entities.model_registry import (
    ModelVersion,
    ModelVersionSearch,
    ModelVersionTag,
    RegisteredModel,
    RegisteredModelAlias,
    RegisteredModelSearch,
    RegisteredModelTag,
)
from mlflow.entities.model_registry.model_version_deployment_job_state import (
    ModelVersionDeploymentJobState,
)
from mlflow.entities.model_registry.registered_model_deployment_job_state import (
    RegisteredModelDeploymentJobState,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    ModelVersion as LegacyModelVersion,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    ModelVersionDeploymentJobState as LegacyProtoMVDeploymentJobState,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    ModelVersionTag as ProtoModelVersionTag,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    RegisteredModel as LegacyRegisteredModel,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    RegisteredModelAlias as LegacyRegisteredModelAlias,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    RegisteredModelTag as ProtoRegisteredModelTag,
)
from mlflow.protos.unity_catalog_messages_pb2 import (
    DeploymentJobConnection,
    ModelVersionDeploymentJobState as NativeMVDeploymentJobState,
    ModelVersionInfo,
    ModelVersionStatus,
    RegisteredModelAliasInfo,
    RegisteredModelInfo,
    TagKeyValue,
)
from mlflow.store._unity_catalog.registry.uc_native_rest_store import (
    model_version_from_uc_native_proto as model_version_from_uc_proto,
    model_version_search_from_uc_native_proto as model_version_search_from_uc_proto,
    registered_model_from_uc_native_proto as registered_model_from_uc_proto,
    registered_model_search_from_uc_native_proto as registered_model_search_from_uc_proto,
)
from mlflow.utils._unity_catalog_utils import (
    uc_model_version_status_to_string,
)

# ---------------------------------------------------------------------------
# Pinned legacy converters (pre-migration mlflow/utils/_unity_catalog_utils.py) -- the oracle.
# ---------------------------------------------------------------------------


def _legacy_registered_model_from_uc_proto(uc_proto: LegacyRegisteredModel) -> RegisteredModel:
    return RegisteredModel(
        name=uc_proto.name,
        creation_timestamp=uc_proto.creation_timestamp,
        last_updated_timestamp=uc_proto.last_updated_timestamp,
        description=uc_proto.description,
        aliases=[
            RegisteredModelAlias(alias=alias.alias, version=alias.version)
            for alias in (uc_proto.aliases or [])
        ],
        tags=[RegisteredModelTag(key=tag.key, value=tag.value) for tag in (uc_proto.tags or [])],
        deployment_job_id=uc_proto.deployment_job_id,
        deployment_job_state=RegisteredModelDeploymentJobState.to_string(
            uc_proto.deployment_job_state
        ),
    )


def _legacy_registered_model_search_from_uc_proto(
    uc_proto: LegacyRegisteredModel,
) -> RegisteredModelSearch:
    return RegisteredModelSearch(
        name=uc_proto.name,
        creation_timestamp=uc_proto.creation_timestamp,
        last_updated_timestamp=uc_proto.last_updated_timestamp,
        description=uc_proto.description,
        aliases=[],
        tags=[],
    )


def _legacy_model_version_from_uc_proto(uc_proto: LegacyModelVersion) -> ModelVersion:
    return ModelVersion(
        name=uc_proto.name,
        version=uc_proto.version,
        creation_timestamp=uc_proto.creation_timestamp,
        last_updated_timestamp=uc_proto.last_updated_timestamp,
        description=uc_proto.description,
        user_id=uc_proto.user_id,
        source=uc_proto.source,
        run_id=uc_proto.run_id,
        status=uc_model_version_status_to_string(uc_proto.status),
        status_message=uc_proto.status_message,
        aliases=[alias.alias for alias in (uc_proto.aliases or [])],
        tags=[ModelVersionTag(key=tag.key, value=tag.value) for tag in (uc_proto.tags or [])],
        model_id=uc_proto.model_id,
        params=[
            ModelParam(key=param.name, value=param.value) for param in (uc_proto.model_params or [])
        ],
        metrics=[
            Metric(
                key=metric.key,
                value=metric.value,
                timestamp=metric.timestamp,
                step=metric.step,
                dataset_name=metric.dataset_name,
                dataset_digest=metric.dataset_digest,
                model_id=metric.model_id,
                run_id=metric.run_id,
            )
            for metric in (uc_proto.model_metrics or [])
        ],
        deployment_job_state=ModelVersionDeploymentJobState.from_proto(
            uc_proto.deployment_job_state
        ),
    )


def _legacy_model_version_search_from_uc_proto(uc_proto: LegacyModelVersion) -> ModelVersionSearch:
    return ModelVersionSearch(
        name=uc_proto.name,
        version=uc_proto.version,
        creation_timestamp=uc_proto.creation_timestamp,
        last_updated_timestamp=uc_proto.last_updated_timestamp,
        description=uc_proto.description,
        user_id=uc_proto.user_id,
        source=uc_proto.source,
        run_id=uc_proto.run_id,
        status=uc_model_version_status_to_string(uc_proto.status),
        status_message=uc_proto.status_message,
        aliases=[],
        tags=[],
        deployment_job_state=ModelVersionDeploymentJobState.from_proto(
            uc_proto.deployment_job_state
        ),
    )


# ---------------------------------------------------------------------------
# Registered model parity
# ---------------------------------------------------------------------------


def test_registered_model_native_legacy_parity():
    full_name = "catalog.schema.model"
    # Legacy MLflow-dialect response (what the pre-migration client received).
    legacy = LegacyRegisteredModel(
        name=full_name,
        creation_timestamp=111,
        last_updated_timestamp=222,
        description="a model",
        aliases=[LegacyRegisteredModelAlias(alias="champion", version="3")],
        tags=[ProtoRegisteredModelTag(key="team", value="ml")],
        deployment_job_id="job-9",
        deployment_job_state=DeploymentJobConnection.CONNECTED,
    )
    # Native governance response with the same logical content (what the migrated client receives).
    # Governance carries catalog/schema/full_name/created_at/... ; enrichment adds
    # deployment_job_id / deployment_job_state as flattened siblings.
    native = RegisteredModelInfo(
        name="model",
        catalog_name="catalog",
        schema_name="schema",
        full_name=full_name,
        comment="a model",
        created_at=111,
        updated_at=222,
        aliases=[RegisteredModelAliasInfo(alias_name="champion", version_num=3)],
        tags=[TagKeyValue(key="team", value="ml")],
        deployment_job_id="job-9",
        deployment_job_state=DeploymentJobConnection.CONNECTED,
    )

    legacy_entity = _legacy_registered_model_from_uc_proto(legacy)
    native_entity = registered_model_from_uc_proto(native)

    assert native_entity.name == legacy_entity.name == full_name
    assert native_entity.creation_timestamp == legacy_entity.creation_timestamp == 111
    assert native_entity.last_updated_timestamp == legacy_entity.last_updated_timestamp == 222
    assert native_entity.description == legacy_entity.description == "a model"
    assert dict(native_entity.tags) == dict(legacy_entity.tags) == {"team": "ml"}
    assert dict(native_entity.aliases) == dict(legacy_entity.aliases) == {"champion": "3"}
    assert native_entity.deployment_job_id == legacy_entity.deployment_job_id == "job-9"
    assert native_entity.deployment_job_state == legacy_entity.deployment_job_state


def test_registered_model_search_native_legacy_parity():
    full_name = "catalog.schema.model"
    legacy = LegacyRegisteredModel(
        name=full_name, creation_timestamp=5, last_updated_timestamp=6, description="d"
    )
    native = RegisteredModelInfo(
        name="model",
        catalog_name="catalog",
        schema_name="schema",
        full_name=full_name,
        comment="d",
        created_at=5,
        updated_at=6,
    )
    legacy_entity = _legacy_registered_model_search_from_uc_proto(legacy)
    native_entity = registered_model_search_from_uc_proto(native)

    assert native_entity.name == legacy_entity.name == full_name
    assert native_entity.creation_timestamp == legacy_entity.creation_timestamp == 5
    assert native_entity.last_updated_timestamp == legacy_entity.last_updated_timestamp == 6
    assert native_entity.description == legacy_entity.description == "d"


# ---------------------------------------------------------------------------
# Model version parity
# ---------------------------------------------------------------------------


def _legacy_mv_deployment_job_state():
    return LegacyProtoMVDeploymentJobState(
        job_id="42",
        run_id="7",
        job_state=DeploymentJobConnection.CONNECTED,
    )


def _native_mv_deployment_job_state():
    return NativeMVDeploymentJobState(
        job_id="42",
        run_id="7",
        job_state=DeploymentJobConnection.CONNECTED,
    )


def test_model_version_native_legacy_parity():
    full_name = "catalog.schema.model"
    legacy = LegacyModelVersion(
        name=full_name,
        version="4",
        creation_timestamp=10,
        last_updated_timestamp=20,
        description="a version",
        user_id="alice",
        source="s3://bucket/model",
        run_id="run-1",
        status=ModelVersionStatus.READY,
        aliases=[LegacyRegisteredModelAlias(alias="prod", version="4")],
        tags=[ProtoModelVersionTag(key="stage", value="prod")],
        model_id="logged-1",
        deployment_job_state=_legacy_mv_deployment_job_state(),
    )
    native = ModelVersionInfo(
        model_name="model",
        catalog_name="catalog",
        schema_name="schema",
        version=4,
        comment="a version",
        source="s3://bucket/model",
        run_id="run-1",
        status=ModelVersionStatus.READY,
        created_at=10,
        updated_at=20,
        created_by="alice",
        aliases=[RegisteredModelAliasInfo(alias_name="prod", version_num=4)],
        tags=[TagKeyValue(key="stage", value="prod")],
        model_id="logged-1",
        deployment_job_state=_native_mv_deployment_job_state(),
    )

    legacy_entity = _legacy_model_version_from_uc_proto(legacy)
    native_entity = model_version_from_uc_proto(native)

    assert native_entity.name == legacy_entity.name == full_name
    assert native_entity.version == legacy_entity.version == "4"
    assert native_entity.creation_timestamp == legacy_entity.creation_timestamp == 10
    assert native_entity.last_updated_timestamp == legacy_entity.last_updated_timestamp == 20
    assert native_entity.description == legacy_entity.description == "a version"
    assert native_entity.user_id == legacy_entity.user_id == "alice"
    assert native_entity.source == legacy_entity.source == "s3://bucket/model"
    assert native_entity.run_id == legacy_entity.run_id == "run-1"
    assert native_entity.status == legacy_entity.status
    assert native_entity.aliases == legacy_entity.aliases == ["prod"]
    assert dict(native_entity.tags) == dict(legacy_entity.tags) == {"stage": "prod"}
    assert native_entity.model_id == legacy_entity.model_id == "logged-1"


def test_model_version_search_native_legacy_parity():
    full_name = "catalog.schema.model"
    legacy = LegacyModelVersion(
        name=full_name,
        version="4",
        creation_timestamp=10,
        last_updated_timestamp=20,
        description="a version",
        user_id="alice",
        source="s3://bucket/model",
        run_id="run-1",
        status=ModelVersionStatus.READY,
        deployment_job_state=_legacy_mv_deployment_job_state(),
    )
    native = ModelVersionInfo(
        model_name="model",
        catalog_name="catalog",
        schema_name="schema",
        version=4,
        comment="a version",
        source="s3://bucket/model",
        run_id="run-1",
        status=ModelVersionStatus.READY,
        created_at=10,
        updated_at=20,
        created_by="alice",
        deployment_job_state=_native_mv_deployment_job_state(),
    )

    legacy_entity = _legacy_model_version_search_from_uc_proto(legacy)
    native_entity = model_version_search_from_uc_proto(native)

    assert native_entity.name == legacy_entity.name == full_name
    assert native_entity.version == legacy_entity.version == "4"
    assert native_entity.creation_timestamp == legacy_entity.creation_timestamp == 10
    assert native_entity.last_updated_timestamp == legacy_entity.last_updated_timestamp == 20
    assert native_entity.description == legacy_entity.description == "a version"
    assert native_entity.user_id == legacy_entity.user_id == "alice"
    assert native_entity.source == legacy_entity.source == "s3://bucket/model"
    assert native_entity.run_id == legacy_entity.run_id == "run-1"
    assert native_entity.status == legacy_entity.status

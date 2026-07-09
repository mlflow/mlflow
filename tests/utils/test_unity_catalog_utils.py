import pytest

from mlflow.entities.model_registry import (
    ModelVersion,
    ModelVersionDeploymentJobState,
    ModelVersionTag,
    RegisteredModel,
    RegisteredModelAlias,
    RegisteredModelTag,
)
from mlflow.entities.model_registry.model_version_search import ModelVersionSearch
from mlflow.entities.model_registry.registered_model_search import RegisteredModelSearch
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    EncryptionDetails,
    SseEncryptionAlgorithm,
    SseEncryptionDetails,
    TemporaryCredentials,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    ModelVersionStatus as ProtoModelVersionStatus,
)
from mlflow.entities.logged_model_parameter import LoggedModelParameter as ModelParam
from mlflow.entities.metric import Metric
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    DeploymentJobConnection,
    ModelVersionInfo,
    RegisteredModelAliasInfo,
    RegisteredModelInfo,
    TagKeyValue,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import ModelMetric as ProtoModelMetric
from mlflow.protos.databricks_uc_registry_messages_pb2 import ModelParam as ProtoModelParam
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    ModelVersionDeploymentJobState as ProtoModelVersionDeploymentJobState,
)
from mlflow.utils._unity_catalog_utils import (
    _parse_aws_sse_credential,
    model_version_from_uc_proto,
    model_version_search_from_uc_proto,
    registered_model_from_uc_proto,
    registered_model_search_from_uc_proto,
)


def test_model_version_and_model_version_search_equality():
    kwargs = {
        "name": "name",
        "version": "1",
        "creation_timestamp": 1,
        "last_updated_timestamp": 2,
        "description": "description",
        "user_id": "user_id",
        "source": "source",
        "run_id": "run_id",
        "status": "READY",
        "status_message": "status_message",
        "aliases": ["alias1", "alias2"],
        "tags": [
            ModelVersionTag(key="key1", value="value"),
            ModelVersionTag(key="key2", value=""),
        ],
    }
    model_version = ModelVersion(**kwargs)
    model_version_search = ModelVersionSearch(**kwargs)

    assert model_version != model_version_search

    kwargs["tags"] = []
    kwargs["aliases"] = []

    model_version_2 = ModelVersion(**kwargs)
    model_version_search_2 = ModelVersionSearch(**kwargs)

    assert model_version_2 == model_version_search_2


def test_registered_model_and_registered_model_search_equality():
    kwargs = {
        "name": "name",
        "creation_timestamp": 1,
        "last_updated_timestamp": 2,
        "description": "description",
        "aliases": [
            RegisteredModelAlias(alias="alias1", version="1"),
            RegisteredModelAlias(alias="alias2", version="2"),
        ],
        "tags": [
            RegisteredModelTag(key="key1", value="value"),
            RegisteredModelTag(key="key2", value=""),
        ],
    }
    registered_model = RegisteredModel(**kwargs)
    registered_model_search = RegisteredModelSearch(**kwargs)

    assert registered_model != registered_model_search

    kwargs["tags"] = []
    kwargs["aliases"] = []

    registered_model_2 = RegisteredModel(**kwargs)
    registered_model_search_2 = RegisteredModelSearch(**kwargs)

    assert registered_model_2 == registered_model_search_2


@pytest.mark.parametrize(
    ("temp_credentials", "parsed"),
    [
        (TemporaryCredentials(), {}),
        (
            TemporaryCredentials(
                encryption_details=EncryptionDetails(
                    sse_encryption_details=SseEncryptionDetails(
                        algorithm=SseEncryptionAlgorithm.SSE_ENCRYPTION_ALGORITHM_UNSPECIFIED
                    )
                )
            ),
            {},
        ),
        (
            TemporaryCredentials(
                encryption_details=EncryptionDetails(
                    sse_encryption_details=SseEncryptionDetails(
                        algorithm=SseEncryptionAlgorithm.AWS_SSE_KMS,
                        aws_kms_key_arn="arn:aws:kms:us-west-2:111111111111:key/test-key-id",
                    )
                )
            ),
            {
                "ServerSideEncryption": "aws:kms",
                "SSEKMSKeyId": "arn:aws:kms:us-west-2:111111111111:key/test-key-id",
            },
        ),
        (
            TemporaryCredentials(
                encryption_details=EncryptionDetails(
                    sse_encryption_details=SseEncryptionDetails(
                        algorithm=SseEncryptionAlgorithm.AWS_SSE_S3,
                    )
                )
            ),
            {
                "ServerSideEncryption": "AES256",
            },
        ),
    ],
)
def test_parse_aws_sse_credential(temp_credentials, parsed):
    assert _parse_aws_sse_credential(temp_credentials) == parsed


def test_registered_model_from_uc_proto():
    expected_registered_model = RegisteredModel(
        name="catalog.schema.name",
        creation_timestamp=1,
        last_updated_timestamp=2,
        description="description",
        aliases=[
            RegisteredModelAlias(alias="champion", version="3"),
            RegisteredModelAlias(alias="challenger", version="5"),
        ],
        tags=[
            RegisteredModelTag(key="key1", value="value"),
            RegisteredModelTag(key="key2", value=""),
        ],
        deployment_job_id="job_42",
        deployment_job_state="CONNECTED",
    )
    # Governance entity (catalog/schema/name split, comment, created_at/updated_at, alias_name/
    # version_num) flattened together with the MLflow enrichment fields.
    uc_proto = RegisteredModelInfo(
        name="name",
        catalog_name="catalog",
        schema_name="schema",
        full_name="catalog.schema.name",
        comment="description",
        created_at=1,
        updated_at=2,
        aliases=[
            RegisteredModelAliasInfo(alias_name="champion", version_num=3),
            RegisteredModelAliasInfo(alias_name="challenger", version_num=5),
        ],
        tags=[
            TagKeyValue(key="key1", value="value"),
            TagKeyValue(key="key2", value=""),
        ],
        deployment_job_id="job_42",
        deployment_job_state=DeploymentJobConnection.State.Value("CONNECTED"),
    )
    assert registered_model_from_uc_proto(uc_proto) == expected_registered_model


def test_model_version_from_uc_proto():
    expected_model_version = ModelVersion(
        name="catalog.schema.model",
        version=3,
        creation_timestamp=1,
        last_updated_timestamp=2,
        description="description",
        user_id="creator",
        source="source",
        run_id="run_id",
        status="READY",
        aliases=["champion"],
        tags=[ModelVersionTag(key="k", value="v")],
        model_id="m-123",
        params=[ModelParam(key="p", value="1")],
        metrics=[
            Metric(
                key="acc",
                value=0.9,
                timestamp=10,
                step=1,
                dataset_name="d",
                dataset_digest="dig",
                model_id="m-123",
                run_id="run_id",
            )
        ],
        deployment_job_state=ModelVersionDeploymentJobState(
            "job_1",
            "run_1",
            "CONNECTED",
            "RUNNING",
            "task",
        ),
    )
    uc_proto = ModelVersionInfo(
        model_name="model",
        catalog_name="catalog",
        schema_name="schema",
        comment="description",
        source="source",
        run_id="run_id",
        status=ProtoModelVersionStatus.Value("READY"),
        version=3,
        created_at=1,
        updated_at=2,
        created_by="creator",
        aliases=[RegisteredModelAliasInfo(alias_name="champion", version_num=3)],
        tags=[TagKeyValue(key="k", value="v")],
        model_id="m-123",
        model_params=[ProtoModelParam(name="p", value="1")],
        model_metrics=[
            ProtoModelMetric(
                key="acc",
                value=0.9,
                timestamp=10,
                step=1,
                dataset_name="d",
                dataset_digest="dig",
                model_id="m-123",
                run_id="run_id",
            )
        ],
        deployment_job_state=ProtoModelVersionDeploymentJobState(
            job_id="job_1",
            run_id="run_1",
            job_state=DeploymentJobConnection.State.Value("CONNECTED"),
            run_state=ProtoModelVersionDeploymentJobState.DeploymentJobRunState.Value("RUNNING"),
            current_task_name="task",
        ),
    )
    assert model_version_from_uc_proto(uc_proto) == expected_model_version


def test_registered_model_search_from_uc_proto():
    expected_registered_model = RegisteredModelSearch(
        name="catalog.schema.name",
        creation_timestamp=1,
        last_updated_timestamp=2,
        description="description",
        aliases=[],
        tags=[],
    )
    uc_proto = RegisteredModelInfo(
        name="name",
        catalog_name="catalog",
        schema_name="schema",
        comment="description",
        created_at=1,
        updated_at=2,
        aliases=[RegisteredModelAliasInfo(alias_name="champion", version_num=3)],
        tags=[TagKeyValue(key="key1", value="value")],
    )
    actual_registered_model = registered_model_search_from_uc_proto(uc_proto)
    assert actual_registered_model == expected_registered_model

    # Search results intentionally drop tags/aliases.
    with pytest.raises(Exception):  # noqa: PT011
        actual_registered_model.tags()
    with pytest.raises(Exception):  # noqa: PT011
        actual_registered_model.aliases()


def test_model_version_search_from_uc_proto():
    expected_model_version = ModelVersionSearch(
        name="catalog.schema.model",
        version=3,
        creation_timestamp=1,
        last_updated_timestamp=2,
        description="description",
        user_id="creator",
        source="source",
        run_id="run_id",
        status="READY",
        aliases=[],
        tags=[],
        deployment_job_state=ModelVersionDeploymentJobState(
            "",
            "",
            "DEPLOYMENT_JOB_CONNECTION_STATE_UNSPECIFIED",
            "DEPLOYMENT_JOB_RUN_STATE_UNSPECIFIED",
            "",
        ),
    )
    uc_proto = ModelVersionInfo(
        model_name="model",
        catalog_name="catalog",
        schema_name="schema",
        comment="description",
        source="source",
        run_id="run_id",
        status=ProtoModelVersionStatus.Value("READY"),
        version=3,
        created_at=1,
        updated_at=2,
        created_by="creator",
        aliases=[RegisteredModelAliasInfo(alias_name="champion", version_num=3)],
        tags=[TagKeyValue(key="k", value="v")],
    )
    actual_model_version = model_version_search_from_uc_proto(uc_proto)
    assert actual_model_version == expected_model_version

    with pytest.raises(Exception):  # noqa: PT011
        actual_model_version.tags()
    with pytest.raises(Exception):  # noqa: PT011
        actual_model_version.aliases()

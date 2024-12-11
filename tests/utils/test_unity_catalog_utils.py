import pytest

from mlflow.entities.model_registry import (
    ModelVersion,
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
from mlflow.protos.databricks_uc_registry_messages_pb2 import ModelVersion as ProtoModelVersion
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    ModelVersionStatus as ProtoModelVersionStatus,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    ModelVersionTag as ProtoModelVersionTag,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    RegisteredModel as ProtoRegisteredModel,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    RegisteredModelAlias as ProtoRegisteredModelAlias,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    RegisteredModelTag as ProtoRegisteredModelTag,
)
from mlflow.utils._unity_catalog_utils import (
    _parse_aws_sse_credential,
    model_version_from_uc_proto,
    model_version_search_from_uc_proto,
    registered_model_from_uc_proto,
    registered_model_search_from_uc_proto,
)


def test_model_version_from_uc_proto():
    expected_model_version = ModelVersion(
        name="name",
        version="1",
        creation_timestamp=1,
        last_updated_timestamp=2,
        description="description",
        user_id="user_id",
        source="source",
        run_id="run_id",
        status="READY",
        status_message="status_message",
        aliases=["alias1", "alias2"],
        tags=[
            ModelVersionTag(key="key1", value="value"),
            ModelVersionTag(key="key2", value=""),
        ],
    )
    uc_proto = ProtoModelVersion(
        name="name",
        version="1",
        creation_timestamp=1,
        last_updated_timestamp=2,
        description="description",
        user_id="user_id",
        source="source",
        run_id="run_id",
        status=ProtoModelVersionStatus.Value("READY"),
        status_message="status_message",
        aliases=[
            ProtoRegisteredModelAlias(alias="alias1", version="1"),
            ProtoRegisteredModelAlias(alias="alias2", version="2"),
        ],
        tags=[
            ProtoModelVersionTag(key="key1", value="value"),
            ProtoModelVersionTag(key="key2", value=""),
        ],
    )
    actual_model_version = model_version_from_uc_proto(uc_proto)
    assert actual_model_version == expected_model_version


def test_model_version_search_from_uc_proto():
    expected_model_version = ModelVersionSearch(
        name="name",
        version="1",
        creation_timestamp=1,
        last_updated_timestamp=2,
        description="description",
        user_id="user_id",
        source="source",
        run_id="run_id",
        status="READY",
        status_message="status_message",
        aliases=[],
        tags=[],
    )
    uc_proto = ProtoModelVersion(
        name="name",
        version="1",
        creation_timestamp=1,
        last_updated_timestamp=2,
        description="description",
        user_id="user_id",
        source="source",
        run_id="run_id",
        status=ProtoModelVersionStatus.Value("READY"),
        status_message="status_message",
        aliases=[
            ProtoRegisteredModelAlias(alias="alias1", version="1"),
            ProtoRegisteredModelAlias(alias="alias2", version="2"),
        ],
        tags=[
            ProtoModelVersionTag(key="key1", value="value"),
            ProtoModelVersionTag(key="key2", value=""),
        ],
    )
    actual_model_version = model_version_search_from_uc_proto(uc_proto)
    assert actual_model_version == expected_model_version

    with pytest.raises(Exception):  # noqa: PT011
        actual_model_version.tags()

    with pytest.raises(Exception):  # noqa: PT011
        actual_model_version.aliases()


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


def test_registered_model_from_uc_proto():
    expected_registered_model = RegisteredModel(
        name="name",
        creation_timestamp=1,
        last_updated_timestamp=2,
        description="description",
        aliases=[
            RegisteredModelAlias(alias="alias1", version="1"),
            RegisteredModelAlias(alias="alias2", version="2"),
        ],
        tags=[
            RegisteredModelTag(key="key1", value="value"),
            RegisteredModelTag(key="key2", value=""),
        ],
    )
    uc_proto = ProtoRegisteredModel(
        name="name",
        creation_timestamp=1,
        last_updated_timestamp=2,
        description="description",
        aliases=[
            ProtoRegisteredModelAlias(alias="alias1", version="1"),
            ProtoRegisteredModelAlias(alias="alias2", version="2"),
        ],
        tags=[
            ProtoRegisteredModelTag(key="key1", value="value"),
            ProtoRegisteredModelTag(key="key2", value=""),
        ],
    )
    actual_registered_model = registered_model_from_uc_proto(uc_proto)
    assert actual_registered_model == expected_registered_model


def test_registered_model_search_from_uc_proto():
    expected_registered_model = RegisteredModelSearch(
        name="name",
        creation_timestamp=1,
        last_updated_timestamp=2,
        description="description",
        aliases=[],
        tags=[],
    )
    uc_proto = ProtoRegisteredModel(
        name="name",
        creation_timestamp=1,
        last_updated_timestamp=2,
        description="description",
        aliases=[
            ProtoRegisteredModelAlias(alias="alias1", version="1"),
            ProtoRegisteredModelAlias(alias="alias2", version="2"),
        ],
        tags=[
            ProtoRegisteredModelTag(key="key1", value="value"),
            ProtoRegisteredModelTag(key="key2", value=""),
        ],
    )
    actual_registered_model = registered_model_search_from_uc_proto(uc_proto)
    assert actual_registered_model == expected_registered_model

    with pytest.raises(Exception):  # noqa: PT011
        actual_registered_model.tags()

    with pytest.raises(Exception):  # noqa: PT011
        actual_registered_model.aliases()


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
                        aws_kms_key_arn="some:arn:test:key/key_id",
                    )
                )
            ),
            {
                "ServerSideEncryption": "aws:kms",
                "SSEKMSKeyId": "key_id",
            },
        ),
    ],
)
def test_parse_aws_sse_credential(temp_credentials, parsed):
    assert _parse_aws_sse_credential(temp_credentials) == parsed

"""Offline coverage for the UC-native artifact credential-dispatch and enrichment-converter arms
that a single (AWS, external-storage) live workspace does not exercise.

These are the pieces that vary workspace-to-workspace:
  * cloud-provider credential arms (AWS / Azure / GCP / R2) selected via TemporaryCredentials oneof
  * SSE/KMS encryption-detail parsing (only populated on encryption-configured workspaces)
  * storage_mode (default-storage vs customer-hosted)
  * enrichment fields (params / metrics / deployment-job-state) that need extra logging setup

We build synthetic protos and assert the client parses/dispatches each correctly, mocking the
concrete ArtifactRepository classes (whose cloud SDKs may be absent) so we validate the dispatch
logic itself, not the cloud transport.
"""

from unittest import mock

import pytest

from mlflow.protos.unity_catalog_messages_pb2 import (
    AwsCredentials,
    AzureUserDelegationSAS,
    DeploymentJobConnection,
    EncryptionDetails,
    GcpOauthToken,
    ModelMetric,
    ModelParam,
    ModelVersionDeploymentJobState,
    ModelVersionInfo,
    R2Credentials,
    RegisteredModelAliasInfo,
    RegisteredModelInfo,
    SseEncryptionAlgorithm,
    SseEncryptionDetails,
    StorageMode,
    TagKeyValue,
    TemporaryCredentials,
)
from mlflow.store._unity_catalog.registry.uc_native_rest_store import (
    model_version_from_uc_native_proto,
    registered_model_from_uc_native_proto,
)
from mlflow.utils._unity_catalog_utils import (
    _get_artifact_repo_from_storage_info,
    _parse_aws_sse_credential,
)

STORAGE = "s3://bucket/models/abc"


def _refresh_returning(token):
    return lambda: token


# ---------------------------------------------------------------------------
# Credential oneof dispatch -- one arm per cloud provider
# ---------------------------------------------------------------------------
def test_dispatch_aws():
    token = TemporaryCredentials(
        aws_temp_credentials=AwsCredentials(
            access_key_id="ak", secret_access_key="sk", session_token="st"
        )
    )
    assert token.WhichOneof("credentials") == "aws_temp_credentials"
    with mock.patch(
        "mlflow.store.artifact.optimized_s3_artifact_repo.OptimizedS3ArtifactRepository"
    ) as repo, mock.patch.dict("sys.modules", {"boto3": mock.MagicMock()}):
        _get_artifact_repo_from_storage_info(STORAGE, token, _refresh_returning(token))
    repo.assert_called_once()
    _, kwargs = repo.call_args
    assert kwargs["access_key_id"] == "ak"
    assert kwargs["secret_access_key"] == "sk"
    assert kwargs["session_token"] == "st"


def test_dispatch_azure():
    token = TemporaryCredentials(
        azure_user_delegation_sas=AzureUserDelegationSAS(sas_token="sas123")
    )
    assert token.WhichOneof("credentials") == "azure_user_delegation_sas"
    fake_azure = mock.MagicMock()
    with mock.patch(
        "mlflow.store.artifact.azure_data_lake_artifact_repo.AzureDataLakeArtifactRepository"
    ) as repo, mock.patch.dict(
        "sys.modules", {"azure": mock.MagicMock(), "azure.core": mock.MagicMock(),
                        "azure.core.credentials": fake_azure}
    ):
        _get_artifact_repo_from_storage_info(STORAGE, token, _refresh_returning(token))
    repo.assert_called_once()


def test_dispatch_gcp():
    token = TemporaryCredentials(gcp_oauth_token=GcpOauthToken(oauth_token="tok"))
    assert token.WhichOneof("credentials") == "gcp_oauth_token"
    gcs_mod = mock.MagicMock()
    oauth_mod = mock.MagicMock()
    with mock.patch(
        "mlflow.store.artifact.gcs_artifact_repo.GCSArtifactRepository"
    ) as repo, mock.patch.dict(
        "sys.modules",
        {
            "google.cloud": mock.MagicMock(),
            "google.cloud.storage": gcs_mod,
            "google.oauth2": mock.MagicMock(),
            "google.oauth2.credentials": oauth_mod,
        },
    ):
        _get_artifact_repo_from_storage_info(STORAGE, token, _refresh_returning(token))
    repo.assert_called_once()


def test_dispatch_r2():
    token = TemporaryCredentials(
        r2_temp_credentials=R2Credentials(
            access_key_id="ak", secret_access_key="sk", session_token="st"
        )
    )
    assert token.WhichOneof("credentials") == "r2_temp_credentials"
    with mock.patch(
        "mlflow.store.artifact.r2_artifact_repo.R2ArtifactRepository"
    ) as repo:
        _get_artifact_repo_from_storage_info(STORAGE, token, _refresh_returning(token))
    repo.assert_called_once()
    _, kwargs = repo.call_args
    assert kwargs["access_key_id"] == "ak"


def test_dispatch_no_credential_raises():
    token = TemporaryCredentials(expiration_time=123)  # no oneof arm set
    assert token.WhichOneof("credentials") is None
    with pytest.raises(Exception, match="unexpected credential type"):
        _get_artifact_repo_from_storage_info(STORAGE, token, _refresh_returning(token))


# ---------------------------------------------------------------------------
# SSE / KMS encryption parsing
# ---------------------------------------------------------------------------
def test_sse_empty_when_no_encryption():
    assert _parse_aws_sse_credential(TemporaryCredentials()) == {}


def test_sse_s3():
    token = TemporaryCredentials(
        encryption_details=EncryptionDetails(
            sse_encryption_details=SseEncryptionDetails(
                algorithm=SseEncryptionAlgorithm.AWS_SSE_S3
            )
        )
    )
    assert _parse_aws_sse_credential(token) == {"ServerSideEncryption": "AES256"}


def test_sse_kms():
    token = TemporaryCredentials(
        encryption_details=EncryptionDetails(
            sse_encryption_details=SseEncryptionDetails(
                algorithm=SseEncryptionAlgorithm.AWS_SSE_KMS, aws_kms_key_arn="arn:aws:kms:key"
            )
        )
    )
    assert _parse_aws_sse_credential(token) == {
        "ServerSideEncryption": "aws:kms",
        "SSEKMSKeyId": "arn:aws:kms:key",
    }


def test_aws_dispatch_threads_sse_into_upload_args():
    token = TemporaryCredentials(
        aws_temp_credentials=AwsCredentials(access_key_id="ak"),
        encryption_details=EncryptionDetails(
            sse_encryption_details=SseEncryptionDetails(
                algorithm=SseEncryptionAlgorithm.AWS_SSE_S3
            )
        ),
    )
    with mock.patch(
        "mlflow.store.artifact.optimized_s3_artifact_repo.OptimizedS3ArtifactRepository"
    ) as repo, mock.patch.dict("sys.modules", {"boto3": mock.MagicMock()}):
        _get_artifact_repo_from_storage_info(STORAGE, token, _refresh_returning(token))
    _, kwargs = repo.call_args
    assert kwargs["s3_upload_extra_args"] == {"ServerSideEncryption": "AES256"}


# ---------------------------------------------------------------------------
# storage_mode field
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (StorageMode.DEFAULT_STORAGE, StorageMode.DEFAULT_STORAGE),
        (StorageMode.CUSTOMER_HOSTED, StorageMode.CUSTOMER_HOSTED),
    ],
)
def test_storage_mode_roundtrip(mode, expected):
    token = TemporaryCredentials(storage_mode=mode)
    assert token.storage_mode == expected


# ---------------------------------------------------------------------------
# Enrichment converters -- fields needing extra logging setup
# ---------------------------------------------------------------------------
def test_native_model_version_params_metrics_deployment():
    proto = ModelVersionInfo(
        catalog_name="c",
        schema_name="s",
        model_name="m",
        version=7,
        model_id="mid",
        aliases=[RegisteredModelAliasInfo(alias_name="champion", version_num=7)],
        tags=[TagKeyValue(key="k", value="v")],
        model_params=[ModelParam(name="lr", value="0.1")],
        model_metrics=[
            ModelMetric(key="acc", value=0.99, timestamp=123, step=2, run_id="r1", model_id="mid")
        ],
        deployment_job_state=ModelVersionDeploymentJobState(
            job_id="j1",
            run_id="run1",
            run_state=ModelVersionDeploymentJobState.DeploymentJobRunState.RUNNING,
        ),
    )
    mv = model_version_from_uc_native_proto(proto)
    assert mv.name == "c.s.m"
    assert mv.version == "7"
    assert mv.model_id == "mid"
    assert mv.aliases == ["champion"]
    assert mv.tags == {"k": "v"}  # ModelVersion.tags is a dict
    assert {p.key: p.value for p in mv.params} == {"lr": "0.1"}
    assert mv.metrics[0].key == "acc"
    assert mv.metrics[0].value == 0.99
    assert mv.metrics[0].run_id == "r1"


def test_native_model_version_empty_enrichment():
    # A minimal response (no params/metrics/aliases) must not crash the converter.
    proto = ModelVersionInfo(catalog_name="c", schema_name="s", model_name="m", version=1)
    mv = model_version_from_uc_native_proto(proto)
    assert mv.name == "c.s.m"
    assert mv.aliases == []
    assert list(mv.params) == []
    assert list(mv.metrics) == []


def test_native_registered_model_deployment_fields():
    proto = RegisteredModelInfo(
        full_name="c.s.m",
        comment="desc",
        aliases=[RegisteredModelAliasInfo(alias_name="prod", version_num=3)],
        tags=[TagKeyValue(key="team", value="ml")],
        deployment_job_id="42",
        deployment_job_state=DeploymentJobConnection.State.CONNECTED,
    )
    rm = registered_model_from_uc_native_proto(proto)
    assert rm.name == "c.s.m"
    assert rm.description == "desc"
    assert rm.aliases == {"prod": "3"}
    assert rm.tags == {"team": "ml"}
    assert rm.deployment_job_id == "42"

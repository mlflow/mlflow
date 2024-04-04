from unittest import mock

import pytest

from mlflow.entities import Run, RunInfo
from mlflow.entities.span_status import SpanStatus
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_status import TraceStatus
from mlflow.tracking._tracking_service.client import TrackingServiceClient


@pytest.fixture
def mock_store():
    with mock.patch("mlflow.tracking._trackingZ_service.utils._get_store") as mock_get_store:
        yield mock_get_store.return_value


def newTrackingServiceClient():
    return TrackingServiceClient("databricks://scope:key")


@pytest.mark.parametrize(
    ("artifact_uri", "databricks_uri", "uri_for_repo"),
    [
        ("dbfs:/path", "databricks://profile", "dbfs://profile@databricks/path"),
        ("dbfs:/path", "databricks://scope:key", "dbfs://scope:key@databricks/path"),
        ("runs:/path", "databricks://scope:key", "runs://scope:key@databricks/path"),
        ("models:/path", "databricks://scope:key", "models://scope:key@databricks/path"),
        # unaffected uri cases
        (
            "dbfs://profile@databricks/path",
            "databricks://scope:key",
            "dbfs://profile@databricks/path",
        ),
        (
            "dbfs://profile@databricks/path",
            "databricks://profile2",
            "dbfs://profile@databricks/path",
        ),
        ("s3:/path", "databricks://profile", "s3:/path"),
        ("ftp://user:pass@host/path", "databricks://profile", "ftp://user:pass@host/path"),
    ],
)
def test_get_artifact_repo(artifact_uri, databricks_uri, uri_for_repo):
    with mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient.get_run",
        return_value=Run(
            RunInfo(
                "uuid", "expr_id", "userid", "status", 0, 10, "active", artifact_uri=artifact_uri
            ),
            None,
        ),
    ), mock.patch(
        "mlflow.tracking._tracking_service.client.get_artifact_repository", return_value=None
    ) as get_repo_mock:
        client = TrackingServiceClient(databricks_uri)
        client._get_artifact_repo("some-run-id")
        get_repo_mock.assert_called_once_with(uri_for_repo)


def test_artifact_repo_is_cached_per_run_id(tmp_path):
    uri = "ftp://user:pass@host/path"
    with mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient.get_run",
        return_value=Run(
            RunInfo("uuid", "expr_id", "userid", "status", 0, 10, "active", artifact_uri=uri),
            None,
        ),
    ):
        tracking_uri = tmp_path.as_uri()
        artifact_repo = TrackingServiceClient(tracking_uri)._get_artifact_repo("some_run_id")
        another_artifact_repo = TrackingServiceClient(tracking_uri)._get_artifact_repo(
            "some_run_id"
        )
        assert artifact_repo is another_artifact_repo


def test_download_trace_data(tmp_path):
    trace_info = TraceInfo(
        request_id="test",
        experiment_id="test",
        timestamp_ms=0,
        execution_time_ms=1,
        status=SpanStatus(TraceStatus.OK),
        request_metadata={},
        tags={"mlflow.artifactLocation": "test"},
    )
    with mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient._get_trace_info",
        return_value=trace_info
    ) as mock_get_trace_info, mock.patch(
        "mlflow.store.artifact.artifact_repo.ArtifactRepository.download_trace_data",
        return_value={"spans": []},
    ) as mock_download_trace_data:
        client = TrackingServiceClient(tmp_path.as_uri())
        trace_data = client._download_trace_data(trace_info=trace_info)
        assert trace_data == TraceData()

        mock_download_trace_data.assert_called_once()
        # The TraceInfo is already fetched prior to the upload_trace_data call,
        # so we should not call _get_trace_info again
        mock_get_trace_info.assert_not_called()


def test_upload_trace_data(tmp_path):
    trace_info = TraceInfo(
        request_id="test",
        experiment_id="test",
        timestamp_ms=0,
        execution_time_ms=1,
        status=SpanStatus(TraceStatus.OK),
        request_metadata={},
        tags={"mlflow.artifactLocation": "test"},
    )
    with mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient._get_trace_info",
        return_value=trace_info
    ) as mock_get_trace_info, mock.patch(
        "mlflow.store.artifact.artifact_repo.ArtifactRepository.upload_trace_data",
    ) as mock_upload_trace_data:
        client = TrackingServiceClient(tmp_path.as_uri())
        client._upload_trace_data(trace_info=trace_info, trace_data=TraceData())
        mock_upload_trace_data.assert_called_once()
        # The TraceInfo is already fetched prior to the upload_trace_data call,
        # so we should not call _get_trace_info again
        mock_get_trace_info.assert_not_called()

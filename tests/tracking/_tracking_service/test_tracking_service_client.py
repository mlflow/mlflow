import json
import uuid
from unittest import mock

import pydantic
import pytest

from mlflow import MlflowClient
from mlflow.entities import Run, RunInfo
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_status import TraceStatus
from mlflow.exceptions import MlflowTraceDataCorrupted, MlflowTraceDataNotFound
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.tracking._tracking_service.client import TrackingServiceClient


@pytest.fixture
def mock_store():
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store") as mock_get_store:
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


def test_download_trace_data(tmp_path, mock_store):
    trace_info = TraceInfo(
        request_id="test",
        experiment_id="test",
        timestamp_ms=0,
        execution_time_ms=1,
        status=TraceStatus.OK,
        request_metadata={},
        tags={"mlflow.artifactLocation": "test"},
    )
    with mock.patch(
        "mlflow.store.artifact.artifact_repo.ArtifactRepository.download_trace_data",
        return_value={"spans": []},
    ) as mock_download_trace_data:
        client = TrackingServiceClient(tmp_path.as_uri())
        trace_data = client._download_trace_data(trace_info=trace_info)
        assert trace_data == TraceData()

        mock_download_trace_data.assert_called_once()
        # The TraceInfo is already fetched prior to the upload_trace_data call,
        # so we should not call get_trace_info again
        mock_store.get_trace_info.assert_not_called()


def test_upload_trace_data(tmp_path, mock_store):
    trace_info = TraceInfo(
        request_id="test",
        experiment_id="test",
        timestamp_ms=0,
        execution_time_ms=1,
        status=TraceStatus.OK,
        request_metadata={},
        tags={"mlflow.artifactLocation": "test"},
    )
    mock_store.start_trace.return_value = trace_info

    class CustomObject(pydantic.BaseModel):
        data: str

    obj = CustomObject(data="test")
    span = MlflowClient().start_trace(
        "span",
        # test non-json serializable objects
        inputs={"data": uuid.uuid4()},
        attributes={SpanAttributeKey.FUNCTION_NAME: "function_name", SpanAttributeKey.OUTPUTS: obj},
    )
    trace_data = TraceData([span])
    trace_data_json = json.dumps(trace_data.to_dict(), cls=TraceJSONEncoder)
    with mock.patch(
        "mlflow.store.artifact.artifact_repo.ArtifactRepository.upload_trace_data",
    ) as mock_upload_trace_data:
        client = TrackingServiceClient(tmp_path.as_uri())
        client._upload_trace_data(trace_info=trace_info, trace_data=trace_data)
        mock_upload_trace_data.assert_called_once_with(trace_data_json)
        # The TraceInfo is already fetched prior to the upload_trace_data call,
        # so we should not call _get_trace_info again
        mock_store.get_trace_info.assert_not_called()


def test_search_traces(tmp_path):
    client = TrackingServiceClient(tmp_path.as_uri())
    with mock.patch.object(
        client,
        "_search_traces",
        side_effect=[
            # Page 1
            (
                [
                    TraceInfo(
                        request_id="test",
                        experiment_id="test",
                        timestamp_ms=0,
                        execution_time_ms=0,
                        status=TraceStatus.OK,
                        request_metadata={},
                        tags={"mlflow.artifactLocation": "test"},
                    ),
                    TraceInfo(
                        request_id="test",
                        experiment_id="test",
                        timestamp_ms=0,
                        execution_time_ms=0,
                        status=TraceStatus.OK,
                        request_metadata={},
                        tags={"mlflow.artifactLocation": "test"},
                    ),
                ],
                "token",
            ),
            # Page 2 (last page)
            (
                [
                    TraceInfo(
                        request_id="test",
                        experiment_id="test",
                        timestamp_ms=1,
                        execution_time_ms=1,
                        status=TraceStatus.OK,
                        request_metadata={},
                        tags={"mlflow.artifactLocation": "test"},
                    ),
                    TraceInfo(
                        request_id="test",
                        experiment_id="test",
                        timestamp_ms=1,
                        execution_time_ms=1,
                        status=TraceStatus.OK,
                        request_metadata={},
                        tags={"mlflow.artifactLocation": "test"},
                    ),
                ],
                None,
            ),
        ],
    ) as mock_search_traces, mock.patch.object(
        client,
        "_download_trace_data",
        side_effect=[
            TraceData(),
            TraceData(),
            TraceData(),
            TraceData(),
        ],
    ) as mock_download_trace_data:
        res1 = client.search_traces(experiment_ids=["0"], max_results=2)
        assert len(res1) == 2
        assert res1.token == "token"

        res2 = client.search_traces(experiment_ids=["0"], max_results=2, page_token=res1.token)
        assert len(res2) == 2
        assert res2.token is None

        assert mock_search_traces.call_count == 2
        assert mock_download_trace_data.call_count == 4


def test_search_traces_download_failures(tmp_path):
    client = TrackingServiceClient(tmp_path.as_uri())

    # Scenario 1: Collect max_results traces before exhausting all pages
    with mock.patch.object(
        client,
        "_search_traces",
        side_effect=[
            # Page 1
            (
                [
                    TraceInfo(
                        request_id="test",
                        experiment_id="test",
                        timestamp_ms=0,
                        execution_time_ms=0,
                        status=TraceStatus.OK,
                        request_metadata={},
                        tags={"mlflow.artifactLocation": "test"},
                    ),
                    TraceInfo(
                        request_id="test",
                        experiment_id="test",
                        timestamp_ms=0,
                        execution_time_ms=0,
                        status=TraceStatus.OK,
                        request_metadata={},
                        tags={"mlflow.artifactLocation": "test"},
                    ),
                ],
                "token1",
            ),
            # Page 2
            (
                [
                    TraceInfo(
                        request_id="test",
                        experiment_id="test",
                        timestamp_ms=1,
                        execution_time_ms=1,
                        status=TraceStatus.OK,
                        request_metadata={},
                        tags={"mlflow.artifactLocation": "test"},
                    )
                ],
                "token2",
            ),
            # Page 3
            (
                [
                    TraceInfo(
                        request_id="test",
                        experiment_id="test",
                        timestamp_ms=1,
                        execution_time_ms=1,
                        status=TraceStatus.OK,
                        request_metadata={},
                        tags={"mlflow.artifactLocation": "test"},
                    )
                ],
                "token3",
            ),
        ],
    ) as mock_search_traces, mock.patch.object(
        client,
        "_download_trace_data",
        side_effect=[
            # Page 1
            TraceData(),
            MlflowTraceDataCorrupted(request_id="test"),
            # Page 2
            MlflowTraceDataNotFound(request_id="test"),
            # Page 3
            TraceData(),
        ],
    ) as mock_download_trace_data:
        res = client.search_traces(experiment_ids=["0"], max_results=2)
        assert len(res) == 2
        assert res.token == "token3"
        assert mock_search_traces.call_count == 3
        assert mock_download_trace_data.call_count == 4

    # Scenario 2: Exhaust all pages before collecting max_results traces
    client = TrackingServiceClient(tmp_path.as_uri())
    with mock.patch.object(
        client,
        "_search_traces",
        side_effect=[
            # Page 1
            (
                [
                    TraceInfo(
                        request_id="test",
                        experiment_id="test",
                        timestamp_ms=0,
                        execution_time_ms=0,
                        status=TraceStatus.OK,
                        request_metadata={},
                        tags={"mlflow.artifactLocation": "test"},
                    ),
                    TraceInfo(
                        request_id="test",
                        experiment_id="test",
                        timestamp_ms=0,
                        execution_time_ms=0,
                        status=TraceStatus.OK,
                        request_metadata={},
                        tags={"mlflow.artifactLocation": "test"},
                    ),
                ],
                "token1",
            ),
            # Page 2 (last page)
            (
                [
                    TraceInfo(
                        request_id="test",
                        experiment_id="test",
                        timestamp_ms=1,
                        execution_time_ms=1,
                        status=TraceStatus.OK,
                        request_metadata={},
                        tags={"mlflow.artifactLocation": "test"},
                    )
                ],
                None,
            ),
        ],
    ) as mock_search_traces, mock.patch.object(
        client,
        "_download_trace_data",
        side_effect=[
            # Page 1
            TraceData(),
            MlflowTraceDataCorrupted(request_id="test"),
            # Page 2
            MlflowTraceDataNotFound(request_id="test"),
        ],
    ) as mock_download_trace_data:
        res = client.search_traces(experiment_ids=["0"], max_results=2)
        assert len(res) == 1
        assert res.token is None
        assert mock_search_traces.call_count == 2
        assert mock_download_trace_data.call_count == 3


def test_search_traces_does_not_swallow_unexpected_exceptions(tmp_path):
    client = TrackingServiceClient(tmp_path.as_uri())
    with mock.patch.object(
        client,
        "_search_traces",
        side_effect=[
            (
                [
                    TraceInfo(
                        request_id="test",
                        experiment_id="test",
                        timestamp_ms=0,
                        execution_time_ms=0,
                        status=TraceStatus.OK,
                        request_metadata={},
                        tags={"mlflow.artifactLocation": "test"},
                    )
                ],
                "token1",
            ),
        ],
    ) as mock_search_traces, mock.patch.object(
        client,
        "_download_trace_data",
        side_effect=[ValueError("Unexpected exception")],
    ) as mock_download_trace_data:
        with pytest.raises(ValueError, match="Unexpected exception"):
            client.search_traces(experiment_ids=["0"], max_results=1)

        mock_search_traces.assert_called_once()
        mock_download_trace_data.assert_called_once()

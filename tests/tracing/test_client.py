import json
import uuid
from unittest import mock

import pydantic
import pytest

from mlflow import MlflowClient
from mlflow.entities.assessment import Assessment, Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_status import TraceStatus
from mlflow.exceptions import MlflowTraceDataCorrupted, MlflowTraceDataNotFound
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY, SpanAttributeKey
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.utils import TraceJSONEncoder


@pytest.fixture
def mock_store():
    with mock.patch("mlflow.tracing.client._get_store") as mock_get_store:
        yield mock_get_store.return_value


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
        "mlflow.store.artifact.local_artifact_repo.LocalArtifactRepository.download_trace_data",
        return_value={"spans": []},
    ) as mock_download_trace_data:
        client = TracingClient(tmp_path.as_uri())
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
    span = start_span_no_context(
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
        client = TracingClient(tmp_path.as_uri())
        client._upload_trace_data(trace_info=trace_info, trace_data=trace_data)
        mock_upload_trace_data.assert_called_once_with(trace_data_json)


def test_search_traces(tmp_path):
    client = TracingClient(tmp_path.as_uri())
    with (
        mock.patch.object(
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
        ) as mock_search_traces,
        mock.patch.object(
            client,
            "_download_trace_data",
            side_effect=[
                TraceData(),
                TraceData(),
                TraceData(),
                TraceData(),
            ],
        ) as mock_download_trace_data,
    ):
        res1 = client.search_traces(experiment_ids=["0"], max_results=2)
        assert len(res1) == 2
        assert res1.token == "token"

        res2 = client.search_traces(experiment_ids=["0"], max_results=2, page_token=res1.token)
        assert len(res2) == 2
        assert res2.token is None

        assert mock_search_traces.call_count == 2
        assert mock_download_trace_data.call_count == 4


def test_search_traces_download_failures(tmp_path):
    client = TracingClient(tmp_path.as_uri())

    # Scenario 1: Collect max_results traces before exhausting all pages
    with (
        mock.patch.object(
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
        ) as mock_search_traces,
        mock.patch.object(
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
        ) as mock_download_trace_data,
    ):
        res = client.search_traces(experiment_ids=["0"], max_results=2)
        assert len(res) == 2
        assert res.token == "token3"
        assert mock_search_traces.call_count == 3
        assert mock_download_trace_data.call_count == 4

    # Scenario 2: Exhaust all pages before collecting max_results traces
    client = TracingClient(tmp_path.as_uri())
    with (
        mock.patch.object(
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
        ) as mock_search_traces,
        mock.patch.object(
            client,
            "_download_trace_data",
            side_effect=[
                # Page 1
                TraceData(),
                MlflowTraceDataCorrupted(request_id="test"),
                # Page 2
                MlflowTraceDataNotFound(request_id="test"),
            ],
        ) as mock_download_trace_data,
    ):
        res = client.search_traces(experiment_ids=["0"], max_results=2)
        assert len(res) == 1
        assert res.token is None
        assert mock_search_traces.call_count == 2
        assert mock_download_trace_data.call_count == 3


def test_search_traces_does_not_swallow_unexpected_exceptions(tmp_path):
    client = TracingClient(tmp_path.as_uri())
    with (
        mock.patch.object(
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
        ) as mock_search_traces,
        mock.patch.object(
            client,
            "_download_trace_data",
            side_effect=[ValueError("Unexpected exception")],
        ) as mock_download_trace_data,
    ):
        with pytest.raises(ValueError, match="Unexpected exception"):
            client.search_traces(experiment_ids=["0"], max_results=1)

        mock_search_traces.assert_called_once()
        mock_download_trace_data.assert_called_once()


def test_search_traces_with_filestore(tmp_path):
    client = TracingClient(tmp_path.as_uri())
    exp_id = MlflowClient(tmp_path.as_uri()).create_experiment("test_search_traces")
    trace_infos = []
    for i in range(3):
        trace_infos.append(
            client.start_trace(
                exp_id,
                i * 1000,
                {
                    SpanAttributeKey.REQUEST_ID: f"request_id_{i}",
                    TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION),
                },
                {},
            ).to_v3()
        )

    with mock.patch.object(
        client,
        "_download_trace_data",
        side_effect=[
            TraceData(),
            TraceData(),
            TraceData(),
            TraceData(),
        ],
    ) as mock_download_trace_data:
        res1 = client.search_traces(experiment_ids=[exp_id], max_results=2)
        assert [res.info for res in res1] == trace_infos[::-1][:2]

        res2 = client.search_traces(experiment_ids=[exp_id], max_results=2, page_token=res1.token)
        assert res2[0].info == trace_infos[0]
        assert res2.token is None
        assert mock_download_trace_data.call_count == 3


@pytest.mark.parametrize("tracking_uri", ["databricks", "databricks://profile"])
def test_search_traces_with_assessments(tracking_uri):
    client = TracingClient(tracking_uri)
    with (
        mock.patch.object(
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
                    None,
                ),
            ],
        ) as mock_search_traces,
        mock.patch.object(
            client,
            "_download_trace_data",
            side_effect=[
                TraceData(),
                TraceData(),
            ],
        ) as mock_download_trace_data,
        mock.patch.object(
            client,
            "get_trace_info",
            return_value=TraceInfo(
                request_id="test",
                experiment_id="test",
                timestamp_ms=0,
                execution_time_ms=0,
                status=TraceStatus.OK,
                request_metadata={},
                tags={"mlflow.artifactLocation": "test"},
                assessments=[
                    Assessment(
                        trace_id="test",
                        name="test",
                        source=AssessmentSource(
                            source_id="test", source_type=AssessmentSourceType.HUMAN
                        ),
                        create_time_ms=0,
                        last_update_time_ms=0,
                        feedback=Feedback("test"),
                    )
                ],
            ),
        ) as mock_get_trace_info,
    ):
        res1 = client.search_traces(experiment_ids=["0"], max_results=2)
        assert len(res1) == 2
        assert res1.token is None

        assert mock_search_traces.call_count == 1
        assert mock_download_trace_data.call_count == 2
        assert mock_get_trace_info.call_count == 2
        assert mock_get_trace_info.call_args_list == [
            mock.call("test", should_query_v3=True),
            mock.call("test", should_query_v3=True),
        ]

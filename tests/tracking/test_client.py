import json
import os
import pickle
import time
import uuid
from pathlib import Path
from unittest import mock
from unittest.mock import Mock, patch

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

import mlflow
from mlflow import MlflowClient, flush_async_logging
from mlflow.config import enable_async_logging
from mlflow.entities import (
    EvaluationDataset,
    ExperimentTag,
    LoggedModel,
    Run,
    RunInfo,
    RunStatus,
    RunTag,
    SourceType,
    Span,
    SpanStatusCode,
    SpanType,
    Trace,
    TraceInfo,
    ViewType,
)
from mlflow.entities.file_info import FileInfo
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.metric import Metric
from mlflow.entities.model_registry import ModelVersion, ModelVersionTag
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.entities.model_registry.prompt_version import IS_PROMPT_TAG_KEY
from mlflow.entities.param import Param
from mlflow.entities.span import create_mlflow_span
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_location import TraceLocation, TraceLocationType, UCSchemaLocation
from mlflow.entities.trace_state import TraceState
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import MLFLOW_TRACKING_USERNAME
from mlflow.exceptions import (
    MlflowException,
    MlflowNotImplementedException,
    MlflowTraceDataCorrupted,
    MlflowTraceDataNotFound,
)
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry.sqlalchemy_store import (
    SqlAlchemyStore as SqlAlchemyModelRegistryStore,
)
from mlflow.store.tracking import SEARCH_EVALUATION_DATASETS_MAX_RESULTS, SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore as SqlAlchemyTrackingStore
from mlflow.tracing.constant import SpansLocation, TraceMetadataKey, TraceTagKey
from mlflow.tracing.provider import _get_tracer, trace_disabled
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.tracking import set_registry_uri
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking._model_registry.utils import (
    _get_store_registry as _get_model_registry_store_registry,
)
from mlflow.tracking._tracking_service.utils import _register, _use_tracking_uri
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID
from mlflow.utils.databricks_utils import _construct_databricks_run_url
from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_COMMIT,
    MLFLOW_PARENT_RUN_ID,
    MLFLOW_PROJECT_ENTRY_POINT,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_USER,
)
from mlflow.utils.os import is_windows

from tests.tracing.conftest import async_logging_enabled  # noqa: F401
from tests.tracing.helper import create_test_trace_info, get_traces


@pytest.fixture(autouse=True)
def reset_registry_uri():
    yield
    set_registry_uri(None)


@pytest.fixture
def mock_store():
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store") as mock_get_store:
        mock_store = mock_get_store.return_value
        with mock.patch("mlflow.tracing.client._get_store", return_value=mock_store):
            yield mock_store


@pytest.fixture
def mock_artifact_repo():
    with mock.patch(
        "mlflow.tracking._tracking_service.client.get_artifact_repository"
    ) as mock_get_repo:
        mock_repo = mock_get_repo.return_value
        with mock.patch("mlflow.tracing.client.get_artifact_repository", return_value=mock_repo):
            yield mock_repo


@pytest.fixture
def mock_registry_store():
    mock_store = mock.MagicMock()
    mock_store.create_model_version = mock.create_autospec(
        SqlAlchemyModelRegistryStore.create_model_version
    )
    with mock.patch("mlflow.tracking._model_registry.utils._get_store", return_value=mock_store):
        yield mock_store


@pytest.fixture
def mock_databricks_tracking_store():
    experiment_id = "test-exp-id"
    run_id = "runid"

    class MockDatabricksTrackingStore:
        def __init__(self, run_id, experiment_id):
            self.run_id = run_id
            self.experiment_id = experiment_id

        def get_run(self, *args, **kwargs):
            return Run(
                RunInfo(self.run_id, self.experiment_id, "userid", "status", 0, 1, None), None
            )

    mock_tracking_store = MockDatabricksTrackingStore(run_id, experiment_id)

    with mock.patch(
        "mlflow.tracking._tracking_service.utils._tracking_store_registry.get_store",
        return_value=mock_tracking_store,
    ):
        yield mock_tracking_store


@pytest.fixture
def mock_store_start_trace():
    def _mock_start_trace(trace_info):
        return create_test_trace_info(
            trace_id="tr-123",
            experiment_id=trace_info.experiment_id,
            request_time=trace_info.request_time,
            execution_duration=trace_info.execution_duration,
            state=trace_info.state,
            trace_metadata=trace_info.trace_metadata,
            tags={
                "mlflow.user": "bob",
                "mlflow.artifactLocation": "test",
                **trace_info.tags,
            },
        )

    with mock.patch(
        "mlflow.tracing.client.TracingClient.start_trace", side_effect=_mock_start_trace
    ) as mock_start_trace:
        yield mock_start_trace


@pytest.fixture
def mock_spark_session():
    with mock.patch(
        "mlflow.utils.databricks_utils._get_active_spark_session"
    ) as mock_spark_session:
        yield mock_spark_session.return_value


@pytest.fixture
def mock_time():
    time = 1552319350.244724
    with mock.patch("time.time", return_value=time):
        yield time


@pytest.fixture
def setup_async_logging():
    enable_async_logging(True)
    yield
    flush_async_logging()
    enable_async_logging(False)


def test_client_create_run(mock_store, mock_time):
    experiment_id = mock.Mock()

    MlflowClient().create_run(experiment_id)

    mock_store.create_run.assert_called_once_with(
        experiment_id=experiment_id,
        user_id="unknown",
        start_time=int(mock_time * 1000),
        tags=[],
        run_name=None,
    )


def test_client_create_run_with_name(mock_store, mock_time):
    experiment_id = mock.Mock()

    MlflowClient().create_run(experiment_id, run_name="my name")

    mock_store.create_run.assert_called_once_with(
        experiment_id=experiment_id,
        user_id="unknown",
        start_time=int(mock_time * 1000),
        tags=[],
        run_name="my name",
    )


def test_client_get_trace(mock_store, mock_artifact_repo):
    trace_id = "trace:/catalog.schema/123"
    mock_store.batch_get_traces.return_value = [
        Trace(
            TraceInfo(
                trace_id=trace_id,
                trace_location=TraceLocation(
                    type=TraceLocationType.UC_SCHEMA,
                    uc_schema=UCSchemaLocation(catalog_name="catalog", schema_name="schema"),
                ),
                request_time=123,
                execution_duration=456,
                state=TraceState.OK,
                tags={"mlflow.artifactLocation": "dbfs:/path/to/artifacts"},
            ),
            TraceData(
                spans=[
                    Span.from_dict(
                        {
                            "name": "predict",
                            "context": {
                                "trace_id": "0x123456789",
                                "span_id": "0x12345",
                            },
                            "parent_id": None,
                            "start_time": 123000000,
                            "end_time": 579000000,
                            "status_code": "OK",
                            "status_message": "",
                            "attributes": {
                                "mlflow.traceRequestId": f'"{trace_id}"',
                                "mlflow.spanType": '"LLM"',
                                "mlflow.spanFunctionName": '"predict"',
                                "mlflow.spanInputs": '{"prompt": "What is the meaning of life?"}',
                                "mlflow.spanOutputs": '{"answer": 42}',
                            },
                            "events": [],
                        }
                    )
                ]
            ),
        )
    ]
    trace = MlflowClient().get_trace(trace_id)
    mock_store.batch_get_traces.assert_called_once_with([trace_id], "catalog.schema")
    mock_artifact_repo.download_trace_data.assert_not_called()

    assert trace.info.trace_id == trace_id
    assert trace.info.trace_location.uc_schema.catalog_name == "catalog"
    assert trace.info.trace_location.uc_schema.schema_name == "schema"
    assert trace.info.timestamp_ms == 123
    assert trace.info.execution_time_ms == 456
    assert trace.info.status == TraceStatus.OK
    assert trace.info.tags == {"mlflow.artifactLocation": "dbfs:/path/to/artifacts"}
    assert trace.data.request == '{"prompt": "What is the meaning of life?"}'
    assert trace.data.response == '{"answer": 42}'
    assert len(trace.data.spans) == 1
    assert trace.data.spans[0].name == "predict"
    assert trace.data.spans[0].trace_id == trace_id
    assert trace.data.spans[0].inputs == {"prompt": "What is the meaning of life?"}
    assert trace.data.spans[0].outputs == {"answer": 42}
    assert trace.data.spans[0].start_time_ns == 123000000
    assert trace.data.spans[0].end_time_ns == 579000000
    assert trace.data.spans[0].status.status_code == SpanStatusCode.OK


def test_client_get_trace_empty_result(mock_store):
    mock_store.batch_get_traces.return_value = []
    with pytest.raises(MlflowException, match="not found"):
        MlflowClient().get_trace("trace:/catalog.schema/123")


def test_client_get_trace_from_artifact_repo(mock_store, mock_artifact_repo):
    mock_store.get_trace_info.return_value = TraceInfo(
        trace_id="tr-1234567",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=123,
        execution_duration=456,
        state=TraceState.OK,
        tags={"mlflow.artifactLocation": "dbfs:/path/to/artifacts"},
    )
    mock_artifact_repo.download_trace_data.return_value = {
        "request": '{"prompt": "What is the meaning of life?"}',
        "response": '{"answer": 42}',
        "spans": [
            {
                "name": "predict",
                "context": {
                    "trace_id": "0x123456789",
                    "span_id": "0x12345",
                },
                "parent_id": None,
                "start_time": 123000000,
                "end_time": 579000000,
                "status_code": "OK",
                "status_message": "",
                "attributes": {
                    "mlflow.traceRequestId": '"tr-1234567"',
                    "mlflow.spanType": '"LLM"',
                    "mlflow.spanFunctionName": '"predict"',
                    "mlflow.spanInputs": '{"prompt": "What is the meaning of life?"}',
                    "mlflow.spanOutputs": '{"answer": 42}',
                },
                "events": [],
            }
        ],
    }
    trace = MlflowClient().get_trace("1234567")
    mock_store.get_trace_info.assert_called_once_with("1234567")
    mock_artifact_repo.download_trace_data.assert_called_once()

    assert trace.info.trace_id == "tr-1234567"
    assert trace.info.experiment_id == "0"
    assert trace.info.timestamp_ms == 123
    assert trace.info.execution_time_ms == 456
    assert trace.info.status == TraceStatus.OK
    assert trace.info.tags == {"mlflow.artifactLocation": "dbfs:/path/to/artifacts"}
    assert trace.data.request == '{"prompt": "What is the meaning of life?"}'
    assert trace.data.response == '{"answer": 42}'
    assert len(trace.data.spans) == 1
    assert trace.data.spans[0].name == "predict"
    assert trace.data.spans[0].trace_id == "tr-1234567"
    assert trace.data.spans[0].inputs == {"prompt": "What is the meaning of life?"}
    assert trace.data.spans[0].outputs == {"answer": 42}
    assert trace.data.spans[0].start_time_ns == 123000000
    assert trace.data.spans[0].end_time_ns == 579000000
    assert trace.data.spans[0].status.status_code == SpanStatusCode.OK


def test_client_get_trace_throws_for_missing_or_corrupted_data(mock_store, mock_artifact_repo):
    mock_store.get_trace_info.return_value = TraceInfo(
        trace_id="1234567",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=123,
        execution_duration=456,
        state=TraceState.OK,
        tags={"mlflow.artifactLocation": "dbfs:/path/to/artifacts"},
    )
    mock_artifact_repo.download_trace_data.side_effect = MlflowTraceDataNotFound("1234567")

    with pytest.raises(
        MlflowException,
        match="Trace with ID 1234567 cannot be loaded because it is missing span data",
    ):
        MlflowClient().get_trace("1234567")

    mock_artifact_repo.download_trace_data.side_effect = MlflowTraceDataCorrupted("1234567")
    with pytest.raises(
        MlflowException,
        match="Trace with ID 1234567 cannot be loaded because its span data is corrupted",
    ):
        MlflowClient().get_trace("1234567")


@pytest.mark.parametrize("include_spans", [True, False])
@pytest.mark.parametrize("num_results", [0, 5])
def test_client_search_traces_with_get_traces(
    mock_store, mock_artifact_repo, include_spans, num_results
):
    mock_trace_infos = [
        TraceInfo(
            trace_id=f"trace:/catalog.schema/{i}",
            trace_location=TraceLocation(
                type=TraceLocationType.UC_SCHEMA,
                uc_schema=UCSchemaLocation(catalog_name="catalog", schema_name="schema"),
            ),
            request_time=123,
            execution_duration=456,
            state=TraceState.OK,
        )
        for i in range(num_results)
    ]
    mock_store.search_traces.return_value = (mock_trace_infos, None)
    mock_store.batch_get_traces.return_value = [
        Trace(info=info, data=TraceData(spans=[])) for info in mock_trace_infos
    ]

    results = MlflowClient().search_traces(
        locations=["catalog.schema"],
        include_spans=include_spans,
    )
    mock_store.search_traces.assert_called_once_with(
        experiment_ids=None,
        filter_string=None,
        max_results=100,
        order_by=None,
        page_token=None,
        model_id=None,
        locations=["catalog.schema"],
    )
    assert len(results) == num_results

    if include_spans and num_results > 0:
        mock_store.batch_get_traces.assert_called_once_with(
            [f"trace:/catalog.schema/{i}" for i in range(num_results)],
            "catalog.schema",
        )
    else:
        mock_store.batch_get_traces.assert_not_called()

    mock_artifact_repo.download_trace_data.assert_not_called()

    # The TraceInfo is already fetched prior to the upload_trace_data call,
    # so we should not call _get_trace_info again
    mock_store.get_trace_info.assert_not_called()


def test_client_search_traces_with_large_results(mock_store, mock_artifact_repo):
    mock_trace_infos = [
        TraceInfo(
            trace_id=f"trace:/catalog.schema/{i}",
            trace_location=TraceLocation(
                type=TraceLocationType.UC_SCHEMA,
                uc_schema=UCSchemaLocation(catalog_name="catalog", schema_name="schema"),
            ),
            request_time=123,
            execution_duration=456,
            state=TraceState.OK,
        )
        for i in range(100)
    ]
    mock_store.search_traces.return_value = (mock_trace_infos, None)

    mock_store.batch_get_traces.return_value = [
        Trace(info=mock_trace_infos[0], data=TraceData(spans=[])) for i in range(10)
    ]

    results = MlflowClient().search_traces(locations=["catalog.schema"])
    mock_store.search_traces.assert_called_once_with(
        experiment_ids=None,
        filter_string=None,
        max_results=100,
        order_by=None,
        page_token=None,
        model_id=None,
        locations=["catalog.schema"],
    )
    assert len(results) == 100
    assert mock_store.batch_get_traces.call_count == 10
    assert mock_store.batch_get_traces.has_calls(
        [
            mock.call([f"trace:/catalog.schema/{j * 10 + i}" for i in range(10)], "catalog.schema")
            for j in range(10)
        ]
    )
    mock_artifact_repo.download_trace_data.assert_not_called()


@pytest.mark.parametrize("include_spans", [True, False])
def test_client_search_traces_mixed(mock_store, mock_artifact_repo, include_spans):
    mock_traces = [
        TraceInfo(
            trace_id="1234567",
            trace_location=TraceLocation(
                type=TraceLocationType.UC_SCHEMA,
                uc_schema=UCSchemaLocation(catalog_name="catalog", schema_name="schema"),
            ),
            request_time=123,
            execution_duration=456,
            state=TraceState.OK,
            tags={"mlflow.artifactLocation": "dbfs:/path/to/artifacts/1"},
        ),
        TraceInfo(
            trace_id="8910",
            trace_location=TraceLocation.from_experiment_id("1"),
            request_time=456,
            execution_duration=789,
            state=TraceState.OK,
            tags={"mlflow.artifactLocation": "dbfs:/path/to/artifacts/2"},
        ),
    ]
    mock_store.search_traces.return_value = (mock_traces, None)
    mock_store.batch_get_traces.return_value = [
        Trace(info=mock_traces[0], data=TraceData(spans=[]))
    ]
    mock_artifact_repo.download_trace_data.return_value = {}
    results = MlflowClient().search_traces(
        locations=["1", "catalog.schema"], include_spans=include_spans
    )

    mock_store.search_traces.assert_called_once_with(
        experiment_ids=None,
        filter_string=None,
        max_results=100,
        order_by=None,
        page_token=None,
        model_id=None,
        locations=["1", "catalog.schema"],
    )
    assert len(results) == 2
    if include_spans:
        mock_store.batch_get_traces.assert_called_once_with(["1234567"], "catalog.schema")
        mock_artifact_repo.download_trace_data.assert_called()
    else:
        mock_store.batch_get_traces.assert_not_called()
        mock_artifact_repo.download_trace_data.assert_not_called()


@pytest.mark.parametrize("include_spans", [True, False])
@pytest.mark.parametrize("num_results", [0, 5])
def test_client_search_traces_with_get_traces_tracking_store(
    mock_store, mock_artifact_repo, include_spans, num_results
):
    mock_trace_infos = [
        TraceInfo(
            trace_id=f"tr-123456789{i}",
            trace_location=TraceLocation.from_experiment_id(f"exp-{i}"),
            request_time=123,
            execution_duration=456,
            state=TraceState.OK,
            tags={TraceTagKey.SPANS_LOCATION: SpansLocation.TRACKING_STORE},
        )
        for i in range(num_results)
    ]
    mock_store.search_traces.return_value = (mock_trace_infos, None)
    mock_store.batch_get_traces.return_value = [
        Trace(info=info, data=TraceData(spans=[])) for info in mock_trace_infos
    ]

    results = MlflowClient().search_traces(
        locations=["exp-0", "exp-1", "exp-2"],
        include_spans=include_spans,
    )
    mock_store.search_traces.assert_called_once_with(
        experiment_ids=None,
        filter_string=None,
        max_results=100,
        order_by=None,
        page_token=None,
        model_id=None,
        locations=["exp-0", "exp-1", "exp-2"],
    )
    assert len(results) == num_results

    if include_spans and num_results > 0:
        mock_store.batch_get_traces.assert_called_once_with(
            [f"tr-123456789{i}" for i in range(num_results)],
            None,
        )
    else:
        mock_store.batch_get_traces.assert_not_called()

    mock_artifact_repo.download_trace_data.assert_not_called()

    # The TraceInfo is already fetched prior to the upload_trace_data call,
    # so we should not call _get_trace_info again
    mock_store.get_trace_info.assert_not_called()


@pytest.mark.parametrize("include_spans", [True, False])
def test_client_search_traces_with_artifact_repo(mock_store, mock_artifact_repo, include_spans):
    mock_traces = [
        TraceInfo(
            trace_id="tr-1234567",
            trace_location=TraceLocation.from_experiment_id("1"),
            request_time=123,
            execution_duration=456,
            state=TraceState.OK,
            tags={"mlflow.artifactLocation": "dbfs:/path/to/artifacts/1"},
        ),
        TraceInfo(
            trace_id="tr-8910",
            trace_location=TraceLocation.from_experiment_id("2"),
            request_time=456,
            execution_duration=789,
            state=TraceState.OK,
            tags={"mlflow.artifactLocation": "dbfs:/path/to/artifacts/2"},
        ),
    ]
    mock_store.search_traces.return_value = (mock_traces, None)
    mock_artifact_repo.download_trace_data.return_value = {}
    results = MlflowClient().search_traces(locations=["1", "2", "3"], include_spans=include_spans)

    mock_store.search_traces.assert_called_once_with(
        experiment_ids=None,
        filter_string=None,
        max_results=100,
        order_by=None,
        page_token=None,
        model_id=None,
        locations=["1", "2", "3"],
    )
    assert len(results) == 2
    if include_spans:
        mock_artifact_repo.download_trace_data.assert_called()
    else:
        mock_artifact_repo.download_trace_data.assert_not_called()

    # The TraceInfo is already fetched prior to the upload_trace_data call,
    # so we should not call _get_trace_info again
    mock_store.get_trace_info.assert_not_called()


@pytest.mark.parametrize("include_spans", [True, False])
def test_client_search_traces_trace_data_download_error(mock_store, include_spans):
    class CustomArtifactRepository(ArtifactRepository):
        def log_artifact(self, local_file, artifact_path=None):
            raise NotImplementedError("Should not be called")

        def log_artifacts(self, local_dir, artifact_path=None):
            raise NotImplementedError("Should not be called")

        def list_artifacts(self, path):
            raise NotImplementedError("Should not be called")

        def _download_file(self, *args, **kwargs):
            raise Exception("Failed to download trace data")

    with mock.patch(
        "mlflow.tracing.client.get_artifact_repository",
        return_value=CustomArtifactRepository("test"),
    ) as mock_get_artifact_repository:
        mock_traces = [
            TraceInfo(
                trace_id="1234567",
                trace_location=TraceLocation.from_experiment_id("1"),
                request_time=123,
                execution_duration=456,
                state=TraceState.OK,
                tags={"mlflow.artifactLocation": "test"},
            ),
        ]
        mock_store.search_traces.return_value = (mock_traces, None)
        traces = MlflowClient().search_traces(experiment_ids=["1"], include_spans=include_spans)

        if include_spans:
            assert traces == []
            mock_get_artifact_repository.assert_called()
        else:
            assert len(traces) == 1
            assert traces[0].info.trace_id == "1234567"
            mock_get_artifact_repository.assert_not_called()


def test_client_search_traces_validates_experiment_ids_type():
    with pytest.raises(MlflowException, match=r"locations must be a list"):
        MlflowClient().search_traces(locations=4)

    with pytest.raises(MlflowException, match=r"locations must be a list"):
        MlflowClient().search_traces(locations="4")


def test_client_delete_traces(mock_store):
    MlflowClient().delete_traces(
        experiment_id="0",
        max_timestamp_millis=1,
        max_traces=2,
        trace_ids=["tr-1234"],
    )
    mock_store.delete_traces.assert_called_once_with(
        experiment_id="0",
        max_timestamp_millis=1,
        max_traces=2,
        trace_ids=["tr-1234"],
    )


@pytest.fixture
def disable_prompt_cache():
    from mlflow.environment_variables import (
        MLFLOW_ALIAS_PROMPT_CACHE_TTL_SECONDS,
        MLFLOW_VERSION_PROMPT_CACHE_TTL_SECONDS,
    )

    MLFLOW_ALIAS_PROMPT_CACHE_TTL_SECONDS.set(0)
    MLFLOW_VERSION_PROMPT_CACHE_TTL_SECONDS.set(0)
    yield
    MLFLOW_ALIAS_PROMPT_CACHE_TTL_SECONDS.unset()
    MLFLOW_VERSION_PROMPT_CACHE_TTL_SECONDS.unset()


@pytest.fixture(params=["file", "sqlalchemy"])
def tracking_uri(request, tmp_path, db_uri):
    """Set an MLflow Tracking URI with different type of backend."""
    if "MLFLOW_SKINNY" in os.environ and request.param == "sqlalchemy":
        pytest.skip("SQLAlchemy store is not available in skinny.")

    original_tracking_uri = mlflow.get_tracking_uri()

    if request.param == "file":
        tracking_uri = tmp_path.joinpath("file").as_uri()
    elif request.param == "sqlalchemy":
        tracking_uri = db_uri

    # NB: MLflow tracer does not handle the change of tracking URI well,
    # so we need to reset the tracer to switch the tracking URI during testing.
    mlflow.tracing.disable()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.tracing.enable()

    yield tracking_uri

    # Reset tracking URI
    mlflow.set_tracking_uri(original_tracking_uri)


@pytest.mark.parametrize("with_active_run", [True, False])
def test_start_and_end_trace(tracking_uri, with_active_run, async_logging_enabled):
    client = MlflowClient(tracking_uri)

    experiment_id = client.create_experiment("test_experiment")

    class TestModel:
        def predict(self, x, y):
            root_span = client.start_trace(
                name="predict",
                inputs={"x": x, "y": y},
                tags={"tag": "tag_value"},
                experiment_id=experiment_id,
            )
            trace_id = root_span.trace_id

            z = x + y

            child_span = client.start_span(
                "child_span_1",
                span_type=SpanType.LLM,
                trace_id=trace_id,
                parent_id=root_span.span_id,
                inputs={"z": z},
            )

            z = z + 2

            client.end_span(
                trace_id=trace_id,
                span_id=child_span.span_id,
                outputs={"output": z},
                attributes={"delta": 2},
            )

            res = self.square(z, trace_id, root_span.span_id)
            client.end_trace(trace_id, outputs={"output": res}, status="OK")
            return res

        def square(self, t, trace_id, parent_id):
            span = client.start_span(
                "child_span_2",
                trace_id=trace_id,
                parent_id=parent_id,
                inputs={"t": t},
            )

            res = t**2
            time.sleep(0.1)

            client.end_span(
                trace_id=trace_id,
                span_id=span.span_id,
                outputs={"output": res},
            )
            return res

    model = TestModel()
    if with_active_run:
        with mlflow.start_run(experiment_id=experiment_id) as run:
            model.predict(1, 2)
            run_id = run.info.run_id
    else:
        model.predict(1, 2)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    trace_id = mlflow.get_trace(mlflow.get_last_active_trace_id()).info.trace_id

    # Validate that trace is logged to the backend
    trace = client.get_trace(trace_id)
    assert trace is not None

    assert trace.info.trace_id is not None
    assert trace.info.execution_time_ms >= 0.1 * 1e3  # at least 0.1 sec
    assert trace.info.status == TraceStatus.OK
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 1, "y": 2}'
    assert trace.info.request_metadata[TraceMetadataKey.OUTPUTS] == '{"output": 25}'
    if with_active_run:
        assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_id
        assert trace.info.experiment_id == run.info.experiment_id
    else:
        assert trace.info.experiment_id == experiment_id

    assert trace.data.request == '{"x": 1, "y": 2}'
    assert trace.data.response == '{"output": 25}'
    assert len(trace.data.spans) == 3

    span_name_to_span = {span.name: span for span in trace.data.spans}
    root_span = span_name_to_span["predict"]
    # NB: Start time of root span and trace info does not match because there is some
    #   latency for starting the trace within the backend
    # assert root_span.start_time_ns // 1e6 == trace.info.timestamp_ms
    assert root_span.parent_id is None
    assert root_span.attributes == {
        "mlflow.experimentId": experiment_id,
        "mlflow.traceRequestId": trace.info.trace_id,
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"x": 1, "y": 2},
        "mlflow.spanOutputs": {"output": 25},
    }

    child_span_1 = span_name_to_span["child_span_1"]
    assert child_span_1.parent_id == root_span.span_id
    assert child_span_1.attributes == {
        "mlflow.traceRequestId": trace.info.trace_id,
        "mlflow.spanType": "LLM",
        "mlflow.spanInputs": {"z": 3},
        "mlflow.spanOutputs": {"output": 5},
        "delta": 2,
    }

    child_span_2 = span_name_to_span["child_span_2"]
    assert child_span_2.parent_id == root_span.span_id
    assert child_span_2.attributes == {
        "mlflow.traceRequestId": trace.info.trace_id,
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"t": 5},
        "mlflow.spanOutputs": {"output": 25},
    }
    assert child_span_2.start_time_ns <= child_span_2.end_time_ns - 0.1 * 1e6


def test_start_and_end_trace_capture_falsy_input_and_output(tracking_uri):
    # This test is to verify that falsy input and output values are correctly logged
    client = MlflowClient(tracking_uri)
    experiment_id = client.create_experiment("test_experiment")

    root = client.start_trace(name="root", experiment_id=experiment_id, inputs=[])
    span = client.start_span(name="child", trace_id=root.trace_id, parent_id=root.span_id, inputs=0)
    client.end_span(trace_id=root.trace_id, span_id=span.span_id, outputs=False)
    client.end_trace(trace_id=root.trace_id, outputs="")

    trace = client.get_trace(root.trace_id)
    assert trace.data.spans[0].inputs == []
    assert trace.data.spans[0].outputs == ""
    assert trace.data.spans[1].inputs == 0
    assert trace.data.spans[1].outputs is False


# TODO: we should investigate whether we need to support this
@pytest.mark.skip(reason="This is not supported by latest span-level export")
@pytest.mark.usefixtures("reset_active_experiment")
def test_start_and_end_trace_before_all_span_end(async_logging_enabled):
    # This test is to verify that the trace is still exported even if some spans are not ended
    exp_id = mlflow.set_experiment("test_experiment_1").experiment_id

    class TestModel:
        def __init__(self):
            self._client = MlflowClient()

        def predict(self, x):
            root_span = self._client.start_trace(name="predict")
            trace_id = root_span.trace_id
            child_span = self._client.start_span(
                "ended-span",
                trace_id=trace_id,
                parent_id=root_span.span_id,
            )
            time.sleep(0.1)
            self._client.end_span(trace_id, child_span.span_id)

            res = self.square(x, trace_id, root_span.span_id)
            self._client.end_trace(trace_id)
            return res

        def square(self, t, trace_id, parent_id):
            self._client.start_span("non-ended-span", trace_id=trace_id, parent_id=parent_id)
            time.sleep(0.1)
            # The span created above is not ended
            return t**2

    model = TestModel()
    model.predict(1)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = MlflowClient().search_traces(experiment_ids=[exp_id])
    assert len(traces) == 1

    trace_info = traces[0].info
    assert trace_info.trace_id is not None
    assert trace_info.experiment_id == exp_id
    assert trace_info.timestamp_ms is not None
    assert trace_info.execution_time_ms is not None
    assert trace_info.status == TraceStatus.OK

    trace_data = traces[0].data
    assert trace_data.request is None
    assert trace_data.response is None
    assert len(trace_data.spans) == 3  # The non-ended span should be also included in the trace

    span_name_to_span = {span.name: span for span in trace_data.spans}
    root_span = span_name_to_span["predict"]
    assert root_span.parent_id is None
    assert root_span.status.status_code == SpanStatusCode.OK

    ended_span = span_name_to_span["ended-span"]
    assert ended_span.parent_id == root_span.span_id
    assert ended_span.start_time_ns < ended_span.end_time_ns
    assert ended_span.status.status_code == SpanStatusCode.OK

    # The non-ended span should have null end_time and UNSET status
    non_ended_span = span_name_to_span["non-ended-span"]
    assert non_ended_span.parent_id == root_span.span_id
    assert non_ended_span.start_time_ns is not None
    assert non_ended_span.end_time_ns is None
    assert non_ended_span.status.status_code == SpanStatusCode.UNSET


def test_log_trace_with_databricks_tracking_uri(mock_store_start_trace, monkeypatch):
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")

    mock_experiment = mock.MagicMock()
    mock_experiment.experiment_id = "test_experiment_id"

    class TestModel:
        def __init__(self):
            self._client = MlflowClient()

        def predict(self, x, y):
            root_span = self._client.start_trace(
                name="predict",
                inputs={"x": x, "y": y},
                # Trying to override mlflow.user tag, which will be ignored
                tags={"tag": "tag_value", "mlflow.user": "unknown"},
            )
            trace_id = root_span.trace_id

            z = x + y

            child_span = self._client.start_span(
                "child_span_1",
                span_type=SpanType.LLM,
                trace_id=trace_id,
                parent_id=root_span.span_id,
                inputs={"z": z},
            )

            z = z + 2

            self._client.end_span(
                trace_id=trace_id,
                span_id=child_span.span_id,
                outputs={"output": z},
                attributes={"delta": 2},
            )
            self._client.end_trace(trace_id, outputs=z, status="OK")
            return z

    model = TestModel()

    with (
        mock.patch("mlflow.get_tracking_uri", return_value="databricks"),
        mock.patch("mlflow.tracking.context.default_context._get_source_name", return_value="test"),
        mock.patch(
            "mlflow.tracing.client.TracingClient._upload_trace_data"
        ) as mock_upload_trace_data,
        mock.patch(
            "mlflow.tracing.client.TracingClient.set_trace_tags",
        ),
        mock.patch(
            "mlflow.tracing.client.TracingClient.set_trace_tag",
        ),
        mock.patch(
            "mlflow.tracing.client.TracingClient.get_trace_info",
        ),
        mock.patch(
            "mlflow.MlflowClient.get_experiment_by_name",
            return_value=mock_experiment,
        ),
    ):
        model.predict(1, 2)
        mlflow.flush_trace_async_logging(terminate=True)

    mock_store_start_trace.assert_called_once()
    mock_upload_trace_data.assert_called_once()


def test_start_and_end_trace_does_not_log_trace_when_disabled(
    tracking_uri, monkeypatch, async_logging_enabled
):
    client = MlflowClient(tracking_uri)
    experiment_id = client.create_experiment("test_experiment")

    @trace_disabled
    def func():
        span = client.start_trace(
            name="predict",
            experiment_id=experiment_id,
            inputs={"x": 1, "y": 2},
            attributes={"attr": "value"},
            tags={"tag": "tag_value"},
        )
        child_span = client.start_span(
            "child_span_1",
            trace_id=span.trace_id,
            parent_id=span.span_id,
        )
        client.end_span(
            trace_id=span.trace_id,
            span_id=child_span.span_id,
            outputs={"output": 5},
        )
        client.end_trace(span.trace_id, outputs=5, status="OK")
        return "done"

    mock_logger = mock.MagicMock()
    monkeypatch.setattr(mlflow.tracking.client, "_logger", mock_logger)

    res = func()

    assert res == "done"
    assert client.search_traces(experiment_ids=[experiment_id]) == []
    # No warning should be issued
    mock_logger.warning.assert_not_called()


def test_start_trace_within_active_run(async_logging_enabled):
    exp_id = mlflow.create_experiment("test")

    client = mlflow.MlflowClient()
    with mlflow.start_run():
        root_span = client.start_trace(
            name="test",
            experiment_id=exp_id,
        )
        client.end_trace(root_span.trace_id)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = client.search_traces(experiment_ids=[exp_id])
    assert len(traces) == 1
    assert traces[0].info.experiment_id == exp_id


def test_start_trace_raise_error_when_active_trace_exists():
    with mlflow.start_span("fluent_span"):
        with pytest.raises(MlflowException, match=r"Another trace is already set in the global"):
            mlflow.tracking.MlflowClient().start_trace("test")


def test_end_trace_raise_error_when_trace_not_exist():
    client = mlflow.tracking.MlflowClient()
    mock_tracing_client = mock.MagicMock()
    mock_tracing_client.get_trace.return_value = None
    client._tracing_client = mock_tracing_client

    with pytest.raises(MlflowException, match=r"Trace with ID test not found"):
        client.end_trace("test")


def test_end_trace_works_for_trace_in_pending_status():
    client = mlflow.tracking.MlflowClient()
    mock_tracing_client = mock.MagicMock()
    mock_tracing_client.get_trace.return_value = Trace(
        info=create_test_trace_info("test", state=TraceState.IN_PROGRESS), data=TraceData()
    )
    client._tracing_client = mock_tracing_client
    client.end_span = lambda *args: None

    client.end_trace("test")


@pytest.mark.parametrize("state", [TraceState.OK, TraceState.ERROR])
def test_end_trace_raise_error_for_trace_in_end_status(state):
    client = mlflow.tracking.MlflowClient()
    mock_tracing_client = mock.MagicMock()
    mock_tracing_client.get_trace.return_value = Trace(
        info=create_test_trace_info("test", state=state), data=TraceData()
    )
    client._tracing_client = mock_tracing_client

    with pytest.raises(MlflowException, match=r"Trace with ID test already finished"):
        client.end_trace("test")


def test_trace_status_either_pending_or_end():
    all_statuses = {status.value for status in TraceStatus}
    pending_or_end_statuses = TraceStatus.pending_statuses() | TraceStatus.end_statuses()
    unclassified_statuses = all_statuses - pending_or_end_statuses
    assert len(unclassified_statuses) == 0, (
        f"Please add {unclassified_statuses} to "
        "either pending_statuses or end_statuses in TraceStatus class definition"
    )


def test_start_span_raise_error_when_parent_id_is_not_provided():
    with pytest.raises(MlflowException, match=r"start_span\(\) must be called with"):
        mlflow.tracking.MlflowClient().start_span("span_name", trace_id="test", parent_id=None)


def test_log_trace(tracking_uri):
    client = MlflowClient(tracking_uri)
    experiment_id = client.create_experiment("test_experiment")

    span = client.start_trace(
        name="test",
        span_type=SpanType.LLM,
        experiment_id=experiment_id,
        tags={"custom_tag": "tag_value"},
    )
    client.end_trace(span.trace_id, status="OK")

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    # Purge all traces in the backend once
    client.delete_traces(experiment_id=experiment_id, trace_ids=[trace.info.trace_id])
    assert client.search_traces(experiment_ids=[experiment_id]) == []

    # Log the trace manually
    new_trace_id = client._log_trace(trace)

    # Validate the trace is added to the backend
    backend_traces = client.search_traces(experiment_ids=[experiment_id])
    assert len(backend_traces) == 1
    assert backend_traces[0].info.trace_id == new_trace_id  # new request ID is assigned
    assert backend_traces[0].info.experiment_id == experiment_id
    assert backend_traces[0].info.status == trace.info.status
    assert backend_traces[0].info.tags["custom_tag"] == "tag_value"
    assert backend_traces[0].info.request_preview == trace.info.request_preview
    assert backend_traces[0].info.response_preview == trace.info.response_preview
    assert len(backend_traces[0].data.spans) == len(trace.data.spans)
    assert backend_traces[0].data.spans[0].name == trace.data.spans[0].name

    # If the experiment ID is None in the given trace, it should be set to the default experiment
    trace.info.experiment_id = None
    new_trace_id = client._log_trace(trace)
    backend_traces = client.search_traces(experiment_ids=[DEFAULT_EXPERIMENT_ID])
    assert len(backend_traces) == 1


def test_search_traces_experiment_ids_deprecation_warning():
    client = MlflowClient()
    exp_id = mlflow.set_experiment("test_experiment_deprecation").experiment_id

    # Test that using experiment_ids shows a deprecation warning
    with pytest.warns(FutureWarning, match="experiment_ids.*deprecated.*use.*locations"):
        result = client.search_traces(experiment_ids=[exp_id])
    assert isinstance(result, list)


def test_ignore_exception_from_tracing_logic(monkeypatch, async_logging_enabled):
    exp_id = mlflow.set_experiment("test_experiment_1").experiment_id
    client = MlflowClient()

    class TestModel:
        def predict(self, x):
            root_span = client.start_trace(experiment_id=exp_id, name="predict")
            trace_id = root_span.trace_id
            child_span = client.start_span(
                name="child", trace_id=trace_id, parent_id=root_span.span_id
            )
            client.end_span(trace_id, child_span.span_id)
            client.end_trace(trace_id)
            return x

    model = TestModel()

    # Mock the span processor's on_end handler to raise an exception
    processor = _get_tracer(__name__).span_processor

    def _always_fail(*args, **kwargs):
        raise ValueError("Some error")

    # Exception while starting the trace should be caught not raise
    monkeypatch.setattr(processor, "on_start", _always_fail)
    response = model.predict(1)
    assert response == 1
    assert len(get_traces()) == 0

    # Exception while ending the trace should be caught not raise
    monkeypatch.setattr(processor, "on_end", _always_fail)
    response = model.predict(1)
    assert response == 1
    assert len(get_traces()) == 0


def test_set_and_delete_trace_tag_on_active_trace(monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")

    client = mlflow.tracking.MlflowClient()

    root_span = client.start_trace(name="test")
    trace_id = root_span.trace_id
    client.set_trace_tag(trace_id, "foo", "bar")
    client.end_trace(trace_id)

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.tags["foo"] == "bar"


def test_set_trace_tag_on_logged_trace(mock_store):
    mlflow.tracking.MlflowClient().set_trace_tag("test", "foo", "bar")
    mlflow.tracking.MlflowClient().set_trace_tag("test", "mlflow.some.reserved.tag", "value")
    mock_store.set_trace_tag.assert_has_calls(
        [
            mock.call("test", "foo", "bar"),
            mock.call("test", "mlflow.some.reserved.tag", "value"),
        ]
    )


def test_delete_trace_tag_on_active_trace(monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")

    client = mlflow.tracking.MlflowClient()
    root_span = client.start_trace(name="test", tags={"foo": "bar", "baz": "qux"})
    trace_id = root_span.trace_id
    client.delete_trace_tag(trace_id, "foo")
    client.end_trace(trace_id)

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert "baz" in trace.info.tags
    assert "foo" not in trace.info.tags


def test_delete_trace_tag_on_logged_trace(mock_store):
    mlflow.tracking.MlflowClient().delete_trace_tag("test", "foo")
    mock_store.delete_trace_tag.assert_called_once_with("test", "foo")


def test_client_create_experiment(mock_store):
    MlflowClient().create_experiment("someName", "someLocation", {"key1": "val1", "key2": "val2"})

    mock_store.create_experiment.assert_called_once_with(
        artifact_location="someLocation",
        tags=[ExperimentTag("key1", "val1"), ExperimentTag("key2", "val2")],
        name="someName",
    )


def test_client_create_run_overrides(mock_store):
    experiment_id = mock.Mock()
    user = mock.Mock()
    start_time = mock.Mock()
    run_name = mock.Mock()
    tags = {
        MLFLOW_USER: user,
        MLFLOW_PARENT_RUN_ID: mock.Mock(),
        MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.JOB),
        MLFLOW_SOURCE_NAME: mock.Mock(),
        MLFLOW_PROJECT_ENTRY_POINT: mock.Mock(),
        MLFLOW_GIT_COMMIT: mock.Mock(),
        "other-key": "other-value",
    }

    MlflowClient().create_run(experiment_id, start_time, tags, run_name)

    mock_store.create_run.assert_called_once_with(
        experiment_id=experiment_id,
        user_id=user,
        start_time=start_time,
        tags=[RunTag(key, value) for key, value in tags.items()],
        run_name=run_name,
    )
    mock_store.reset_mock()
    MlflowClient().create_run(experiment_id, start_time, tags)
    mock_store.create_run.assert_called_once_with(
        experiment_id=experiment_id,
        user_id=user,
        start_time=start_time,
        tags=[RunTag(key, value) for key, value in tags.items()],
        run_name=None,
    )


def test_client_set_terminated_no_change_name(mock_store):
    experiment_id = mock.Mock()
    run = MlflowClient().create_run(experiment_id, run_name="my name")
    MlflowClient().set_terminated(run.info.run_id)
    _, kwargs = mock_store.update_run_info.call_args
    assert kwargs["run_name"] is None


def test_client_search_runs_defaults(mock_store):
    MlflowClient().search_runs([1, 2, 3])
    mock_store.search_runs.assert_called_once_with(
        experiment_ids=[1, 2, 3],
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
        order_by=None,
        page_token=None,
    )


def test_client_search_runs_filter(mock_store):
    MlflowClient().search_runs(["a", "b", "c"], "my filter")
    mock_store.search_runs.assert_called_once_with(
        experiment_ids=["a", "b", "c"],
        filter_string="my filter",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
        order_by=None,
        page_token=None,
    )


def test_client_search_runs_view_type(mock_store):
    MlflowClient().search_runs(["a", "b", "c"], "my filter", ViewType.DELETED_ONLY)
    mock_store.search_runs.assert_called_once_with(
        experiment_ids=["a", "b", "c"],
        filter_string="my filter",
        run_view_type=ViewType.DELETED_ONLY,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
        order_by=None,
        page_token=None,
    )


def test_client_search_runs_max_results(mock_store):
    MlflowClient().search_runs([5], "my filter", ViewType.ALL, 2876)
    mock_store.search_runs.assert_called_once_with(
        experiment_ids=[5],
        filter_string="my filter",
        run_view_type=ViewType.ALL,
        max_results=2876,
        order_by=None,
        page_token=None,
    )


def test_client_search_runs_int_experiment_id(mock_store):
    MlflowClient().search_runs(123)
    mock_store.search_runs.assert_called_once_with(
        experiment_ids=[123],
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
        order_by=None,
        page_token=None,
    )


def test_client_search_runs_string_experiment_id(mock_store):
    MlflowClient().search_runs("abc")
    mock_store.search_runs.assert_called_once_with(
        experiment_ids=["abc"],
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
        order_by=None,
        page_token=None,
    )


def test_client_search_runs_order_by(mock_store):
    MlflowClient().search_runs([5], order_by=["a", "b"])
    mock_store.search_runs.assert_called_once_with(
        experiment_ids=[5],
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
        order_by=["a", "b"],
        page_token=None,
    )


def test_client_search_runs_page_token(mock_store):
    MlflowClient().search_runs([5], page_token="blah")
    mock_store.search_runs.assert_called_once_with(
        experiment_ids=[5],
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
        order_by=None,
        page_token="blah",
    )


def test_update_registered_model(mock_registry_store):
    """
    Update registered model no longer supports name change.
    """
    expected_return_value = "some expected return value."
    mock_registry_store.rename_registered_model.return_value = expected_return_value
    expected_return_value_2 = "other expected return value."
    mock_registry_store.update_registered_model.return_value = expected_return_value_2
    res = MlflowClient(registry_uri="sqlite:///somedb.db").update_registered_model(
        name="orig name", description="new description"
    )
    assert expected_return_value_2 == res
    mock_registry_store.update_registered_model.assert_called_once_with(
        name="orig name", description="new description", deployment_job_id=None
    )
    mock_registry_store.rename_registered_model.assert_not_called()


def test_create_model_version(mock_registry_store):
    """
    Basic test for create model version.
    """
    mock_registry_store.create_model_version.return_value = _default_model_version()
    res = MlflowClient(registry_uri="sqlite:///somedb.db").create_model_version(
        "orig name", "source", "run-id", tags={"key": "value"}, description="desc"
    )
    assert res == _default_model_version()
    mock_registry_store.create_model_version.assert_called_once_with(
        "orig name",
        "source",
        "run-id",
        [ModelVersionTag(key="key", value="value")],
        None,
        "desc",
        local_model_path=None,
        model_id=None,
    )


def test_update_model_version(mock_registry_store):
    """
    Update registered model no longer support state changes.
    """
    mock_registry_store.update_model_version.return_value = _default_model_version()
    res = MlflowClient(registry_uri="sqlite:///somedb.db").update_model_version(
        name="orig name", version="1", description="desc"
    )
    assert _default_model_version() == res
    mock_registry_store.update_model_version.assert_called_once_with(
        name="orig name", version="1", description="desc"
    )
    mock_registry_store.transition_model_version_stage.assert_not_called()


def test_transition_model_version_stage(mock_registry_store):
    name = "Model 1"
    version = "12"
    stage = "Production"
    expected_result = ModelVersion(name, version, creation_timestamp=123, current_stage=stage)
    mock_registry_store.transition_model_version_stage.return_value = expected_result
    actual_result = MlflowClient(registry_uri="sqlite:///somedb.db").transition_model_version_stage(
        name, version, stage
    )
    mock_registry_store.transition_model_version_stage.assert_called_once_with(
        name=name, version=version, stage=stage, archive_existing_versions=False
    )
    assert expected_result == actual_result


def test_registry_uri_set_as_param():
    uri = "sqlite:///somedb.db"
    client = MlflowClient(tracking_uri="databricks://tracking", registry_uri=uri)
    assert client._registry_uri == uri


def test_registry_uri_from_set_registry_uri():
    uri = "sqlite:///somedb.db"
    set_registry_uri(uri)
    client = MlflowClient(tracking_uri="databricks://tracking")
    assert client._registry_uri == uri
    set_registry_uri(None)


def test_registry_uri_from_tracking_uri_param():
    tracking_uri = "databricks://tracking_vhawoierj"
    with mock.patch(
        "mlflow.tracking._tracking_service.utils.get_tracking_uri",
        return_value=tracking_uri,
    ):
        client = MlflowClient(tracking_uri=tracking_uri)
        # For databricks tracking URIs, registry URI defaults to Unity Catalog with profile
        assert client._registry_uri == "databricks-uc://tracking_vhawoierj"


def test_registry_uri_from_implicit_tracking_uri():
    tracking_uri = "databricks://tracking_wierojasdf"
    with mock.patch(
        "mlflow.tracking._tracking_service.utils.get_tracking_uri",
        return_value=tracking_uri,
    ):
        client = MlflowClient()
        # For databricks tracking URIs, registry URI defaults to Unity Catalog with profile
        assert client._registry_uri == "databricks-uc://tracking_wierojasdf"


def test_create_model_version_nondatabricks_source_no_runlink(mock_registry_store):
    run_id = "runid"
    client = MlflowClient(tracking_uri="http://10.123.1231.11")
    mock_registry_store.create_model_version.return_value = ModelVersion(
        "name",
        1,
        0,
        1,
        source="source",
        run_id=run_id,
    )
    model_version = client.create_model_version("name", "source", "runid")
    assert model_version.name == "name"
    assert model_version.source == "source"
    assert model_version.run_id == "runid"
    # verify that the store was not provided a run link
    mock_registry_store.create_model_version.assert_called_once_with(
        "name", "source", "runid", [], None, None, local_model_path=None, model_id=None
    )


def test_create_model_version_nondatabricks_source_no_run_id(mock_registry_store):
    client = MlflowClient(tracking_uri="http://10.123.1231.11")
    mock_registry_store.create_model_version.return_value = ModelVersion(
        "name", 1, 0, 1, source="source"
    )
    model_version = client.create_model_version("name", "source")
    assert model_version.name == "name"
    assert model_version.source == "source"
    assert model_version.run_id is None
    # verify that the store was not provided a run id
    mock_registry_store.create_model_version.assert_called_once_with(
        "name", "source", None, [], None, None, local_model_path=None, model_id=None
    )


def test_create_model_version_explicitly_set_run_link(
    mock_registry_store, mock_databricks_tracking_store
):
    run_link = "my-run-link"
    hostname = "https://workspace.databricks.com/"
    workspace_id = "10002"
    mock_registry_store.create_model_version.return_value = ModelVersion(
        "name",
        1,
        0,
        1,
        source="source",
        run_id=mock_databricks_tracking_store.run_id,
        run_link=run_link,
    )

    # mocks to make sure that even if you're in a notebook, this setting is respected.
    with (
        mock.patch("mlflow.utils.databricks_utils.is_in_databricks_notebook", return_value=True),
        mock.patch(
            "mlflow.utils.databricks_utils.get_workspace_info_from_dbutils",
            return_value=(hostname, workspace_id),
        ),
    ):
        client = MlflowClient(tracking_uri="databricks", registry_uri="otherplace")
        model_version = client.create_model_version("name", "source", "runid", run_link=run_link)
        assert model_version.run_link == run_link
        # verify that the store was provided with the explicitly passed in run link
        mock_registry_store.create_model_version.assert_called_once_with(
            "name", "source", "runid", [], run_link, None, local_model_path=None, model_id=None
        )


def test_create_model_version_run_link_in_notebook_with_default_profile(
    mock_registry_store, mock_databricks_tracking_store
):
    hostname = "https://workspace.databricks.com/"
    workspace_id = "10002"
    workspace_url = _construct_databricks_run_url(
        hostname,
        mock_databricks_tracking_store.experiment_id,
        mock_databricks_tracking_store.run_id,
        workspace_id,
    )

    with (
        mock.patch("mlflow.utils.databricks_utils.is_in_databricks_notebook", return_value=True),
        mock.patch(
            "mlflow.utils.databricks_utils.get_workspace_info_from_dbutils",
            return_value=(hostname, workspace_id),
        ),
    ):
        client = MlflowClient(tracking_uri="databricks", registry_uri="otherplace")
        mock_registry_store.create_model_version.return_value = ModelVersion(
            "name",
            1,
            0,
            1,
            source="source",
            run_id=mock_databricks_tracking_store.run_id,
            run_link=workspace_url,
        )
        model_version = client.create_model_version("name", "source", "runid")
        assert model_version.run_link == workspace_url
        # verify that the client generated the right URL
        mock_registry_store.create_model_version.assert_called_once_with(
            "name", "source", "runid", [], workspace_url, None, local_model_path=None, model_id=None
        )


def test_create_model_version_with_source(mock_registry_store, mock_databricks_tracking_store):
    model_id = "model_id"
    mock_registry_store.create_model_version.return_value = ModelVersion(
        "name",
        1,
        0,
        1,
        source="/path/to/source",
        run_id=mock_databricks_tracking_store.run_id,
        run_link=None,
        model_id=model_id,
    )
    mock_logged_model = LoggedModel(
        experiment_id="exp_id",
        model_id="model_id",
        name="name",
        artifact_location="/path/to/source",
        creation_timestamp=0,
        last_updated_timestamp=0,
    )

    with mock.patch(
        "mlflow.tracking.client.MlflowClient.get_logged_model", return_value=mock_logged_model
    ):
        client = MlflowClient(tracking_uri="databricks")
        model_version = client.create_model_version(
            "name", f"models:/{model_id}", "runid", run_link=None, model_id=model_id
        )
        assert model_version.model_id == model_id
        mock_registry_store.create_model_version.assert_called_once_with(
            "name",
            f"models:/{model_id}",
            "runid",
            [],
            None,
            None,
            local_model_path=None,
            model_id="model_id",
        )

    mock_registry_store.create_model_version.reset_mock()
    with mock.patch(
        "mlflow.tracking.client.MlflowClient.get_logged_model", return_value=mock_logged_model
    ):
        client = MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")
        model_version = client.create_model_version(
            "name", f"models:/{model_id}", "runid", run_link=None, model_id=model_id
        )
        assert model_version.model_id == model_id
        mock_registry_store.create_model_version.assert_called_once_with(
            "name",
            f"models:/{model_id}",
            "runid",
            [],
            None,
            None,
            local_model_path=None,
            model_id="model_id",
        )


def test_create_model_version_with_nondatabricks_source_uc_registry(mock_registry_store):
    name = "name"
    model_id = "model_id"
    run_id = "runid"
    source = "/path/to/source"
    model_uri = f"models:/{model_id}"
    mock_registry_store.create_model_version.return_value = ModelVersion(
        "name",
        1,
        0,
        1,
        source=source,
        run_id=run_id,
        run_link=None,
        model_id=model_id,
    )
    mock_logged_model = LoggedModel(
        experiment_id="exp_id",
        model_id=model_id,
        name=name,
        artifact_location=source,
        creation_timestamp=0,
        last_updated_timestamp=0,
    )

    with mock.patch(
        "mlflow.tracking.client.MlflowClient.get_logged_model", return_value=mock_logged_model
    ):
        client = MlflowClient(tracking_uri="http://10.123.1231.11", registry_uri="databricks-uc")
        model_version = client.create_model_version(
            name, model_uri, run_id, run_link=None, model_id=model_id
        )
        assert model_version.model_id == model_id
        mock_registry_store.create_model_version.assert_called_once_with(
            name,
            source,
            run_id,
            [],
            None,
            None,
            local_model_path=None,
            model_id=None,
        )


def test_creation_default_values_in_unity_catalog(mock_registry_store):
    client = MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")
    mock_registry_store.create_model_version.return_value = ModelVersion(
        "name",
        1,
        0,
        1,
        source="source",
        run_id="runid",
    )
    client.create_model_version("name", "source", "runid")
    # verify that registry store was called with tags=[] and run_link=None
    mock_registry_store.create_model_version.assert_called_once_with(
        "name", "source", "runid", [], None, None, local_model_path=None, model_id=None
    )
    client.create_registered_model(name="name", description="description")
    # verify that registry store was called with tags=[]
    mock_registry_store.create_registered_model.assert_called_once_with(
        "name", [], "description", None
    )


def test_await_model_version_creation(mock_registry_store):
    mv = ModelVersion(
        name="name",
        version=1,
        creation_timestamp=123,
        status=ModelVersionStatus.to_string(ModelVersionStatus.FAILED_REGISTRATION),
    )
    mock_registry_store.create_model_version.return_value = mv

    client = MlflowClient(tracking_uri="http://10.123.1231.11")

    client.create_model_version("name", "source")
    mock_registry_store._await_model_version_creation.assert_called_once_with(
        mv, DEFAULT_AWAIT_MAX_SLEEP_SECONDS
    )


def test_create_model_version_run_link_with_configured_profile(
    mock_registry_store, mock_databricks_tracking_store
):
    hostname = "https://workspace.databricks.com/"
    workspace_id = "10002"
    workspace_url = _construct_databricks_run_url(
        hostname,
        mock_databricks_tracking_store.experiment_id,
        mock_databricks_tracking_store.run_id,
        workspace_id,
    )

    with (
        mock.patch("mlflow.utils.databricks_utils.is_in_databricks_notebook", return_value=False),
        mock.patch(
            "mlflow.utils.databricks_utils.get_workspace_info_from_databricks_secrets",
            return_value=(hostname, workspace_id),
        ),
    ):
        client = MlflowClient(tracking_uri="databricks", registry_uri="otherplace")
        mock_registry_store.create_model_version.return_value = ModelVersion(
            "name",
            1,
            0,
            1,
            source="source",
            run_id=mock_databricks_tracking_store.run_id,
            run_link=workspace_url,
        )
        model_version = client.create_model_version("name", "source", "runid")
        assert model_version.run_link == workspace_url
        # verify that the client generated the right URL
        mock_registry_store.create_model_version.assert_called_once_with(
            "name", "source", "runid", [], workspace_url, None, local_model_path=None, model_id=None
        )


def test_create_model_version_copy_called_db_to_db(mock_registry_store):
    client = MlflowClient(
        tracking_uri="databricks://tracking",
        registry_uri="databricks://registry:workspace",
    )
    mock_registry_store.create_model_version.return_value = _default_model_version()
    with mock.patch("mlflow.tracking.client._upload_artifacts_to_databricks") as upload_mock:
        client.create_model_version(
            "model name",
            "dbfs:/source",
            "run_12345",
            run_link="not:/important/for/test",
        )
        upload_mock.assert_called_once_with(
            "dbfs:/source",
            "run_12345",
            "databricks://tracking",
            "databricks://registry:workspace",
        )


def test_create_model_version_copy_called_nondb_to_db(mock_registry_store):
    client = MlflowClient(
        tracking_uri="https://tracking", registry_uri="databricks://registry:workspace"
    )
    mock_registry_store.create_model_version.return_value = _default_model_version()
    with mock.patch("mlflow.tracking.client._upload_artifacts_to_databricks") as upload_mock:
        client.create_model_version(
            "model name", "s3:/source", "run_12345", run_link="not:/important/for/test"
        )
        upload_mock.assert_called_once_with(
            "s3:/source",
            "run_12345",
            "https://tracking",
            "databricks://registry:workspace",
        )


def test_create_model_version_copy_not_called_to_db(mock_registry_store):
    client = MlflowClient(
        tracking_uri="databricks://registry:workspace",
        registry_uri="databricks://registry:workspace",
    )
    mock_registry_store.create_model_version.return_value = _default_model_version()
    with mock.patch("mlflow.tracking.client._upload_artifacts_to_databricks") as upload_mock:
        client.create_model_version(
            "model name",
            "dbfs:/source",
            "run_12345",
            run_link="not:/important/for/test",
        )
        upload_mock.assert_not_called()


def test_create_model_version_copy_not_called_to_nondb(mock_registry_store):
    client = MlflowClient(tracking_uri="databricks://tracking", registry_uri="https://registry")
    mock_registry_store.create_model_version.return_value = _default_model_version()
    with mock.patch("mlflow.tracking.client._upload_artifacts_to_databricks") as upload_mock:
        client.create_model_version(
            "model name",
            "dbfs:/source",
            "run_12345",
            run_link="not:/important/for/test",
        )
        upload_mock.assert_not_called()


def _default_model_version():
    return ModelVersion("model name", 1, creation_timestamp=123, status="READY")


def test_client_can_be_serialized_with_pickle(tmp_path):
    """
    Verifies that instances of `MlflowClient` can be serialized using pickle, even if the underlying
    Tracking and Model Registry stores used by the client are not serializable using pickle
    """

    class MockUnpickleableTrackingStore(SqlAlchemyTrackingStore):
        pass

    class MockUnpickleableModelRegistryStore(SqlAlchemyModelRegistryStore):
        pass

    backend_store_path = tmp_path.joinpath("test.db")
    artifact_store_path = tmp_path.joinpath("artifacts")

    mock_tracking_store = MockUnpickleableTrackingStore(
        f"sqlite:///{backend_store_path}", str(artifact_store_path)
    )
    mock_model_registry_store = MockUnpickleableModelRegistryStore(
        f"sqlite:///{backend_store_path}"
    )

    # Verify that the mock stores cannot be pickled because they are defined within a function
    # (i.e. the test function)
    with pytest.raises(AttributeError, match="<locals>.MockUnpickleableTrackingStore'"):
        pickle.dumps(mock_tracking_store)

    with pytest.raises(AttributeError, match="<locals>.MockUnpickleableModelRegistryStore'"):
        pickle.dumps(mock_model_registry_store)

    _register("pickle", lambda *args, **kwargs: mock_tracking_store)
    _get_model_registry_store_registry().register(
        "pickle", lambda *args, **kwargs: mock_model_registry_store
    )

    # Create an MlflowClient with the store that cannot be pickled, perform
    # tracking & model registry operations, and verify that the client can still be pickled
    client = MlflowClient("pickle://foo")
    client.create_experiment("test_experiment")
    client.create_registered_model("test_model")
    pickle.dumps(client)


@pytest.fixture
def mock_registry_store_with_get_latest_version(mock_registry_store):
    mock_get_latest_versions = mock.Mock()
    mock_get_latest_versions.return_value = [
        ModelVersion(
            "model_name",
            1,
            0,
        )
    ]

    mock_registry_store.get_latest_versions = mock_get_latest_versions
    return mock_registry_store


def test_set_model_version_tag(mock_registry_store_with_get_latest_version):
    # set_model_version_tag using version
    MlflowClient().set_model_version_tag("model_name", 1, "tag1", "foobar")
    mock_registry_store_with_get_latest_version.set_model_version_tag.assert_called_once_with(
        "model_name", 1, ModelVersionTag(key="tag1", value="foobar")
    )

    mock_registry_store_with_get_latest_version.set_model_version_tag.reset_mock()

    # set_model_version_tag using stage
    MlflowClient().set_model_version_tag("model_name", key="tag1", value="foobar", stage="Staging")
    mock_registry_store_with_get_latest_version.set_model_version_tag.assert_called_once_with(
        "model_name", 1, ModelVersionTag(key="tag1", value="foobar")
    )

    # set_model_version_tag with version and stage set
    with pytest.raises(MlflowException, match="version and stage cannot be set together"):
        MlflowClient().set_model_version_tag("model_name", 1, "tag1", "foobar", stage="Staging")

    # set_model_version_tag with version and stage not set
    with pytest.raises(MlflowException, match="version or stage must be set"):
        MlflowClient().set_model_version_tag("model_name", key="tag1", value="foobar")


def test_delete_model_version_tag(mock_registry_store_with_get_latest_version):
    # delete_model_version_tag using version
    MlflowClient().delete_model_version_tag("model_name", 1, "tag1")
    mock_registry_store_with_get_latest_version.delete_model_version_tag.assert_called_once_with(
        "model_name", 1, "tag1"
    )

    mock_registry_store_with_get_latest_version.delete_model_version_tag.reset_mock()

    # delete_model_version_tag using stage
    MlflowClient().delete_model_version_tag("model_name", key="tag1", stage="Staging")
    mock_registry_store_with_get_latest_version.delete_model_version_tag.assert_called_once_with(
        "model_name", 1, "tag1"
    )

    # delete_model_version_tag with version and stage set
    with pytest.raises(MlflowException, match="version and stage cannot be set together"):
        MlflowClient().delete_model_version_tag(
            "model_name", version=1, key="tag1", stage="staging"
        )

    # delete_model_version_tag with version and stage not set
    with pytest.raises(MlflowException, match="version or stage must be set"):
        MlflowClient().delete_model_version_tag("model_name", key="tag1")


def test_set_registered_model_alias(mock_registry_store):
    MlflowClient().set_registered_model_alias("model_name", "test_alias", 1)
    mock_registry_store.set_registered_model_alias.assert_called_once_with(
        "model_name", "test_alias", 1
    )


def test_delete_registered_model_alias(mock_registry_store):
    MlflowClient().delete_registered_model_alias("model_name", "test_alias")
    mock_registry_store.delete_registered_model_alias.assert_called_once_with(
        "model_name", "test_alias"
    )


def test_get_model_version_by_alias(mock_registry_store):
    mock_registry_store.get_model_version_by_alias.return_value = _default_model_version()
    res = MlflowClient().get_model_version_by_alias("model_name", "test_alias")
    assert res == _default_model_version()
    mock_registry_store.get_model_version_by_alias.assert_called_once_with(
        "model_name", "test_alias"
    )


def test_update_run(mock_store):
    MlflowClient().update_run(run_id="run_id", status="FINISHED", name="my name")
    mock_store.update_run_info.assert_called_once_with(
        run_id="run_id",
        run_status=RunStatus.from_string("FINISHED"),
        end_time=mock.ANY,
        run_name="my name",
    )


def test_client_log_metric_params_tags_overrides(mock_store):
    experiment_id = mock.Mock()
    start_time = mock.Mock()
    run_name = mock.Mock()
    run = MlflowClient().create_run(experiment_id, start_time, tags={}, run_name=run_name)
    run_id = run.info.run_id

    run_operation = MlflowClient().log_metric(run_id, "m1", 0.87, 123456789, 1, synchronous=False)
    run_operation.wait()

    run_operation = MlflowClient().log_param(run_id, "p1", "pv1", synchronous=False)
    run_operation.wait()

    run_operation = MlflowClient().set_tag(run_id, "t1", "tv1", synchronous=False)
    run_operation.wait()

    mock_store.log_metric_async.assert_called_once_with(run_id, Metric("m1", 0.87, 123456789, 1))
    mock_store.log_param_async.assert_called_once_with(run_id, Param("p1", "pv1"))
    mock_store.set_tag_async.assert_called_once_with(run_id, RunTag("t1", "tv1"))

    mock_store.reset_mock()

    # log_batch_async
    MlflowClient().create_run(experiment_id, start_time, {})
    metrics = [Metric("m1", 0.87, 123456789, 1), Metric("m2", 0.87, 123456789, 1)]
    tags = [RunTag("t1", "tv1"), RunTag("t2", "tv2")]
    params = [Param("p1", "pv1"), Param("p2", "pv2")]
    run_operation = MlflowClient().log_batch(run_id, metrics, params, tags, synchronous=False)
    run_operation.wait()

    mock_store.log_batch_async.assert_called_once_with(
        run_id=run_id, metrics=metrics, params=params, tags=tags
    )


def test_invalid_run_id_log_artifact():
    with pytest.raises(
        MlflowException,
        match=r"Invalid run id.*",
    ):
        MlflowClient().log_artifact("tr-123", "path")


def test_enable_async_logging(mock_store, setup_async_logging):
    MlflowClient().log_param(run_id="run_id", key="key", value="val")
    mock_store.log_param_async.assert_called_once_with("run_id", Param("key", "val"))

    MlflowClient().log_metric(run_id="run_id", key="key", value="val", step=1, timestamp=1)
    mock_store.log_metric_async.assert_called_once_with("run_id", Metric("key", "val", 1, 1))


def test_file_store_download_upload_trace_data(tmp_path):
    with _use_tracking_uri(tmp_path.joinpath("mlruns").as_uri()):
        client = MlflowClient()
        span = client.start_trace("test", inputs={"test": 1})
        client.end_trace(span.trace_id, outputs={"result": 2})
        trace = mlflow.get_trace(span.trace_id)
        trace_data = client.get_trace(span.trace_id).data
        assert trace_data.request == trace.data.request
        assert trace_data.response == trace.data.response


def test_get_trace_throw_if_trace_id_is_online_trace_id():
    client = MlflowClient("databricks")
    trace_id = "3a3c3b56-910a-4721-8d02-0333eda5f37e"
    with pytest.raises(MlflowException, match="Traces from inference tables can only be loaded"):
        client.get_trace(trace_id)

    another_client = MlflowClient("mlruns")
    with pytest.raises(MlflowException, match=r"Trace with ID '[\w-]+' not found"):
        another_client.get_trace(trace_id)


@pytest.fixture(params=["file", "sqlalchemy"])
def registry_uri(request, tmp_path, db_uri):
    """Set an MLflow Model Registry URI with different type of backend."""
    if "MLFLOW_SKINNY" in os.environ and request.param == "sqlalchemy":
        pytest.skip("SQLAlchemy store is not available in skinny.")

    original_registry_uri = mlflow.get_registry_uri()

    if request.param == "file":
        registry_uri = tmp_path.joinpath("file").as_uri()
    elif request.param == "sqlalchemy":
        registry_uri = db_uri

    yield registry_uri

    # Reset tracking URI
    mlflow.set_tracking_uri(original_registry_uri)


def test_crud_prompts(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    client.register_prompt(
        name="prompt_1",
        template="Hi, {{title}} {{name}}! How are you today?",
        commit_message="A friendly greeting",
    )

    prompt = client.load_prompt("prompt_1", version=1)
    assert prompt.name == "prompt_1"
    assert prompt.template == "Hi, {{title}} {{name}}! How are you today?"
    assert prompt.commit_message == "A friendly greeting"

    client.register_prompt(
        name="prompt_1",
        template="Hi, {{title}} {{name}}! What's up?",
        commit_message="New greeting",
    )

    prompt = client.load_prompt("prompt_1", version=2)
    assert prompt.template == "Hi, {{title}} {{name}}! What's up?"

    prompt = client.load_prompt("prompt_1", version=1)
    assert prompt.template == "Hi, {{title}} {{name}}! How are you today?"

    prompt = client.load_prompt("prompts:/prompt_1/2")
    assert prompt.template == "Hi, {{title}} {{name}}! What's up?"

    # Test loading non-existent prompts
    assert mlflow.load_prompt("does_not_exist", version=1, allow_missing=True) is None


def test_create_prompt_with_tags_and_metadata(tracking_uri, disable_prompt_cache):
    client = MlflowClient(tracking_uri=tracking_uri)

    # Create prompt with version-specific tags
    client.register_prompt(
        name="prompt_1",
        template="Hi, {{name}}!",
        tags={"author": "Alice"},  # This will be version-level tags now
    )

    # Set some prompt-level tags separately
    client.set_prompt_tag("prompt_1", "application", "greeting")
    client.set_prompt_tag("prompt_1", "language", "en")

    # Test version 1
    prompt_v1 = client.load_prompt("prompt_1", version=1)
    assert prompt_v1.template == "Hi, {{name}}!"
    # Version tags are separate from prompt tags
    assert prompt_v1.tags == {"author": "Alice"}

    # Test prompt-level tags (separate from version)
    prompt_entity = client.get_prompt("prompt_1")
    # Note: Currently includes the version tags too, but we expect this behavior to change
    assert prompt_entity.tags == {
        "author": "Alice",  # This appears due to current implementation
        "application": "greeting",
        "language": "en",
    }

    # Create version 2 with different version-level tags
    client.register_prompt(
        name="prompt_1",
        template="こんにちは、{{name}}!",
        tags={"author": "Bob", "date": "2022-01-01"},  # Version-level tags
    )

    # Update some prompt-level tags
    client.set_prompt_tag("prompt_1", "project", "toy")
    client.set_prompt_tag("prompt_1", "language", "ja")

    # Test version 2
    prompt_v2 = client.load_prompt("prompt_1", version=2)
    assert prompt_v2.template == "こんにちは、{{name}}!"
    # Version 2 has its own version tags (decoupled from prompt and version 1)
    assert prompt_v2.tags == {"author": "Bob", "date": "2022-01-01"}

    # Verify prompt-level tags are updated and separate
    prompt_entity_updated = client.get_prompt("prompt_1")
    # Note: Currently the prompt tags get overwritten by the newest version's tags
    assert prompt_entity_updated.tags == {
        "author": "Bob",  # This appears due to current implementation
        "date": "2022-01-01",  # This appears due to current implementation
        "application": "greeting",
        "project": "toy",
        "language": "ja",
    }

    # Version 1 tags should be unchanged (decoupled from prompt tags)
    prompt_v1_after_update = client.load_prompt("prompt_1", version=1)
    assert prompt_v1_after_update.tags == {"author": "Alice"}  # Unchanged


def test_create_prompt_error_handling(tracking_uri, disable_prompt_cache):
    client = MlflowClient(tracking_uri=tracking_uri)

    # Exceeds the max length
    with pytest.raises(MlflowException, match=r"Prompt text exceeds max length of"):
        client.register_prompt(name="prompt_1", template="a" * 100_001)

    # When the first version creation fails, RegisteredModel should not be created
    with pytest.raises(MlflowException, match=r"Prompt with name=prompt_1 not found"):
        client.load_prompt("prompt_1", version=1)

    client.register_prompt("prompt_1", template="Hi, {{title}} {{name}}!")
    assert client.load_prompt("prompt_1", version=1) is not None

    # When the subsequent version creation fails, RegisteredModel should remain
    with pytest.raises(MlflowException, match=r"Prompt text exceeds max length of"):
        client.register_prompt(name="prompt_1", template="a" * 100_001)

    assert client.load_prompt("prompt_1", version=1) is not None


def test_create_prompt_with_invalid_name(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    with pytest.raises(MlflowException, match=r"Prompt name must be a non-empty string"):
        client.register_prompt(name="", template="Hi, {{name}}!")

    with pytest.raises(MlflowException, match=r"Prompt name must be a non-empty string"):
        client.register_prompt(name=123, template="Hi, {{name}}!")

    for invalid_pattern in [
        "prompt_1/2",
        "m%6fdel",
        "prompt?!?",
        "prompt with space",
    ]:
        with pytest.raises(MlflowException, match=r"Prompt name can only contain alphanumeric"):
            client.register_prompt(name=invalid_pattern, template="Hi, {{name}}!")

    # Name conflicts with a model
    client.create_registered_model("model")
    with pytest.raises(MlflowException, match=r"Model 'model' exists with the same name."):
        client.register_prompt(name="model", template="Hi, {{name}}!")


def test_load_prompt_error(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    with pytest.raises(MlflowException, match=r"Prompt with name=test not found"):
        client.load_prompt("test", version=1)

    # Both file and sqlalchemy return the same error format now
    error_msg = r"Prompt with name=test not found"

    with pytest.raises(MlflowException, match=error_msg):
        client.load_prompt("test", version=2)

    with pytest.raises(MlflowException, match=error_msg):
        client.load_prompt("test", version=2, allow_missing=False)

    # Load prompt with a model name
    client.create_registered_model("model")
    client.create_model_version("model", "source")

    with pytest.raises(MlflowException, match=r"Name `model` is registered as a model"):
        client.load_prompt("model", version=1)

    with pytest.raises(MlflowException, match=r"Name `model` is registered as a model"):
        client.load_prompt("model", version=1)

    with pytest.raises(MlflowException, match=r"Name `model` is registered as a model"):
        client.load_prompt("model", version=1, allow_missing=False)

    with pytest.raises(MlflowException, match=r"Name `model` is registered as a model"):
        client.load_prompt("model", version=1, allow_missing=False)


def test_link_prompt_version_to_run(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    prompt = client.register_prompt("prompt", template="Hi, {{name}}!")

    # Create actual runs to link to
    run1 = client.create_run(experiment_id="0").info.run_id
    run2 = client.create_run(experiment_id="0").info.run_id

    # Test that the method can be called without error
    client.link_prompt_version_to_run(run1, prompt)
    client.link_prompt_version_to_run(run2, prompt)

    # Verify tag was set by checking the run data
    run_data = client.get_run(run1)
    linked_prompts_tag = run_data.data.tags.get("mlflow.linkedPrompts")
    assert linked_prompts_tag is not None

    # Verify the JSON structure
    linked_prompts = json.loads(linked_prompts_tag)
    assert any(p["name"] == "prompt" and p["version"] == "1" for p in linked_prompts)

    # Test error case
    with pytest.raises(MlflowException, match=r"The `prompt` argument must be"):
        client.link_prompt_version_to_run(run1, 123)


@pytest.mark.parametrize("registry_uri", ["databricks"])
def test_crud_prompt_on_unsupported_registry(registry_uri):
    client = MlflowClient(registry_uri=registry_uri)

    with pytest.raises(MlflowException, match=r"The 'register_prompt' API is not supported"):
        client.register_prompt(
            name="prompt_1",
            template="Hi, {{title}} {{name}}! How are you today?",
            commit_message="A friendly greeting",
            tags={"model": "my-model"},
        )

    with pytest.raises(MlflowException, match=r"The 'load_prompt' API is not supported"):
        client.load_prompt("prompt_1")


def test_block_create_model_with_prompt_tag(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    with pytest.raises(MlflowException, match=r"Prompts cannot be registered"):
        client.create_registered_model(
            name="model",
            tags={IS_PROMPT_TAG_KEY: "true"},
        )

    with pytest.raises(MlflowException, match=r"Prompts cannot be registered"):
        client.create_model_version(
            name="model",
            source="source",
            tags={IS_PROMPT_TAG_KEY: "false"},
        )


def test_block_create_prompt_with_existing_model_name(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    client.create_registered_model("model")

    with pytest.raises(MlflowException, match=r"Model 'model' exists with"):
        client.register_prompt(
            name="model",
            template="Hi, {{title}} {{name}}! How are you today?",
            commit_message="A friendly greeting",
            tags={"model": "my-model"},
        )


def test_block_handling_prompt_with_model_apis(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)
    client.register_prompt("prompt", template="Hi, {{name}}!")
    client.set_prompt_alias("prompt", alias="alias", version=1)
    # Validate the prompt is registered
    prompt = client.load_prompt("prompt", version=1)
    assert prompt.name == "prompt"
    assert prompt.aliases == ["alias"]

    apis_to_args = [
        (client.rename_registered_model, ["prompt", "new_name"]),
        (client.update_registered_model, ["prompt", "new_description"]),
        (client.delete_registered_model, ["prompt"]),
        (client.get_registered_model, ["prompt"]),
        (client.get_latest_versions, ["prompt"]),
        (client.set_registered_model_tag, ["prompt", "tag", "value"]),
        (client.delete_registered_model_tag, ["prompt", "tag"]),
        (client.update_model_version, ["prompt", 1, "new_description"]),
        (client.transition_model_version_stage, ["prompt", 1, "Production"]),
        (client.delete_model_version, ["prompt", 1]),
        (client.get_model_version, ["prompt", 1]),
        (client.get_model_version_download_uri, ["prompt", 1]),
        (client.set_model_version_tag, ["prompt", 1, "tag", "value"]),
        (client.delete_model_version_tag, ["prompt", 1, "tag"]),
        (client.set_registered_model_alias, ["prompt", "alias", 1]),
        (client.delete_registered_model_alias, ["prompt", "alias"]),
        (client.get_model_version_by_alias, ["prompt", "alias"]),
    ]

    for api, args in apis_to_args:
        with pytest.raises(MlflowException, match=r"Registered Model with name='prompt' not found"):
            api(*args)

    with pytest.raises(MlflowException, match=r"Model with uri 'models:/prompt/1' not found"):
        client.copy_model_version("models:/prompt/1", "new_model")


def test_log_and_detach_prompt(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    client.register_prompt(name="p1", template="Hi, there!")
    time.sleep(0.001)  # To avoid timestamp precision issue in Windows
    client.register_prompt(name="p2", template="Hi, {{name}}!")

    run_id = client.create_run(experiment_id="0").info.run_id

    # Check that initially no prompts are linked to the run
    run = client.get_run(run_id)
    linked_prompts_tag = run.data.tags.get(TraceTagKey.LINKED_PROMPTS)
    assert linked_prompts_tag is None

    client.link_prompt_version_to_run(run_id, "prompts:/p1/1")
    run = client.get_run(run_id)
    linked_prompts_tag = run.data.tags.get(TraceTagKey.LINKED_PROMPTS)
    assert linked_prompts_tag is not None
    prompts = json.loads(linked_prompts_tag)
    assert len(prompts) == 1
    assert prompts[0]["name"] == "p1"

    client.link_prompt_version_to_run(run_id, "prompts:/p2/1")
    run = client.get_run(run_id)
    linked_prompts_tag = run.data.tags.get(TraceTagKey.LINKED_PROMPTS)
    prompts = json.loads(linked_prompts_tag)
    assert len(prompts) == 2
    prompt_names = [p["name"] for p in prompts]
    assert "p1" in prompt_names
    assert "p2" in prompt_names


def test_search_prompt(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    client.register_prompt(name="prompt_1", template="Hi, {{name}}!")
    client.register_prompt(name="prompt_2", template="Hello, {{name}}!")
    client.register_prompt(name="prompt_3", template="Greetings, {{name}}!")
    client.register_prompt(name="prompt_4", template="Howdy, {{name}}!")
    client.register_prompt(name="prompt_5", template="Salutations, {{name}}!")
    client.register_prompt(name="prompt_6", template="Bonjour, {{name}}!")
    client.register_prompt(name="test", template="Test Template")
    client.register_prompt(name="new", template="Bonjour, {{name}}!")

    prompts = client.search_prompts(filter_string="name='prompt_1'")
    assert len(prompts) == 1
    assert prompts[0].name == "prompt_1"

    prompts = client.search_prompts(filter_string="name LIKE '%prompt%'")
    assert len(prompts) == 6
    assert all("prompt" in prompt.name for prompt in prompts)

    prompts = client.search_prompts()
    assert len(prompts) == 8

    prompts = client.search_prompts(max_results=3)
    assert len(prompts) == 3


def test_delete_prompt_version_no_auto_cleanup(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    # Create prompt and version
    client.register_prompt(name="test_prompt", template="Hello {{name}}!")

    # Verify prompt and version exist
    prompt = client.get_prompt("test_prompt")
    assert prompt is not None
    assert prompt.name == "test_prompt"

    prompt_version = client.get_prompt_version("test_prompt", 1)
    assert prompt_version is not None
    assert prompt_version.version == 1

    # Delete the version - prompt should remain
    client.delete_prompt_version("test_prompt", "1")

    # Prompt should still exist even though it has no versions
    prompt = client.get_prompt("test_prompt")
    assert prompt is not None
    assert prompt.name == "test_prompt"

    # Version should be gone
    with pytest.raises(MlflowException, match=r"Prompt.*name=test_prompt.*version=1.*not found"):
        client.get_prompt_version("test_prompt", 1)


def test_delete_prompt_with_no_versions(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    # Create prompt and version, then delete version
    client.register_prompt(name="empty_prompt", template="Hello {{name}}!")
    client.delete_prompt_version("empty_prompt", "1")

    # Verify prompt exists but has no versions
    prompt = client.get_prompt("empty_prompt")
    assert prompt is not None

    # Delete the prompt - should work regardless of registry type
    client.delete_prompt("empty_prompt")

    # Prompt should be gone
    prompt = client.get_prompt("empty_prompt")
    assert prompt is None


def test_delete_prompt_complete_workflow(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    # Create prompt with multiple versions
    client.register_prompt(name="workflow_prompt", template="Version 1: {{name}}")
    client.register_prompt(name="workflow_prompt", template="Version 2: {{name}}")
    client.register_prompt(name="workflow_prompt", template="Version 3: {{name}}")

    # Verify all versions exist
    v1 = client.get_prompt_version("workflow_prompt", 1)
    v2 = client.get_prompt_version("workflow_prompt", 2)
    v3 = client.get_prompt_version("workflow_prompt", 3)
    assert v1.template == "Version 1: {{name}}"
    assert v2.template == "Version 2: {{name}}"
    assert v3.template == "Version 3: {{name}}"

    # Delete versions one by one
    client.delete_prompt_version("workflow_prompt", "1")
    client.delete_prompt_version("workflow_prompt", "2")
    client.delete_prompt_version("workflow_prompt", "3")

    # Prompt should still exist with no versions
    prompt = client.get_prompt("workflow_prompt")
    assert prompt is not None

    # Now delete the prompt itself
    client.delete_prompt("workflow_prompt")

    # Prompt should be completely gone
    prompt = client.get_prompt("workflow_prompt")
    assert prompt is None


def test_delete_prompt_error_handling(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    # Test deleting non-existent prompt
    with pytest.raises(MlflowException, match=r"Prompt with name=nonexistent not found"):
        client.delete_prompt("nonexistent")

    # Test deleting non-existent version
    client.register_prompt(name="test_errors", template="Hello {{name}}!")
    with pytest.raises(MlflowException, match=r"Prompt.*name=test_errors.*version=999.*not found"):
        client.delete_prompt_version("test_errors", "999")


def test_delete_prompt_version_behavior_consistency(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    # Create multiple prompts with versions
    for i in range(3):
        prompt_name = f"consistency_test_{i}"
        client.register_prompt(name=prompt_name, template=f"Template {i}: {{{{name}}}}")

        # Delete the version immediately
        client.delete_prompt_version(prompt_name, "1")

        # Prompt should remain but have no versions
        prompt = client.get_prompt(prompt_name)
        assert prompt is not None
        assert prompt.name == prompt_name

        # Version should be gone
        with pytest.raises(MlflowException, match=r"Prompt.*version.*not found"):
            client.get_prompt_version(prompt_name, 1)

    # Clean up - delete all prompts
    for i in range(3):
        client.delete_prompt(f"consistency_test_{i}")
        prompt = client.get_prompt(f"consistency_test_{i}")
        assert prompt is None


@pytest.mark.parametrize("registry_uri", ["databricks-uc"])
def test_delete_prompt_with_versions_unity_catalog_error(registry_uri):
    # Mock Unity Catalog behavior
    client = MlflowClient(registry_uri=registry_uri)

    # Mock the search_prompt_versions to return versions
    mock_response = Mock()
    mock_response.prompt_versions = [Mock(version="1")]

    with (
        patch.object(client, "search_prompt_versions", return_value=mock_response),
        patch.object(client, "_registry_uri", registry_uri),
    ):
        with pytest.raises(
            MlflowException, match=r"Cannot delete prompt .* because it still has undeleted"
        ):
            client.delete_prompt("test_prompt")


def test_link_prompt_version_to_model_smoke_test(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    # Create an experiment and a run to have a proper context
    experiment_id = client.create_experiment("test_experiment")
    with mlflow.start_run(experiment_id=experiment_id):
        # Create a model with a run context
        model = client.create_logged_model(experiment_id=experiment_id)

        # Register a prompt
        client.register_prompt(name="test_prompt", template="Hello, {{name}}!")

        # Link the prompt version to the model (this should not raise an exception)
        # This is the main assertion - that the method call succeeds
        client.link_prompt_version_to_model(
            name="test_prompt", version="1", model_id=model.model_id
        )


def test_link_prompts_to_trace_smoke_test(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    # Create an experiment and a run to have a proper context
    experiment_id = client.create_experiment("test_experiment")
    with mlflow.start_run(experiment_id=experiment_id):
        # Create a simple trace for testing
        trace_info = client.start_trace("test_trace")
        trace_id = trace_info.request_id

        # Register a prompt
        client.register_prompt(name="test_prompt", template="Hello, {{name}}!")

        # Get the prompt version and link to the trace (this should not raise an exception)
        # This is the main assertion - that the method call succeeds
        prompt_version = client.get_prompt_version("test_prompt", "1")
        client.link_prompt_versions_to_trace(prompt_versions=[prompt_version], trace_id=trace_id)


def test_log_model_artifact(tmp_path: Path, tracking_uri: str) -> None:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment("test")
    model = client.create_logged_model(experiment_id=experiment_id)
    tmp_path = tmp_path.joinpath("artifacts")
    tmp_path.mkdir()
    tmp_file = tmp_path.joinpath("file")
    tmp_file.write_text("a")
    client.log_model_artifact(model_id=model.model_id, local_path=str(tmp_file))
    artifacts = client.list_logged_model_artifacts(model_id=model.model_id)
    assert artifacts == [FileInfo(path="file", is_dir=False, file_size=1)]
    another_tmp_file = tmp_path.joinpath("another_file")
    another_tmp_file.write_text("aa")
    client.log_model_artifact(model_id=model.model_id, local_path=str(another_tmp_file))
    artifacts = client.list_logged_model_artifacts(model_id=model.model_id)
    artifacts = sorted(artifacts, key=lambda x: x.path)
    assert artifacts == [
        FileInfo(path="another_file", is_dir=False, file_size=2),
        FileInfo(path="file", is_dir=False, file_size=1),
    ]


def test_log_model_artifacts(tmp_path: Path, tracking_uri: str) -> None:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment("test")
    model = client.create_logged_model(experiment_id=experiment_id)
    tmp_path = tmp_path.joinpath("artifacts")
    tmp_path.mkdir()
    tmp_file = tmp_path.joinpath("file")
    tmp_file.write_text("a")
    tmp_dir = tmp_path.joinpath("dir")
    tmp_dir.mkdir()
    another_file = tmp_dir.joinpath("another_file")
    another_file.write_text("aa")
    client.log_model_artifacts(model_id=model.model_id, local_dir=str(tmp_path))
    artifacts = client.list_logged_model_artifacts(model_id=model.model_id)
    artifacts = sorted(artifacts, key=lambda x: x.path)
    assert artifacts == [
        FileInfo(path="dir", is_dir=True, file_size=None),
        FileInfo(path="file", is_dir=False, file_size=1),
    ]
    artifacts = client.list_logged_model_artifacts(model_id=model.model_id, path="dir")
    assert artifacts == [FileInfo(path="dir/another_file", is_dir=False, file_size=2)]


def test_logged_model_model_id_required(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    with pytest.raises(MlflowException, match="`model_id` must be a non-empty string, but got ''"):
        client.finalize_logged_model("", LoggedModelStatus.READY)

    with pytest.raises(MlflowException, match="`model_id` must be a non-empty string, but got ''"):
        client.get_logged_model("")

    with pytest.raises(MlflowException, match="`model_id` must be a non-empty string, but got ''"):
        client.delete_logged_model("")

    with pytest.raises(MlflowException, match="`model_id` must be a non-empty string, but got ''"):
        client.set_logged_model_tags("", {})

    with pytest.raises(MlflowException, match="`model_id` must be a non-empty string, but got ''"):
        client.delete_logged_model_tag("", "")


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny client does not support the np or pandas dependencies",
)
def test_log_metric_link_to_active_model(tracking_uri):
    model = mlflow.create_external_model(name="test_model")
    mlflow.set_active_model(name=model.name)
    client = MlflowClient(tracking_uri=tracking_uri)
    with mlflow.start_run() as run:
        client.log_metric(run.info.run_id, "metric", 1)
    logged_model = mlflow.get_logged_model(model_id=model.model_id)
    assert logged_model.name == model.name
    assert logged_model.model_id == model.model_id
    assert logged_model.metrics[0].key == "metric"
    assert logged_model.metrics[0].value == 1


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny client does not support the np or pandas dependencies",
)
def test_log_batch_link_to_active_model(tracking_uri):
    model = mlflow.create_external_model(name="test_model")
    mlflow.set_active_model(name=model.name)
    client = MlflowClient(tracking_uri=tracking_uri)
    with mlflow.start_run() as run:
        client.log_batch(run.info.run_id, [Metric("metric1", 1, 0, 0), Metric("metric2", 2, 0, 0)])
    logged_model = mlflow.get_logged_model(model_id=model.model_id)
    assert logged_model.name == model.name
    assert logged_model.model_id == model.model_id
    assert {m.key: m.value for m in logged_model.metrics} == {
        "metric1": 1,
        "metric2": 2,
    }


def test_load_prompt_with_alias_uri(tracking_uri, disable_prompt_cache):
    client = MlflowClient(tracking_uri=tracking_uri)

    # Register two versions of a prompt
    client.register_prompt(name="alias_prompt", template="Hello, world!")
    client.register_prompt(name="alias_prompt", template="Hello, {{name}}!")

    # Assign alias to version 1
    client.set_prompt_alias("alias_prompt", alias="production", version=1)
    prompt = client.load_prompt("prompts:/alias_prompt@production")
    assert prompt.template == "Hello, world!"
    assert "production" in prompt.aliases

    # Reassign alias to version 2
    client.set_prompt_alias("alias_prompt", alias="production", version=2)
    prompt = client.load_prompt("prompts:/alias_prompt@production")
    assert prompt.template == "Hello, {{name}}!"
    assert "production" in prompt.aliases

    # Delete alias and verify loading fails
    client.delete_prompt_alias("alias_prompt", alias="production")
    with pytest.raises(
        MlflowException, match=r"Prompt (.*) does not exist.|Prompt alias (.*) not found."
    ):
        client.load_prompt("prompts:/alias_prompt@production")

    # Loading with the 'latest' alias
    prompt = client.load_prompt("prompts:/alias_prompt@latest")
    assert prompt.template == "Hello, {{name}}!"


def test_load_prompt_allow_missing_name_version(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    # Non-existent prompt by name+version should return None when allow_missing=True
    result = client.load_prompt("nonexistent_prompt", version=1, allow_missing=True)
    assert result is None

    # Non-existent prompt by name+version should raise exception when allow_missing=False
    with pytest.raises(MlflowException, match="Prompt with name=nonexistent_prompt not found"):
        client.load_prompt("nonexistent_prompt", version=1, allow_missing=False)

    # Existing prompt with non-existent version should return None when allow_missing=True
    client.register_prompt(name="existing_prompt", template="Hello, world!")
    result = client.load_prompt("existing_prompt", version=999, allow_missing=True)
    assert result is None

    # Existing prompt with non-existent version should raise exception when allow_missing=False
    with pytest.raises(
        MlflowException, match=r"Prompt \(name=existing_prompt, version=999\) not found"
    ):
        client.load_prompt("existing_prompt", version=999, allow_missing=False)


def test_load_prompt_allow_missing_uri_version(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    # Non-existent prompt by URI+version should return None when allow_missing=True
    result = client.load_prompt("prompts:/nonexistent_prompt/1", allow_missing=True)
    assert result is None

    # Non-existent prompt by URI+version should raise exception when allow_missing=False
    with pytest.raises(MlflowException, match="Prompt with name=nonexistent_prompt not found"):
        client.load_prompt("prompts:/nonexistent_prompt/1", allow_missing=False)

    # Existing prompt with non-existent version via URI should return None when allow_missing=True
    client.register_prompt(name="existing_prompt", template="Hello, world!")
    result = client.load_prompt("prompts:/existing_prompt/999", allow_missing=True)
    assert result is None

    # Existing prompt with non-existent version via URI should raise when allow_missing=False
    with pytest.raises(
        MlflowException, match=r"Prompt \(name=existing_prompt, version=999\) not found"
    ):
        client.load_prompt("prompts:/existing_prompt/999", allow_missing=False)


def test_load_prompt_allow_missing_uri_alias(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)

    # Non-existent prompt with alias should return None when allow_missing=True
    result = client.load_prompt("prompts:/nonexistent_prompt@production", allow_missing=True)
    assert result is None

    # Non-existent prompt with alias should raise exception when allow_missing=False
    with pytest.raises(MlflowException, match="Prompt with name=nonexistent_prompt not found"):
        client.load_prompt("prompts:/nonexistent_prompt@production", allow_missing=False)

    # Existing prompt with non-existent alias should return None when allow_missing=True
    client.register_prompt(name="existing_prompt", template="Hello, world!")
    result = client.load_prompt("prompts:/existing_prompt@nonexistent_alias", allow_missing=True)
    assert result is None

    # Existing prompt with non-existent alias should raise exception when allow_missing=False
    with pytest.raises(MlflowException, match="Prompt alias nonexistent_alias not found"):
        client.load_prompt("prompts:/existing_prompt@nonexistent_alias", allow_missing=False)


def test_create_prompt_chat_format_client_integration():
    chat_template = [
        {"role": "system", "content": "You are a {{style}} assistant."},
        {"role": "user", "content": "{{question}}"},
    ]

    response_format = {"type": "string"}

    # Use client to create prompt
    client = MlflowClient()
    prompt = client.register_prompt(
        name="test_chat_client",
        template=chat_template,
        response_format=response_format,
        commit_message="Test chat prompt via client",
    )

    assert prompt.template == chat_template
    assert prompt.response_format == response_format

    # Load via client
    loaded_prompt = client.get_prompt_version("test_chat_client", 1)
    assert not loaded_prompt.is_text_prompt
    assert loaded_prompt.template == chat_template
    assert loaded_prompt.response_format == response_format


def test_link_chat_prompt_version_to_run():
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello {{name}}!"},
    ]

    client = MlflowClient()
    prompt = client.register_prompt(name="test_chat_link", template=chat_template)

    # Create run and link prompt
    run = client.create_run(client.create_experiment("test_exp"))
    client.link_prompt_version_to_run(run.info.run_id, prompt)

    # Verify linking
    run_data = client.get_run(run.info.run_id)
    linked_prompts_tag = run_data.data.tags.get(TraceTagKey.LINKED_PROMPTS)
    assert linked_prompts_tag is not None

    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 1
    assert linked_prompts[0]["name"] == "test_chat_link"
    assert linked_prompts[0]["version"] == "1"


def test_create_prompt_with_pydantic_response_format_client():
    from pydantic import BaseModel

    class ResponseSchema(BaseModel):
        answer: str
        confidence: float

    client = MlflowClient()
    prompt = client.register_prompt(
        name="test_pydantic_client",
        template="What is {{question}}?",
        response_format=ResponseSchema,
        commit_message="Test Pydantic response format via client",
    )

    assert prompt.response_format == ResponseSchema.model_json_schema()
    assert prompt.commit_message == "Test Pydantic response format via client"

    # Load and verify
    loaded_prompt = client.get_prompt_version("test_pydantic_client", 1)
    assert loaded_prompt.response_format == ResponseSchema.model_json_schema()


def test_create_prompt_with_dict_response_format_client():
    response_format = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "key_points": {"type": "array", "items": {"type": "string"}},
        },
    }

    client = MlflowClient()
    prompt = client.register_prompt(
        name="test_dict_response_client",
        template="Analyze this: {{text}}",
        response_format=response_format,
        tags={"analysis_type": "text"},
    )

    assert prompt.response_format == response_format
    assert prompt.tags["analysis_type"] == "text"

    # Load and verify
    loaded_prompt = client.get_prompt_version("test_dict_response_client", 1)
    assert loaded_prompt.response_format == response_format


def test_create_prompt_text_backward_compatibility_client():
    client = MlflowClient()
    prompt = client.register_prompt(
        name="test_text_backward_client",
        template="Hello {{name}}!",
        commit_message="Test backward compatibility via client",
    )

    assert prompt.is_text_prompt
    assert prompt.template == "Hello {{name}}!"
    assert prompt.commit_message == "Test backward compatibility via client"

    # Load and verify
    loaded_prompt = client.get_prompt_version("test_text_backward_client", 1)
    assert loaded_prompt.is_text_prompt
    assert loaded_prompt.template == "Hello {{name}}!"


def test_create_prompt_complex_chat_template_client():
    chat_template = [
        {
            "role": "system",
            "content": "You are a {{style}} assistant named {{name}}.",
        },
        {"role": "user", "content": "{{greeting}}! {{question}}"},
        {
            "role": "assistant",
            "content": "I understand you're asking about {{topic}}.",
        },
    ]

    client = MlflowClient()
    prompt = client.register_prompt(
        name="test_complex_chat_client",
        template=chat_template,
        tags={"complexity": "high"},
    )

    assert prompt.template == chat_template
    assert prompt.tags["complexity"] == "high"

    # Load and verify
    loaded_prompt = client.get_prompt_version("test_complex_chat_client", 1)
    assert not loaded_prompt.is_text_prompt
    assert loaded_prompt.template == chat_template


def test_create_prompt_with_none_response_format_client():
    client = MlflowClient()
    prompt = client.register_prompt(
        name="test_none_response_client",
        template="Hello {{name}}!",
        response_format=None,
    )

    assert prompt.response_format is None

    # Load and verify
    loaded_prompt = client.get_prompt_version("test_none_response_client", 1)
    assert loaded_prompt.response_format is None


def test_create_prompt_with_empty_chat_template_client():
    client = MlflowClient()
    prompt = client.register_prompt(name="test_empty_chat_client", template=[])

    assert prompt.is_text_prompt
    assert prompt.template == "[]"  # Empty list serialized as string

    # Load and verify
    loaded_prompt = client.get_prompt_version("test_empty_chat_client", 1)
    assert loaded_prompt.is_text_prompt


def test_create_prompt_with_single_message_chat_client():
    chat_template = [{"role": "user", "content": "Hello {{name}}!"}]

    client = MlflowClient()
    prompt = client.register_prompt(name="test_single_message_client", template=chat_template)

    assert prompt.template == chat_template
    assert prompt.variables == {"name"}

    # Load and verify
    loaded_prompt = client.get_prompt_version("test_single_message_client", 1)
    assert not loaded_prompt.is_text_prompt
    assert loaded_prompt.template == chat_template


def test_create_prompt_with_multiple_variables_in_chat_client():
    chat_template = [
        {
            "role": "system",
            "content": "You are a {{style}} assistant named {{name}}.",
        },
        {"role": "user", "content": "{{greeting}}! {{question}}"},
        {
            "role": "assistant",
            "content": "I understand you're asking about {{topic}}.",
        },
    ]

    client = MlflowClient()
    prompt = client.register_prompt(name="test_multiple_variables_client", template=chat_template)

    expected_variables = {"style", "name", "greeting", "question", "topic"}
    assert prompt.variables == expected_variables

    # Load and verify
    loaded_prompt = client.get_prompt_version("test_multiple_variables_client", 1)
    assert loaded_prompt.variables == expected_variables


def test_create_prompt_with_mixed_content_types_client():
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello {{name}}!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
    ]

    client = MlflowClient()
    prompt = client.register_prompt(name="test_mixed_content_client", template=chat_template)

    assert prompt.template == chat_template
    assert prompt.variables == {"name"}

    # Load and verify
    loaded_prompt = client.get_prompt_version("test_mixed_content_client", 1)
    assert not loaded_prompt.is_text_prompt
    assert loaded_prompt.template == chat_template


def test_create_prompt_with_nested_variables_client():
    chat_template = [
        {
            "role": "system",
            "content": "You are a {{user.preferences.style}} assistant.",
        },
        {
            "role": "user",
            "content": "Hello {{user.name}}! {{user.preferences.greeting}}",
        },
    ]

    client = MlflowClient()
    prompt = client.register_prompt(name="test_nested_variables_client", template=chat_template)

    expected_variables = {
        "user.preferences.style",
        "user.name",
        "user.preferences.greeting",
    }
    assert prompt.variables == expected_variables

    # Load and verify
    loaded_prompt = client.get_prompt_version("test_nested_variables_client", 1)
    assert loaded_prompt.variables == expected_variables


def test_link_prompt_with_response_format_to_run():
    response_format = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }
    client = MlflowClient()
    prompt = client.register_prompt(
        name="test_response_link",
        template="What is {{question}}?",
        response_format=response_format,
    )

    # Create run and link prompt
    run = client.create_run(client.create_experiment("test_exp"))
    client.link_prompt_version_to_run(run.info.run_id, prompt)

    # Verify linking
    run_data = client.get_run(run.info.run_id)
    linked_prompts_tag = run_data.data.tags.get(TraceTagKey.LINKED_PROMPTS)
    assert linked_prompts_tag is not None

    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 1
    assert linked_prompts[0]["name"] == "test_response_link"
    assert linked_prompts[0]["version"] == "1"


def test_link_multiple_prompt_types_to_run():
    client = MlflowClient()

    # Create text prompt
    text_prompt = client.register_prompt(name="test_text_link", template="Hello {{name}}!")

    # Create chat prompt
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "{{question}}"},
    ]
    chat_prompt = client.register_prompt(name="test_chat_link_multiple", template=chat_template)

    # Create run and link both prompts
    run = client.create_run(client.create_experiment("test_exp"))
    client.link_prompt_version_to_run(run.info.run_id, text_prompt)
    client.link_prompt_version_to_run(run.info.run_id, chat_prompt)

    # Verify linking
    run_data = client.get_run(run.info.run_id)
    linked_prompts_tag = run_data.data.tags.get(TraceTagKey.LINKED_PROMPTS)
    assert linked_prompts_tag is not None

    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 2

    expected_prompts = [
        {"name": "test_text_link", "version": "1"},
        {"name": "test_chat_link_multiple", "version": "1"},
    ]
    for expected_prompt in expected_prompts:
        assert expected_prompt in linked_prompts


def test_mlflow_client_create_dataset(mock_store):
    created_dataset = EvaluationDataset(
        dataset_id="test_dataset_id",
        name="test_dataset",
        digest="abcdef123456",
        created_time=1234567890,
        last_update_time=1234567890,
        tags={"environment": "production", "version": "1.0"},
    )
    created_dataset.experiment_ids = ["exp1", "exp2"]
    mock_store.create_dataset.return_value = created_dataset

    # Mock context registry to return empty tags so mlflow.user is not auto-added
    with mock.patch(
        "mlflow.tracking._tracking_service.client.context_registry.resolve_tags", return_value={}
    ):
        dataset = MlflowClient().create_dataset(
            name="qa_evaluation",
            experiment_id=["exp1", "exp2"],
            tags={"environment": "production", "version": "1.0"},
        )

    assert dataset.dataset_id == "test_dataset_id"
    assert dataset.name == "test_dataset"
    assert dataset.tags == {"environment": "production", "version": "1.0"}

    mock_store.create_dataset.assert_called_once_with(
        name="qa_evaluation",
        tags={"environment": "production", "version": "1.0"},
        experiment_ids=["exp1", "exp2"],
    )


def test_mlflow_client_create_evaluation_dataset_minimal(mock_store):
    created_dataset = EvaluationDataset(
        dataset_id="test_dataset_id",
        name="test_dataset",
        digest="abcdef123456",
        created_time=1234567890,
        last_update_time=1234567890,
    )
    mock_store.create_dataset.return_value = created_dataset

    # Mock context registry to return empty tags so mlflow.user is not auto-added
    with mock.patch(
        "mlflow.tracking._tracking_service.client.context_registry.resolve_tags", return_value={}
    ):
        dataset = MlflowClient().create_dataset(name="test_dataset")

    assert dataset.dataset_id == "test_dataset_id"
    assert dataset.name == "test_dataset"

    mock_store.create_dataset.assert_called_once_with(
        name="test_dataset",
        tags=None,
        experiment_ids=None,
    )


def test_mlflow_client_get_dataset(mock_store):
    mock_store.get_dataset.return_value = EvaluationDataset(
        dataset_id="dataset_123",
        name="test_dataset",
        digest="abcdef123456",
        created_time=1234567890,
        last_update_time=1234567890,
        tags={"source": "human-annotated"},
    )

    dataset = MlflowClient().get_dataset("dataset_123")

    assert dataset.dataset_id == "dataset_123"
    assert dataset.name == "test_dataset"
    assert dataset.tags == {"source": "human-annotated"}

    mock_store.get_dataset.assert_called_once_with("dataset_123")


def test_mlflow_client_delete_dataset(mock_store):
    MlflowClient().delete_dataset("dataset_123")

    mock_store.delete_dataset.assert_called_once_with("dataset_123")


def test_mlflow_client_search_datasets(mock_store):
    mock_store.search_datasets.return_value = PagedList(
        [
            EvaluationDataset(
                dataset_id="dataset_1",
                name="dataset_1",
                digest="digest1",
                created_time=1234567890,
                last_update_time=1234567890,
            ),
            EvaluationDataset(
                dataset_id="dataset_2",
                name="dataset_2",
                digest="digest2",
                created_time=1234567890,
                last_update_time=1234567890,
            ),
        ],
        "next_token",
    )

    result = MlflowClient().search_datasets(
        experiment_ids=["exp1", "exp2"],
        filter_string="name LIKE 'qa_%'",
        max_results=100,
        order_by=["created_time DESC"],
        page_token="page_token_123",
    )

    assert len(result) == 2
    assert result[0].dataset_id == "dataset_1"
    assert result[1].dataset_id == "dataset_2"
    assert result.token == "next_token"

    mock_store.search_datasets.assert_called_once_with(
        experiment_ids=["exp1", "exp2"],
        filter_string="name LIKE 'qa_%'",
        max_results=100,
        order_by=["created_time DESC"],
        page_token="page_token_123",
    )


def test_mlflow_client_search_datasets_empty_results(mock_store):
    mock_store.search_datasets.return_value = PagedList([], None)

    result = MlflowClient().search_datasets(
        experiment_ids=["exp1"], filter_string="name = 'nonexistent'"
    )

    assert len(result) == 0
    assert result.token is None


def test_mlflow_client_search_datasets_defaults(mock_store):
    mock_store.search_datasets.return_value = PagedList([], None)

    result = MlflowClient().search_datasets()

    assert len(result) == 0
    assert result.token is None

    mock_store.search_datasets.assert_called_once_with(
        experiment_ids=None,
        filter_string=None,
        max_results=SEARCH_EVALUATION_DATASETS_MAX_RESULTS,
        order_by=None,
        page_token=None,
    )


@pytest.mark.skipif(is_windows(), reason="FileStore URI handling issues on Windows")
def test_mlflow_client_datasets_filestore_not_supported(tmp_path):
    file_store_uri = str(tmp_path)
    client = MlflowClient(tracking_uri=file_store_uri)

    with pytest.raises(MlflowException, match="is not supported with FileStore") as exc_info:
        client.create_dataset(name="test_dataset")
    assert exc_info.value.error_code == "FEATURE_DISABLED"

    with pytest.raises(MlflowException, match="is not supported with FileStore") as exc_info:
        client.get_dataset("dataset_123")
    assert exc_info.value.error_code == "FEATURE_DISABLED"

    with pytest.raises(MlflowException, match="is not supported with FileStore") as exc_info:
        client.delete_dataset("dataset_123")
    assert exc_info.value.error_code == "FEATURE_DISABLED"

    with pytest.raises(MlflowException, match="is not supported with FileStore") as exc_info:
        client.search_datasets()
    assert exc_info.value.error_code == "FEATURE_DISABLED"

    with pytest.raises(MlflowException, match="is not supported with FileStore") as exc_info:
        client.set_dataset_tags("dataset_123", {"tag1": "value1"})
    assert exc_info.value.error_code == "FEATURE_DISABLED"

    with pytest.raises(MlflowException, match="is not supported with FileStore") as exc_info:
        client.delete_dataset_tag("dataset_123", "tag1")
    assert exc_info.value.error_code == "FEATURE_DISABLED"

    with pytest.raises(MlflowException, match="is not supported with FileStore") as exc_info:
        client.add_dataset_to_experiments("dataset_123", ["1", "2"])
    assert exc_info.value.error_code == "FEATURE_DISABLED"

    with pytest.raises(MlflowException, match="is not supported with FileStore") as exc_info:
        client.remove_dataset_from_experiments("dataset_123", ["1", "2"])
    assert exc_info.value.error_code == "FEATURE_DISABLED"


def test_mlflow_client_set_dataset_tags(mock_store):
    MlflowClient().set_dataset_tags(
        dataset_id="dataset_123",
        tags={"env": "prod", "version": "2.0"},
    )

    mock_store.set_dataset_tags.assert_called_once_with(
        dataset_id="dataset_123",
        tags={"env": "prod", "version": "2.0"},
    )


def test_mlflow_client_delete_dataset_tag(mock_store):
    MlflowClient().delete_dataset_tag(
        dataset_id="dataset_123",
        key="deprecated",
    )

    mock_store.delete_dataset_tag.assert_called_once_with(
        dataset_id="dataset_123",
        key="deprecated",
    )


def test_mlflow_client_delete_dataset_records(mock_store):
    mock_store.delete_dataset_records.return_value = 2

    result = MlflowClient().delete_dataset_records(
        dataset_id="dataset_123",
        dataset_record_ids=["record_1", "record_2"],
    )

    assert result == 2
    mock_store.delete_dataset_records.assert_called_once_with(
        dataset_id="dataset_123",
        dataset_record_ids=["record_1", "record_2"],
    )


def test_mlflow_client_delete_dataset_records_empty(mock_store):
    mock_store.delete_dataset_records.return_value = 0

    result = MlflowClient().delete_dataset_records(
        dataset_id="dataset_123",
        dataset_record_ids=["nonexistent_record"],
    )

    assert result == 0


def test_mlflow_client_add_dataset_to_experiments(mock_store):
    mock_dataset = Mock(spec=EvaluationDataset)
    mock_dataset.dataset_id = "dataset_123"
    mock_dataset.experiment_ids = ["1", "2", "3"]
    mock_store.add_dataset_to_experiments.return_value = mock_dataset

    client = MlflowClient()
    result = client.add_dataset_to_experiments(
        dataset_id="dataset_123",
        experiment_ids=["2", "3"],
    )

    assert result == mock_dataset
    assert result.experiment_ids == ["1", "2", "3"]
    mock_store.add_dataset_to_experiments.assert_called_once_with("dataset_123", ["2", "3"])


def test_mlflow_client_remove_dataset_from_experiments(mock_store):
    mock_dataset = Mock(spec=EvaluationDataset)
    mock_dataset.dataset_id = "dataset_123"
    mock_dataset.experiment_ids = ["1"]
    mock_store.remove_dataset_from_experiments.return_value = mock_dataset

    client = MlflowClient()
    result = client.remove_dataset_from_experiments(
        dataset_id="dataset_123",
        experiment_ids=["2", "3"],
    )

    assert result == mock_dataset
    assert result.experiment_ids == ["1"]
    mock_store.remove_dataset_from_experiments.assert_called_once_with("dataset_123", ["2", "3"])


def test_mlflow_client_dataset_associations_databricks_blocking(mock_store):
    with mock.patch("mlflow.utils.databricks_utils.is_databricks_uri") as mock_is_dbx:
        mock_is_dbx.return_value = True
        client = MlflowClient(tracking_uri="databricks")

        with pytest.raises(
            MlflowException, match="not supported when tracking URI is 'databricks'"
        ) as exc_info:
            client.add_dataset_to_experiments("dataset_123", ["1", "2"])
        assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"

        with pytest.raises(
            MlflowException, match="not supported when tracking URI is 'databricks'"
        ) as exc_info:
            client.remove_dataset_from_experiments("dataset_123", ["1", "2"])
        assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"


def test_log_spans_and_get_trace_with_sqlalchemy_store(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path}/test.db"

    with _use_tracking_uri(tracking_uri):
        client = MlflowClient()

        assert isinstance(client._tracking_client.store, SqlAlchemyTrackingStore)

        experiment_id = client.create_experiment("test_log_spans_get_trace")
        trace_id = f"tr-{uuid.uuid4().hex}"

        # Create test spans using OpenTelemetry format
        otel_span1 = OTelReadableSpan(
            name="parent_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=111,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={
                "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
                "llm.model_name": "test-model",
                "custom.attribute": "parent-value",
            },
            start_time=1_000_000_000,
            end_time=2_000_000_000,
            resource=None,
        )

        otel_span2 = OTelReadableSpan(
            name="child_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=222,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=trace_api.SpanContext(
                trace_id=12345,
                span_id=111,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            attributes={
                "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
                "operation.type": "database_query",
                "custom.attribute": "child-value",
            },
            start_time=1_200_000_000,
            end_time=1_800_000_000,
            resource=None,
        )

        # Convert to MLflow spans
        mlflow_spans = [
            create_mlflow_span(otel_span1, trace_id, "LLM"),
            create_mlflow_span(otel_span2, trace_id, "LLM"),
        ]

        # Log spans directly to the store (simulating OTLP endpoint)
        store = client._tracking_client.store
        logged_spans = store.log_spans(experiment_id, mlflow_spans)

        # Verify spans were logged
        assert len(logged_spans) == 2

        # Verify the trace has the spans location tag set
        trace_info = store.get_trace_info(trace_id)
        assert trace_info.tags.get(TraceTagKey.SPANS_LOCATION) == SpansLocation.TRACKING_STORE

        # Now test that mlflow.get_trace() works and loads spans from the database
        trace = mlflow.get_trace(trace_id)

        # Verify trace structure
        assert trace.info.trace_id == trace_id
        assert trace.info.tags.get(TraceTagKey.SPANS_LOCATION) == SpansLocation.TRACKING_STORE

        # Verify spans were loaded from database
        assert len(trace.data.spans) == 2

        # Sort spans by start time for consistent testing
        spans_by_start_time = sorted(trace.data.spans, key=lambda s: s.start_time_ns)

        # Verify parent span
        parent_span = spans_by_start_time[0]
        assert parent_span.name == "parent_span"
        assert parent_span.trace_id == trace_id
        assert parent_span.start_time_ns == 1_000_000_000
        assert parent_span.end_time_ns == 2_000_000_000
        assert parent_span.attributes.get("llm.model_name") == "test-model"
        assert parent_span.attributes.get("custom.attribute") == "parent-value"

        # Verify child span
        child_span = spans_by_start_time[1]
        assert child_span.name == "child_span"
        assert child_span.trace_id == trace_id
        assert child_span.start_time_ns == 1_200_000_000
        assert child_span.end_time_ns == 1_800_000_000
        assert child_span.attributes.get("operation.type") == "database_query"
        assert child_span.attributes.get("custom.attribute") == "child-value"


def test_mlflow_get_trace_with_sqlalchemy_store(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path}/test.db"

    with _use_tracking_uri(tracking_uri):
        client = MlflowClient()

        assert isinstance(client._tracking_client.store, SqlAlchemyTrackingStore)

        with mlflow.start_span() as span:
            pass

        trace_id = span.trace_id
        sql_alchemy_store_module = "mlflow.store.tracking.sqlalchemy_store.SqlAlchemyStore"
        with (
            mock.patch(f"{sql_alchemy_store_module}.get_trace") as mock_get_trace,
        ):
            mlflow.get_trace(trace_id)

        mock_get_trace.assert_called_once_with(trace_id)

        with (
            mock.patch(
                f"{sql_alchemy_store_module}.get_trace",
                side_effect=MlflowNotImplementedException,
            ) as mock_get_trace,
            mock.patch(f"{sql_alchemy_store_module}.batch_get_traces") as mock_batch_get_traces,
        ):
            mlflow.get_trace(trace_id)

        mock_get_trace.assert_called_once_with(trace_id)
        mock_batch_get_traces.assert_called_once_with([trace_id])

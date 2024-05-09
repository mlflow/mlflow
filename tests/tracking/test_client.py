import pickle
import time
from unittest import mock

import pytest

import mlflow
from mlflow import MlflowClient, flush_async_logging
from mlflow.config import enable_async_logging
from mlflow.entities import (
    ExperimentTag,
    Run,
    RunInfo,
    RunStatus,
    RunTag,
    SourceType,
    SpanStatusCode,
    SpanType,
    Trace,
    TraceInfo,
    ViewType,
)
from mlflow.entities.metric import Metric
from mlflow.entities.model_registry import ModelVersion, ModelVersionTag
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.entities.param import Param
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import MLFLOW_TRACKING_USERNAME
from mlflow.exceptions import MlflowException, MlflowTraceDataCorrupted, MlflowTraceDataNotFound
from mlflow.store.model_registry.sqlalchemy_store import (
    SqlAlchemyStore as SqlAlchemyModelRegistryStore,
)
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore as SqlAlchemyTrackingStore
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracking import set_registry_uri
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking._model_registry.utils import (
    _get_store_registry as _get_model_registry_store_registry,
)
from mlflow.tracking._tracking_service.utils import _register
from mlflow.utils.databricks_utils import _construct_databricks_run_url
from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_COMMIT,
    MLFLOW_PARENT_RUN_ID,
    MLFLOW_PROJECT_ENTRY_POINT,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_USER,
)

from tests.tracing.conftest import clear_singleton  # noqa: F401
from tests.tracing.conftest import mock_store as mock_store_for_tracing  # noqa: F401
from tests.tracing.helper import create_test_trace_info, get_traces


@pytest.fixture(autouse=True)
def reset_registry_uri():
    yield
    set_registry_uri(None)


@pytest.fixture
def mock_store():
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store") as mock_get_store:
        yield mock_get_store.return_value


@pytest.fixture
def mock_artifact_repo():
    with mock.patch(
        "mlflow.tracking._tracking_service.client.get_artifact_repository"
    ) as mock_get_repo:
        yield mock_get_repo.return_value


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
    mock_store.get_trace_info.return_value = TraceInfo(
        request_id="tr-1234567",
        experiment_id="0",
        timestamp_ms=123,
        execution_time_ms=456,
        status=TraceStatus.OK,
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

    assert trace.info.request_id == "tr-1234567"
    assert trace.info.experiment_id == "0"
    assert trace.info.timestamp_ms == 123
    assert trace.info.execution_time_ms == 456
    assert trace.info.status == TraceStatus.OK
    assert trace.info.tags == {"mlflow.artifactLocation": "dbfs:/path/to/artifacts"}
    assert trace.data.request == '{"prompt": "What is the meaning of life?"}'
    assert trace.data.response == '{"answer": 42}'
    assert len(trace.data.spans) == 1
    assert trace.data.spans[0].name == "predict"
    assert trace.data.spans[0].request_id == "tr-1234567"
    assert trace.data.spans[0].inputs == {"prompt": "What is the meaning of life?"}
    assert trace.data.spans[0].outputs == {"answer": 42}
    assert trace.data.spans[0].start_time_ns == 123000000
    assert trace.data.spans[0].end_time_ns == 579000000
    assert trace.data.spans[0].status.status_code == SpanStatusCode.OK


def test_client_get_trace_throws_for_missing_or_corrupted_data(mock_store, mock_artifact_repo):
    mock_store.get_trace_info.return_value = TraceInfo(
        request_id="1234567",
        experiment_id="0",
        timestamp_ms=123,
        execution_time_ms=456,
        status=TraceStatus.OK,
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


def test_client_search_traces(mock_store, mock_artifact_repo):
    mock_traces = [
        TraceInfo(
            request_id="1234567",
            experiment_id="1",
            timestamp_ms=123,
            execution_time_ms=456,
            status=TraceStatus.OK,
            tags={"mlflow.artifactLocation": "dbfs:/path/to/artifacts/1"},
        ),
        TraceInfo(
            request_id="8910",
            experiment_id="2",
            timestamp_ms=456,
            execution_time_ms=789,
            status=TraceStatus.OK,
            tags={"mlflow.artifactLocation": "dbfs:/path/to/artifacts/2"},
        ),
    ]
    mock_store.search_traces.return_value = (mock_traces, None)

    MlflowClient().search_traces(experiment_ids=["1", "2", "3"])

    mock_store.search_traces.assert_called_once_with(
        experiment_ids=["1", "2", "3"],
        filter_string=None,
        max_results=100,
        order_by=None,
        page_token=None,
    )
    mock_artifact_repo.download_trace_data.assert_called()
    # The TraceInfo is already fetched prior to the upload_trace_data call,
    # so we should not call _get_trace_info again
    mock_store.get_trace_info.assert_not_called()


def test_client_delete_traces(mock_store):
    MlflowClient().delete_traces(
        experiment_id="0",
        max_timestamp_millis=1,
        max_traces=2,
        request_ids=["tr-1234"],
    )
    mock_store.delete_traces.assert_called_once_with(
        experiment_id="0",
        max_timestamp_millis=1,
        max_traces=2,
        request_ids=["tr-1234"],
    )


@pytest.mark.parametrize("with_active_run", [True, False])
def test_start_and_end_trace(clear_singleton, with_active_run):
    class TestModel:
        def __init__(self):
            self._client = MlflowClient()
            self._exp_id = self._client.create_experiment("test_experiment")

        def predict(self, x, y):
            root_span = self._client.start_trace(
                name="predict",
                inputs={"x": x, "y": y},
                tags={"tag": "tag_value"},
                experiment_id=self._exp_id,
            )
            request_id = root_span.request_id

            z = x + y

            child_span = self._client.start_span(
                "child_span_1",
                span_type=SpanType.LLM,
                request_id=request_id,
                parent_id=root_span.span_id,
                inputs={"z": z},
            )

            z = z + 2

            self._client.end_span(
                request_id=request_id,
                span_id=child_span.span_id,
                outputs={"output": z},
                attributes={"delta": 2},
            )

            res = self.square(z, request_id, root_span.span_id)
            self._client.end_trace(request_id, outputs={"output": res}, status="OK")
            return res

        def square(self, t, request_id, parent_id):
            span = self._client.start_span(
                "child_span_2",
                request_id=request_id,
                parent_id=parent_id,
                inputs={"t": t},
            )

            res = t**2
            time.sleep(0.1)

            self._client.end_span(
                request_id=request_id,
                span_id=span.span_id,
                outputs={"output": res},
            )
            return res

    model = TestModel()
    if with_active_run:
        with mlflow.start_run() as run:
            model.predict(1, 2)
            run_id = run.info.run_id
    else:
        model.predict(1, 2)

    traces = get_traces()
    assert len(traces) == 1
    trace_info = traces[0].info
    assert trace_info.request_id is not None
    assert trace_info.execution_time_ms >= 0.1 * 1e3  # at least 0.1 sec
    assert trace_info.status == TraceStatus.OK
    assert trace_info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 1, "y": 2}'
    assert trace_info.request_metadata[TraceMetadataKey.OUTPUTS] == '{"output": 25}'
    if with_active_run:
        assert trace_info.request_metadata["mlflow.sourceRun"] == run_id
        assert trace_info.experiment_id == run.info.experiment_id
    else:
        assert trace_info.experiment_id == model._exp_id

    trace_data = traces[0].data
    assert trace_data.request == '{"x": 1, "y": 2}'
    assert trace_data.response == '{"output": 25}'
    assert len(trace_data.spans) == 3

    span_name_to_span = {span.name: span for span in trace_data.spans}
    root_span = span_name_to_span["predict"]
    assert root_span.start_time_ns // 1e6 == trace_info.timestamp_ms
    assert (root_span.end_time_ns - root_span.start_time_ns) // 1e6 == trace_info.execution_time_ms
    assert root_span.parent_id is None
    assert root_span.attributes == {
        "mlflow.experimentId": model._exp_id,
        "mlflow.traceRequestId": trace_info.request_id,
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"x": 1, "y": 2},
        "mlflow.spanOutputs": {"output": 25},
    }

    child_span_1 = span_name_to_span["child_span_1"]
    assert child_span_1.parent_id == root_span.span_id
    assert child_span_1.attributes == {
        "mlflow.traceRequestId": trace_info.request_id,
        "mlflow.spanType": "LLM",
        "mlflow.spanInputs": {"z": 3},
        "mlflow.spanOutputs": {"output": 5},
        "delta": 2,
    }

    child_span_2 = span_name_to_span["child_span_2"]
    assert child_span_2.parent_id == root_span.span_id
    assert child_span_2.attributes == {
        "mlflow.traceRequestId": trace_info.request_id,
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"t": 5},
        "mlflow.spanOutputs": {"output": 25},
    }
    assert child_span_2.start_time_ns <= child_span_2.end_time_ns - 0.1 * 1e6


@pytest.mark.usefixtures("reset_active_experiment")
def test_start_and_end_trace_before_all_span_end(clear_singleton):
    # This test is to verify that the trace is still exported even if some spans are not ended
    exp_id = mlflow.set_experiment("test_experiment_1").experiment_id

    class TestModel:
        def __init__(self):
            self._client = MlflowClient()

        def predict(self, x):
            root_span = self._client.start_trace(name="predict")
            request_id = root_span.request_id
            child_span = self._client.start_span(
                "ended-span",
                request_id=request_id,
                parent_id=root_span.span_id,
            )
            time.sleep(0.1)
            self._client.end_span(request_id, child_span.span_id)

            res = self.square(x, request_id, root_span.span_id)
            self._client.end_trace(request_id)
            return res

        def square(self, t, request_id, parent_id):
            self._client.start_span("non-ended-span", request_id=request_id, parent_id=parent_id)
            time.sleep(0.1)
            # The span created above is not ended
            return t**2

    model = TestModel()
    model.predict(1)

    traces = get_traces()
    assert len(traces) == 1

    trace_info = traces[0].info
    assert trace_info.request_id is not None
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
    assert root_span.start_time_ns // 1e6 == trace_info.timestamp_ms
    assert (root_span.end_time_ns - root_span.start_time_ns) // 1e6 == trace_info.execution_time_ms

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


@mock.patch("mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks")
def test_log_trace_with_databricks_tracking_uri(
    clear_singleton, mock_store_for_tracing, monkeypatch
):
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")

    mock_experiment = mock.MagicMock()
    mock_experiment.experiment_id = "test_experiment_id"
    monkeypatch.setattr(
        mock_store_for_tracing,
        "get_experiment_by_name",
        mock.MagicMock(return_value=mock_experiment),
    )

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
            request_id = root_span.request_id

            z = x + y

            child_span = self._client.start_span(
                "child_span_1",
                span_type=SpanType.LLM,
                request_id=request_id,
                parent_id=root_span.span_id,
                inputs={"z": z},
            )

            z = z + 2

            self._client.end_span(
                request_id=request_id,
                span_id=child_span.span_id,
                outputs={"output": z},
                attributes={"delta": 2},
            )
            self._client.end_trace(request_id, outputs=z, status="OK")
            return z

    model = TestModel()

    def _mock_update_trace_info(trace_info):
        trace_manager = InMemoryTraceManager.get_instance()
        with trace_manager.get_trace("tr-12345") as trace:
            trace.info.tags.update({"tag": "tag_value"})

    with mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient._upload_trace_data"
    ) as mock_upload_trace_data, mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient.set_trace_tags",
    ), mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient.get_trace_info",
    ), mock.patch(
        "mlflow.tracing.trace_manager.InMemoryTraceManager.update_trace_info",
        side_effect=_mock_update_trace_info,
    ):
        model.predict(1, 2)

    traces = get_traces()
    assert len(traces) == 1
    trace_info = traces[0].info
    assert trace_info.request_id == "tr-12345"
    assert trace_info.experiment_id == "test_experiment_id"
    assert trace_info.status == TraceStatus.OK
    assert trace_info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 1, "y": 2}'
    assert trace_info.request_metadata[TraceMetadataKey.OUTPUTS] == "5"
    assert trace_info.tags == {
        "mlflow.traceName": "predict",
        "mlflow.artifactLocation": "test",
        "mlflow.user": "bob",
        "mlflow.source.name": "test",
        "mlflow.source.type": "LOCAL",
        "tag": "tag_value",
    }

    trace_data = traces[0].data
    assert trace_data.request == '{"x": 1, "y": 2}'
    assert trace_data.response == "5"
    assert len(trace_data.spans) == 2

    mock_store_for_tracing.start_trace.assert_called_once()
    mock_store_for_tracing.end_trace.assert_called_once()
    mock_upload_trace_data.assert_called_once()


def test_start_trace_raise_error_when_active_trace_exists(clear_singleton):
    with mlflow.start_span("fluent_span"):
        with pytest.raises(MlflowException, match=r"Another trace is already set in the global"):
            mlflow.tracking.MlflowClient().start_trace("test")


def test_end_trace_raise_error_when_trace_not_exist(clear_singleton):
    client = mlflow.tracking.MlflowClient()
    mock_tracking_client = mock.MagicMock()
    mock_tracking_client.get_trace.return_value = None
    client._tracking_client = mock_tracking_client

    with pytest.raises(MlflowException, match=r"Trace with ID test not found"):
        client.end_trace("test")


def test_end_trace_raise_error_when_trace_finished_twice(clear_singleton):
    client = mlflow.tracking.MlflowClient()
    mock_tracking_client = mock.MagicMock()
    mock_tracking_client.get_trace.return_value = Trace(
        info=create_test_trace_info("test"), data=None
    )
    client._tracking_client = mock_tracking_client

    with pytest.raises(MlflowException, match=r"Trace with ID test already finished"):
        client.end_trace("test")


def test_start_span_raise_error_when_parent_id_is_not_provided():
    with pytest.raises(MlflowException, match=r"start_span\(\) must be called with"):
        mlflow.tracking.MlflowClient().start_span("span_name", request_id="test", parent_id=None)


def test_set_and_delete_trace_tag_on_active_trace(clear_singleton, monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")

    client = mlflow.tracking.MlflowClient()

    root_span = client.start_trace(name="test")
    request_id = root_span.request_id
    client.set_trace_tag(request_id, "foo", "bar")
    client.end_trace(request_id)

    trace = get_traces()[0]
    assert trace.info.tags["mlflow.traceName"] == "test"
    assert trace.info.tags["foo"] == "bar"
    assert trace.info.tags["mlflow.source.name"] == "test"
    assert trace.info.tags["mlflow.source.type"] == "LOCAL"


def test_set_trace_tag_on_logged_trace(mock_store, clear_singleton):
    mlflow.tracking.MlflowClient().set_trace_tag("test", "foo", "bar")
    mock_store.set_trace_tag.assert_called_once_with("test", "foo", "bar")


def test_delete_trace_tag_on_active_trace(clear_singleton, monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")

    client = mlflow.tracking.MlflowClient()
    root_span = client.start_trace(name="test", tags={"foo": "bar", "baz": "qux"})
    request_id = root_span.request_id
    client.delete_trace_tag(request_id, "foo")
    client.end_trace(request_id)

    trace = get_traces()[0]
    assert trace.info.tags["baz"] == "qux"
    assert trace.info.tags["mlflow.traceName"] == "test"
    assert trace.info.tags["mlflow.source.name"] == "test"
    assert trace.info.tags["mlflow.source.type"] == "LOCAL"


def test_delete_trace_tag_on_logged_trace(mock_store, clear_singleton):
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
        name="orig name", description="new description"
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
        assert client._registry_uri == tracking_uri


def test_registry_uri_from_implicit_tracking_uri():
    tracking_uri = "databricks://tracking_wierojasdf"
    with mock.patch(
        "mlflow.tracking._tracking_service.utils.get_tracking_uri",
        return_value=tracking_uri,
    ):
        client = MlflowClient()
        assert client._registry_uri == tracking_uri


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
        "name", "source", "runid", [], None, None, local_model_path=None
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
        "name", "source", None, [], None, None, local_model_path=None
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
    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_notebook", return_value=True
    ), mock.patch(
        "mlflow.utils.databricks_utils.get_workspace_info_from_dbutils",
        return_value=(hostname, workspace_id),
    ):
        client = MlflowClient(tracking_uri="databricks", registry_uri="otherplace")
        model_version = client.create_model_version("name", "source", "runid", run_link=run_link)
        assert model_version.run_link == run_link
        # verify that the store was provided with the explicitly passed in run link
        mock_registry_store.create_model_version.assert_called_once_with(
            "name", "source", "runid", [], run_link, None, local_model_path=None
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

    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_notebook", return_value=True
    ), mock.patch(
        "mlflow.utils.databricks_utils.get_workspace_info_from_dbutils",
        return_value=(hostname, workspace_id),
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
            "name", "source", "runid", [], workspace_url, None, local_model_path=None
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
        "name", "source", "runid", [], None, None, local_model_path=None
    )
    client.create_registered_model(name="name", description="description")
    # verify that registry store was called with tags=[]
    mock_registry_store.create_registered_model.assert_called_once_with("name", [], "description")


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

    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_notebook", return_value=False
    ), mock.patch(
        "mlflow.utils.databricks_utils.get_workspace_info_from_databricks_secrets",
        return_value=(hostname, workspace_id),
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
            "name", "source", "runid", [], workspace_url, None, local_model_path=None
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

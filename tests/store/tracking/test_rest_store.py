import json
import math
import time
from unittest import mock

import pytest

import mlflow
from mlflow.entities import (
    Dataset,
    DatasetInput,
    EvaluationDataset,
    Experiment,
    ExperimentTag,
    GatewayResourceType,
    InputTag,
    LifecycleStage,
    LoggedModelParameter,
    Metric,
    Param,
    RunTag,
    SourceType,
    ViewType,
)
from mlflow.entities.assessment import (
    Assessment,
    Expectation,
    ExpectationValue,
    Feedback,
    FeedbackValue,
)
from mlflow.entities.assessment_error import AssessmentError
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.span import LiveSpan
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_info_v2 import TraceInfoV2
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_metrics import (
    AggregationType,
    MetricAggregation,
    MetricDataPoint,
    MetricViewType,
)
from mlflow.entities.trace_state import TraceState
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import (
    _MLFLOW_CREATE_LOGGED_MODEL_PARAMS_BATCH_SIZE,
    _MLFLOW_LOG_LOGGED_MODEL_PARAMS_BATCH_SIZE,
    MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT,
)
from mlflow.exceptions import MlflowException, MlflowNotImplementedException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.protos.service_pb2 import (
    AddDatasetToExperiments,
    AttachModelToGatewayEndpoint,
    CalculateTraceFilterCorrelation,
    CreateAssessment,
    CreateDataset,
    CreateGatewayEndpoint,
    CreateGatewayEndpointBinding,
    CreateGatewayModelDefinition,
    CreateGatewaySecret,
    CreateLoggedModel,
    CreateRun,
    DeleteDataset,
    DeleteDatasetTag,
    DeleteExperiment,
    DeleteGatewayEndpoint,
    DeleteGatewayEndpointBinding,
    DeleteGatewayModelDefinition,
    DeleteGatewaySecret,
    DeleteRun,
    DeleteScorer,
    DeleteTag,
    DeleteTraces,
    DetachModelFromGatewayEndpoint,
    EndTrace,
    GetDataset,
    GetDatasetExperimentIds,
    GetDatasetRecords,
    GetExperimentByName,
    GetGatewayEndpoint,
    GetGatewayModelDefinition,
    GetGatewaySecretInfo,
    GetLoggedModel,
    GetScorer,
    GetTrace,
    GetTraceInfoV3,
    LinkPromptsToTrace,
    ListGatewayEndpointBindings,
    ListGatewayEndpoints,
    ListGatewayModelDefinitions,
    ListGatewaySecretInfos,
    ListScorers,
    ListScorerVersions,
    LogBatch,
    LogInputs,
    LogLoggedModelParamsRequest,
    LogMetric,
    LogModel,
    LogParam,
    RegisterScorer,
    RemoveDatasetFromExperiments,
    RestoreExperiment,
    RestoreRun,
    SearchEvaluationDatasets,
    SearchExperiments,
    SearchRuns,
    SetDatasetTags,
    SetExperimentTag,
    SetTag,
    SetTraceTag,
    StartTrace,
    StartTraceV3,
    UpdateGatewayEndpoint,
    UpdateGatewayModelDefinition,
    UpdateGatewaySecret,
    UpsertDatasetRecords,
)
from mlflow.protos.service_pb2 import RunTag as ProtoRunTag
from mlflow.protos.service_pb2 import TraceRequestMetadata as ProtoTraceRequestMetadata
from mlflow.protos.service_pb2 import TraceTag as ProtoTraceTag
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracing.analysis import TraceFilterCorrelationResult
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION_KEY
from mlflow.tracking.request_header.default_request_header_provider import (
    DefaultRequestHeaderProvider,
)
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    _V3_TRACE_REST_API_PATH_PREFIX,
    MlflowHostCreds,
    get_logged_model_endpoint,
)

from tests.tracing.helper import create_mock_otel_span


class MyCoolException(Exception):
    pass


class CustomErrorHandlingRestStore(RestStore):
    def _call_endpoint(self, api, json_body):
        raise MyCoolException("cool")


def mock_http_request():
    return mock.patch(
        "mlflow.utils.rest_utils.http_request",
        return_value=mock.MagicMock(status_code=200, text="{}"),
    )


def test_successful_http_request():
    def mock_request(*args, **kwargs):
        # Filter out None arguments
        assert args == ("POST", "https://hello/api/2.0/mlflow/experiments/search")
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        assert kwargs == {
            "allow_redirects": True,
            "json": {"view_type": "ACTIVE_ONLY"},
            "headers": DefaultRequestHeaderProvider().request_headers(),
            "verify": True,
            "timeout": 120,
        }
        response = mock.MagicMock()
        response.status_code = 200
        response.text = '{"experiments": [{"name": "Exp!", "lifecycle_stage": "active"}]}'
        return response

    with mock.patch("requests.Session.request", side_effect=mock_request):
        store = RestStore(lambda: MlflowHostCreds("https://hello"))
        experiments = store.search_experiments()
        assert experiments[0].name == "Exp!"


def test_failed_http_request():
    response = mock.MagicMock()
    response.status_code = 404
    response.text = '{"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "No experiment"}'
    with mock.patch("requests.Session.request", return_value=response):
        store = RestStore(lambda: MlflowHostCreds("https://hello"))
        with pytest.raises(MlflowException, match="RESOURCE_DOES_NOT_EXIST: No experiment"):
            store.search_experiments()


def test_failed_http_request_custom_handler():
    response = mock.MagicMock()
    response.status_code = 404
    response.text = '{"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "No experiment"}'

    with mock.patch("requests.Session.request", return_value=response):
        store = CustomErrorHandlingRestStore(lambda: MlflowHostCreds("https://hello"))
        with pytest.raises(MyCoolException, match="cool"):
            store.search_experiments()


def test_response_with_unknown_fields():
    experiment_json = {
        "experiment_id": "1",
        "name": "My experiment",
        "artifact_location": "foo",
        "lifecycle_stage": "deleted",
        "OMG_WHAT_IS_THIS_FIELD": "Hooly cow",
    }

    response = mock.MagicMock()
    response.status_code = 200
    experiments = {"experiments": [experiment_json]}
    response.text = json.dumps(experiments)
    with mock.patch("requests.Session.request", return_value=response):
        store = RestStore(lambda: MlflowHostCreds("https://hello"))
        experiments = store.search_experiments()
        assert len(experiments) == 1
        assert experiments[0].name == "My experiment"


def _args(host_creds, endpoint, method, json_body, use_v3=False, retry_timeout_seconds=None):
    version = "3.0" if use_v3 else "2.0"
    res = {
        "host_creds": host_creds,
        "endpoint": f"/api/{version}/mlflow/{endpoint}",
        "method": method,
    }
    if retry_timeout_seconds is not None:
        res["retry_timeout_seconds"] = retry_timeout_seconds
    if method == "GET":
        res["params"] = json.loads(json_body)
    else:
        res["json"] = json.loads(json_body)
    return res


def _verify_requests(
    http_request, host_creds, endpoint, method, json_body, use_v3=False, retry_timeout_seconds=None
):
    """
    Verify HTTP requests in tests.

    Args:
        http_request: The mocked HTTP request object
        host_creds: MlflowHostCreds object
        endpoint: The endpoint being called (e.g., "traces/123")
        method: The HTTP method (e.g., "GET", "POST")
        json_body: The request body as a JSON string
        use_v3: If True, verify using /api/3.0/mlflow/ prefix instead of /api/2.0/mlflow/
                This is used for trace-related endpoints that use the V3 API.
        retry_timeout_seconds: The retry timeout seconds to use for the request
    """
    http_request.assert_any_call(
        **(_args(host_creds, endpoint, method, json_body, use_v3, retry_timeout_seconds))
    )


def test_requestor():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    user_name = "mock user"
    source_name = "rest test"
    run_name = "my name"

    source_name_patch = mock.patch(
        "mlflow.tracking.context.default_context._get_source_name", return_value=source_name
    )
    source_type_patch = mock.patch(
        "mlflow.tracking.context.default_context._get_source_type",
        return_value=SourceType.LOCAL,
    )
    with (
        mock_http_request() as mock_http,
        mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=store),
        mock.patch("mlflow.tracking.context.default_context._get_user", return_value=user_name),
        mock.patch("time.time", return_value=13579),
        source_name_patch,
        source_type_patch,
    ):
        with mlflow.start_run(experiment_id="43", run_name=run_name):
            cr_body = message_to_json(
                CreateRun(
                    experiment_id="43",
                    user_id=user_name,
                    run_name=run_name,
                    start_time=13579000,
                    tags=[
                        ProtoRunTag(key="mlflow.source.name", value=source_name),
                        ProtoRunTag(key="mlflow.source.type", value="LOCAL"),
                        ProtoRunTag(key="mlflow.user", value=user_name),
                        ProtoRunTag(key="mlflow.runName", value=run_name),
                    ],
                )
            )
            expected_kwargs = _args(creds, "runs/create", "POST", cr_body)

            assert mock_http.call_count == 1
            actual_kwargs = mock_http.call_args[1]

            # Test the passed tag values separately from the rest of the request
            # Tag order is inconsistent on Python 2 and 3, but the order does not matter
            expected_tags = expected_kwargs["json"].pop("tags")
            actual_tags = actual_kwargs["json"].pop("tags")

            assert sorted(expected_tags, key=lambda t: t["key"]) == sorted(
                actual_tags, key=lambda t: t["key"]
            )
            assert expected_kwargs == actual_kwargs

    with mock_http_request() as mock_http:
        store.log_param("some_uuid", Param("k1", "v1"))
        body = message_to_json(
            LogParam(run_uuid="some_uuid", run_id="some_uuid", key="k1", value="v1")
        )
        _verify_requests(mock_http, creds, "runs/log-parameter", "POST", body)

    with mock_http_request() as mock_http:
        store.set_experiment_tag("some_id", ExperimentTag("t1", "abcd" * 1000))
        body = message_to_json(
            SetExperimentTag(experiment_id="some_id", key="t1", value="abcd" * 1000)
        )
        _verify_requests(mock_http, creds, "experiments/set-experiment-tag", "POST", body)

    with mock_http_request() as mock_http:
        store.set_tag("some_uuid", RunTag("t1", "abcd" * 1000))
        body = message_to_json(
            SetTag(run_uuid="some_uuid", run_id="some_uuid", key="t1", value="abcd" * 1000)
        )
        _verify_requests(mock_http, creds, "runs/set-tag", "POST", body)

    with mock_http_request() as mock_http:
        store.delete_tag("some_uuid", "t1")
        body = message_to_json(DeleteTag(run_id="some_uuid", key="t1"))
        _verify_requests(mock_http, creds, "runs/delete-tag", "POST", body)

    with mock_http_request() as mock_http:
        store.log_metric("u2", Metric("m1", 0.87, 12345, 3))
        body = message_to_json(
            LogMetric(run_uuid="u2", run_id="u2", key="m1", value=0.87, timestamp=12345, step=3)
        )
        _verify_requests(mock_http, creds, "runs/log-metric", "POST", body)

    with mock_http_request() as mock_http:
        metrics = [
            Metric("m1", 0.87, 12345, 0),
            Metric("m2", 0.49, 12345, -1),
            Metric("m3", 0.58, 12345, 2),
        ]
        params = [Param("p1", "p1val"), Param("p2", "p2val")]
        tags = [RunTag("t1", "t1val"), RunTag("t2", "t2val")]
        store.log_batch(run_id="u2", metrics=metrics, params=params, tags=tags)
        metric_protos = [metric.to_proto() for metric in metrics]
        param_protos = [param.to_proto() for param in params]
        tag_protos = [tag.to_proto() for tag in tags]
        body = message_to_json(
            LogBatch(run_id="u2", metrics=metric_protos, params=param_protos, tags=tag_protos)
        )
        _verify_requests(mock_http, creds, "runs/log-batch", "POST", body)

    with mock_http_request() as mock_http:
        dataset = Dataset(name="name", digest="digest", source_type="st", source="source")
        tag = InputTag(key="k1", value="v1")
        dataset_input = DatasetInput(dataset=dataset, tags=[tag])
        store.log_inputs("some_uuid", [dataset_input])
        body = message_to_json(LogInputs(run_id="some_uuid", datasets=[dataset_input.to_proto()]))
        _verify_requests(mock_http, creds, "runs/log-inputs", "POST", body)

    with mock_http_request() as mock_http:
        store.delete_run("u25")
        _verify_requests(
            mock_http, creds, "runs/delete", "POST", message_to_json(DeleteRun(run_id="u25"))
        )

    with mock_http_request() as mock_http:
        store.restore_run("u76")
        _verify_requests(
            mock_http, creds, "runs/restore", "POST", message_to_json(RestoreRun(run_id="u76"))
        )

    with mock_http_request() as mock_http:
        store.delete_experiment("0")
        _verify_requests(
            mock_http,
            creds,
            "experiments/delete",
            "POST",
            message_to_json(DeleteExperiment(experiment_id="0")),
        )

    with mock_http_request() as mock_http:
        store.restore_experiment("0")
        _verify_requests(
            mock_http,
            creds,
            "experiments/restore",
            "POST",
            message_to_json(RestoreExperiment(experiment_id="0")),
        )

    with mock.patch("mlflow.utils.rest_utils.http_request") as mock_http:
        response = mock.MagicMock()
        response.status_code = 200
        response.text = '{"runs": ["1a", "2b", "3c"], "next_page_token": "67890fghij"}'
        mock_http.return_value = response
        result = store.search_runs(
            ["0", "1"],
            "params.p1 = 'a'",
            ViewType.ACTIVE_ONLY,
            max_results=10,
            order_by=["a"],
            page_token="12345abcde",
        )

        expected_message = SearchRuns(
            experiment_ids=["0", "1"],
            filter="params.p1 = 'a'",
            run_view_type=ViewType.to_proto(ViewType.ACTIVE_ONLY),
            max_results=10,
            order_by=["a"],
            page_token="12345abcde",
        )
        _verify_requests(mock_http, creds, "runs/search", "POST", message_to_json(expected_message))
        assert result.token == "67890fghij"

    with mock_http_request() as mock_http:
        run_id = "run_id"
        m = Model(artifact_path="model/path", run_id="run_id", flavors={"tf": "flavor body"})
        store.record_logged_model("run_id", m)
        expected_message = LogModel(run_id=run_id, model_json=json.dumps(m.get_tags_dict()))
        _verify_requests(
            mock_http, creds, "runs/log-model", "POST", message_to_json(expected_message)
        )

    # if model has config, it should be removed from the model_json before sending to the server
    with mock_http_request() as mock_http:
        run_id = "run_id"
        flavors_with_config = {
            "tf": "flavor body",
            "python_function": {"config": {"a": 1}, "code": "code"},
        }
        m_with_config = Model(
            artifact_path="model/path", run_id="run_id", flavors=flavors_with_config
        )
        store.record_logged_model("run_id", m_with_config)
        flavors = m_with_config.get_tags_dict().get("flavors", {})
        assert all("config" not in v for v in flavors.values())
        expected_message = LogModel(
            run_id=run_id, model_json=json.dumps(m_with_config.get_tags_dict())
        )
        _verify_requests(
            mock_http,
            creds,
            "runs/log-model",
            "POST",
            message_to_json(expected_message),
        )

    with mock_http_request() as mock_http:
        trace_id = "tr-123"
        store.get_trace_info(trace_id)

        # Verify the V3 API was called
        v3_expected_message = GetTraceInfoV3(trace_id=trace_id)
        _verify_requests(
            mock_http,
            creds,
            "traces/tr-123",
            "GET",
            message_to_json(v3_expected_message),
            use_v3=True,
        )


def test_get_experiment_by_name():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    with mock.patch("mlflow.utils.rest_utils.http_request") as mock_http:
        response = mock.MagicMock()
        response.status_code = 200
        experiment = Experiment(
            experiment_id="123",
            name="abc",
            artifact_location="/abc",
            lifecycle_stage=LifecycleStage.ACTIVE,
        )
        response.text = json.dumps(
            {"experiment": json.loads(message_to_json(experiment.to_proto()))}
        )
        mock_http.return_value = response
        result = store.get_experiment_by_name("abc")
        expected_message0 = GetExperimentByName(experiment_name="abc")
        _verify_requests(
            mock_http,
            creds,
            "experiments/get-by-name",
            "GET",
            message_to_json(expected_message0),
        )
        assert result.experiment_id == experiment.experiment_id
        assert result.name == experiment.name
        assert result.artifact_location == experiment.artifact_location
        assert result.lifecycle_stage == experiment.lifecycle_stage
        # Test GetExperimentByName against nonexistent experiment
        mock_http.reset_mock()
        nonexistent_exp_response = mock.MagicMock()
        nonexistent_exp_response.status_code = 404
        nonexistent_exp_response.text = MlflowException(
            "Exp doesn't exist!", RESOURCE_DOES_NOT_EXIST
        ).serialize_as_json()
        mock_http.return_value = nonexistent_exp_response
        assert store.get_experiment_by_name("nonexistent-experiment") is None
        expected_message1 = GetExperimentByName(experiment_name="nonexistent-experiment")
        _verify_requests(
            mock_http,
            creds,
            "experiments/get-by-name",
            "GET",
            message_to_json(expected_message1),
        )
        assert mock_http.call_count == 1


def test_search_experiments():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.search_experiments(
            view_type=ViewType.DELETED_ONLY,
            max_results=5,
            filter_string="name",
            order_by=["name"],
            page_token="abc",
        )
        _verify_requests(
            mock_http,
            creds,
            "experiments/search",
            "POST",
            message_to_json(
                SearchExperiments(
                    view_type=ViewType.DELETED_ONLY,
                    max_results=5,
                    filter="name",
                    order_by=["name"],
                    page_token="abc",
                )
            ),
        )


def _mock_response_with_200_status_code():
    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    return mock_response


def test_get_metric_history_paginated():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    response_1 = _mock_response_with_200_status_code()
    response_2 = _mock_response_with_200_status_code()
    response_payload_1 = {
        "metrics": [
            {"key": "a_metric", "value": 42, "timestamp": 123456777, "step": 0},
            {"key": "a_metric", "value": 46, "timestamp": 123456797, "step": 1},
        ],
        "next_page_token": "AcursorForTheRestofTheData",
    }
    response_1.text = json.dumps(response_payload_1)
    response_payload_2 = {
        "metrics": [
            {"key": "a_metric", "value": 40, "timestamp": 123456877, "step": 2},
            {"key": "a_metric", "value": 56, "timestamp": 123456897, "step": 3},
        ],
        "next_page_token": "",
    }
    response_2.text = json.dumps(response_payload_2)
    with mock.patch(
        "requests.Session.request", side_effect=[response_1, response_2]
    ) as mock_request:
        # Fetch the first page
        metrics = store.get_metric_history(
            run_id="2", metric_key="a_metric", max_results=2, page_token=None
        )
        mock_request.assert_called_once()
        assert mock_request.call_args.kwargs["params"] == {
            "max_results": 2,
            "metric_key": "a_metric",
            "run_id": "2",
            "run_uuid": "2",
        }
        assert len(metrics) == 2
        assert metrics[0] == Metric(key="a_metric", value=42, timestamp=123456777, step=0)
        assert metrics[1] == Metric(key="a_metric", value=46, timestamp=123456797, step=1)
        assert metrics.token == "AcursorForTheRestofTheData"
        # Fetch the second page
        mock_request.reset_mock()
        metrics = store.get_metric_history(
            run_id="2", metric_key="a_metric", max_results=2, page_token=metrics.token
        )
        mock_request.assert_called_once()
        assert mock_request.call_args.kwargs["params"] == {
            "max_results": 2,
            "page_token": "AcursorForTheRestofTheData",
            "metric_key": "a_metric",
            "run_id": "2",
            "run_uuid": "2",
        }
        assert len(metrics) == 2
        assert metrics[0] == Metric(key="a_metric", value=40, timestamp=123456877, step=2)
        assert metrics[1] == Metric(key="a_metric", value=56, timestamp=123456897, step=3)
        assert metrics.token is None


def test_get_metric_history_on_non_existent_metric_key():
    creds = MlflowHostCreds("https://hello")
    rest_store = RestStore(lambda: creds)
    empty_metric_response = _mock_response_with_200_status_code()
    empty_metric_response.text = json.dumps({})
    with mock.patch(
        "requests.Session.request", side_effect=[empty_metric_response]
    ) as mock_request:
        metrics = rest_store.get_metric_history(run_id="1", metric_key="test_metric")
        mock_request.assert_called_once()
        assert metrics == []


def test_deprecated_start_trace_v2():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    request_id = "tr-123"
    experiment_id = "447585625682310"
    timestamp_ms = 123
    # Metadata/tags values should be string, but should not break for other types too
    metadata = {"key1": "val1", "key2": "val2", "key3": 123, TRACE_SCHEMA_VERSION_KEY: "2"}
    tags = {"tag1": "tv1", "tag2": "tv2", "tag3": None}
    expected_request = StartTrace(
        experiment_id=experiment_id,
        timestamp_ms=123,
        request_metadata=[
            ProtoTraceRequestMetadata(key=k, value=str(v)) for k, v in metadata.items()
        ],
        tags=[ProtoTraceTag(key=k, value=str(v)) for k, v in tags.items()],
    )
    response = mock.MagicMock()
    response.status_code = 200
    response.text = json.dumps(
        {
            "trace_info": {
                "request_id": request_id,
                "experiment_id": experiment_id,
                "timestamp_ms": timestamp_ms,
                "execution_time_ms": None,
                "status": 0,  # Running
                "request_metadata": [{"key": k, "value": str(v)} for k, v in metadata.items()],
                "tags": [{"key": k, "value": str(v)} for k, v in tags.items()],
            }
        }
    )
    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        res = store.deprecated_start_trace_v2(
            experiment_id=experiment_id,
            timestamp_ms=timestamp_ms,
            request_metadata=metadata,
            tags=tags,
        )
        _verify_requests(mock_http, creds, "traces", "POST", message_to_json(expected_request))
        assert isinstance(res, TraceInfoV2)
        assert res.request_id == request_id
        assert res.experiment_id == experiment_id
        assert res.timestamp_ms == timestamp_ms
        assert res.execution_time_ms == 0
        assert res.status == TraceStatus.UNSPECIFIED
        assert res.request_metadata == {k: str(v) for k, v in metadata.items()}
        assert res.tags == {k: str(v) for k, v in tags.items()}


def test_start_trace(monkeypatch):
    monkeypatch.setenv(MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT.name, "1")

    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    trace = Trace(
        info=TraceInfo(
            trace_id="tr-123",
            trace_location=TraceLocation.from_experiment_id("123"),
            request_time=123,
            execution_duration=10,
            state=TraceState.OK,
            request_preview="",
            response_preview="",
            trace_metadata={},
        ),
        data=TraceData(),
    )

    response = mock.MagicMock()
    response.status_code = 200
    response.text = json.dumps({})

    expected_request = StartTraceV3(trace=trace.to_proto())

    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        store.start_trace(trace.info)
        _verify_requests(
            mock_http,
            creds,
            "traces",
            "POST",
            message_to_json(expected_request),
            use_v3=True,
            retry_timeout_seconds=1,
        )


def test_deprecated_end_trace_v2():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    experiment_id = "447585625682310"
    request_id = "tr-123"
    timestamp_ms = 123
    status = TraceStatus.OK
    metadata = {"key1": "val1", "key2": "val2", TRACE_SCHEMA_VERSION_KEY: "2"}
    tags = {"tag1": "tv1", "tag2": "tv2"}
    expected_request = EndTrace(
        request_id=request_id,
        timestamp_ms=123,
        status=status,
        request_metadata=[ProtoTraceRequestMetadata(key=k, value=v) for k, v in metadata.items()],
        tags=[ProtoTraceTag(key=k, value=v) for k, v in tags.items()],
    )
    response = mock.MagicMock()
    response.status_code = 200
    response.text = json.dumps(
        {
            "trace_info": {
                "request_id": request_id,
                "experiment_id": experiment_id,
                "timestamp_ms": timestamp_ms,
                "execution_time_ms": 12345,
                "status": 1,  # OK
                "request_metadata": [{"key": k, "value": v} for k, v in metadata.items()],
                "tags": [{"key": k, "value": v} for k, v in tags.items()],
            }
        }
    )

    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        res = store.deprecated_end_trace_v2(
            request_id=request_id,
            timestamp_ms=timestamp_ms,
            status=status,
            request_metadata=metadata,
            tags=tags,
        )
        _verify_requests(
            mock_http,
            creds,
            f"traces/{request_id}",
            "PATCH",
            message_to_json(expected_request),
            use_v3=False,
        )
        assert isinstance(res, TraceInfoV2)
        assert res.request_id == request_id
        assert res.experiment_id == experiment_id
        assert res.timestamp_ms == timestamp_ms
        assert res.execution_time_ms == 12345
        assert res.status == TraceStatus.OK
        assert res.request_metadata == metadata
        assert res.tags == tags


def test_search_traces():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200

    # Format the response
    response.text = json.dumps(
        {
            "traces": [
                {
                    "trace_id": "tr-1234",
                    "trace_location": {
                        "type": "MLFLOW_EXPERIMENT",
                        "mlflow_experiment": {"experiment_id": "1234"},
                    },
                    "request_time": "1970-01-01T00:00:00.123Z",
                    "execution_duration_ms": 456,
                    "state": "OK",
                    "trace_metadata": {"key": "value"},
                    "tags": {"k": "v"},
                }
            ],
            "next_page_token": "token",
        }
    )

    # Parameters for search_traces
    experiment_ids = ["1234"]
    filter_string = "state = 'OK'"
    max_results = 10
    order_by = ["request_time DESC"]
    page_token = "12345abcde"

    # Test with databricks tracking URI (using v3 endpoint)
    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        trace_infos, token = store.search_traces(
            locations=experiment_ids,
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
            page_token=page_token,
        )

        # Verify the correct endpoint was called
        call_args = mock_http.call_args[1]
        assert call_args["endpoint"] == f"{_V3_TRACE_REST_API_PATH_PREFIX}/search"

        # Verify the correct parameters were passed
        json_body = call_args["json"]
        # The field name should now be 'locations' instead of 'trace_locations'
        assert "locations" in json_body
        # The experiment_ids are converted to trace_locations
        assert len(json_body["locations"]) == 1
        assert json_body["locations"][0]["mlflow_experiment"]["experiment_id"] == experiment_ids[0]
        assert json_body["filter"] == filter_string
        assert json_body["max_results"] == max_results
        assert json_body["order_by"] == order_by
        assert json_body["page_token"] == page_token

    # Verify the correct parameters were passed and the correct trace info objects were returned
    # for either endpoint
    assert len(trace_infos) == 1
    assert isinstance(trace_infos[0], TraceInfo)
    assert trace_infos[0].trace_id == "tr-1234"
    assert trace_infos[0].experiment_id == "1234"
    assert trace_infos[0].request_time == 123
    # V3's state maps to V2's status
    assert trace_infos[0].state == TraceStatus.OK.to_state()
    # This is correct because TraceInfoV3.from_proto converts the repeated field tags to a dict
    assert trace_infos[0].tags == {"k": "v"}
    assert trace_infos[0].trace_metadata == {"key": "value"}
    assert token == "token"


def test_search_traces_errors():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    with pytest.raises(
        MlflowException,
        match="Locations must be a list of experiment IDs",
    ):
        store.search_traces(locations=["catalog.schema"])

    with pytest.raises(
        MlflowException,
        match="Searching traces by model_id is not supported on the current tracking server.",
    ):
        store.search_traces(model_id="model_id")


def test_get_artifact_uri_for_trace_compatibility():
    from mlflow.tracing.utils.artifact_utils import get_artifact_uri_for_trace

    # Create a TraceInfo (v2) object
    trace_info_v2 = TraceInfoV2(
        request_id="tr-1234",
        experiment_id="1234",
        timestamp_ms=123,
        execution_time_ms=456,
        status=TraceStatus.OK,
        request_metadata={"key": "value"},
        tags={MLFLOW_ARTIFACT_LOCATION: "s3://bucket/trace-v2-path"},
    )

    # Create a TraceInfoV3 object
    trace_location = TraceLocation.from_experiment_id("5678")
    trace_info_v3 = TraceInfo(
        trace_id="tr-5678",
        trace_location=trace_location,
        request_time=789,
        state=TraceState.OK,
        trace_metadata={"key3": "value3"},
        tags={MLFLOW_ARTIFACT_LOCATION: "s3://bucket/trace-v3-path"},
    )

    # Test that get_artifact_uri_for_trace works with TraceInfo (v2)
    v2_uri = get_artifact_uri_for_trace(trace_info_v2)
    assert v2_uri == "s3://bucket/trace-v2-path"

    # Test that get_artifact_uri_for_trace works with TraceInfoV3
    v3_uri = get_artifact_uri_for_trace(trace_info_v3)
    assert v3_uri == "s3://bucket/trace-v3-path"

    # Test that get_artifact_uri_for_trace raises the expected exception when tag is missing
    trace_info_no_tag = TraceInfoV2(
        request_id="tr-1234",
        experiment_id="1234",
        timestamp_ms=123,
        execution_time_ms=456,
        status=TraceStatus.OK,
        tags={},
    )
    with pytest.raises(MlflowException, match="Unable to determine trace artifact location"):
        get_artifact_uri_for_trace(trace_info_no_tag)


@pytest.mark.parametrize(
    "delete_traces_kwargs",
    [
        {"experiment_id": "0", "request_ids": ["tr-1234"]},
        {"experiment_id": "0", "max_timestamp_millis": 1, "max_traces": 2},
    ],
)
def test_delete_traces(delete_traces_kwargs):
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    request = DeleteTraces(**delete_traces_kwargs)
    response.text = json.dumps({"traces_deleted": 1})
    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        if "request_ids" in delete_traces_kwargs:
            delete_traces_kwargs["trace_ids"] = delete_traces_kwargs.pop("request_ids")
        res = store.delete_traces(**delete_traces_kwargs)
        _verify_requests(mock_http, creds, "traces/delete-traces", "POST", message_to_json(request))
        assert res == 1


def test_delete_traces_with_batching():
    from mlflow.environment_variables import _MLFLOW_DELETE_TRACES_MAX_BATCH_SIZE

    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200

    # Create 250 trace IDs to test batching (should create 3 batches: 100, 100, 50)
    num_traces = 250
    trace_ids = [f"tr-{i}" for i in range(num_traces)]

    # Each batch returns some number of deleted traces
    response.text = json.dumps({"traces_deleted": 100})

    batch_size = _MLFLOW_DELETE_TRACES_MAX_BATCH_SIZE.get()

    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        store.delete_traces(experiment_id="0", trace_ids=trace_ids)

        # Verify that we made 3 API calls (250 / 100 = 3 batches)
        expected_num_calls = math.ceil(num_traces / batch_size)
        assert mock_http.call_count == expected_num_calls

        # Verify that batch sizes are [100, 100, 50]
        batch_sizes = [len(call[1]["json"]["request_ids"]) for call in mock_http.call_args_list]
        assert batch_sizes == [100, 100, 50]


def test_set_trace_tag():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    trace_id = "tr-1234"
    request = SetTraceTag(
        key="k",
        value="v",
    )
    response.text = "{}"

    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        res = store.set_trace_tag(
            trace_id=trace_id,
            key=request.key,
            value=request.value,
        )
        _verify_requests(
            mock_http,
            creds,
            f"traces/{trace_id}/tags",
            "PATCH",
            message_to_json(request),
            use_v3=False,
        )
        assert res is None


@pytest.mark.parametrize("is_databricks", [True, False])
def test_log_assessment_feedback(is_databricks):
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    response.text = json.dumps(
        {
            "assessment": {
                "assessment_id": "1234",
                "assessment_name": "assessment_name",
                "trace_id": "tr-1234",
                "source": {
                    "source_type": "LLM_JUDGE",
                    "source_id": "gpt-4o-mini",
                },
                "create_time": "2025-02-20T05:47:23Z",
                "last_update_time": "2025-02-20T05:47:23Z",
                "feedback": {"value": True},
                "rationale": "rationale",
                "metadata": {"model": "gpt-4o-mini"},
                "error": None,
                "span_id": None,
            }
        }
    )

    feedback = Feedback(
        trace_id="tr-1234",
        name="assessment_name",
        value=True,
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4o-mini"
        ),
        create_time_ms=int(time.time() * 1000),
        last_update_time_ms=int(time.time() * 1000),
        rationale="rationale",
        metadata={"model": "gpt-4o-mini"},
        span_id=None,
    )

    request = CreateAssessment(assessment=feedback.to_proto())
    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        res = store.create_assessment(feedback)

        _verify_requests(
            mock_http,
            creds,
            "traces/tr-1234/assessments",
            "POST",
            message_to_json(request),
            use_v3=True,
        )
        assert isinstance(res, Feedback)
        assert res.assessment_id is not None
        assert res.value == feedback.value


@pytest.mark.parametrize("is_databricks", [True, False])
def test_log_assessment_expectation(is_databricks):
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    response.text = json.dumps(
        {
            "assessment": {
                "assessment_id": "1234",
                "assessment_name": "assessment_name",
                "trace_id": "tr-1234",
                "source": {
                    "source_type": "HUMAN",
                    "source_id": "me",
                },
                "create_time": "2025-02-20T05:47:23Z",
                "last_update_time": "2025-02-20T05:47:23Z",
                "expectation": {
                    "serialized_value": {
                        "value": '{"key1": "value1", "key2": "value2"}',
                        "serialization_format": "JSON_FORMAT",
                    }
                },
                "error": None,
                "span_id": None,
            }
        }
    )

    expectation = Expectation(
        trace_id="tr-1234",
        name="assessment_name",
        value={"key1": "value1", "key2": "value2"},
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="me"),
        create_time_ms=int(time.time() * 1000),
        last_update_time_ms=int(time.time() * 1000),
        span_id=None,
    )

    request = CreateAssessment(assessment=expectation.to_proto())
    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        res = store.create_assessment(expectation)

        _verify_requests(
            mock_http,
            creds,
            "traces/tr-1234/assessments",
            "POST",
            message_to_json(request),
            use_v3=True,
        )
        assert isinstance(res, Expectation)
        assert res.assessment_id is not None
        assert res.value == expectation.value


@pytest.mark.parametrize("is_databricks", [True, False])
@pytest.mark.parametrize(
    ("updates", "expected_request_json"),
    [
        (
            {"name": "updated_name"},
            {
                "assessment": {
                    "assessment_id": "1234",
                    "trace_id": "tr-1234",
                    "assessment_name": "updated_name",
                },
                "update_mask": "assessmentName",
            },
        ),
        (
            {"expectation": ExpectationValue(value="updated_value")},
            {
                "assessment": {
                    "assessment_id": "1234",
                    "trace_id": "tr-1234",
                    "expectation": {"value": "updated_value"},
                },
                "update_mask": "expectation",
            },
        ),
        (
            {
                "feedback": FeedbackValue(value=0.5),
                "rationale": "update",
                "metadata": {"model": "gpt-4o-mini"},
            },
            {
                "assessment": {
                    "assessment_id": "1234",
                    "trace_id": "tr-1234",
                    "feedback": {"value": 0.5},
                    "rationale": "update",
                    "metadata": {"model": "gpt-4o-mini"},
                },
                "update_mask": "feedback,rationale,metadata",
            },
        ),
    ],
)
def test_update_assessment(updates, expected_request_json, is_databricks):
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    response.text = json.dumps(
        {
            "assessment": {
                "assessment_id": "1234",
                "assessment_name": "assessment_name",
                "trace_id": "tr-1234",
                "source": {
                    "source_type": "LLM_JUDGE",
                    "source_id": "gpt-4o-mini",
                },
                "create_time": "2025-02-20T05:47:23Z",
                "last_update_time": "2025-02-25T01:23:45Z",
                "feedback": {"value": True},
                "rationale": "rationale",
                "metadata": {"model": "gpt-4o-mini"},
                "error": None,
                "span_id": None,
            }
        }
    )

    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        res = store.update_assessment(
            trace_id="tr-1234",
            assessment_id="1234",
            **updates,
        )

        _verify_requests(
            mock_http,
            creds,
            "traces/tr-1234/assessments/1234",
            "PATCH",
            json.dumps(expected_request_json),
            use_v3=True,
        )
        assert isinstance(res, Assessment)


@pytest.mark.parametrize("is_databricks", [True, False])
def test_get_assessment(is_databricks):
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    response.text = json.dumps(
        {
            "assessment": {
                "assessment_id": "1234",
                "assessment_name": "assessment_name",
                "trace_id": "tr-1234",
                "source": {
                    "source_type": "LLM_JUDGE",
                    "source_id": "gpt-4o-mini",
                },
                "create_time": "2025-02-20T05:47:23Z",
                "last_update_time": "2025-02-25T01:23:45Z",
                "feedback": {"value": "test value"},
                "rationale": "rationale",
                "metadata": {"model": "gpt-4o-mini"},
                "error": None,
                "span_id": None,
            }
        }
    )

    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        res = store.get_assessment(trace_id="tr-1234", assessment_id="1234")

    expected_request_json = {"assessment_id": "1234", "trace_id": "tr-1234"}
    _verify_requests(
        mock_http,
        creds,
        "traces/tr-1234/assessments/1234",
        "GET",
        json.dumps(expected_request_json),
        use_v3=True,
    )

    assert isinstance(res, Feedback)
    assert res.assessment_id == "1234"
    assert res.name == "assessment_name"
    assert res.trace_id == "tr-1234"
    assert res.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert res.source.source_id == "gpt-4o-mini"
    assert res.value == "test value"


@pytest.mark.parametrize("is_databricks", [True, False])
def test_delete_assessment(is_databricks):
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    response.text = "{}"

    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        store.delete_assessment(trace_id="tr-1234", assessment_id="1234")

    expected_request_json = {"assessment_id": "1234", "trace_id": "tr-1234"}
    _verify_requests(
        mock_http,
        creds,
        "traces/tr-1234/assessments/1234",
        "DELETE",
        json.dumps(expected_request_json),
        use_v3=True,
    )


def test_update_assessment_invalid_update():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with pytest.raises(MlflowException, match="Exactly one of `expectation` or `feedback`"):
        store.update_assessment(
            trace_id="tr-1234",
            assessment_id="1234",
            expectation=ExpectationValue(value="updated_value"),
            feedback=FeedbackValue(value=0.5),
        )


def test_get_trace_info():
    # Generate a sample trace in v3 format
    with mlflow.start_span(name="test_span") as span:
        span.set_inputs({"input": "value"})
        span.set_outputs({"output": "value"})

    trace = mlflow.get_trace(span.trace_id)
    trace.info.trace_metadata = {"key1": "value1"}
    trace.info.tags = {"tag1": "value1"}
    trace.info.assessments = [
        Feedback(name="feedback", value=0.9, trace_id=span.trace_id),
        Feedback(
            name="feedback_error",
            value=None,
            error=AssessmentError(error_code="500", error_message="error message"),
            trace_id=span.trace_id,
        ),
        Expectation(name="expectation", value=True, trace_id=span.trace_id, span_id=span.span_id),
        Expectation(
            name="complex_expectation",
            value={"complex": [{"key": "value"}]},
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4o-mini"
            ),
            trace_id=span.trace_id,
        ),
    ]
    trace_proto = trace.to_proto()
    mock_response = GetTraceInfoV3.Response(trace=trace_proto)

    store = RestStore(lambda: MlflowHostCreds("https://hello"))

    with mock.patch.object(store, "_call_endpoint", return_value=mock_response):
        result = store.get_trace_info(span.trace_id)

        # Verify we get the expected object back
        assert isinstance(result, TraceInfo)
        assert result.trace_id == span.trace_id
        assert result.experiment_id == "0"
        assert result.trace_metadata == {"key1": "value1"}
        assert result.tags == {"tag1": "value1"}
        assert result.state == TraceState.OK
        assert len(result.assessments) == 4
        assert result.assessments[0].name == "feedback"
        assert result.assessments[1].name == "feedback_error"
        assert result.assessments[2].name == "expectation"
        assert result.assessments[3].name == "complex_expectation"


def test_get_trace():
    # Generate a sample trace with spans
    with mlflow.start_span(name="root_span") as span:
        span.set_inputs({"input": "value"})
        span.set_outputs({"output": "value"})
        with mlflow.start_span(name="child_span") as child:
            child.set_inputs({"child_input": "child_value"})

    trace = mlflow.get_trace(span.trace_id)
    trace_proto = trace.to_proto()
    mock_response = GetTrace.Response(trace=trace_proto)

    store = RestStore(lambda: MlflowHostCreds("https://hello"))

    with mock.patch.object(store, "_call_endpoint", return_value=mock_response) as mock_call:
        result = store.get_trace(span.trace_id, allow_partial=True)

        # Verify we get the expected object back
        assert isinstance(result, Trace)
        assert result.info.trace_id == span.trace_id
        assert len(result.data.spans) == 2

        # Verify the endpoint was called with correct parameters
        mock_call.assert_called_once()
        call_args = mock_call.call_args
        assert call_args[0][0] == GetTrace
        # Check the request body contains the trace_id and allow_partial
        request_body_json = json.loads(call_args[0][1])
        assert request_body_json["trace_id"] == span.trace_id
        assert request_body_json["allow_partial"] is True


def test_get_trace_with_allow_partial_false():
    # Generate a sample trace
    with mlflow.start_span(name="test_span") as span:
        span.set_inputs({"input": "value"})

    trace = mlflow.get_trace(span.trace_id)
    trace_proto = trace.to_proto()
    mock_response = GetTrace.Response(trace=trace_proto)

    store = RestStore(lambda: MlflowHostCreds("https://hello"))

    with mock.patch.object(store, "_call_endpoint", return_value=mock_response) as mock_call:
        result = store.get_trace(span.trace_id, allow_partial=False)

        # Verify we get the expected object back
        assert isinstance(result, Trace)
        assert result.info.trace_id == span.trace_id

        # Verify the endpoint was called with allow_partial=False
        call_args = mock_call.call_args
        request_body_json = json.loads(call_args[0][1])
        assert request_body_json["allow_partial"] is False


def test_get_trace_handles_old_server_routing_conflict():
    store = RestStore(lambda: MlflowHostCreds("https://hello"))

    # Simulate old server returning "Trace with ID 'get' not found"
    error_response = MlflowException(
        "Trace with ID 'get' not found",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )

    with mock.patch.object(store, "_call_endpoint", side_effect=error_response):
        with pytest.raises(MlflowNotImplementedException):  # noqa: PT011
            store.get_trace("some-trace-id")


def test_get_trace_raises_other_errors():
    store = RestStore(lambda: MlflowHostCreds("https://hello"))

    error_message = "Trace with ID 'abc123' not found"
    genuine_error = MlflowException(
        error_message,
        error_code=RESOURCE_DOES_NOT_EXIST,
    )

    with mock.patch.object(store, "_call_endpoint", side_effect=genuine_error):
        with pytest.raises(MlflowException, match=error_message):
            store.get_trace("abc123")


def test_log_logged_model_params():
    with mock.patch("mlflow.store.tracking.rest_store.call_endpoint") as mock_call_endpoint:
        # Create test data
        model_id = "model_123"
        params = [
            LoggedModelParameter(key=f"param_{i}", value=f"value_{i}")
            for i in range(250)  # Create enough params to test batching
        ]

        batches = [
            message_to_json(
                LogLoggedModelParamsRequest(
                    model_id=model_id,
                    params=[p.to_proto() for p in params[:100]],
                )
            ),
            message_to_json(
                LogLoggedModelParamsRequest(
                    model_id=model_id,
                    params=[p.to_proto() for p in params[100:200]],
                )
            ),
            message_to_json(
                LogLoggedModelParamsRequest(
                    model_id=model_id,
                    params=[p.to_proto() for p in params[200:]],
                )
            ),
        ]

        store = RestStore(lambda: None)

        store.log_logged_model_params(model_id=model_id, params=params)

        # Verify call_endpoint was called with the correct arguments
        assert mock_call_endpoint.call_count == 3
        for i, call in enumerate(mock_call_endpoint.call_args_list):
            _, endpoint, method, json_body, response_proto = call.args

            # Verify endpoint and method are correct
            assert endpoint, method == RestStore._METHOD_TO_INFO[LogLoggedModelParamsRequest]
            assert json_body == batches[i]


@pytest.mark.parametrize(
    ("params_count", "expected_call_count", "create_batch_size", "log_batch_size"),
    [
        (None, 1, 100, 100),  # None params - only CreateLoggedModel
        (0, 1, 100, 100),  # No params - only CreateLoggedModel
        (5, 1, 100, 100),  # Few params - only CreateLoggedModel
        (100, 1, 100, 100),  # Exactly 100 params - only CreateLoggedModel
        (
            150,
            3,
            100,
            100,
        ),  # 150 params - CreateLoggedModel + LogLoggedModelParamsRequest + GetLoggedModel
        (
            250,
            4,
            100,
            100,
        ),  # 250 params - CreateLoggedModel + 2 LogLoggedModelParamsRequest calls + GetLoggedModel
        (
            250,
            3,
            200,
            100,
        ),  # 250 params with larger create batch - CreateLoggedModel
        # + 1 LogLoggedModelParamsRequest + GetLoggedModel
        (
            250,
            5,
            100,
            50,
        ),  # 250 params with smaller log batch - CreateLoggedModel
        # + 4 LogLoggedModelParamsRequest calls + GetLoggedModel
    ],
)
def test_create_logged_models_with_params(
    monkeypatch, params_count, expected_call_count, create_batch_size, log_batch_size
):
    # Set environment variables using monkeypatch
    monkeypatch.setenv(_MLFLOW_CREATE_LOGGED_MODEL_PARAMS_BATCH_SIZE.name, str(create_batch_size))
    monkeypatch.setenv(_MLFLOW_LOG_LOGGED_MODEL_PARAMS_BATCH_SIZE.name, str(log_batch_size))

    store = RestStore(lambda: None)
    with (
        mock.patch("mlflow.entities.logged_model.LoggedModel.from_proto") as mock_from_proto,
        mock.patch.object(store, "_call_endpoint") as mock_call_endpoint,
    ):
        # Setup mocks
        mock_model = mock.MagicMock()
        model_id = "model_123"
        mock_model.model_id = model_id
        mock_from_proto.return_value = mock_model
        mock_response = mock.MagicMock()
        mock_response.model = mock.MagicMock()
        mock_call_endpoint.return_value = mock_response

        # Create params
        params = (
            [LoggedModelParameter(key=f"key_{i}", value=f"value_{i}") for i in range(params_count)]
            if params_count
            else None
        )

        # Call the method
        store.create_logged_model("experiment_id", params=params)

        # Verify calls
        endpoint = get_logged_model_endpoint(model_id)

        # CreateLoggedModel should always be called
        initial_params = [p.to_proto() for p in params[:create_batch_size]] if params else None
        mock_call_endpoint.assert_any_call(
            CreateLoggedModel,
            message_to_json(
                CreateLoggedModel(
                    experiment_id="experiment_id",
                    params=initial_params,
                )
            ),
        )

        # If params > create_batch_size, additional calls should be made
        if params_count and params_count > create_batch_size:
            # LogLoggedModelParamsRequest should be called for remaining params
            remaining_params = params[create_batch_size:]
            for i in range(0, len(remaining_params), log_batch_size):
                batch = remaining_params[i : i + log_batch_size]
                mock_call_endpoint.assert_any_call(
                    LogLoggedModelParamsRequest,
                    json_body=message_to_json(
                        LogLoggedModelParamsRequest(
                            model_id=model_id,
                            params=[p.to_proto() for p in batch],
                        )
                    ),
                    endpoint=f"{endpoint}/params",
                )

            # GetLoggedModel should be called to get the updated model
            mock_call_endpoint.assert_any_call(GetLoggedModel, endpoint=endpoint)

        # Verify total number of calls
        assert mock_call_endpoint.call_count == expected_call_count


def test_create_evaluation_dataset():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    with mock.patch.object(store, "_call_endpoint") as mock_call:
        create_response = CreateDataset.Response()
        create_response.dataset.dataset_id = "d-1234567890abcdef1234567890abcdef"
        create_response.dataset.name = "test_dataset"
        create_response.dataset.created_time = 1234567890
        create_response.dataset.last_update_time = 1234567890
        create_response.dataset.digest = "abc123"
        create_response.dataset.tags = json.dumps({"env": "test"})

        mock_call.side_effect = [create_response]

        store.create_dataset(
            name="test_dataset",
            tags={"env": "test"},
            experiment_ids=["0", "1"],
        )

        assert mock_call.call_count == 1

        create_req = CreateDataset(
            name="test_dataset",
            experiment_ids=["0", "1"],
            tags=json.dumps({"env": "test"}),
        )
        mock_call.assert_called_once_with(
            CreateDataset,
            message_to_json(create_req),
            endpoint="/api/3.0/mlflow/datasets/create",
        )


def test_create_dataset_without_experiment_ids():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    with mock.patch.object(store, "_call_endpoint") as mock_call:
        create_response = CreateDataset.Response()
        create_response.dataset.dataset_id = "d-abcdef1234567890abcdef1234567890"
        create_response.dataset.name = "test_dataset_no_exp"
        create_response.dataset.created_time = 1234567890
        create_response.dataset.last_update_time = 1234567890
        create_response.dataset.digest = "xyz789"

        mock_call.side_effect = [create_response]

        store.create_dataset(
            name="test_dataset_no_exp",
            tags={"env": "prod"},
        )

        assert mock_call.call_count == 1

        create_req = CreateDataset(
            name="test_dataset_no_exp",
            tags=json.dumps({"env": "prod"}),
        )
        mock_call.assert_called_once_with(
            CreateDataset,
            message_to_json(create_req),
            endpoint="/api/3.0/mlflow/datasets/create",
        )


def test_create_dataset_with_empty_experiment_ids():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    with mock.patch.object(store, "_call_endpoint") as mock_call:
        create_response = CreateDataset.Response()
        create_response.dataset.dataset_id = "d-fedcba0987654321fedcba0987654321"
        create_response.dataset.name = "test_dataset_empty"
        create_response.dataset.created_time = 1234567890
        create_response.dataset.last_update_time = 1234567890
        create_response.dataset.digest = "empty123"

        mock_call.side_effect = [create_response]

        store.create_dataset(
            name="test_dataset_empty",
            experiment_ids=[],
            tags={"env": "staging"},
        )

        assert mock_call.call_count == 1

        create_req = CreateDataset(
            name="test_dataset_empty",
            experiment_ids=[],
            tags=json.dumps({"env": "staging"}),
        )
        mock_call.assert_called_once_with(
            CreateDataset,
            message_to_json(create_req),
            endpoint="/api/3.0/mlflow/datasets/create",
        )


def test_get_evaluation_dataset():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    dataset_id = "d-1234567890abcdef1234567890abcdef"

    with mock.patch.object(store, "_call_endpoint") as mock_call:
        response = GetDataset.Response()
        response.dataset.dataset_id = dataset_id
        response.dataset.name = "test_dataset"
        response.dataset.digest = "abc123"
        response.dataset.created_time = 1234567890
        response.dataset.last_update_time = 1234567890
        mock_call.return_value = response

        result = store.get_dataset(dataset_id)

        assert result.dataset_id == dataset_id
        assert result.name == "test_dataset"

        mock_call.assert_called_once_with(
            GetDataset,
            None,
            endpoint=f"/api/3.0/mlflow/datasets/{dataset_id}",
        )


def test_delete_evaluation_dataset():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    dataset_id = "d-1234567890abcdef1234567890abcdef"

    with mock.patch.object(store, "_call_endpoint") as mock_call:
        mock_call.return_value = DeleteDataset.Response()

        store.delete_dataset(dataset_id)

        mock_call.assert_called_once_with(
            DeleteDataset,
            None,
            endpoint=f"/api/3.0/mlflow/datasets/{dataset_id}",
        )


def test_search_evaluation_datasets():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.search_datasets(
            experiment_ids=["0", "1"],
            filter_string='name = "dataset1"',
            max_results=10,
            order_by=["name DESC"],
            page_token="token123",
        )
        _verify_requests(
            mock_http,
            creds,
            "datasets/search",
            "POST",
            message_to_json(
                SearchEvaluationDatasets(
                    experiment_ids=["0", "1"],
                    filter_string='name = "dataset1"',
                    max_results=10,
                    order_by=["name DESC"],
                    page_token="token123",
                )
            ),
            use_v3=True,
        )


def test_set_evaluation_dataset_tags():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    dataset_id = "d-1234567890abcdef1234567890abcdef"
    tags = {"env": "production", "version": "2.0", "deprecated": None}

    with mock.patch.object(store, "_call_endpoint") as mock_call:
        mock_call.return_value = mock.Mock()
        store.set_dataset_tags(
            dataset_id=dataset_id,
            tags=tags,
        )

        req = SetDatasetTags(
            tags=json.dumps(tags),
        )
        expected_json = message_to_json(req)

        mock_call.assert_called_once_with(
            SetDatasetTags,
            expected_json,
            endpoint=f"/api/3.0/mlflow/datasets/{dataset_id}/tags",
        )


def test_delete_dataset_tag():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    dataset_id = "d-1234567890abcdef1234567890abcdef"
    key = "deprecated_tag"

    with mock.patch.object(store, "_call_endpoint") as mock_call:
        mock_call.return_value = mock.Mock()
        store.delete_dataset_tag(
            dataset_id=dataset_id,
            key=key,
        )

        mock_call.assert_called_once_with(
            DeleteDatasetTag,
            None,
            endpoint=f"/api/3.0/mlflow/datasets/{dataset_id}/tags/{key}",
        )


def test_dataset_apis_blocked_in_databricks():
    # Test that the decorator blocks dataset APIs when using a Databricks tracking URI
    # Mock the tracking URI to return a Databricks URI
    with mock.patch("mlflow.tracking.get_tracking_uri", return_value="databricks://profile"):
        creds = MlflowHostCreds("https://workspace.cloud.databricks.com")
        store = RestStore(lambda: creds)

        with pytest.raises(
            MlflowException,
            match="Evaluation dataset APIs is not supported in Databricks environments",
        ):
            store.create_dataset(name="test", experiment_id=["0"])

    # Test that APIs work when not using Databricks tracking URI
    with mock.patch("mlflow.tracking.get_tracking_uri", return_value="http://localhost:5000"):
        non_databricks_creds = MlflowHostCreds("http://localhost:5000")
        non_databricks_store = RestStore(lambda: non_databricks_creds)

        mock_response = mock.MagicMock()
        mock_response.dataset.tags = "{}"
        non_databricks_store._call_endpoint = mock.MagicMock(return_value=mock_response)

        # This should not raise an error
        result = non_databricks_store.get_dataset("d-123")
        assert result is not None


def test_upsert_evaluation_dataset_records():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    dataset_id = "d-1234567890abcdef1234567890abcdef"
    records = [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"accuracy": 0.95},
            "source": {
                "source_type": "HUMAN",
                "source_data": {"user": "user123"},
            },
        },
        {
            "inputs": {"question": "How to use MLflow?"},
            "expectations": {"accuracy": 0.9},
            "source": {
                "source_type": "TRACE",
                "source_data": {"trace_id": "trace123"},
            },
        },
    ]

    with mock.patch.object(store, "_call_endpoint") as mock_call:
        response = UpsertDatasetRecords.Response()
        response.inserted_count = 2
        response.updated_count = 0
        mock_call.return_value = response

        result = store.upsert_dataset_records(
            dataset_id=dataset_id,
            records=records,
        )

        assert result == {"inserted": 2, "updated": 0}

        req = UpsertDatasetRecords(
            records=json.dumps(records),
        )
        expected_json = message_to_json(req)

        mock_call.assert_called_once_with(
            UpsertDatasetRecords,
            expected_json,
            endpoint=f"/api/3.0/mlflow/datasets/{dataset_id}/records",
        )


def test_get_evaluation_dataset_experiment_ids():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    dataset_id = "d-1234567890abcdef1234567890abcdef"

    with mock.patch.object(store, "_call_endpoint") as mock_call:
        response = GetDatasetExperimentIds.Response()
        response.experiment_ids.extend(["exp1", "exp2", "exp3"])
        mock_call.return_value = response

        result = store.get_dataset_experiment_ids(dataset_id)

        assert result == ["exp1", "exp2", "exp3"]

        mock_call.assert_called_once_with(
            GetDatasetExperimentIds,
            None,
            endpoint=f"/api/3.0/mlflow/datasets/{dataset_id}/experiment-ids",
        )


def test_evaluation_dataset_error_handling():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    with mock.patch("mlflow.utils.rest_utils.http_request") as mock_http:
        error_response = {
            "error_code": "RESOURCE_DOES_NOT_EXIST",
            "message": "Evaluation dataset not found",
        }
        response = mock.MagicMock()
        response.status_code = 404
        response.text = json.dumps(error_response)
        mock_http.return_value = response

        with pytest.raises(MlflowException, match="Evaluation dataset not found"):
            store.get_dataset("d-nonexistent")


def test_evaluation_dataset_comprehensive_workflow():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    dataset_id = "d-1234567890abcdef1234567890abcdef"

    with mock.patch.object(store, "_call_endpoint") as mock_call:
        create_response = CreateDataset.Response()
        create_response.dataset.dataset_id = dataset_id
        create_response.dataset.name = "test_dataset"
        create_response.dataset.created_time = 1234567890
        create_response.dataset.last_update_time = 1234567890
        create_response.dataset.digest = "abc123"
        create_response.dataset.tags = json.dumps({"env": "test", "version": "1.0"})

        get_response1 = GetDataset.Response()
        get_response1.dataset.CopyFrom(create_response.dataset)
        get_response1.dataset.tags = json.dumps({"env": "staging", "version": "1.1", "team": "ml"})

        upsert_response1 = UpsertDatasetRecords.Response()
        upsert_response1.inserted_count = 2
        upsert_response1.updated_count = 0

        get_response2 = GetDataset.Response()
        get_response2.dataset.CopyFrom(get_response1.dataset)
        get_response2.dataset.tags = json.dumps({"env": "production", "version": "2.0"})

        upsert_response2 = UpsertDatasetRecords.Response()
        upsert_response2.inserted_count = 1
        upsert_response2.updated_count = 2

        mock_call.side_effect = [
            create_response,  # Create with tags
            None,  # First tag update
            get_response1,  # Get after first tag update
            upsert_response1,  # First record upsert
            None,  # Second tag update (remove team tag)
            get_response2,  # Get after second tag update
            upsert_response2,  # Second record upsert
        ]

        dataset = store.create_dataset(
            name="test_dataset",
            tags={"env": "test", "version": "1.0"},
            experiment_ids=["exp1"],
        )
        assert dataset.tags == {"env": "test", "version": "1.0"}

        store.set_dataset_tags(
            dataset_id=dataset_id,
            tags={"env": "staging", "version": "1.1", "team": "ml"},
        )
        updated_dataset = store.get_dataset(dataset_id)
        assert updated_dataset.tags == {"env": "staging", "version": "1.1", "team": "ml"}

        records1 = [
            {"inputs": {"q": "What is MLflow?"}, "expectations": {"score": 0.9}},
            {"inputs": {"q": "How to track?"}, "expectations": {"score": 0.8}},
        ]
        result1 = store.upsert_dataset_records(dataset_id, records1)
        assert result1 == {"inserted": 2, "updated": 0}

        store.set_dataset_tags(
            dataset_id=dataset_id,
            tags={"env": "production", "version": "2.0", "team": None},
        )
        final_dataset = store.get_dataset(dataset_id)
        assert final_dataset.tags == {"env": "production", "version": "2.0"}

        records2 = [
            {"inputs": {"q": "What is tracking?"}, "expectations": {"score": 0.95}},  # New
            {"inputs": {"q": "What is MLflow?"}, "expectations": {"score": 0.95}},  # Update
            {"inputs": {"q": "How to track?"}, "expectations": {"score": 0.85}},  # Update
        ]
        result2 = store.upsert_dataset_records(dataset_id, records2)
        assert result2 == {"inserted": 1, "updated": 2}

        assert mock_call.call_count == 7


def test_evaluation_dataset_merge_records():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    dataset_id = "d-1234567890abcdef1234567890abcdef"

    eval_dataset = EvaluationDataset(
        dataset_id=dataset_id,
        name="test_dataset",
        digest="abc123",
        created_time=1234567890,
        last_update_time=1234567890,
    )

    with mock.patch("mlflow.tracking._tracking_service.utils._get_store") as mock_get_store:
        mock_get_store.return_value = store

        with mock.patch.object(store, "get_dataset") as mock_get:
            mock_get.return_value = eval_dataset

            with mock.patch.object(store, "_call_endpoint") as mock_call:
                upsert_response = UpsertDatasetRecords.Response()
                upsert_response.inserted_count = 2
                upsert_response.updated_count = 0
                mock_call.return_value = upsert_response

                records = [
                    {
                        "inputs": {"question": "What is MLflow?", "temperature": 0.7},
                        "expectations": {"accuracy": 0.95},
                    },
                    {
                        "inputs": {"question": "How to track?", "model": "gpt-4"},
                        "expectations": {"clarity": 0.85},
                    },
                ]

                result = eval_dataset.merge_records(records)

                assert result is eval_dataset

                assert mock_get.call_count == 1
                assert mock_call.call_count == 1

                call_args = mock_call.call_args_list[0]
                assert call_args[0][0] == UpsertDatasetRecords

                # Check the endpoint path contains the dataset_id
                endpoint = call_args[1].get("endpoint")
                assert endpoint == f"/api/3.0/mlflow/datasets/{dataset_id}/records"

                # Check the request body contains the records
                upsert_req_json = call_args[0][1]
                upsert_req_dict = json.loads(upsert_req_json)

                sent_records = json.loads(upsert_req_dict["records"])
                assert len(sent_records) == 2


def test_evaluation_dataset_get_records():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    dataset_id = "d-1234567890abcdef1234567890abcdef"

    with mock.patch.object(store, "_call_endpoint") as mock_call:
        response = GetDatasetRecords.Response()
        records = [
            {
                "dataset_id": dataset_id,
                "dataset_record_id": "r-001",
                "inputs": {"question": "What is MLflow?"},
                "expectations": {"accuracy": 0.95},
                "tags": {"source": "test"},
                "source_type": "HUMAN",
                "source_id": "user123",
                "created_time": 1234567890,
                "last_update_time": 1234567890,
            },
            {
                "dataset_id": dataset_id,
                "dataset_record_id": "r-002",
                "inputs": {"question": "How to track?"},
                "expectations": {"clarity": 0.85},
                "tags": {},
                "source_type": "TRACE",
                "source_id": "trace456",
                "created_time": 1234567891,
                "last_update_time": 1234567891,
            },
        ]
        response.records = json.dumps(records)
        response.next_page_token = ""
        mock_call.return_value = response

        records, next_page_token = store._load_dataset_records(dataset_id)

        assert len(records) == 2
        assert records[0].dataset_record_id == "r-001"
        assert records[0].inputs == {"question": "What is MLflow?"}
        assert records[1].dataset_record_id == "r-002"
        assert next_page_token is None

        req = GetDatasetRecords(
            max_results=1000,
        )
        expected_json = message_to_json(req)

        mock_call.assert_called_once_with(
            GetDatasetRecords,
            expected_json,
            endpoint=f"/api/3.0/mlflow/datasets/{dataset_id}/records",
        )


def test_evaluation_dataset_lazy_loading_records():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    dataset_id = "d-1234567890abcdef1234567890abcdef"

    eval_dataset = EvaluationDataset(
        dataset_id=dataset_id,
        name="test_dataset",
        digest="abc123",
        created_time=1234567890,
        last_update_time=1234567890,
    )

    with mock.patch("mlflow.tracking._tracking_service.utils._get_store") as mock_get_store:
        mock_get_store.return_value = store

        with mock.patch.object(store, "_load_dataset_records") as mock_load:
            from mlflow.entities.dataset_record import DatasetRecord

            mock_records = [
                DatasetRecord(
                    dataset_id=dataset_id,
                    dataset_record_id="r-001",
                    inputs={"q": "test"},
                    expectations={"score": 0.9},
                    tags={},
                    created_time=1234567890,
                    last_update_time=1234567890,
                )
            ]
            mock_load.return_value = (mock_records, None)

            records = eval_dataset.records

            assert len(records) == 1
            assert records[0].dataset_record_id == "r-001"

            mock_load.assert_called_once_with(dataset_id, max_results=None)


def test_evaluation_dataset_pagination():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.search_datasets(max_results=10)
        _verify_requests(
            mock_http,
            creds,
            "datasets/search",
            "POST",
            message_to_json(
                SearchEvaluationDatasets(
                    experiment_ids=[],
                    filter_string=None,
                    max_results=10,
                    order_by=[],
                    page_token=None,
                )
            ),
            use_v3=True,
        )

    with mock_http_request() as mock_http:
        store.search_datasets(max_results=10, page_token="page2")
        _verify_requests(
            mock_http,
            creds,
            "datasets/search",
            "POST",
            message_to_json(
                SearchEvaluationDatasets(
                    experiment_ids=[],
                    filter_string=None,
                    max_results=10,
                    order_by=[],
                    page_token="page2",
                )
            ),
            use_v3=True,
        )


def test_load_dataset_records_pagination():
    store = RestStore(lambda: None)
    dataset_id = "d-1234567890abcdef1234567890abcdef"

    with mock.patch.object(store, "_call_endpoint") as mock_call_endpoint:
        mock_response = mock.MagicMock()

        mock_record1 = mock.MagicMock()
        mock_record1.dataset_record_id = "r-001"
        mock_record1.inputs = json.dumps({"q": "Question 1"})
        mock_record1.expectations = json.dumps({"a": "Answer 1"})
        mock_record1.tags = "{}"
        mock_record1.source_type = "TRACE"
        mock_record1.source_id = "trace-1"
        mock_record1.created_time = 1609459200

        mock_record2 = mock.MagicMock()
        mock_record2.dataset_record_id = "r-002"
        mock_record2.inputs = json.dumps({"q": "Question 2"})
        mock_record2.expectations = json.dumps({"a": "Answer 2"})
        mock_record2.tags = "{}"
        mock_record2.source_type = "TRACE"
        mock_record2.source_id = "trace-2"
        mock_record2.created_time = 1609459201

        mock_response.records = json.dumps(
            [
                {
                    "dataset_id": dataset_id,
                    "dataset_record_id": "r-001",
                    "inputs": {"q": "Question 1"},
                    "expectations": {"a": "Answer 1"},
                    "tags": {},
                    "source_type": "TRACE",
                    "source_id": "trace-1",
                    "created_time": 1609459200,
                    "last_update_time": 1609459200,
                },
                {
                    "dataset_id": dataset_id,
                    "dataset_record_id": "r-002",
                    "inputs": {"q": "Question 2"},
                    "expectations": {"a": "Answer 2"},
                    "tags": {},
                    "source_type": "TRACE",
                    "source_id": "trace-2",
                    "created_time": 1609459201,
                    "last_update_time": 1609459201,
                },
            ]
        )
        mock_response.next_page_token = "token_page2"
        mock_call_endpoint.return_value = mock_response

        records, next_token = store._load_dataset_records(
            dataset_id, max_results=2, page_token=None
        )

        assert len(records) == 2
        assert records[0].dataset_record_id == "r-001"
        assert records[1].dataset_record_id == "r-002"
        assert next_token == "token_page2"

        mock_call_endpoint.assert_called_once_with(
            GetDatasetRecords,
            message_to_json(GetDatasetRecords(max_results=2)),
            endpoint=f"/api/3.0/mlflow/datasets/{dataset_id}/records",
        )

        mock_call_endpoint.reset_mock()
        mock_response.records = json.dumps(
            [
                {
                    "dataset_id": dataset_id,
                    "dataset_record_id": "r-003",
                    "inputs": {"q": "Question 3"},
                    "expectations": {"a": "Answer 3"},
                    "tags": {},
                    "source_type": "TRACE",
                    "source_id": "trace-3",
                    "created_time": 1609459202,
                    "last_update_time": 1609459202,
                }
            ]
        )
        mock_response.next_page_token = ""

        records, next_token = store._load_dataset_records(
            dataset_id, max_results=2, page_token="token_page2"
        )

        assert len(records) == 1
        assert records[0].dataset_record_id == "r-003"
        assert next_token is None

        req_with_token = GetDatasetRecords(max_results=2)
        req_with_token.page_token = "token_page2"
        mock_call_endpoint.assert_called_once_with(
            GetDatasetRecords,
            message_to_json(req_with_token),
            endpoint=f"/api/3.0/mlflow/datasets/{dataset_id}/records",
        )


def test_evaluation_dataset_created_by_and_updated_by():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    with mock.patch.object(store, "_call_endpoint") as mock_call:
        created_response = CreateDataset.Response()
        created_response.dataset.dataset_id = "d-test123"
        created_response.dataset.name = "test_dataset"
        created_response.dataset.created_time = 1234567890000
        created_response.dataset.last_update_time = 1234567890000
        created_response.dataset.created_by = "user1"
        created_response.dataset.last_updated_by = "user1"
        created_response.dataset.digest = "abc123"
        created_response.dataset.tags = json.dumps({"mlflow.user": "user1", "environment": "test"})

        mock_call.return_value = created_response

        dataset = store.create_dataset(
            name="test_dataset",
            experiment_ids=["exp1"],
            tags={"mlflow.user": "user1", "environment": "test"},
        )

        assert dataset.created_by == "user1"
        assert dataset.last_updated_by == "user1"
        assert dataset.tags["mlflow.user"] == "user1"

        upsert_response = UpsertDatasetRecords.Response()
        upsert_response.inserted_count = 1
        upsert_response.updated_count = 0

        get_response = GetDataset.Response()
        get_response.dataset.dataset_id = "d-test123"
        get_response.dataset.name = "test_dataset"
        get_response.dataset.created_time = 1234567890000
        get_response.dataset.last_update_time = 1234567900000
        get_response.dataset.created_by = "user1"
        get_response.dataset.last_updated_by = "user2"
        get_response.dataset.digest = "def456"
        get_response.dataset.tags = json.dumps({"mlflow.user": "user1", "environment": "test"})

        mock_call.side_effect = [upsert_response, get_response]

        records = [
            {
                "inputs": {"question": "Test?"},
                "expectations": {"score": 0.9},
                "tags": {"mlflow.user": "user2"},
            }
        ]
        result = store.upsert_dataset_records("d-test123", records)

        assert result["inserted"] == 1
        assert result["updated"] == 0

        updated_dataset = store.get_dataset("d-test123")
        assert updated_dataset.created_by == "user1"
        assert updated_dataset.last_updated_by == "user2"

        created_response_no_user = CreateDataset.Response()
        created_response_no_user.dataset.dataset_id = "d-test456"
        created_response_no_user.dataset.name = "test_dataset_no_user"
        created_response_no_user.dataset.created_time = 1234567890000
        created_response_no_user.dataset.last_update_time = 1234567890000
        created_response_no_user.dataset.digest = "ghi789"
        created_response_no_user.dataset.tags = json.dumps({"environment": "production"})

        mock_call.side_effect = None
        mock_call.return_value = created_response_no_user

        dataset_no_user = store.create_dataset(
            name="test_dataset_no_user",
            experiment_ids=["exp2"],
            tags={"environment": "production"},
        )

        assert dataset_no_user.created_by is None
        assert dataset_no_user.last_updated_by is None

        set_tags_response = SetDatasetTags.Response()
        set_tags_response.dataset.dataset_id = "d-test123"
        set_tags_response.dataset.name = "test_dataset"
        set_tags_response.dataset.created_time = 1234567890000
        set_tags_response.dataset.last_update_time = 1234567890000
        set_tags_response.dataset.created_by = "user1"
        set_tags_response.dataset.last_updated_by = "user1"
        set_tags_response.dataset.digest = "abc123"
        set_tags_response.dataset.tags = json.dumps(
            {"mlflow.user": "user3", "environment": "staging", "version": "2.0"}
        )

        mock_call.return_value = set_tags_response

        store.set_dataset_tags(
            "d-test123",
            {"mlflow.user": "user3", "version": "2.0", "environment": "staging"},
        )

        call_args = mock_call.call_args_list[-1]
        api, json_body = call_args[0]
        assert api == SetDatasetTags
        request_dict = json.loads(json_body)
        tags_dict = json.loads(request_dict["tags"])
        assert "mlflow.user" in tags_dict
        assert tags_dict["mlflow.user"] == "user3"


def test_evaluation_dataset_user_tracking_search():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    with mock.patch.object(store, "_call_endpoint") as mock_call:
        search_response = SearchEvaluationDatasets.Response()

        dataset1 = search_response.datasets.add()
        dataset1.dataset_id = "d-dataset1"
        dataset1.name = "dataset1"
        dataset1.created_time = 1234567890000
        dataset1.last_update_time = 1234567900000
        dataset1.created_by = "user1"
        dataset1.last_updated_by = "user2"
        dataset1.digest = "search1"

        dataset2 = search_response.datasets.add()
        dataset2.dataset_id = "d-dataset2"
        dataset2.name = "dataset2"
        dataset2.created_time = 1234567891000
        dataset2.last_update_time = 1234567891000
        dataset2.created_by = "user2"
        dataset2.last_updated_by = "user2"
        dataset2.digest = "search2"

        mock_call.return_value = search_response

        results = store.search_datasets(filter_string="created_by = 'user1'")

        assert len(results) == 2
        assert results[0].created_by == "user1"
        assert results[0].last_updated_by == "user2"

        call_args = mock_call.call_args_list[-1]
        api, json_body = call_args[0]
        assert api == SearchEvaluationDatasets
        request_json = json.loads(json_body)
        assert request_json["filter_string"] == "created_by = 'user1'"

        results = store.search_datasets(filter_string="last_updated_by = 'user2'")

        call_args = mock_call.call_args_list[-1]
        api, json_body = call_args[0]
        request_json = json.loads(json_body)
        assert request_json["filter_string"] == "last_updated_by = 'user2'"


def test_add_dataset_to_experiments():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    with mock.patch.object(store, "_call_endpoint") as mock_call:
        response = AddDatasetToExperiments.Response()
        response.dataset.dataset_id = "d-1234567890abcdef1234567890abcdef"
        response.dataset.name = "test_dataset"
        response.dataset.experiment_ids.extend(["0", "1", "3", "4"])
        response.dataset.created_time = 1234567890
        response.dataset.last_update_time = 1234567890
        response.dataset.digest = "abc123"

        mock_call.side_effect = [response]

        result = store.add_dataset_to_experiments(
            dataset_id="d-1234567890abcdef1234567890abcdef",
            experiment_ids=["3", "4"],
        )

        assert mock_call.call_count == 1
        assert result.dataset_id == "d-1234567890abcdef1234567890abcdef"
        assert "3" in result.experiment_ids
        assert "4" in result.experiment_ids
        assert "0" in result.experiment_ids
        assert "1" in result.experiment_ids

        req = AddDatasetToExperiments(
            dataset_id="d-1234567890abcdef1234567890abcdef",
            experiment_ids=["3", "4"],
        )
        mock_call.assert_called_once_with(
            AddDatasetToExperiments,
            message_to_json(req),
            endpoint="/api/3.0/mlflow/datasets/d-1234567890abcdef1234567890abcdef/add-experiments",
        )


def test_remove_dataset_from_experiments():
    creds = MlflowHostCreds("https://test-server")
    store = RestStore(lambda: creds)

    with mock.patch.object(store, "_call_endpoint") as mock_call:
        response = RemoveDatasetFromExperiments.Response()
        response.dataset.dataset_id = "d-1234567890abcdef1234567890abcdef"
        response.dataset.name = "test_dataset"
        response.dataset.experiment_ids.extend(["0", "1"])
        response.dataset.created_time = 1234567890
        response.dataset.last_update_time = 1234567890
        response.dataset.digest = "abc123"

        mock_call.side_effect = [response]

        result = store.remove_dataset_from_experiments(
            dataset_id="d-1234567890abcdef1234567890abcdef",
            experiment_ids=["3"],
        )

        assert mock_call.call_count == 1
        assert result.dataset_id == "d-1234567890abcdef1234567890abcdef"
        assert "3" not in result.experiment_ids
        assert "0" in result.experiment_ids
        assert "1" in result.experiment_ids

        req = RemoveDatasetFromExperiments(
            dataset_id="d-1234567890abcdef1234567890abcdef",
            experiment_ids=["3"],
        )
        mock_call.assert_called_once_with(
            RemoveDatasetFromExperiments,
            message_to_json(req),
            endpoint="/api/3.0/mlflow/datasets/d-1234567890abcdef1234567890abcdef/remove-experiments",
        )


def test_register_scorer():
    store = RestStore(lambda: None)

    with mock.patch.object(store, "_call_endpoint") as mock_call_endpoint:
        experiment_id = "123"
        name = "accuracy_scorer"
        serialized_scorer = "serialized_scorer_data"

        mock_response = mock.MagicMock()
        mock_response.version = 1
        mock_response.scorer_id = "test-scorer-id"
        mock_response.experiment_id = experiment_id
        mock_response.name = name
        mock_response.serialized_scorer = serialized_scorer
        mock_response.creation_time = 1234567890
        mock_call_endpoint.return_value = mock_response

        scorer_version = store.register_scorer(experiment_id, name, serialized_scorer)

        assert scorer_version.scorer_version == 1
        assert scorer_version.scorer_id == "test-scorer-id"
        assert scorer_version.experiment_id == experiment_id
        assert scorer_version.scorer_name == name
        assert scorer_version._serialized_scorer == serialized_scorer
        assert scorer_version.creation_time == 1234567890

        mock_call_endpoint.assert_called_once_with(
            RegisterScorer,
            message_to_json(
                RegisterScorer(
                    experiment_id=experiment_id, name=name, serialized_scorer=serialized_scorer
                )
            ),
            endpoint="/api/3.0/mlflow/scorers/register",
        )


def test_list_scorers():
    store = RestStore(lambda: None)

    with mock.patch.object(store, "_call_endpoint") as mock_call_endpoint:
        experiment_id = "123"

        # Mock response
        mock_scorer1 = mock.MagicMock()
        mock_scorer1.experiment_id = 123
        mock_scorer1.scorer_name = "accuracy_scorer"
        mock_scorer1.scorer_version = 1
        mock_scorer1.serialized_scorer = "serialized_accuracy_scorer"

        mock_scorer2 = mock.MagicMock()
        mock_scorer2.experiment_id = 123
        mock_scorer2.scorer_name = "safety_scorer"
        mock_scorer2.scorer_version = 2
        mock_scorer2.serialized_scorer = "serialized_safety_scorer"

        mock_response = mock.MagicMock()
        mock_response.scorers = [mock_scorer1, mock_scorer2]
        mock_call_endpoint.return_value = mock_response

        # Call the method
        scorers = store.list_scorers(experiment_id)

        # Verify result
        assert len(scorers) == 2
        assert scorers[0].scorer_name == "accuracy_scorer"
        assert scorers[0].scorer_version == 1
        assert scorers[0]._serialized_scorer == "serialized_accuracy_scorer"
        assert scorers[1].scorer_name == "safety_scorer"
        assert scorers[1].scorer_version == 2
        assert scorers[1]._serialized_scorer == "serialized_safety_scorer"

        # Verify API call
        mock_call_endpoint.assert_called_once_with(
            ListScorers,
            message_to_json(ListScorers(experiment_id=experiment_id)),
            endpoint="/api/3.0/mlflow/scorers/list",
        )


def test_list_scorer_versions():
    store = RestStore(lambda: None)

    with mock.patch.object(store, "_call_endpoint") as mock_call_endpoint:
        experiment_id = "123"
        name = "accuracy_scorer"

        # Mock response
        mock_scorer1 = mock.MagicMock()
        mock_scorer1.experiment_id = 123
        mock_scorer1.scorer_name = "accuracy_scorer"
        mock_scorer1.scorer_version = 1
        mock_scorer1.serialized_scorer = "serialized_accuracy_scorer_v1"

        mock_scorer2 = mock.MagicMock()
        mock_scorer2.experiment_id = 123
        mock_scorer2.scorer_name = "accuracy_scorer"
        mock_scorer2.scorer_version = 2
        mock_scorer2.serialized_scorer = "serialized_accuracy_scorer_v2"

        mock_response = mock.MagicMock()
        mock_response.scorers = [mock_scorer1, mock_scorer2]
        mock_call_endpoint.return_value = mock_response

        # Call the method
        scorers = store.list_scorer_versions(experiment_id, name)

        # Verify result
        assert len(scorers) == 2
        assert scorers[0].scorer_version == 1
        assert scorers[0]._serialized_scorer == "serialized_accuracy_scorer_v1"
        assert scorers[1].scorer_version == 2
        assert scorers[1]._serialized_scorer == "serialized_accuracy_scorer_v2"

        # Verify API call
        mock_call_endpoint.assert_called_once_with(
            ListScorerVersions,
            message_to_json(ListScorerVersions(experiment_id=experiment_id, name=name)),
            endpoint="/api/3.0/mlflow/scorers/versions",
        )


def test_get_scorer_with_version():
    store = RestStore(lambda: None)

    with mock.patch.object(store, "_call_endpoint") as mock_call_endpoint:
        experiment_id = "123"
        name = "accuracy_scorer"
        version = 2

        # Mock response
        mock_response = mock.MagicMock()
        mock_scorer = mock.MagicMock()
        mock_scorer.experiment_id = 123
        mock_scorer.scorer_name = "accuracy_scorer"
        mock_scorer.scorer_version = 2
        mock_scorer.serialized_scorer = "serialized_accuracy_scorer_v2"
        mock_scorer.creation_time = 1640995200000
        mock_response.scorer = mock_scorer
        mock_call_endpoint.return_value = mock_response

        # Call the method
        result = store.get_scorer(experiment_id, name, version=version)

        # Verify result
        assert result._serialized_scorer == "serialized_accuracy_scorer_v2"
        assert result.scorer_version == 2
        assert result.scorer_name == "accuracy_scorer"

        # Verify API call
        mock_call_endpoint.assert_called_once_with(
            GetScorer,
            message_to_json(GetScorer(experiment_id=experiment_id, name=name, version=version)),
            endpoint="/api/3.0/mlflow/scorers/get",
        )


def test_get_scorer_without_version():
    store = RestStore(lambda: None)

    with mock.patch.object(store, "_call_endpoint") as mock_call_endpoint:
        experiment_id = "123"
        name = "accuracy_scorer"

        # Mock response
        mock_response = mock.MagicMock()
        mock_scorer = mock.MagicMock()
        mock_scorer.experiment_id = 123
        mock_scorer.scorer_name = "accuracy_scorer"
        mock_scorer.scorer_version = 3
        mock_scorer.serialized_scorer = "serialized_accuracy_scorer_latest"
        mock_scorer.creation_time = 1640995200000
        mock_response.scorer = mock_scorer
        mock_call_endpoint.return_value = mock_response

        # Call the method
        result = store.get_scorer(experiment_id, name)

        # Verify result
        assert result._serialized_scorer == "serialized_accuracy_scorer_latest"
        assert result.scorer_version == 3
        assert result.scorer_name == "accuracy_scorer"

        # Verify API call
        mock_call_endpoint.assert_called_once_with(
            GetScorer,
            message_to_json(GetScorer(experiment_id=experiment_id, name=name)),
            endpoint="/api/3.0/mlflow/scorers/get",
        )


def test_delete_scorer_with_version():
    store = RestStore(lambda: None)

    with mock.patch.object(store, "_call_endpoint") as mock_call_endpoint:
        experiment_id = "123"
        name = "accuracy_scorer"
        version = 2

        # Mock response (empty response for delete operations)
        mock_response = mock.MagicMock()
        mock_call_endpoint.return_value = mock_response

        # Call the method
        store.delete_scorer(experiment_id, name, version=version)

        # Verify API call
        mock_call_endpoint.assert_called_once_with(
            DeleteScorer,
            message_to_json(DeleteScorer(experiment_id=experiment_id, name=name, version=version)),
            endpoint="/api/3.0/mlflow/scorers/delete",
        )


def test_delete_scorer_without_version():
    store = RestStore(lambda: None)

    with mock.patch.object(store, "_call_endpoint") as mock_call_endpoint:
        experiment_id = "123"
        name = "accuracy_scorer"

        # Mock response (empty response for delete operations)
        mock_response = mock.MagicMock()
        mock_call_endpoint.return_value = mock_response

        # Call the method
        store.delete_scorer(experiment_id, name)

        # Verify API call
        mock_call_endpoint.assert_called_once_with(
            DeleteScorer,
            message_to_json(DeleteScorer(experiment_id=experiment_id, name=name)),
            endpoint="/api/3.0/mlflow/scorers/delete",
        )


def test_calculate_trace_filter_correlation():
    store = RestStore(lambda: None)

    with mock.patch.object(store, "_call_endpoint") as mock_call_endpoint:
        experiment_ids = ["123", "456"]
        filter_string1 = "span.type = 'LLM'"
        filter_string2 = "feedback.quality > 0.8"
        base_filter = "request_time > 1000"

        mock_response = mock.MagicMock()
        mock_response.npmi = 0.456
        mock_response.npmi_smoothed = 0.445
        mock_response.filter1_count = 100
        mock_response.filter2_count = 80
        mock_response.joint_count = 50
        mock_response.total_count = 200
        mock_response.HasField = lambda field: field in ["npmi", "npmi_smoothed"]
        mock_call_endpoint.return_value = mock_response

        result = store.calculate_trace_filter_correlation(
            experiment_ids=experiment_ids,
            filter_string1=filter_string1,
            filter_string2=filter_string2,
            base_filter=base_filter,
        )

        assert isinstance(result, TraceFilterCorrelationResult)
        assert result.npmi == 0.456
        assert result.npmi_smoothed == 0.445
        assert result.filter1_count == 100
        assert result.filter2_count == 80
        assert result.joint_count == 50
        assert result.total_count == 200

        expected_request = CalculateTraceFilterCorrelation(
            experiment_ids=experiment_ids,
            filter_string1=filter_string1,
            filter_string2=filter_string2,
            base_filter=base_filter,
        )
        mock_call_endpoint.assert_called_once_with(
            CalculateTraceFilterCorrelation,
            message_to_json(expected_request),
            "/api/3.0/mlflow/traces/calculate-filter-correlation",
        )


def test_calculate_trace_filter_correlation_without_base_filter():
    store = RestStore(lambda: None)

    with mock.patch.object(store, "_call_endpoint") as mock_call_endpoint:
        experiment_ids = ["123"]
        filter_string1 = "span.type = 'LLM'"
        filter_string2 = "feedback.quality > 0.8"

        mock_response = mock.MagicMock()
        mock_response.filter1_count = 0
        mock_response.filter2_count = 0
        mock_response.joint_count = 0
        mock_response.total_count = 100
        mock_response.HasField = lambda field: False
        mock_call_endpoint.return_value = mock_response

        result = store.calculate_trace_filter_correlation(
            experiment_ids=experiment_ids,
            filter_string1=filter_string1,
            filter_string2=filter_string2,
        )

        assert isinstance(result, TraceFilterCorrelationResult)
        assert math.isnan(result.npmi)
        assert result.npmi_smoothed is None
        assert result.filter1_count == 0
        assert result.filter2_count == 0
        assert result.joint_count == 0
        assert result.total_count == 100

        expected_request = CalculateTraceFilterCorrelation(
            experiment_ids=experiment_ids,
            filter_string1=filter_string1,
            filter_string2=filter_string2,
        )
        mock_call_endpoint.assert_called_once_with(
            CalculateTraceFilterCorrelation,
            message_to_json(expected_request),
            "/api/3.0/mlflow/traces/calculate-filter-correlation",
        )


def _create_mock_response(status_code: int = 200, text: str = "{}") -> mock.MagicMock:
    """Helper to create a mock HTTP response."""
    response = mock.MagicMock()
    response.status_code = status_code
    response.text = text
    return response


def _create_test_spans() -> list[LiveSpan]:
    """Helper to create test spans for log_spans tests."""
    otel_span = create_mock_otel_span(
        trace_id=123,
        span_id=1,
        name="test_span",
        start_time=1000000,
        end_time=2000000,
    )
    return [LiveSpan(otel_span, trace_id="tr-123")]


def test_log_spans_with_version_check():
    spans = _create_test_spans()
    experiment_id = "exp-123"

    # Test 1: Server version is None (failed to retrieve)
    # Use unique host to avoid cache conflicts
    creds1 = MlflowHostCreds("https://host1")
    store1 = RestStore(lambda: creds1)
    with mock.patch(
        "mlflow.store.tracking.rest_store.http_request", side_effect=Exception("Connection error")
    ):
        with pytest.raises(NotImplementedError, match="could not identify MLflow server version"):
            store1.log_spans(experiment_id, spans)

    # Test 2: Server version is less than 3.4
    creds2 = MlflowHostCreds("https://host2")
    store2 = RestStore(lambda: creds2)
    with mock.patch(
        "mlflow.store.tracking.rest_store.http_request",
        return_value=_create_mock_response(text="3.3.0"),
    ):
        with pytest.raises(
            NotImplementedError, match="MLflow server version 3.3.0 is less than 3.4"
        ):
            store2.log_spans(experiment_id, spans)

    # Test 3: Server version is exactly 3.4.0 - should succeed
    creds3 = MlflowHostCreds("https://host3")
    store3 = RestStore(lambda: creds3)
    with mock.patch(
        "mlflow.store.tracking.rest_store.http_request",
        side_effect=[
            # First call is to /version, second is to OTLP endpoint
            _create_mock_response(text="3.4.0"),  # version response
            _create_mock_response(),  # OTLP response
        ],
    ):
        result = store3.log_spans(experiment_id, spans)
        assert result == spans

    # Test 4: Server version is greater than 3.4 - should succeed
    creds4 = MlflowHostCreds("https://host4")
    store4 = RestStore(lambda: creds4)
    with mock.patch(
        "mlflow.store.tracking.rest_store.http_request",
        side_effect=[
            # First call is to /version, second is to OTLP endpoint
            _create_mock_response(text="3.5.0"),  # version response
            _create_mock_response(),  # OTLP response
        ],
    ):
        result = store4.log_spans(experiment_id, spans)
        assert result == spans

    # Test 5: Real timeout test - verify that timeout works properly without mocking
    # Using a non-existent host that will trigger timeout
    creds5 = MlflowHostCreds("https://host5")
    store5 = RestStore(lambda: creds5)
    start_time = time.time()
    with pytest.raises(NotImplementedError, match="could not identify MLflow server version"):
        store5.log_spans(experiment_id, spans)
    elapsed_time = time.time() - start_time
    # Should timeout within 3 seconds (plus some buffer for processing)
    assert elapsed_time < 5, f"Version check took {elapsed_time}s, should timeout within 3s"


def test_server_version_check_caching():
    spans = _create_test_spans()
    experiment_id = "exp-123"

    # Use the same host credentials for all stores to test caching
    creds = MlflowHostCreds("https://cached-host")
    store1 = RestStore(lambda: creds)
    store2 = RestStore(lambda: creds)  # Different store instance, same creds

    # First call - should fetch version and then call OTLP
    with mock.patch(
        "mlflow.store.tracking.rest_store.http_request",
        side_effect=[
            _create_mock_response(text="3.5.0"),  # version response
            _create_mock_response(),  # OTLP response
        ],
    ) as mock_http:
        # We call log_spans because it performs a server version check via _get_server_version
        result1 = store1.log_spans(experiment_id, spans)
        assert result1 == spans

        # Should have called /version first, then /v1/traces
        mock_http.assert_any_call(
            host_creds=creds,
            endpoint="/version",
            method="GET",
            timeout=3,
            max_retries=0,
            retry_timeout_seconds=1,
            raise_on_status=True,
        )
        mock_http.assert_any_call(
            host_creds=creds,
            endpoint="/v1/traces",
            method="POST",
            data=mock.ANY,
            extra_headers=mock.ANY,
        )
        assert mock_http.call_count == 2

    # Second call with same store - should use cached version, only call OTLP
    with mock.patch(
        "mlflow.store.tracking.rest_store.http_request", return_value=_create_mock_response()
    ) as mock_http:
        result2 = store1.log_spans(experiment_id, spans)
        assert result2 == spans

        # Should only call OTLP, not version (cached)
        mock_http.assert_called_once_with(
            host_creds=creds,
            endpoint="/v1/traces",
            method="POST",
            data=mock.ANY,
            extra_headers=mock.ANY,
        )

    # Third call with different store but same creds - should still use cached version
    with mock.patch(
        "mlflow.store.tracking.rest_store.http_request", return_value=_create_mock_response()
    ) as mock_http:
        result3 = store2.log_spans(experiment_id, spans)
        assert result3 == spans

        # Should only call OTLP, not version (cached across instances)
        mock_http.assert_called_once_with(
            host_creds=creds,
            endpoint="/v1/traces",
            method="POST",
            data=mock.ANY,
            extra_headers=mock.ANY,
        )


def test_link_prompts_to_trace():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    response.text = "{}"

    trace_id = "tr-1234"
    prompt_versions = [
        PromptVersion(name="prompt1", version=1, template="template1"),
        PromptVersion(name="prompt2", version=2, template="template2"),
    ]

    request = LinkPromptsToTrace(
        trace_id=trace_id,
        prompt_versions=[
            LinkPromptsToTrace.PromptVersionRef(name=pv.name, version=str(pv.version))
            for pv in prompt_versions
        ],
    )

    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        store.link_prompts_to_trace(trace_id=trace_id, prompt_versions=prompt_versions)
        _verify_requests(
            mock_http,
            creds,
            "traces/link-prompts",
            "POST",
            message_to_json(request),
        )


def test_create_gateway_secret():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.create_gateway_secret(
            secret_name="test-key",
            secret_value={"api_key": "sk-test-12345"},
            provider="openai",
        )
        body = message_to_json(
            CreateGatewaySecret(
                secret_name="test-key",
                secret_value={"api_key": "sk-test-12345"},
                provider="openai",
            )
        )
        _verify_requests(mock_http, creds, "gateway/secrets/create", "POST", body, use_v3=True)


def test_get_gateway_secret_info():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.get_secret_info(secret_id="secret-123")
        body = message_to_json(GetGatewaySecretInfo(secret_id="secret-123"))
        _verify_requests(mock_http, creds, "gateway/secrets/get", "GET", body, use_v3=True)


def test_update_gateway_secret():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.update_gateway_secret(
            secret_id="secret-123",
            secret_value={"api_key": "sk-new-value"},
            auth_config={"region": "us-east-1"},
        )
        body = message_to_json(
            UpdateGatewaySecret(
                secret_id="secret-123",
                secret_value={"api_key": "sk-new-value"},
                auth_config_json='{"region": "us-east-1"}',
            )
        )
        _verify_requests(mock_http, creds, "gateway/secrets/update", "POST", body, use_v3=True)


def test_delete_gateway_secret():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.delete_gateway_secret(secret_id="secret-123")
        body = message_to_json(DeleteGatewaySecret(secret_id="secret-123"))
        _verify_requests(mock_http, creds, "gateway/secrets/delete", "DELETE", body, use_v3=True)


def test_list_gateway_secret_infos():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.list_secret_infos()
        body = message_to_json(ListGatewaySecretInfos())
        _verify_requests(mock_http, creds, "gateway/secrets/list", "GET", body, use_v3=True)


def test_create_gateway_endpoint():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.create_gateway_endpoint(
            name="my-endpoint",
            model_definition_ids=["model-def-123"],
        )
        body = message_to_json(
            CreateGatewayEndpoint(
                name="my-endpoint",
                model_definition_ids=["model-def-123"],
            )
        )
        _verify_requests(mock_http, creds, "gateway/endpoints/create", "POST", body, use_v3=True)


def test_get_gateway_endpoint():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.get_gateway_endpoint(endpoint_id="endpoint-123")
        body = message_to_json(GetGatewayEndpoint(endpoint_id="endpoint-123"))
        _verify_requests(mock_http, creds, "gateway/endpoints/get", "GET", body, use_v3=True)


def test_update_gateway_endpoint():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.update_gateway_endpoint(endpoint_id="endpoint-123", name="new-name")
        body = message_to_json(UpdateGatewayEndpoint(endpoint_id="endpoint-123", name="new-name"))
        _verify_requests(mock_http, creds, "gateway/endpoints/update", "POST", body, use_v3=True)


def test_delete_gateway_endpoint():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.delete_gateway_endpoint(endpoint_id="endpoint-123")
        body = message_to_json(DeleteGatewayEndpoint(endpoint_id="endpoint-123"))
        _verify_requests(mock_http, creds, "gateway/endpoints/delete", "DELETE", body, use_v3=True)


def test_list_gateway_endpoints():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.list_gateway_endpoints()
        body = message_to_json(ListGatewayEndpoints())
        _verify_requests(mock_http, creds, "gateway/endpoints/list", "GET", body, use_v3=True)


def test_create_gateway_model_definition():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.create_gateway_model_definition(
            name="my-model-def",
            secret_id="secret-456",
            provider="anthropic",
            model_name="claude-3-5-sonnet",
        )
        body = message_to_json(
            CreateGatewayModelDefinition(
                name="my-model-def",
                secret_id="secret-456",
                provider="anthropic",
                model_name="claude-3-5-sonnet",
            )
        )
        _verify_requests(
            mock_http, creds, "gateway/model-definitions/create", "POST", body, use_v3=True
        )


def test_get_gateway_model_definition():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.get_gateway_model_definition(model_definition_id="model-def-123")
        body = message_to_json(GetGatewayModelDefinition(model_definition_id="model-def-123"))
        _verify_requests(
            mock_http, creds, "gateway/model-definitions/get", "GET", body, use_v3=True
        )


def test_list_gateway_model_definitions():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.list_gateway_model_definitions()
        body = message_to_json(ListGatewayModelDefinitions())
        _verify_requests(
            mock_http, creds, "gateway/model-definitions/list", "GET", body, use_v3=True
        )


def test_update_gateway_model_definition():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.update_gateway_model_definition(
            model_definition_id="model-def-123",
            name="updated-name",
            model_name="gpt-4o-mini",
        )
        body = message_to_json(
            UpdateGatewayModelDefinition(
                model_definition_id="model-def-123",
                name="updated-name",
                model_name="gpt-4o-mini",
            )
        )
        _verify_requests(
            mock_http, creds, "gateway/model-definitions/update", "POST", body, use_v3=True
        )


def test_delete_gateway_model_definition():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.delete_gateway_model_definition(model_definition_id="model-def-123")
        body = message_to_json(DeleteGatewayModelDefinition(model_definition_id="model-def-123"))
        _verify_requests(
            mock_http, creds, "gateway/model-definitions/delete", "DELETE", body, use_v3=True
        )


def test_attach_model_to_gateway_endpoint():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.attach_model_to_endpoint(
            endpoint_id="endpoint-123",
            model_definition_id="model-def-456",
        )
        body = message_to_json(
            AttachModelToGatewayEndpoint(
                endpoint_id="endpoint-123",
                model_definition_id="model-def-456",
                weight=1,
            )
        )
        _verify_requests(
            mock_http, creds, "gateway/endpoints/models/attach", "POST", body, use_v3=True
        )


def test_detach_model_from_gateway_endpoint():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.detach_model_from_endpoint(
            endpoint_id="endpoint-123",
            model_definition_id="model-def-456",
        )
        body = message_to_json(
            DetachModelFromGatewayEndpoint(
                endpoint_id="endpoint-123",
                model_definition_id="model-def-456",
            )
        )
        _verify_requests(
            mock_http, creds, "gateway/endpoints/models/detach", "POST", body, use_v3=True
        )


def test_create_gateway_endpoint_binding():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    response_json = json.dumps({"binding": {"resource_type": "scorer_job"}})
    with mock.patch(
        "mlflow.utils.rest_utils.http_request",
        return_value=mock.MagicMock(status_code=200, text=response_json),
    ) as mock_http:
        store.create_endpoint_binding(
            endpoint_id="endpoint-123",
            resource_type=GatewayResourceType.SCORER_JOB,
            resource_id="job-456",
        )
        body = message_to_json(
            CreateGatewayEndpointBinding(
                endpoint_id="endpoint-123",
                resource_type="scorer_job",
                resource_id="job-456",
            )
        )
        _verify_requests(
            mock_http, creds, "gateway/endpoints/bindings/create", "POST", body, use_v3=True
        )


def test_delete_gateway_endpoint_binding():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.delete_endpoint_binding(
            endpoint_id="endpoint-123",
            resource_type="scorer_job",
            resource_id="job-456",
        )
        body = message_to_json(
            DeleteGatewayEndpointBinding(
                endpoint_id="endpoint-123",
                resource_type="scorer_job",
                resource_id="job-456",
            )
        )
        _verify_requests(
            mock_http, creds, "gateway/endpoints/bindings/delete", "DELETE", body, use_v3=True
        )


def test_list_gateway_endpoint_bindings():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    with mock_http_request() as mock_http:
        store.list_endpoint_bindings(
            endpoint_id="endpoint-123",
            resource_type=GatewayResourceType.SCORER_JOB,
            resource_id="job-456",
        )
        body = message_to_json(
            ListGatewayEndpointBindings(
                endpoint_id="endpoint-123",
                resource_type="scorer_job",
                resource_id="job-456",
            )
        )
        _verify_requests(
            mock_http, creds, "gateway/endpoints/bindings/list", "GET", body, use_v3=True
        )


def test_query_trace_metrics():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200

    # Format the response
    response.text = json.dumps(
        {
            "data_points": [
                {
                    "metric_name": "latency",
                    "dimensions": {"span_name": "chat", "status": "OK"},
                    "values": {"AVG": 123.45, "COUNT": 10.0},
                },
                {
                    "metric_name": "latency",
                    "dimensions": {"span_name": "embeddings", "status": "OK"},
                    "values": {"AVG": 50.0, "COUNT": 5.0},
                },
            ],
            "next_page_token": "next_token",
        }
    )

    # Parameters for query_trace_metrics
    experiment_ids = ["1234", "5678"]
    view_type = MetricViewType.SPANS
    metric_name = "latency"
    aggregations = [
        MetricAggregation(AggregationType.AVG),
        MetricAggregation(AggregationType.COUNT),
    ]
    dimensions = ["span_name", "status"]
    filters = ["status = 'OK'"]
    max_results = 100
    page_token = "page_token_123"

    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        result = store.query_trace_metrics(
            experiment_ids=experiment_ids,
            view_type=view_type,
            metric_name=metric_name,
            aggregations=aggregations,
            dimensions=dimensions,
            filters=filters,
            max_results=max_results,
            page_token=page_token,
        )

        # Verify the correct endpoint was called
        call_args = mock_http.call_args[1]
        assert call_args["endpoint"] == f"{_V3_TRACE_REST_API_PATH_PREFIX}/metrics"

        # Verify the correct parameters were passed
        json_body = call_args["json"]
        assert json_body["experiment_ids"] == experiment_ids
        assert json_body["view_type"] == "SPANS"
        assert json_body["metric_name"] == metric_name
        assert json_body["max_results"] == max_results
        assert json_body["dimensions"] == dimensions
        assert json_body["filters"] == filters
        assert json_body["page_token"] == page_token
        assert len(json_body["aggregations"]) == 2

    # Verify the correct data points were returned
    assert len(result) == 2
    assert isinstance(result[0], MetricDataPoint)
    assert result[0].metric_name == "latency"
    assert result[0].dimensions == {"span_name": "chat", "status": "OK"}
    assert result[0].values == {"AVG": 123.45, "COUNT": 10.0}

    assert result[1].metric_name == "latency"
    assert result[1].dimensions == {"span_name": "embeddings", "status": "OK"}
    assert result[1].values == {"AVG": 50.0, "COUNT": 5.0}

    # Verify pagination token
    assert result.token == "next_token"

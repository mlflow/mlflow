import json
from unittest import mock

import pytest

import mlflow
from mlflow.entities import (
    Dataset,
    DatasetInput,
    Experiment,
    ExperimentTag,
    InputTag,
    LifecycleStage,
    Metric,
    Param,
    RunTag,
    SourceType,
    ViewType,
)
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_status import TraceStatus
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.protos.service_pb2 import (
    CreateRun,
    DeleteExperiment,
    DeleteRun,
    DeleteTag,
    DeleteTraces,
    EndTrace,
    GetExperimentByName,
    LogBatch,
    LogInputs,
    LogMetric,
    LogModel,
    LogParam,
    RestoreExperiment,
    RestoreRun,
    SearchExperiments,
    SearchRuns,
    SearchTraces,
    SetExperimentTag,
    SetTag,
    SetTraceTag,
    StartTrace,
)
from mlflow.protos.service_pb2 import RunTag as ProtoRunTag
from mlflow.protos.service_pb2 import TraceRequestMetadata as ProtoTraceRequestMetadata
from mlflow.protos.service_pb2 import TraceTag as ProtoTraceTag
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking.request_header.default_request_header_provider import (
    DefaultRequestHeaderProvider,
)
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import MlflowHostCreds


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


@mock.patch("requests.Session.request")
def test_successful_http_request(request):
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

    request.side_effect = mock_request

    store = RestStore(lambda: MlflowHostCreds("https://hello"))
    experiments = store.search_experiments()
    assert experiments[0].name == "Exp!"


@mock.patch("requests.Session.request")
def test_failed_http_request(request):
    response = mock.MagicMock()
    response.status_code = 404
    response.text = '{"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "No experiment"}'
    request.return_value = response

    store = RestStore(lambda: MlflowHostCreds("https://hello"))
    with pytest.raises(MlflowException, match="RESOURCE_DOES_NOT_EXIST: No experiment"):
        store.search_experiments()


@mock.patch("requests.Session.request")
def test_failed_http_request_custom_handler(request):
    response = mock.MagicMock()
    response.status_code = 404
    response.text = '{"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "No experiment"}'
    request.return_value = response

    store = CustomErrorHandlingRestStore(lambda: MlflowHostCreds("https://hello"))
    with pytest.raises(MyCoolException, match="cool"):
        store.search_experiments()


@mock.patch("requests.Session.request")
def test_response_with_unknown_fields(request):
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
    request.return_value = response

    store = RestStore(lambda: MlflowHostCreds("https://hello"))
    experiments = store.search_experiments()
    assert len(experiments) == 1
    assert experiments[0].name == "My experiment"


def _args(host_creds, endpoint, method, json_body):
    res = {
        "host_creds": host_creds,
        "endpoint": f"/api/2.0/mlflow/{endpoint}",
        "method": method,
    }
    if method == "GET":
        res["params"] = json.loads(json_body)
    else:
        res["json"] = json.loads(json_body)
    return res


def _verify_requests(http_request, host_creds, endpoint, method, json_body):
    http_request.assert_any_call(**(_args(host_creds, endpoint, method, json_body)))


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
    with mock_http_request() as mock_http, mock.patch(
        "mlflow.tracking._tracking_service.utils._get_store", return_value=store
    ), mock.patch(
        "mlflow.tracking.context.default_context._get_user", return_value=user_name
    ), mock.patch("time.time", return_value=13579), source_name_patch, source_type_patch:
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


def test_start_trace():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    request_id = "tr-123"
    experiment_id = "447585625682310"
    timestamp_ms = 123
    # Metadata/tags values should be string, but should not break for other types too
    metadata = {"key1": "val1", "key2": "val2", "key3": 123}
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
        res = store.start_trace(
            experiment_id=experiment_id,
            timestamp_ms=timestamp_ms,
            request_metadata=metadata,
            tags=tags,
        )
        _verify_requests(mock_http, creds, "traces", "POST", message_to_json(expected_request))
        assert isinstance(res, TraceInfo)
        assert res.request_id == request_id
        assert res.experiment_id == experiment_id
        assert res.timestamp_ms == timestamp_ms
        assert res.execution_time_ms == 0
        assert res.status == TraceStatus.UNSPECIFIED
        assert res.request_metadata == {k: str(v) for k, v in metadata.items()}
        assert res.tags == {k: str(v) for k, v in tags.items()}


def test_end_trace():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)

    experiment_id = "447585625682310"
    request_id = "tr-123"
    timestamp_ms = 123
    status = TraceStatus.OK
    metadata = {"key1": "val1", "key2": "val2"}
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
        res = store.end_trace(
            request_id=request_id,
            timestamp_ms=timestamp_ms,
            status=status,
            request_metadata=metadata,
            tags=tags,
        )
        _verify_requests(
            mock_http, creds, f"traces/{request_id}", "PATCH", message_to_json(expected_request)
        )
        assert isinstance(res, TraceInfo)
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
    request = SearchTraces(
        experiment_ids=["0", "1"],
        filter="trace.status = 'ERROR'",
        max_results=1,
        order_by=["timestamp_ms DESC"],
        page_token="12345abcde",
    )
    response.text = json.dumps(
        {
            "traces": [
                {
                    "request_id": "tr-1234",
                    "experiment_id": "1234",
                    "timestamp_ms": 123,
                    "execution_time_ms": 456,
                    "status": "ERROR",
                    "tags": [
                        {"key": "k", "value": "v"},
                    ],
                },
            ],
            "next_page_token": "token",
        }
    )
    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        trace_infos, token = store.search_traces(
            experiment_ids=request.experiment_ids,
            filter_string=request.filter,
            max_results=request.max_results,
            order_by=request.order_by,
            page_token=request.page_token,
        )
        _verify_requests(mock_http, creds, "traces", "GET", message_to_json(request))
        assert trace_infos == [
            TraceInfo(
                request_id="tr-1234",
                experiment_id="1234",
                timestamp_ms=123,
                execution_time_ms=456,
                status="ERROR",
                request_metadata={},
                tags={"k": "v"},
            )
        ]
        assert token == "token"


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
        res = store.delete_traces(**delete_traces_kwargs)
        _verify_requests(mock_http, creds, "traces/delete-traces", "POST", message_to_json(request))
        assert res == 1


def test_set_trace_tag():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    response = mock.MagicMock()
    response.status_code = 200
    request_id = "tr-1234"
    request = SetTraceTag(
        key="k",
        value="v",
    )
    response.text = "{}"
    with mock.patch("mlflow.utils.rest_utils.http_request", return_value=response) as mock_http:
        res = store.set_trace_tag(
            request_id=request_id,
            key=request.key,
            value=request.value,
        )
        _verify_requests(
            mock_http, creds, f"traces/{request_id}/tags", "PATCH", message_to_json(request)
        )
        assert res is None

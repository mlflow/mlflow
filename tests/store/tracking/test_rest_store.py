import json
import unittest

from unittest import mock
import pytest

import mlflow
from mlflow.entities import (
    Param,
    Metric,
    RunTag,
    SourceType,
    ViewType,
    ExperimentTag,
    Experiment,
    LifecycleStage,
)
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.service_pb2 import (
    CreateRun,
    DeleteExperiment,
    DeleteRun,
    LogBatch,
    LogMetric,
    LogParam,
    RestoreExperiment,
    RestoreRun,
    RunTag as ProtoRunTag,
    SearchRuns,
    SetTag,
    DeleteTag,
    SetExperimentTag,
    GetExperimentByName,
    ListExperiments,
    LogModel,
)
from mlflow.protos.databricks_pb2 import (
    RESOURCE_DOES_NOT_EXIST,
    ENDPOINT_NOT_FOUND,
    REQUEST_LIMIT_EXCEEDED,
    INTERNAL_ERROR,
    ErrorCode,
)
from mlflow.store.tracking.rest_store import (
    RestStore,
    DatabricksRestStore,
)
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import MlflowHostCreds, _DEFAULT_HEADERS


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


class TestRestStore:
    @mock.patch("requests.Session.request")
    def test_successful_http_request(self, request):
        def mock_request(*args, **kwargs):
            # Filter out None arguments
            assert args == ("GET", "https://hello/api/2.0/mlflow/experiments/list")
            kwargs = dict((k, v) for k, v in kwargs.items() if v is not None)
            assert kwargs == {
                "params": {"view_type": "ACTIVE_ONLY"},
                "headers": _DEFAULT_HEADERS,
                "verify": True,
                "timeout": 120,
            }
            response = mock.MagicMock()
            response.status_code = 200
            response.text = '{"experiments": [{"name": "Exp!", "lifecycle_stage": "active"}]}'
            return response

        request.side_effect = mock_request

        store = RestStore(lambda: MlflowHostCreds("https://hello"))
        experiments = store.list_experiments()
        assert experiments[0].name == "Exp!"

    @mock.patch("requests.Session.request")
    def test_failed_http_request(self, request):
        response = mock.MagicMock()
        response.status_code = 404
        response.text = '{"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "No experiment"}'
        request.return_value = response

        store = RestStore(lambda: MlflowHostCreds("https://hello"))
        with pytest.raises(MlflowException, match="RESOURCE_DOES_NOT_EXIST: No experiment"):
            store.list_experiments()

    @mock.patch("requests.Session.request")
    def test_failed_http_request_custom_handler(self, request):
        response = mock.MagicMock()
        response.status_code = 404
        response.text = '{"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "No experiment"}'
        request.return_value = response

        store = CustomErrorHandlingRestStore(lambda: MlflowHostCreds("https://hello"))
        with pytest.raises(MyCoolException, match="cool"):
            store.list_experiments()

    @mock.patch("requests.Session.request")
    def test_response_with_unknown_fields(self, request):
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
        experiments = store.list_experiments()
        assert len(experiments) == 1
        assert experiments[0].name == "My experiment"

    def _args(self, host_creds, endpoint, method, json_body):
        res = {
            "host_creds": host_creds,
            "endpoint": "/api/2.0/mlflow/%s" % endpoint,
            "method": method,
        }
        if method == "GET":
            res["params"] = json.loads(json_body)
        else:
            res["json"] = json.loads(json_body)
        return res

    def _verify_requests(self, http_request, host_creds, endpoint, method, json_body):
        http_request.assert_any_call(**(self._args(host_creds, endpoint, method, json_body)))

    def test_requestor(self):
        creds = MlflowHostCreds("https://hello")
        store = RestStore(lambda: creds)

        user_name = "mock user"
        source_name = "rest test"

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
        ), mock.patch(
            "time.time", return_value=13579
        ), source_name_patch, source_type_patch:
            with mlflow.start_run(experiment_id="43"):
                cr_body = message_to_json(
                    CreateRun(
                        experiment_id="43",
                        user_id=user_name,
                        start_time=13579000,
                        tags=[
                            ProtoRunTag(key="mlflow.source.name", value=source_name),
                            ProtoRunTag(key="mlflow.source.type", value="LOCAL"),
                            ProtoRunTag(key="mlflow.user", value=user_name),
                        ],
                    )
                )
                expected_kwargs = self._args(creds, "runs/create", "POST", cr_body)

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
            self._verify_requests(mock_http, creds, "runs/log-parameter", "POST", body)

        with mock_http_request() as mock_http:
            store.set_experiment_tag("some_id", ExperimentTag("t1", "abcd" * 1000))
            body = message_to_json(
                SetExperimentTag(experiment_id="some_id", key="t1", value="abcd" * 1000)
            )
            self._verify_requests(mock_http, creds, "experiments/set-experiment-tag", "POST", body)

        with mock_http_request() as mock_http:
            store.set_tag("some_uuid", RunTag("t1", "abcd" * 1000))
            body = message_to_json(
                SetTag(run_uuid="some_uuid", run_id="some_uuid", key="t1", value="abcd" * 1000)
            )
            self._verify_requests(mock_http, creds, "runs/set-tag", "POST", body)

        with mock_http_request() as mock_http:
            store.delete_tag("some_uuid", "t1")
            body = message_to_json(DeleteTag(run_id="some_uuid", key="t1"))
            self._verify_requests(mock_http, creds, "runs/delete-tag", "POST", body)

        with mock_http_request() as mock_http:
            store.log_metric("u2", Metric("m1", 0.87, 12345, 3))
            body = message_to_json(
                LogMetric(run_uuid="u2", run_id="u2", key="m1", value=0.87, timestamp=12345, step=3)
            )
            self._verify_requests(mock_http, creds, "runs/log-metric", "POST", body)

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
            self._verify_requests(mock_http, creds, "runs/log-batch", "POST", body)

        with mock_http_request() as mock_http:
            store.delete_run("u25")
            self._verify_requests(
                mock_http, creds, "runs/delete", "POST", message_to_json(DeleteRun(run_id="u25"))
            )

        with mock_http_request() as mock_http:
            store.restore_run("u76")
            self._verify_requests(
                mock_http, creds, "runs/restore", "POST", message_to_json(RestoreRun(run_id="u76"))
            )

        with mock_http_request() as mock_http:
            store.delete_experiment("0")
            self._verify_requests(
                mock_http,
                creds,
                "experiments/delete",
                "POST",
                message_to_json(DeleteExperiment(experiment_id="0")),
            )

        with mock_http_request() as mock_http:
            store.restore_experiment("0")
            self._verify_requests(
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
            self._verify_requests(
                mock_http, creds, "runs/search", "POST", message_to_json(expected_message)
            )
            assert result.token == "67890fghij"

        with mock_http_request() as mock_http:
            run_id = "run_id"
            m = Model(artifact_path="model/path", run_id="run_id", flavors={"tf": "flavor body"})
            store.record_logged_model("run_id", m)
            expected_message = LogModel(run_id=run_id, model_json=m.to_json())
            self._verify_requests(
                mock_http, creds, "runs/log-model", "POST", message_to_json(expected_message)
            )

    @pytest.mark.parametrize("store_class", [RestStore, DatabricksRestStore])
    def test_get_experiment_by_name(self, store_class):
        creds = MlflowHostCreds("https://hello")
        store = store_class(lambda: creds)
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
            self._verify_requests(
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
            self._verify_requests(
                mock_http,
                creds,
                "experiments/get-by-name",
                "GET",
                message_to_json(expected_message1),
            )
            assert mock_http.call_count == 1

            # Test REST client behavior against a mocked old server, which has handler for
            # ListExperiments but not GetExperimentByName
            mock_http.reset_mock()
            list_exp_response = mock.MagicMock()
            list_exp_response.text = json.dumps(
                {"experiments": [json.loads(message_to_json(experiment.to_proto()))]}
            )
            list_exp_response.status_code = 200

            def response_fn(*args, **kwargs):
                # pylint: disable=unused-argument
                if kwargs.get("endpoint") == "/api/2.0/mlflow/experiments/get-by-name":
                    raise MlflowException(
                        "GetExperimentByName is not implemented", ENDPOINT_NOT_FOUND
                    )
                else:
                    return list_exp_response

            mock_http.side_effect = response_fn
            result = store.get_experiment_by_name("abc")
            expected_message2 = ListExperiments(view_type=ViewType.ALL)
            self._verify_requests(
                mock_http,
                creds,
                "experiments/get-by-name",
                "GET",
                message_to_json(expected_message0),
            )
            self._verify_requests(
                mock_http, creds, "experiments/list", "GET", message_to_json(expected_message2)
            )
            assert result.experiment_id == experiment.experiment_id
            assert result.name == experiment.name
            assert result.artifact_location == experiment.artifact_location
            assert result.lifecycle_stage == experiment.lifecycle_stage

            # Verify that REST client won't fall back to ListExperiments for 429 errors (hitting
            # rate limits)
            mock_http.reset_mock()

            def rate_limit_response_fn(*args, **kwargs):
                # pylint: disable=unused-argument
                raise MlflowException(
                    "Hit rate limit on GetExperimentByName", REQUEST_LIMIT_EXCEEDED
                )

            mock_http.side_effect = rate_limit_response_fn
            with pytest.raises(MlflowException, match="Hit rate limit") as exc_info:
                store.get_experiment_by_name("imspamming")
            assert exc_info.value.error_code == ErrorCode.Name(REQUEST_LIMIT_EXCEEDED)
            assert mock_http.call_count == 1

    def test_databricks_rest_store_get_experiment_by_name(self):
        creds = MlflowHostCreds("https://hello")
        store = DatabricksRestStore(lambda: creds)
        with mock.patch("mlflow.utils.rest_utils.http_request") as mock_http:
            # Verify that Databricks REST client won't fall back to ListExperiments for 500-level
            # errors that are not ENDPOINT_NOT_FOUND

            def rate_limit_response_fn(*args, **kwargs):
                # pylint: disable=unused-argument
                raise MlflowException("Some internal error!", INTERNAL_ERROR)

            mock_http.side_effect = rate_limit_response_fn
            with pytest.raises(MlflowException, match="Some internal error!") as exc_info:
                store.get_experiment_by_name("abc")
            assert exc_info.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
            expected_message0 = GetExperimentByName(experiment_name="abc")
            self._verify_requests(
                mock_http,
                creds,
                "experiments/get-by-name",
                "GET",
                message_to_json(expected_message0),
            )
            assert mock_http.call_count == 1

    def test_databricks_paginate_list_experiments(self):
        creds = MlflowHostCreds("https://hello")
        store = DatabricksRestStore(lambda: creds)

        list_exp_responses = []
        next_page_tokens = ["a", "b", None]
        for next_page_token in next_page_tokens:
            experiment = Experiment(
                experiment_id="123",
                name=str(next_page_token),
                artifact_location="/abc",
                lifecycle_stage=LifecycleStage.ACTIVE,
            )
            list_exp_response = mock.MagicMock()
            list_exp_response.text = json.dumps(
                {
                    "experiments": [json.loads(message_to_json(experiment.to_proto()))],
                    "next_page_token": next_page_token,
                }
            )
            list_exp_response.status_code = 200
            list_exp_responses.append(list_exp_response)

        with mock.patch("mlflow.utils.rest_utils.http_request", side_effect=list_exp_responses):
            for idx, experiments in enumerate(
                store._paginate_list_experiments(ViewType.ACTIVE_ONLY)
            ):
                assert experiments[0].name == str(next_page_tokens[idx])
                assert experiments.token == next_page_tokens[idx]


if __name__ == "__main__":
    unittest.main()

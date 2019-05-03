import json
import unittest

import mock
import six

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.entities import Param, Metric, RunTag, SourceType
from mlflow.protos.service_pb2 import DeleteExperiment, RestoreExperiment, LogParam, LogMetric, \
    SetTag, DeleteRun, RestoreRun, CreateRun, RunTag as ProtoRunTag, LogBatch
from mlflow.store.rest_store import RestStore
from mlflow.utils.proto_json_utils import message_to_json

from mlflow.utils.rest_utils import MlflowHostCreds, _DEFAULT_HEADERS


class TestRestStore(unittest.TestCase):
    @mock.patch('requests.request')
    def test_successful_http_request(self, request):
        def mock_request(**kwargs):
            # Filter out None arguments
            kwargs = dict((k, v) for k, v in six.iteritems(kwargs) if v is not None)
            assert kwargs == {
                'method': 'GET',
                'params': {'view_type': 'ACTIVE_ONLY'},
                'url': 'https://hello/api/2.0/preview/mlflow/experiments/list',
                'headers': _DEFAULT_HEADERS,
                'verify': True,
            }
            response = mock.MagicMock
            response.status_code = 200
            response.text = '{"experiments": [{"name": "Exp!", "lifecycle_stage": "active"}]}'
            return response

        request.side_effect = mock_request

        store = RestStore(lambda: MlflowHostCreds('https://hello'))
        experiments = store.list_experiments()
        assert experiments[0].name == "Exp!"

    @mock.patch('requests.request')
    def test_failed_http_request(self, request):
        response = mock.MagicMock
        response.status_code = 404
        response.text = '{"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "No experiment"}'
        request.return_value = response

        store = RestStore(lambda: MlflowHostCreds('https://hello'))
        with self.assertRaises(MlflowException) as cm:
            store.list_experiments()
        self.assertIn("RESOURCE_DOES_NOT_EXIST: No experiment", str(cm.exception))

    @mock.patch('requests.request')
    def test_response_with_unknown_fields(self, request):
        experiment_json = {
            "experiment_id": "1",
            "name": "My experiment",
            "artifact_location": "foo",
            "lifecycle_stage": "deleted",
            "OMG_WHAT_IS_THIS_FIELD": "Hooly cow",
        }

        response = mock.MagicMock
        response.status_code = 200
        experiments = {"experiments": [experiment_json]}
        response.text = json.dumps(experiments)
        request.return_value = response

        store = RestStore(lambda: MlflowHostCreds('https://hello'))
        experiments = store.list_experiments()
        assert len(experiments) == 1
        assert experiments[0].name == 'My experiment'

    def _args(self, host_creds, endpoint, method, json_body):
        return {'host_creds': host_creds,
                'endpoint': "/api/2.0/preview/mlflow/%s" % endpoint,
                'method': method,
                'json': json.loads(json_body)}

    def _verify_requests(self, http_request, host_creds, endpoint, method, json_body):
        http_request.assert_called_with(**(self._args(host_creds, endpoint, method, json_body)))

    def _verify_request_has_calls(self, http_request, host_creds, call_args):
        http_request.assert_has_calls(calls=[mock.call(**(self._args(host_creds, endpoint, method,
                                                                     json_body)))
                                             for endpoint, method, json_body in call_args],
                                      any_order=True)

    @mock.patch('requests.request')
    def test_requestor(self, request):
        response = mock.MagicMock
        response.status_code = 200
        response.text = '{}'
        request.return_value = response

        creds = MlflowHostCreds('https://hello')
        store = RestStore(lambda: creds)

        user_name = "mock user"
        source_name = "rest test"

        source_name_patch = mock.patch(
            "mlflow.tracking.context._get_source_name", return_value=source_name
        )
        source_type_patch = mock.patch(
            "mlflow.tracking.context._get_source_type", return_value=SourceType.LOCAL
        )
        with mock.patch('mlflow.store.rest_store.http_request_safe') as mock_http, \
                mock.patch('mlflow.tracking.utils._get_store', return_value=store), \
                mock.patch('mlflow.tracking.client._get_user_id', return_value=user_name), \
                mock.patch('time.time', return_value=13579), \
                source_name_patch, source_type_patch:
            with mlflow.start_run(experiment_id="43"):
                cr_body = message_to_json(CreateRun(experiment_id="43",
                                                    user_id=user_name, start_time=13579000,
                                                    tags=[ProtoRunTag(key='mlflow.source.name',
                                                                      value=source_name),
                                                          ProtoRunTag(key='mlflow.source.type',
                                                                      value='LOCAL')]))
                assert mock_http.call_count == 1
                exp_calls = [("runs/create", "POST", cr_body)]
                self._verify_request_has_calls(mock_http, creds, exp_calls)

        with mock.patch('mlflow.store.rest_store.http_request_safe') as mock_http:
            store.log_param("some_uuid", Param("k1", "v1"))
            body = message_to_json(LogParam(
                run_uuid="some_uuid", run_id="some_uuid", key="k1", value="v1"))
            self._verify_requests(mock_http, creds,
                                  "runs/log-parameter", "POST", body)

        with mock.patch('mlflow.store.rest_store.http_request_safe') as mock_http:
            store.set_tag("some_uuid", RunTag("t1", "abcd"*1000))
            body = message_to_json(SetTag(
                run_uuid="some_uuid", run_id="some_uuid", key="t1", value="abcd"*1000))
            self._verify_requests(mock_http, creds,
                                  "runs/set-tag", "POST", body)

        with mock.patch('mlflow.store.rest_store.http_request_safe') as mock_http:
            store.log_metric("u2", Metric("m1", 0.87, 12345, 3))
            body = message_to_json(LogMetric(
                run_uuid="u2", run_id="u2", key="m1", value=0.87, timestamp=12345, step=3))
            self._verify_requests(mock_http, creds,
                                  "runs/log-metric", "POST", body)

        with mock.patch('mlflow.store.rest_store.http_request_safe') as mock_http:
            metrics = [Metric("m1", 0.87, 12345, 0), Metric("m2", 0.49, 12345, -1),
                       Metric("m3", 0.58, 12345, 2)]
            params = [Param("p1", "p1val"), Param("p2", "p2val")]
            tags = [RunTag("t1", "t1val"), RunTag("t2", "t2val")]
            store.log_batch(run_id="u2", metrics=metrics, params=params, tags=tags)
            metric_protos = [metric.to_proto() for metric in metrics]
            param_protos = [param.to_proto() for param in params]
            tag_protos = [tag.to_proto() for tag in tags]
            body = message_to_json(LogBatch(run_id="u2", metrics=metric_protos,
                                            params=param_protos, tags=tag_protos))
            self._verify_requests(mock_http, creds,
                                  "runs/log-batch", "POST", body)

        with mock.patch('mlflow.store.rest_store.http_request_safe') as mock_http:
            store.delete_run("u25")
            self._verify_requests(mock_http, creds,
                                  "runs/delete", "POST",
                                  message_to_json(DeleteRun(run_id="u25")))

        with mock.patch('mlflow.store.rest_store.http_request_safe') as mock_http:
            store.restore_run("u76")
            self._verify_requests(mock_http, creds,
                                  "runs/restore", "POST",
                                  message_to_json(RestoreRun(run_id="u76")))

        with mock.patch('mlflow.store.rest_store.http_request_safe') as mock_http:
            store.delete_experiment("0")
            self._verify_requests(mock_http, creds,
                                  "experiments/delete", "POST",
                                  message_to_json(DeleteExperiment(experiment_id="0")))

        with mock.patch('mlflow.store.rest_store.http_request_safe') as mock_http:
            store.restore_experiment("0")
            self._verify_requests(mock_http, creds,
                                  "experiments/restore", "POST",
                                  message_to_json(RestoreExperiment(experiment_id="0")))


if __name__ == '__main__':
    unittest.main()

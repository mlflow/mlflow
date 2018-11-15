import json
import unittest

import mock
import six

from mlflow.exceptions import MlflowException
from mlflow.entities import Param, Metric, RunTag
from mlflow.protos.service_pb2 import DeleteExperiment, RestoreExperiment, LogParam, LogMetric, \
    SetTag, DeleteRun, RestoreRun
from mlflow.store.rest_store import RestStore
from mlflow.utils.proto_json_utils import message_to_json

from mlflow.utils.rest_utils import MlflowHostCreds


class TestRestStore(unittest.TestCase):
    @mock.patch('requests.request')
    def test_successful_http_request(self, request):
        def mock_request(**kwargs):
            # Filter out None arguments
            kwargs = dict((k, v) for k, v in six.iteritems(kwargs) if v is not None)
            assert kwargs == {
                'method': 'GET',
                'json': {'view_type': 'ACTIVE_ONLY'},
                'url': 'https://hello/api/2.0/preview/mlflow/experiments/list',
                'headers': {},
                'verify': True,
            }
            response = mock.MagicMock
            response.status_code = 200
            response.text = '{"experiments": [{"name": "Exp!"}]}'
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
            "experiment_id": 1,
            "name": "My experiment",
            "artifact_location": "foo",
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

    def _verify_requests(self, http_request, host_creds, endpoint, method, json_body):
        http_request.assert_called_with(host_creds=host_creds,
                                        endpoint="/api/2.0/preview/mlflow/%s" % endpoint,
                                        method=method,
                                        json=json.loads(json_body))

    @mock.patch('requests.request')
    def test_requestor(self, request):
        response = mock.MagicMock
        response.status_code = 200
        response.text = '{}'
        request.return_value = response

        creds = MlflowHostCreds('https://hello')
        store = RestStore(lambda: creds)

        with mock.patch('mlflow.store.rest_store.http_request_safe') as mock_http:
            store.log_param("some_uuid", Param("k1", "v1"))
            body = message_to_json(LogParam(run_uuid="some_uuid", key="k1", value="v1"))
            self._verify_requests(mock_http, creds,
                                  "runs/log-parameter", "POST", body)

        with mock.patch('mlflow.store.rest_store.http_request_safe') as mock_http:
            store.set_tag("some_uuid", RunTag("t1", "abcd"*1000))
            body = message_to_json(SetTag(run_uuid="some_uuid", key="t1", value="abcd"*1000))
            self._verify_requests(mock_http, creds,
                                  "runs/set-tag", "POST", body)

        with mock.patch('mlflow.store.rest_store.http_request_safe') as mock_http:
            store.log_metric("u2", Metric("m1", 0.87, 12345))
            body = message_to_json(LogMetric(run_uuid="u2", key="m1", value=0.87, timestamp=12345))
            self._verify_requests(mock_http, creds,
                                  "runs/log-metric", "POST", body)

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
            store.delete_experiment(0)
            self._verify_requests(mock_http, creds,
                                  "experiments/delete", "POST",
                                  message_to_json(DeleteExperiment(experiment_id=0)))

        with mock.patch('mlflow.store.rest_store.http_request_safe') as mock_http:
            store.restore_experiment(0)
            self._verify_requests(mock_http, creds,
                                  "experiments/restore", "POST",
                                  message_to_json(RestoreExperiment(experiment_id=0)))


if __name__ == '__main__':
    unittest.main()

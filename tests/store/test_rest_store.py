import json
import unittest

import mock
import six

from mlflow.protos.service_pb2 import DeleteExperiment, RestoreExperiment
from mlflow.store.rest_store import RestStore, RestException
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
        with self.assertRaises(RestException) as cm:
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
        assert http_request.assert_called_with(host_creds=host_creds,
                                               endpoint=endpoint,
                                               method=method,
                                               json=json_body)

    def test_requestor(self):
        store = RestStore(lambda: MlflowHostCreds('https://hello'))

        with mock.patch('mlflow.utils.rest_utils.http_request') as mock_http:
            store.delete_experiment(0)
            self._verify_requests(mock_http,
                                  store.get_host_creds(),
                                  "/api/2.0/preview/mlflow/experiments/delete",
                                  "POST",
                                  message_to_json(DeleteExperiment(experiment_id=0)))

        with mock.patch('mlflow.utils.rest_utils.http_request') as mock_http:
            store.restore_experiment(0)
            self._verify_requests(mock_http,
                                  store.get_host_creds(),
                                  "/api/2.0/preview/mlflow/experiments/restore",
                                  "POST",
                                  message_to_json(RestoreExperiment(experiment_id=0)))


if __name__ == '__main__':
    unittest.main()

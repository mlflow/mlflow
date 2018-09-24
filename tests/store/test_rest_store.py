import json
import mock
import six
import unittest

from mlflow.store.rest_store import RestStore, RestException
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


if __name__ == '__main__':
    unittest.main()

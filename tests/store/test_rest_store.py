import mock
import six
import unittest

from mlflow.store.rest_store import RestStore, RestException


class TestRestStore(unittest.TestCase):
    @mock.patch('requests.request')
    def test_successful_http_request(self, request):
        def mock_request(**kwargs):
            # Filter out None arguments
            kwargs = dict((k, v) for k, v in six.iteritems(kwargs) if v is not None)
            assert kwargs == {
                'method': 'GET',
                'url': 'https://hello/api/2.0/preview/mlflow/experiments/list',
                'verify': True,
            }
            response = mock.MagicMock
            response.status_code = 200
            response.text = '{"experiments": [{"name": "Exp!"}]}'
            return response
        request.side_effect = mock_request

        store = RestStore({'hostname': 'https://hello'})
        experiments = store.list_experiments()
        assert experiments[0].name == "Exp!"

    @mock.patch('requests.request')
    def test_failed_http_request(self, request):
        def mock_request(**_):
            response = mock.MagicMock
            response.status_code = 404
            response.text = '{"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "No experiment"}'
            return response
        request.side_effect = mock_request

        store = RestStore({'hostname': 'https://hello'})
        with self.assertRaises(RestException) as cm:
            store.list_experiments()
        self.assertIn("RESOURCE_DOES_NOT_EXIST: No experiment", str(cm.exception))


if __name__ == '__main__':
    unittest.main()

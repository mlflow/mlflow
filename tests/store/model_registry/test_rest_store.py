import unittest

import json
import mock
import pytest
import uuid

from mlflow.entities.model_registry import RegisteredModel, ModelVersion
from mlflow.protos.model_registry_pb2 import CreateRegisteredModel, \
    UpdateRegisteredModel, DeleteRegisteredModel, ListRegisteredModels, \
    GetRegisteredModelDetails, GetLatestVersions, CreateModelVersion, UpdateModelVersion, \
    DeleteModelVersion, GetModelVersionDetails, GetModelVersionDownloadUri, SearchModelVersions, \
    GetModelVersionStages
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import MlflowHostCreds


@pytest.fixture(scope="class")
def request():
    with mock.patch('requests.request') as request:
        response = mock.MagicMock
        response.status_code = 200
        response.text = '{}'
        request.return_value = response
        yield request


@pytest.mark.usefixtures("request")
class TestRestStore(unittest.TestCase):
    def setUp(self):
        self.creds = MlflowHostCreds('https://hello')
        self.store = RestStore(lambda: self.creds)

    def tearDown(self):
        pass

    def _args(self, host_creds, endpoint, method, json_body):
        res = {'host_creds': host_creds,
               'endpoint': "/api/2.0/preview/mlflow/%s" % endpoint,
               'method': method}
        if method == "GET":
            res["params"] = json.loads(json_body)
        else:
            res["json"] = json.loads(json_body)
        return res

    def _verify_requests(self, http_request, endpoint, method, proto_message):
        print(http_request.call_args_list)
        json_body = message_to_json(proto_message)
        http_request.assert_any_call(**(self._args(self.creds, endpoint, method, json_body)))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_create_registered_model(self, mock_http):
        self.store.create_registered_model("model_1")
        self._verify_requests(mock_http, "registered-models/create", "POST",
                              CreateRegisteredModel(name="model_1"))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_update_registered_model_name(self, mock_http):
        rm = RegisteredModel("model_1")
        self.store.update_registered_model(registered_model=rm, new_name="model_2")
        self._verify_requests(mock_http, "registered-models/update", "PATCH",
                              UpdateRegisteredModel(registered_model=rm.to_proto(),
                                                    name="model_2"))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_update_registered_model_description(self, mock_http):
        rm = RegisteredModel("model_1")
        self.store.update_registered_model(registered_model=rm, description="test model")
        self._verify_requests(mock_http, "registered-models/update", "PATCH",
                              UpdateRegisteredModel(registered_model=rm.to_proto(),
                                                    description="test model"))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_update_registered_model_all(self, mock_http):
        rm = RegisteredModel("model_1")
        self.store.update_registered_model(registered_model=rm,
                                           new_name="model_3",
                                           description="rename and describe")
        self._verify_requests(mock_http, "registered-models/update", "PATCH",
                              UpdateRegisteredModel(registered_model=rm.to_proto(),
                                                    name="model_3",
                                                    description="rename and describe"))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_delete_registered_model(self, mock_http):
        rm = RegisteredModel("model_1")
        self.store.delete_registered_model(registered_model=rm)
        self._verify_requests(mock_http, "registered-models/delete", "DELETE",
                              DeleteRegisteredModel(registered_model=rm.to_proto()))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_list_registered_model(self, mock_http):
        self.store.list_registered_models()
        self._verify_requests(mock_http, "registered-models/list", "GET", ListRegisteredModels())

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_get_registered_model_detailed(self, mock_http):
        rm = RegisteredModel("model_1")
        self.store.get_registered_model_details(registered_model=rm)
        self._verify_requests(mock_http, "registered-models/get-details", "POST",
                              GetRegisteredModelDetails(registered_model=rm.to_proto()))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_get_latest_versions(self, mock_http):
        rm = RegisteredModel("model_1")
        self.store.get_latest_versions(registered_model=rm)
        self._verify_requests(mock_http, "registered-models/get-latest-versions", "POST",
                              GetLatestVersions(registered_model=rm.to_proto()))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_get_latest_versions_with_stages(self, mock_http):
        rm = RegisteredModel("model_1")
        self.store.get_latest_versions(registered_model=rm, stages=["blaah"])
        self._verify_requests(mock_http, "registered-models/get-latest-versions", "POST",
                              GetLatestVersions(registered_model=rm.to_proto(), stages=["blaah"]))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_create_model_version(self, mock_http):
        run_id = uuid.uuid4().hex
        self.store.create_model_version("model_1", "path/to/source", run_id)
        self._verify_requests(mock_http, "model-versions/create", "POST",
                              CreateModelVersion(name="model_1", source="path/to/source",
                                                 run_id=run_id))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_update_model_version_stage(self, mock_http):
        rm = RegisteredModel("model_1")
        mv = ModelVersion(rm, 5)
        self.store.update_model_version(model_version=mv, stage="prod")
        self._verify_requests(mock_http, "model-versions/update", "PATCH",
                              UpdateModelVersion(model_version=mv.to_proto(), stage="prod"))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_update_model_version_decription(self, mock_http):
        rm = RegisteredModel("model_1")
        mv = ModelVersion(rm, 5)
        self.store.update_model_version(model_version=mv, description="test model version")
        self._verify_requests(mock_http, "model-versions/update", "PATCH",
                              UpdateModelVersion(model_version=mv.to_proto(),
                                                 description="test model version"))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_update_model_version_all(self, mock_http):
        rm = RegisteredModel("model_1")
        mv = ModelVersion(rm, 5)
        self.store.update_model_version(model_version=mv, stage="5%", description="A|B test")
        self._verify_requests(mock_http, "model-versions/update", "PATCH",
                              UpdateModelVersion(model_version=mv.to_proto(),
                                                 stage="5%", description="A|B test"))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_delete_model_version(self, mock_http):
        rm = RegisteredModel("model_1")
        mv = ModelVersion(rm, 12)
        self.store.delete_model_version(model_version=mv)
        self._verify_requests(mock_http, "model-versions/delete", "DELETE",
                              DeleteModelVersion(model_version=mv.to_proto()))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_get_model_version_details(self, mock_http):
        rm = RegisteredModel("model_11")
        mv = ModelVersion(rm, 8)
        self.store.get_model_version_details(model_version=mv)
        self._verify_requests(mock_http, "model-versions/get-details", "POST",
                              GetModelVersionDetails(model_version=mv.to_proto()))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_get_model_version_download_uri(self, mock_http):
        rm = RegisteredModel("model_11")
        mv = ModelVersion(rm, 8)
        self.store.get_model_version_download_uri(model_version=mv)
        self._verify_requests(mock_http, "model-versions/get-download-uri", "POST",
                              GetModelVersionDownloadUri(model_version=mv.to_proto()))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_search_model_versions(self, mock_http):
        self.store.search_model_versions(filter_string="name='model_12'")
        self._verify_requests(mock_http, "model-versions/search", "GET",
                              SearchModelVersions(filter="name='model_12'"))

    @mock.patch('mlflow.utils.rest_utils.http_request')
    def test_get_model_version_stages(self, mock_http):
        rm = RegisteredModel("model_11")
        mv = ModelVersion(rm, 8)
        self.store.get_model_version_stages(model_version=mv)
        self._verify_requests(mock_http, "model-versions/get-stages", "POST",
                              GetModelVersionStages(model_version=mv.to_proto()))

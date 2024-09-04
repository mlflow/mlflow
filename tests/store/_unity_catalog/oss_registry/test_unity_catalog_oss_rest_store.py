import json
from unittest import mock

import pytest

from mlflow.protos.unity_catalog_oss_messages_pb2 import (
    RegisteredModelInfo,
)
from mlflow.store._unity_catalog.registry_oss.rest_store_oss import UnityCatalogOssStore
from mlflow.utils.proto_json_utils import message_to_json

from tests.helper_functions import mock_http_200
from tests.store._unity_catalog.conftest import _REGISTRY_HOST_CREDS


@pytest.fixture
def store(mock_databricks_uc_oss_host_creds):
    with mock.patch("mlflow.utils.databricks_utils.get_databricks_host_creds"):
        return UnityCatalogOssStore(store_uri="databricks-uc")


def _args(endpoint, method, json_body, host_creds, extra_headers):
    res = {
        "host_creds": host_creds,
        "endpoint": f"/api/2.1/unity-catalog/{endpoint}",
        "method": method,
        "extra_headers": extra_headers,
    }
    if extra_headers is None:
        del res["extra_headers"]
    if method == "GET":
        res["params"] = json.loads(json_body)
    else:
        res["json"] = json.loads(json_body)
    return res


def _verify_requests(
    http_request,
    endpoint,
    method,
    proto_message,
    host_creds=_REGISTRY_HOST_CREDS,
    extra_headers=None,
):
    json_body = message_to_json(proto_message)
    call_args = _args(endpoint, method, json_body, host_creds, extra_headers)
    http_request.assert_any_call(**call_args)


@mock_http_200
def test_create_registered_model(mock_http, store):
    description = "best model ever"
    store.create_registered_model(
        name="catalog_1.schema_1.model_1",
        description=description,
    )
    _verify_requests(
        mock_http,
        "models",
        "POST",
        RegisteredModelInfo(
            name="model_1",
            catalog_name="catalog_1",
            schema_name="schema_1",
            comment=description,
        ),
    )

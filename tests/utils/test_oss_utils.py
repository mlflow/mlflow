import unittest
import pytest
from mlflow.utils.rest_utils import MlflowHostCreds
from mlflow.utils.uri import _DATABRICKS_UNITY_CATALOG_SCHEME
from mlflow.utils.oss_utils import get_oss_host_creds
from unittest import mock

def test_get_oss_host_creds():
    # Test case: When the scheme is "uc" and the new scheme is not "_DATABRICKS_UNITY_CATALOG_SCHEME"
    server_uri = "uc:http://localhost:8081"
    expected_creds = MlflowHostCreds(host="http://localhost:8081")
    actual_creds = get_oss_host_creds(server_uri)
    assert actual_creds == expected_creds

def test_get_databricks_host_creds():
    # Test case: When the scheme is "uc" and the new scheme is "_DATABRICKS_UNITY_CATALOG_SCHEME"
    server_uri = "uc:databricks-uc://profile:prefix"
    with mock.patch("mlflow.utils.oss_utils.get_databricks_host_creds",
                    return_value=mock.MagicMock()) as mock_get_databricks_host_creds:
        expected_creds = get_oss_host_creds(server_uri)
        assert mock_get_databricks_host_creds.call_args_list == [mock.call('databricks-uc://profile:prefix')]

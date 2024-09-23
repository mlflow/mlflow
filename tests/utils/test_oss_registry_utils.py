from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.utils.oss_registry_utils import get_oss_host_creds
from mlflow.utils.rest_utils import MlflowHostCreds


@pytest.mark.parametrize(
    ("server_uri", "expected_creds"),
    [
        ("uc:databricks-uc", MlflowHostCreds(host="databricks-uc")),
        ("uc:http://localhost:8081", MlflowHostCreds(host="http://localhost:8081")),
        ("invalid_scheme:http://localhost:8081", MlflowException),
        ("databricks-uc", MlflowException),
    ],
)
def test_get_oss_host_creds(server_uri, expected_creds):
    with mock.patch(
        "mlflow.utils.oss_registry_utils.get_databricks_host_creds",
        return_value=MlflowHostCreds(host="databricks-uc"),
    ):
        if expected_creds == MlflowException:
            with pytest.raises(
                MlflowException, match="The scheme of the server_uri should be 'uc'"
            ):
                get_oss_host_creds(server_uri)
        else:
            actual_creds = get_oss_host_creds(server_uri)
            assert actual_creds == expected_creds


def test_get_databricks_host_creds():
    # Test case: When the scheme is "uc" and the new scheme is "_DATABRICKS_UNITY_CATALOG_SCHEME"
    server_uri = "uc:databricks-uc"
    with mock.patch(
        "mlflow.utils.oss_registry_utils.get_databricks_host_creds", return_value=mock.MagicMock()
    ) as mock_get_databricks_host_creds:
        get_oss_host_creds(server_uri)
        assert mock_get_databricks_host_creds.call_args_list == [mock.call("databricks-uc")]

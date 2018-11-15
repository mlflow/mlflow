import json

import boto3
import pytest

from tests.sagemaker.mock import mock_sagemaker


@pytest.fixture
def sagemaker_client():
    return boto3.client("sagemaker", region_name="us-west-2")


@mock_sagemaker
def test_created_endpoint_config_is_listed_by_list_endpoints_function(sagemaker_client):
    endpoint_config_name = "sample-config"

    sagemaker_client.create_endpoint_config(
        EndpointConfigName='sample-config',
        ProductionVariants=[
            {
                'VariantName': 'sample-variant',
                'ModelName': 'sample-model',
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m4.xlarge', 
                'InitialVariantWeight': 1.0, 
            },
        ],
    )

    endpoint_configs_response = sagemaker_client.list_endpoint_configs()
    assert "EndpointConfigs" in endpoint_configs_response
    endpoint_configs = endpoint_configs_response["EndpointConfigs"]
    assert all([
        "EndpointConfigName" in endpoint_config for endpoint_config in endpoint_configs])
    assert endpoint_config_name in [
        endpoint_config["EndpointConfigName"] for endpoint_config in endpoint_configs
    ]

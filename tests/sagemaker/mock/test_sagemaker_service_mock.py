import json

import boto3
import pytest

from tests.sagemaker.mock import mock_sagemaker


@pytest.fixture
def sagemaker_client():
    return boto3.client("sagemaker", region_name="us-west-2")


@mock_sagemaker
def sagemaker_model(sagemaker_client):
    


@mock_sagemaker
def test_create_model_with_valid_parameters_succeeds(sagemaker_client):



@mock_sagemaker
def test_created_endpoint_config_is_listed_by_list_endpoints_function(sagemaker_client):
    endpoint_config_name = "sample-config"

    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
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


@mock_sagemaker
def test_create_endpoint_config_returns_endpoint_arn_containing_config_name(sagemaker_client):
    endpoint_config_name = "sample-config"

    create_config_response = sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
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
    assert "EndpointConfigArn" in create_config_response
    assert endpoint_config_name in create_config_response["EndpointConfigArn"]


@mock_sagemaker
def test_creating_endpoint_config_with_name_already_in_use_raises_exception(sagemaker_client):
    endpoint_config_name = "sample-config"

    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
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

    with pytest.raises(ValueError):
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
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


@mock_sagemaker
def test_all_endpoint_configs_are_listed_after_creating_many_configs(sagemaker_client):
    endpoint_config_names = []

    for i in range(100):
        endpoint_config_name = "sample-config-{idx}".format(idx=i)
        endpoint_config_names.append(endpoint_config_name)

        sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
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

    listed_endpoint_configs = sagemaker_client.list_endpoint_configs()["EndpointConfigs"]
    listed_endpoint_config_names = [
            endpoint_config["EndpointConfigName"] 
            for endpoint_config in listed_endpoint_configs]
    for endpoint_config_name in endpoint_config_names:
        assert endpoint_config_name in listed_endpoint_config_names

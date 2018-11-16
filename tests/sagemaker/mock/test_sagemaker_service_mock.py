import json

import boto3
import pytest

from tests.sagemaker.mock import mock_sagemaker


@pytest.fixture
def sagemaker_client():
    return boto3.client("sagemaker", region_name="us-west-2")


def create_sagemaker_model(sagemaker_client, model_name):
    return sagemaker_client.create_model(
        ExecutionRoleArn='arn:aws:iam::012345678910:role/sample-role',
        ModelName=model_name,
        PrimaryContainer={
            'Image': '012345678910.dkr.ecr.us-west-2.amazonaws.com/sample-container', 
        }
    )


def create_endpoint_config(sagemaker_client, endpoint_config_name, model_name):
    return sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'sample-variant',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m4.xlarge',
                'InitialVariantWeight': 1.0,
            },
        ],
    )


@mock_sagemaker
def test_created_model_is_listed_by_list_models_function(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(
            sagemaker_client=sagemaker_client, model_name=model_name)

    models_response = sagemaker_client.list_models()
    assert "Models" in models_response 
    models = models_response["Models"]
    assert all(["ModelName" in model for model in models])
    assert model_name in [model["ModelName"] for model in models]


@mock_sagemaker
def test_create_model_returns_arn_containing_model_name(sagemaker_client):
    model_name = "sample-model"
    model_create_response = create_sagemaker_model(
            sagemaker_client=sagemaker_client, model_name=model_name)
    assert "ModelArn" in model_create_response
    assert model_name in model_create_response["ModelArn"]


@mock_sagemaker
def test_creating_model_with_name_already_in_use_raises_exception(sagemaker_client):
    model_name = "sample-model-name"

    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    with pytest.raises(ValueError):
        create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)


@mock_sagemaker
def test_all_models_are_listed_after_creating_many_models(sagemaker_client):
    model_names = []

    for i in range(100):
        model_name = "sample-model-{idx}".format(idx=i)
        model_names.append(model_name)

        create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    listed_models = sagemaker_client.list_models()["Models"]
    listed_model_names = [model["ModelName"] for model in listed_models]
    for model_name in model_names:
        assert model_name in listed_model_names 


@mock_sagemaker
def test_created_endpoint_config_is_listed_by_list_endpoints_function(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    endpoint_config_name = "sample-config"
    create_endpoint_config(
            sagemaker_client=sagemaker_client, 
            endpoint_config_name=endpoint_config_name, 
            model_name=model_name)

    endpoint_configs_response = sagemaker_client.list_endpoint_configs()
    assert "EndpointConfigs" in endpoint_configs_response
    endpoint_configs = endpoint_configs_response["EndpointConfigs"]
    assert all([
        "EndpointConfigName" in endpoint_config for endpoint_config in endpoint_configs])
    assert endpoint_config_name in [
        endpoint_config["EndpointConfigName"] for endpoint_config in endpoint_configs
    ]


@mock_sagemaker
def test_create_endpoint_config_returns_arn_containing_config_name(
        sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    endpoint_config_name = "sample-config"
    create_config_response = create_endpoint_config(
            sagemaker_client=sagemaker_client, 
            endpoint_config_name=endpoint_config_name, 
            model_name=model_name)
    
    assert "EndpointConfigArn" in create_config_response
    assert endpoint_config_name in create_config_response["EndpointConfigArn"]


@mock_sagemaker
def test_creating_endpoint_config_with_name_already_in_use_raises_exception(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    endpoint_config_name = "sample-config"
    create_endpoint_config(
            sagemaker_client=sagemaker_client, 
            endpoint_config_name=endpoint_config_name, 
            model_name=model_name)

    with pytest.raises(ValueError):
        create_endpoint_config(
                sagemaker_client=sagemaker_client, 
                endpoint_config_name=endpoint_config_name, 
                model_name=model_name)


@mock_sagemaker
def test_all_endpoint_configs_are_listed_after_creating_many_configs(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)
    endpoint_config_names = []

    for i in range(100):
        endpoint_config_name = "sample-config-{idx}".format(idx=i)
        endpoint_config_names.append(endpoint_config_name)

        create_endpoint_config(
                sagemaker_client=sagemaker_client, 
                endpoint_config_name=endpoint_config_name, 
                model_name=model_name)

    listed_endpoint_configs = sagemaker_client.list_endpoint_configs()["EndpointConfigs"]
    listed_endpoint_config_names = [
            endpoint_config["EndpointConfigName"] 
            for endpoint_config in listed_endpoint_configs]
    for endpoint_config_name in endpoint_config_names:
        assert endpoint_config_name in listed_endpoint_config_names


@mock_sagemaker
def test_created_endpoint_is_listed_by_list_endpoints_function(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    endpoint_config_name = "sample-config"
    create_endpoint_config(
            sagemaker_client=sagemaker_client, 
            endpoint_config_name=endpoint_config_name, 
            model_name=model_name)

    endpoint_name = "sample-endpoint"

    sagemaker_client.create_endpoint(
        EndpointConfigName=endpoint_config_name,
        EndpointName=endpoint_name,
        Tags=[
            {
                "Key": "Some Key",
                "Value": "Some Value",
            },
        ],
    )

    endpoints_response = sagemaker_client.list_endpoints()
    assert "Endpoints" in endpoints_response
    endpoints = endpoints_response["Endpoints"]
    assert all(["EndpointName" in endpoint for endpoint in endpoints])
    assert endpoint_name in [endpoint["EndpointName"] for endpoint in endpoints]


@mock_sagemaker
def test_create_endpoin_returns_arn_containing_endpoint_name(
        sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    endpoint_config_name = "sample-config"
    create_endpoint_config(
            sagemaker_client=sagemaker_client, 
            endpoint_config_name=endpoint_config_name, 
            model_name=model_name)

    endpoint_name = "sample-endpoint"

    create_endpoint_response = sagemaker_client.create_endpoint(
        EndpointConfigName=endpoint_config_name,
        EndpointName=endpoint_name,
        Tags=[
            {
                "Key": "Some Key",
                "Value": "Some Value",
            },
        ],
    )
    
    assert "EndpointArn" in create_endpoint_response 
    assert endpoint_name in create_endpoint_response["EndpointArn"]


@mock_sagemaker
def test_creating_endpoint_with_name_already_in_use_raises_exception(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    endpoint_config_name = "sample-config"
    create_endpoint_config(
            sagemaker_client=sagemaker_client, 
            endpoint_config_name=endpoint_config_name, 
            model_name=model_name)

    endpoint_name = "sample-endpoint"

    sagemaker_client.create_endpoint(
        EndpointConfigName=endpoint_config_name,
        EndpointName=endpoint_name,
        Tags=[
            {
                "Key": "Some Key",
                "Value": "Some Value",
            },
        ],
    )

    with pytest.raises(ValueError):
        sagemaker_client.create_endpoint(
            EndpointConfigName=endpoint_config_name,
            EndpointName=endpoint_name,
            Tags=[
                {
                    "Key": "Some Key",
                    "Value": "Some Value",
                },
            ],
        )


@mock_sagemaker
def test_all_endpoint_are_listed_after_creating_many_endpoints(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    endpoint_config_name = "sample-config"
    create_endpoint_config(
            sagemaker_client=sagemaker_client, 
            endpoint_config_name=endpoint_config_name, 
            model_name=model_name)

    endpoint_names = []

    for i in range(100):
        endpoint_name = "sample-endpoint-{idx}".format(idx=i)
        endpoint_names.append(endpoint_name)

        sagemaker_client.create_endpoint(
            EndpointConfigName=endpoint_config_name,
            EndpointName=endpoint_name,
            Tags=[
                {
                    "Key": "Some Key",
                    "Value": "Some Value",
                },
            ],
        )

    listed_endpoints = sagemaker_client.list_endpoints()["Endpoints"]
    listed_endpoint_names = [endpoint["EndpointName"] for endpoint in listed_endpoints]
    for endpoint_name in endpoint_names:
        assert endpoint_name in listed_endpoint_names




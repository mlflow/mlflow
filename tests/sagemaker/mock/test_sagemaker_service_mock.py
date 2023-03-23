import boto3
import pytest

from tests.helper_functions import set_boto_credentials  # pylint: disable=unused-import
from tests.sagemaker.mock import mock_sagemaker


@pytest.fixture
def sagemaker_client():
    return boto3.client("sagemaker", region_name="us-west-2")


def create_sagemaker_model(sagemaker_client, model_name):
    return sagemaker_client.create_model(
        ExecutionRoleArn="arn:aws:iam::012345678910:role/sample-role",
        ModelName=model_name,
        PrimaryContainer={
            "Image": "012345678910.dkr.ecr.us-west-2.amazonaws.com/sample-container",
        },
    )


def create_endpoint_config(sagemaker_client, endpoint_config_name, model_name):
    return sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "sample-variant",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": "ml.m4.xlarge",
                "InitialVariantWeight": 1.0,
            },
        ],
    )


@mock_sagemaker
def test_created_model_is_listed_by_list_models_function(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    models_response = sagemaker_client.list_models()
    assert "Models" in models_response
    models = models_response["Models"]
    assert all("ModelName" in model for model in models)
    assert model_name in [model["ModelName"] for model in models]


@mock_sagemaker
def test_create_model_returns_arn_containing_model_name(sagemaker_client):
    model_name = "sample-model"
    model_create_response = create_sagemaker_model(
        sagemaker_client=sagemaker_client, model_name=model_name
    )
    assert "ModelArn" in model_create_response
    assert model_name in model_create_response["ModelArn"]


@mock_sagemaker
def test_creating_model_with_name_already_in_use_raises_exception(sagemaker_client):
    model_name = "sample-model-name"

    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    with pytest.raises(ValueError, match="Attempted to create a model"):
        create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)


@mock_sagemaker
def test_all_models_are_listed_after_creating_many_models(sagemaker_client):
    model_names = []

    for i in range(100):
        model_name = f"sample-model-{i}"
        model_names.append(model_name)

        create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    listed_models = sagemaker_client.list_models()["Models"]
    listed_model_names = [model["ModelName"] for model in listed_models]
    for model_name in model_names:
        assert model_name in listed_model_names


@mock_sagemaker
def test_describe_model_response_contains_expected_attributes(sagemaker_client):
    model_name = "sample-model"
    execution_role_arn = "arn:aws:iam::012345678910:role/sample-role"
    primary_container = {
        "Image": "012345678910.dkr.ecr.us-west-2.amazonaws.com/sample-container",
    }

    sagemaker_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=execution_role_arn,
        PrimaryContainer=primary_container,
    )

    describe_model_response = sagemaker_client.describe_model(ModelName=model_name)
    assert "CreationTime" in describe_model_response
    assert "ModelArn" in describe_model_response
    assert "ExecutionRoleArn" in describe_model_response
    assert describe_model_response["ExecutionRoleArn"] == execution_role_arn
    assert "ModelName" in describe_model_response
    assert describe_model_response["ModelName"] == model_name
    assert "PrimaryContainer" in describe_model_response
    assert describe_model_response["PrimaryContainer"] == primary_container


@mock_sagemaker
def test_describe_model_throws_exception_for_nonexistent_model(sagemaker_client):
    with pytest.raises(ValueError, match="Attempted to describe a model"):
        sagemaker_client.describe_model(ModelName="nonexistent-model")


@mock_sagemaker
def test_model_is_no_longer_listed_after_deletion(sagemaker_client):
    model_name = "sample-model-name"

    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    sagemaker_client.delete_model(ModelName=model_name)

    listed_models = sagemaker_client.list_models()["Models"]
    listed_model_names = [model["ModelName"] for model in listed_models]
    assert model_name not in listed_model_names


@mock_sagemaker
def test_created_endpoint_config_is_listed_by_list_endpoints_function(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    endpoint_config_name = "sample-config"
    create_endpoint_config(
        sagemaker_client=sagemaker_client,
        endpoint_config_name=endpoint_config_name,
        model_name=model_name,
    )

    endpoint_configs_response = sagemaker_client.list_endpoint_configs()
    assert "EndpointConfigs" in endpoint_configs_response
    endpoint_configs = endpoint_configs_response["EndpointConfigs"]
    assert all("EndpointConfigName" in endpoint_config for endpoint_config in endpoint_configs)
    assert endpoint_config_name in [
        endpoint_config["EndpointConfigName"] for endpoint_config in endpoint_configs
    ]


@mock_sagemaker
def test_create_endpoint_config_returns_arn_containing_config_name(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    endpoint_config_name = "sample-config"
    create_config_response = create_endpoint_config(
        sagemaker_client=sagemaker_client,
        endpoint_config_name=endpoint_config_name,
        model_name=model_name,
    )

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
        model_name=model_name,
    )

    with pytest.raises(ValueError, match="Attempted to create an endpoint configuration"):
        create_endpoint_config(
            sagemaker_client=sagemaker_client,
            endpoint_config_name=endpoint_config_name,
            model_name=model_name,
        )


@mock_sagemaker
def test_all_endpoint_configs_are_listed_after_creating_many_configs(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)
    endpoint_config_names = []

    for i in range(100):
        endpoint_config_name = f"sample-config-{i}"
        endpoint_config_names.append(endpoint_config_name)

        create_endpoint_config(
            sagemaker_client=sagemaker_client,
            endpoint_config_name=endpoint_config_name,
            model_name=model_name,
        )

    listed_endpoint_configs = sagemaker_client.list_endpoint_configs()["EndpointConfigs"]
    listed_endpoint_config_names = [
        endpoint_config["EndpointConfigName"] for endpoint_config in listed_endpoint_configs
    ]
    for endpoint_config_name in endpoint_config_names:
        assert endpoint_config_name in listed_endpoint_config_names


@mock_sagemaker
def test_describe_endpoint_config_response_contains_expected_attributes(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    endpoint_config_name = "sample-config"
    production_variants = [
        {
            "VariantName": "sample-variant",
            "ModelName": model_name,
            "InitialInstanceCount": 1,
            "InstanceType": "ml.m4.xlarge",
            "InitialVariantWeight": 1.0,
        },
    ]
    async_inference_config = {
        "ClientConfig": {"MaxConcurrentInvocationsPerInstance": 4},
        "OutputConfig": {"S3OutputPath": "s3://bucket_name/", "NotificationConfig": {}},
    }
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=production_variants,
        AsyncInferenceConfig=async_inference_config,
    )

    describe_endpoint_config_response = sagemaker_client.describe_endpoint_config(
        EndpointConfigName=endpoint_config_name
    )
    assert "CreationTime" in describe_endpoint_config_response
    assert "EndpointConfigArn" in describe_endpoint_config_response
    assert "EndpointConfigName" in describe_endpoint_config_response
    assert describe_endpoint_config_response["EndpointConfigName"] == endpoint_config_name
    assert "ProductionVariants" in describe_endpoint_config_response
    assert describe_endpoint_config_response["ProductionVariants"] == production_variants
    assert "AsyncInferenceConfig" in describe_endpoint_config_response
    assert describe_endpoint_config_response["AsyncInferenceConfig"] == async_inference_config


@mock_sagemaker
def test_describe_endpoint_config_throws_exception_for_nonexistent_config(sagemaker_client):
    with pytest.raises(ValueError, match="Attempted to describe an endpoint config"):
        sagemaker_client.describe_endpoint_config(EndpointConfigName="nonexistent-config")


@mock_sagemaker
def test_endpoint_config_is_no_longer_listed_after_deletion(sagemaker_client):
    model_name = "sample-model-name"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    endpoint_config_name = "sample-config"
    create_endpoint_config(
        sagemaker_client=sagemaker_client,
        endpoint_config_name=endpoint_config_name,
        model_name=model_name,
    )

    sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)

    listed_endpoint_configs = sagemaker_client.list_endpoint_configs()["EndpointConfigs"]
    listed_endpoint_config_names = [
        endpoint_config["EndpointConfigName"] for endpoint_config in listed_endpoint_configs
    ]
    assert endpoint_config_name not in listed_endpoint_config_names


@mock_sagemaker
def test_created_endpoint_is_listed_by_list_endpoints_function(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    endpoint_config_name = "sample-config"
    create_endpoint_config(
        sagemaker_client=sagemaker_client,
        endpoint_config_name=endpoint_config_name,
        model_name=model_name,
    )

    endpoint_name = "sample-endpoint"

    sagemaker_client.create_endpoint(
        EndpointConfigName=endpoint_config_name,
        EndpointName=endpoint_name,
        Tags=[{"Key": "Some Key", "Value": "Some Value"}],
    )

    endpoints_response = sagemaker_client.list_endpoints()
    assert "Endpoints" in endpoints_response
    endpoints = endpoints_response["Endpoints"]
    assert all("EndpointName" in endpoint for endpoint in endpoints)
    assert endpoint_name in [endpoint["EndpointName"] for endpoint in endpoints]


@mock_sagemaker
def test_create_endpoint_returns_arn_containing_endpoint_name(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    endpoint_config_name = "sample-config"
    create_endpoint_config(
        sagemaker_client=sagemaker_client,
        endpoint_config_name=endpoint_config_name,
        model_name=model_name,
    )

    endpoint_name = "sample-endpoint"

    create_endpoint_response = sagemaker_client.create_endpoint(
        EndpointConfigName=endpoint_config_name,
        EndpointName=endpoint_name,
        Tags=[{"Key": "Some Key", "Value": "Some Value"}],
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
        model_name=model_name,
    )

    endpoint_name = "sample-endpoint"

    sagemaker_client.create_endpoint(
        EndpointConfigName=endpoint_config_name,
        EndpointName=endpoint_name,
        Tags=[{"Key": "Some Key", "Value": "Some Value"}],
    )

    with pytest.raises(ValueError, match="Attempted to create an endpoint"):
        sagemaker_client.create_endpoint(
            EndpointConfigName=endpoint_config_name,
            EndpointName=endpoint_name,
            Tags=[{"Key": "Some Key", "Value": "Some Value"}],
        )


@mock_sagemaker
def test_all_endpoint_are_listed_after_creating_many_endpoints(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    endpoint_config_name = "sample-config"
    create_endpoint_config(
        sagemaker_client=sagemaker_client,
        endpoint_config_name=endpoint_config_name,
        model_name=model_name,
    )

    endpoint_names = []

    for i in range(100):
        endpoint_name = f"sample-endpoint-{i}"
        endpoint_names.append(endpoint_name)

        sagemaker_client.create_endpoint(
            EndpointConfigName=endpoint_config_name,
            EndpointName=endpoint_name,
            Tags=[{"Key": "Some Key", "Value": "Some Value"}],
        )

    listed_endpoints = sagemaker_client.list_endpoints()["Endpoints"]
    listed_endpoint_names = [endpoint["EndpointName"] for endpoint in listed_endpoints]
    for endpoint_name in endpoint_names:
        assert endpoint_name in listed_endpoint_names


@mock_sagemaker
def test_describe_endpoint_response_contains_expected_attributes(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    endpoint_config_name = "sample-config"
    production_variants = [
        {
            "VariantName": "sample-variant",
            "ModelName": model_name,
            "InitialInstanceCount": 1,
            "InstanceType": "ml.m4.xlarge",
            "InitialVariantWeight": 1.0,
        },
    ]
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=production_variants,
    )

    endpoint_name = "sample-endpoint"
    sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )

    describe_endpoint_response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    assert "CreationTime" in describe_endpoint_response
    assert "LastModifiedTime" in describe_endpoint_response
    assert "EndpointArn" in describe_endpoint_response
    assert "EndpointStatus" in describe_endpoint_response
    assert "ProductionVariants" in describe_endpoint_response


@mock_sagemaker
def test_describe_endpoint_throws_exception_for_nonexistent_endpoint(sagemaker_client):
    with pytest.raises(ValueError, match="Attempted to describe an endpoint"):
        sagemaker_client.describe_endpoint(EndpointName="nonexistent-endpoint")


@mock_sagemaker
def test_endpoint_is_no_longer_listed_after_deletion(sagemaker_client):
    model_name = "sample-model-name"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    endpoint_config_name = "sample-config"
    create_endpoint_config(
        sagemaker_client=sagemaker_client,
        endpoint_config_name=endpoint_config_name,
        model_name=model_name,
    )

    endpoint_name = "sample-endpoint"
    sagemaker_client.create_endpoint(
        EndpointConfigName=endpoint_config_name,
        EndpointName=endpoint_name,
    )

    sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

    listed_endpoints = sagemaker_client.list_endpoints()["Endpoints"]
    listed_endpoint_names = [endpoint["EndpointName"] for endpoint in listed_endpoints]
    assert endpoint_name not in listed_endpoint_names


@mock_sagemaker
def test_update_endpoint_modifies_config_correctly(sagemaker_client):
    model_name = "sample-model-name"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    first_endpoint_config_name = "sample-config-1"
    create_endpoint_config(
        sagemaker_client=sagemaker_client,
        endpoint_config_name=first_endpoint_config_name,
        model_name=model_name,
    )

    second_endpoint_config_name = "sample-config-2"
    create_endpoint_config(
        sagemaker_client=sagemaker_client,
        endpoint_config_name=second_endpoint_config_name,
        model_name=model_name,
    )

    endpoint_name = "sample-endpoint"
    sagemaker_client.create_endpoint(
        EndpointConfigName=first_endpoint_config_name,
        EndpointName=endpoint_name,
    )

    first_describe_endpoint_response = sagemaker_client.describe_endpoint(
        EndpointName=endpoint_name
    )
    assert first_describe_endpoint_response["EndpointConfigName"] == first_endpoint_config_name

    sagemaker_client.update_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=second_endpoint_config_name
    )

    second_describe_endpoint_response = sagemaker_client.describe_endpoint(
        EndpointName=endpoint_name
    )
    assert second_describe_endpoint_response["EndpointConfigName"] == second_endpoint_config_name


@mock_sagemaker
def test_update_endpoint_with_nonexistent_config_throws_exception(sagemaker_client):
    model_name = "sample-model-name"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    endpoint_config_name = "sample-config"
    create_endpoint_config(
        sagemaker_client=sagemaker_client,
        endpoint_config_name=endpoint_config_name,
        model_name=model_name,
    )

    endpoint_name = "sample-endpoint"
    sagemaker_client.create_endpoint(
        EndpointConfigName=endpoint_config_name,
        EndpointName=endpoint_name,
    )

    with pytest.raises(ValueError, match="Attempted to update an endpoint"):
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name, EndpointConfigName="nonexistent-config"
        )


@mock_sagemaker
def test_created_transform_job_is_listed_by_list_transform_jobs_function(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    transform_input = {
        "DataSource": {"S3DataSource": {"S3DataType": "Some Data Type", "S3Uri": "Some Input Uri"}}
    }

    transform_output = {"S3OutputPath": "Some Output Path"}

    transform_resources = {"InstanceType": "Some Instance Type", "InstanceCount": 1}

    job_name = "sample-job"

    sagemaker_client.create_transform_job(
        TransformJobName=job_name,
        ModelName=model_name,
        TransformInput=transform_input,
        TransformOutput=transform_output,
        TransformResources=transform_resources,
        Tags=[{"Key": "Some Key", "Value": "Some Value"}],
    )

    transform_jobs_response = sagemaker_client.list_transform_jobs()
    assert "TransformJobSummaries" in transform_jobs_response
    transform_jobs = transform_jobs_response["TransformJobSummaries"]
    assert all("TransformJobName" in transform_job for transform_job in transform_jobs)
    assert job_name in [transform_job["TransformJobName"] for transform_job in transform_jobs]


@mock_sagemaker
def test_create_transform_job_returns_arn_containing_transform_job_name(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    transform_input = {
        "DataSource": {"S3DataSource": {"S3DataType": "Some Data Type", "S3Uri": "Some Input Uri"}}
    }

    transform_output = {"S3OutputPath": "Some Output Path"}

    transform_resources = {"InstanceType": "Some Instance Type", "InstanceCount": 1}

    job_name = "sample-job"

    create_transform_job_response = sagemaker_client.create_transform_job(
        TransformJobName=job_name,
        ModelName=model_name,
        TransformInput=transform_input,
        TransformOutput=transform_output,
        TransformResources=transform_resources,
        Tags=[{"Key": "Some Key", "Value": "Some Value"}],
    )

    assert "TransformJobArn" in create_transform_job_response
    assert job_name in create_transform_job_response["TransformJobArn"]


@mock_sagemaker
def test_creating_transform_job_with_name_already_in_use_raises_exception(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    transform_input = {
        "DataSource": {"S3DataSource": {"S3DataType": "Some Data Type", "S3Uri": "Some Input Uri"}}
    }

    transform_output = {"S3OutputPath": "Some Output Path"}

    transform_resources = {"InstanceType": "Some Instance Type", "InstanceCount": 1}

    job_name = "sample-job"

    sagemaker_client.create_transform_job(
        TransformJobName=job_name,
        ModelName=model_name,
        TransformInput=transform_input,
        TransformOutput=transform_output,
        TransformResources=transform_resources,
        Tags=[{"Key": "Some Key", "Value": "Some Value"}],
    )

    with pytest.raises(ValueError, match="Attempted to create a transform job"):
        sagemaker_client.create_transform_job(
            TransformJobName=job_name,
            ModelName=model_name,
            TransformInput=transform_input,
            TransformOutput=transform_output,
            TransformResources=transform_resources,
            Tags=[{"Key": "Some Key", "Value": "Some Value"}],
        )


@mock_sagemaker
def test_all_transform_jobs_are_listed_after_creating_many_transform_jobs(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    transform_input = {
        "DataSource": {"S3DataSource": {"S3DataType": "Some Data Type", "S3Uri": "Some Input Uri"}}
    }

    transform_output = {"S3OutputPath": "Some Output Path"}

    transform_resources = {"InstanceType": "Some Instance Type", "InstanceCount": 1}

    job_names = []

    for i in range(100):
        job_name = f"sample-job-{i}"
        job_names.append(job_name)

        sagemaker_client.create_transform_job(
            TransformJobName=job_name,
            ModelName=model_name,
            TransformInput=transform_input,
            TransformOutput=transform_output,
            TransformResources=transform_resources,
            Tags=[{"Key": "Some Key", "Value": "Some Value"}],
        )

    listed_transform_jobs = sagemaker_client.list_transform_jobs()["TransformJobSummaries"]
    listed_transform_job_names = [
        transform_job["TransformJobName"] for transform_job in listed_transform_jobs
    ]
    for job_name in job_names:
        assert job_name in listed_transform_job_names


@mock_sagemaker
def test_describe_transform_job_response_contains_expected_attributes(sagemaker_client):
    model_name = "sample-model"
    create_sagemaker_model(sagemaker_client=sagemaker_client, model_name=model_name)

    transform_input = {
        "DataSource": {"S3DataSource": {"S3DataType": "Some Data Type", "S3Uri": "Some Input Uri"}}
    }

    transform_output = {"S3OutputPath": "Some Output Path"}

    transform_resources = {"InstanceType": "Some Instance Type", "InstanceCount": 1}

    job_name = "sample-job"

    sagemaker_client.create_transform_job(
        TransformJobName=job_name,
        ModelName=model_name,
        TransformInput=transform_input,
        TransformOutput=transform_output,
        TransformResources=transform_resources,
        Tags=[{"Key": "Some Key", "Value": "Some Value"}],
    )

    describe_transform_job_response = sagemaker_client.describe_transform_job(
        TransformJobName=job_name
    )
    assert "TransformJobName" in describe_transform_job_response
    assert "CreationTime" in describe_transform_job_response
    assert "TransformJobArn" in describe_transform_job_response
    assert "TransformJobStatus" in describe_transform_job_response
    assert "ModelName" in describe_transform_job_response


@mock_sagemaker
def test_describe_transform_job_throws_exception_for_nonexistent_transform_job(sagemaker_client):
    with pytest.raises(ValueError, match="Attempted to describe a transform job"):
        sagemaker_client.describe_transform_job(TransformJobName="nonexistent-job")

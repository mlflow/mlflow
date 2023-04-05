import os
import pytest
import time
from collections import namedtuple
from io import BytesIO
from unittest import mock

import json
import boto3
import botocore
import numpy as np
import pandas as pd
from click.testing import CliRunner
from sklearn.linear_model import LogisticRegression
from moto.core import DEFAULT_ACCOUNT_ID

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.sagemaker as mfs
from mlflow.deployments.cli import commands as cli_commands
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import (
    ErrorCode,
    RESOURCE_DOES_NOT_EXIST,
    INVALID_PARAMETER_VALUE,
    INTERNAL_ERROR,
)
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri

from tests.helper_functions import set_boto_credentials  # pylint: disable=unused-import
from tests.sagemaker.mock import mock_sagemaker, Endpoint, EndpointOperation

TrainedModel = namedtuple("TrainedModel", ["model_path", "run_id", "model_uri"])


@pytest.fixture
def pretrained_model():
    model_path = "model"
    with mlflow.start_run():
        X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 1, 0])
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(X, y)
        mlflow.sklearn.log_model(lr, model_path)
        run_id = mlflow.active_run().info.run_id
        model_uri = "runs:/" + run_id + "/" + model_path
        return TrainedModel(model_path, run_id, model_uri)


@pytest.fixture
def sagemaker_client():
    return boto3.client("sagemaker", region_name="us-west-2")


@pytest.fixture
def sagemaker_deployment_client():
    return mfs.SageMakerDeploymentClient(
        "sagemaker:/us-west-2/arn:aws:iam::123456789012:role/assumed_role"
    )


def create_sagemaker_deployment_through_cli(
    app_name, model_uri, region_name, env=None, config=None
):
    if env is None:
        env = {"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}
    if config is not None:
        _config = []
        for c in config:
            _config += ["-C", c]
    else:
        _config = []
    result = CliRunner(env=env).invoke(
        cli_commands,
        [
            "create",
            "--target",
            f"sagemaker:/{region_name}",
            "--name",
            app_name,
            "--model-uri",
            model_uri,
        ]
        + _config,
    )
    assert result.exit_code == 0


def get_sagemaker_backend(region_name):
    return mock_sagemaker.backends[DEFAULT_ACCOUNT_ID][region_name]


def mock_sagemaker_aws_services(fn):
    from functools import wraps
    from moto import mock_s3, mock_ecr, mock_sts, mock_iam

    @mock_ecr
    @mock_iam
    @mock_s3
    @mock_sagemaker
    @mock_sts
    @wraps(fn)
    def mock_wrapper(*args, **kwargs):
        # Create an ECR repository for the `mlflow-pyfunc` SageMaker docker image
        ecr_client = boto3.client("ecr", region_name="us-west-2")
        ecr_client.create_repository(repositoryName=mfs.DEFAULT_IMAGE_NAME)

        # Create the moto IAM role
        role_policy = """
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "*",
                    "Resource": "*"
                }
            ]
        }
        """
        iam_client = boto3.client("iam", region_name="us-west-2")
        iam_client.create_role(RoleName="moto", AssumeRolePolicyDocument=role_policy)

        # Create IAM role to be assumed (could be in another AWS account)
        iam_client.create_role(RoleName="assumed_role", AssumeRolePolicyDocument=role_policy)
        return fn(*args, **kwargs)

    return mock_wrapper


def test_initialize_sagemaker_deployment_client_with_only_target_name():
    plugin = mfs.SageMakerDeploymentClient("sagemaker")

    assert plugin.region_name == mfs.DEFAULT_REGION_NAME
    assert plugin.assumed_role_arn is None


def test_initialize_sagemaker_deployment_client_with_empty_path():
    plugin = mfs.SageMakerDeploymentClient("sagemaker:/")

    assert plugin.region_name == mfs.DEFAULT_REGION_NAME
    assert plugin.assumed_role_arn is None


def test_initialize_sagemaker_deployment_client_with_region_name():
    plugin = mfs.SageMakerDeploymentClient("sagemaker:/us-east-1")

    assert plugin.region_name == "us-east-1"
    assert plugin.assumed_role_arn is None


def test_initialize_sagemaker_deployment_client_with_region_name_and_iam_role_arn():
    plugin = mfs.SageMakerDeploymentClient(
        "sagemaker:/us-east-1/////////arn:aws:iam::123456789012:role/dummy.company.com/assumed_role"
    )

    assert plugin.region_name == "us-east-1"
    assert (
        plugin.assumed_role_arn == "arn:aws:iam::123456789012:role/dummy.company.com/assumed_role"
    )


def test_init_sagemaker_deployment_client_with_iam_role_arn_but_no_region_name_raises_exception():
    match = "A region name must be provided when the target_uri contains a role ARN."
    with pytest.raises(MlflowException, match=match) as exc:
        mfs.SageMakerDeploymentClient(
            "sagemaker:/arn:aws:iam::123456789012:role/dummy.company.com/assumed_role"
        )

    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.parametrize("field_name", ["instance_count", "timeout_seconds"])
def test__apply_custom_config_converts_from_string_to_int_for_int_fields(
    field_name, sagemaker_deployment_client
):
    config = {field_name: 0}
    custom_config = {field_name: "5"}

    sagemaker_deployment_client._apply_custom_config(config, custom_config)

    assert config[field_name] == 5


@pytest.mark.parametrize("field_name", ["synchronous", "archive"])
def test__apply_custom_config_converts_from_string_to_bool_for_bool_fields(
    field_name, sagemaker_deployment_client
):
    config = {field_name: True}
    custom_config = {field_name: "False"}

    sagemaker_deployment_client._apply_custom_config(config, custom_config)

    assert config[field_name] is False


def test__apply_custom_config_converts_from_string_to_dict_for_dict_fields(
    sagemaker_deployment_client,
):
    vpc_config = {
        "SecurityGroupIds": [
            "sg-123456abc",
        ],
        "Subnets": [
            "subnet-123456abc",
        ],
    }
    env_config = {
        "GUNICORN_CMD_ARGS": "--timeout=60",
    }
    tags_config = {
        "tag1": "value1",
    }
    config = {"vpc_config": None, "env": None, "tags": None}
    custom_config = {
        "vpc_config": json.dumps(vpc_config),
        "env": json.dumps(env_config),
        "tags": json.dumps(tags_config),
    }

    sagemaker_deployment_client._apply_custom_config(config, custom_config)

    assert config["vpc_config"] == vpc_config
    assert config["env"] == env_config
    assert config["tags"] == tags_config


def test__apply_custom_config_does_not_change_type_of_string_fields(sagemaker_deployment_client):
    config = {"region_name": "us-west-1"}
    custom_config = {"region_name": "us-east-3"}

    sagemaker_deployment_client._apply_custom_config(config, custom_config)

    assert config["region_name"] == "us-east-3"


@mock_sagemaker_aws_services
def test_create_deployment_with_non_existent_assume_role_arn_raises_exception(pretrained_model):
    plugin = mfs.SageMakerDeploymentClient(
        "sagemaker:/us-west-2/arn:aws:iam::123456789012:role/non-existent-role-arn"
    )
    match = (
        r"An error occurred \(NoSuchEntity\) when calling the GetRole "
        r"operation: Role non-existent-role-arn not found"
    )
    with pytest.raises(botocore.exceptions.ClientError, match=match):
        plugin.create_deployment(
            name="bad_assume_role_arn",
            model_uri=pretrained_model.model_uri,
        )


@mock_sagemaker_aws_services
def test_create_deployment_with_assume_role_arn(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    app_name = "deploy_with_assume_role_arn"
    sagemaker_deployment_client.create_deployment(
        name=app_name,
        model_uri=pretrained_model.model_uri,
    )
    assert app_name in [
        endpoint["EndpointName"] for endpoint in sagemaker_client.list_endpoints()["Endpoints"]
    ]


@mock_sagemaker_aws_services
def test_create_deployment_with_async_config(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    app_name = "deploy_with_async_config"
    expected_async_inference_config = {
        "ClientConfig": {"MaxConcurrentInvocationsPerInstance": 4},
        "OutputConfig": {"S3OutputPath": "s3://bucket_name/", "NotificationConfig": {}},
    }
    sagemaker_deployment_client.create_deployment(
        name=app_name,
        model_uri=pretrained_model.model_uri,
        config={"async_inference_config": expected_async_inference_config},
    )
    configs = sagemaker_client.list_endpoint_configs()
    target_config = None
    for config in configs["EndpointConfigs"]:
        if app_name in config["EndpointConfigName"]:
            target_config = config
    if target_config is None:
        raise Exception("Endpoint config not found")
    endpoint_config = sagemaker_client.describe_endpoint_config(
        EndpointConfigName=target_config["EndpointConfigName"]
    )
    assert "AsyncInferenceConfig" in endpoint_config
    assert endpoint_config["AsyncInferenceConfig"] == expected_async_inference_config


@mock_sagemaker_aws_services
def test_create_deployment_without_async_config(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    app_name = "deploy_without_endpoint_config"
    sagemaker_deployment_client.create_deployment(
        name=app_name,
        model_uri=pretrained_model.model_uri,
    )
    configs = sagemaker_client.list_endpoint_configs()
    target_config = None
    for config in configs["EndpointConfigs"]:
        if app_name in config["EndpointConfigName"]:
            target_config = config
    if target_config is None:
        raise Exception("Endpoint config not found")
    assert "AsyncInferenceConfig" not in target_config


@mock_sagemaker_aws_services
def test_update_deployment_with_async_config_when_endpoint_exists(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    app_name = "update_deploy_with_async_config"
    expected_async_inference_config = {
        "ClientConfig": {"MaxConcurrentInvocationsPerInstance": 4},
        "OutputConfig": {"S3OutputPath": "s3://bucket_name/", "NotificationConfig": {}},
    }
    sagemaker_deployment_client.create_deployment(
        name=app_name, model_uri=pretrained_model.model_uri
    )
    sagemaker_deployment_client.update_deployment(
        name=app_name,
        model_uri=pretrained_model.model_uri,
        config={"async_inference_config": expected_async_inference_config},
    )
    configs = sagemaker_client.list_endpoint_configs()
    target_config = None
    for config in configs["EndpointConfigs"]:
        if app_name in config["EndpointConfigName"]:
            target_config = config
    if target_config is None:
        raise Exception("Endpoint config not found")
    endpoint_config = sagemaker_client.describe_endpoint_config(
        EndpointConfigName=target_config["EndpointConfigName"]
    )
    assert "AsyncInferenceConfig" in endpoint_config
    assert endpoint_config["AsyncInferenceConfig"] == expected_async_inference_config


@mock_sagemaker_aws_services
def test_update_deployment_without_async_config(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    app_name = "deploy_without_async_config"
    sagemaker_deployment_client.update_deployment(
        name=app_name,
        model_uri=pretrained_model.model_uri,
    )
    configs = sagemaker_client.list_endpoint_configs()
    target_config = None
    for config in configs["EndpointConfigs"]:
        if app_name in config["EndpointConfigName"]:
            target_config = config
    if target_config is None:
        raise Exception("Endpoint config not found")
    assert "AsyncInferenceConfig" not in target_config


def test_create_deployment_with_unsupported_flavor_raises_exception(
    pretrained_model, sagemaker_deployment_client
):
    unsupported_flavor = "this is not a valid flavor"
    match = "The specified flavor: `this is not a valid flavor` is not supported for deployment"
    with pytest.raises(MlflowException, match=match) as exc:
        sagemaker_deployment_client.create_deployment(
            name="bad_flavor", model_uri=pretrained_model.model_uri, flavor=unsupported_flavor
        )

    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_create_deployment_with_missing_flavor_raises_exception(
    pretrained_model, sagemaker_deployment_client
):
    missing_flavor = "mleap"
    match = "The specified model does not contain the specified deployment flavor"
    with pytest.raises(MlflowException, match=match) as exc:
        sagemaker_deployment_client.create_deployment(
            name="missing-flavor", model_uri=pretrained_model.model_uri, flavor=missing_flavor
        )

    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_create_deployment_of_model_with_no_supported_flavors_raises_exception(
    pretrained_model, sagemaker_deployment_client
):
    logged_model_path = _download_artifact_from_uri(pretrained_model.model_uri)
    model_config_path = os.path.join(logged_model_path, "MLmodel")
    model_config = Model.load(model_config_path)
    del model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
    model_config.save(path=model_config_path)

    match = "The specified model does not contain any of the supported flavors for deployment"
    with pytest.raises(MlflowException, match=match) as exc:
        sagemaker_deployment_client.create_deployment(
            name="missing-flavor", model_uri=logged_model_path, flavor=None
        )

    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_attempting_to_deploy_in_asynchronous_mode_without_archiving_throws_exception(
    pretrained_model, sagemaker_deployment_client
):
    with pytest.raises(MlflowException, match="Resources must be archived") as exc:
        sagemaker_deployment_client.create_deployment(
            name="test-app",
            model_uri=pretrained_model.model_uri,
            config={"archive": False, "synchronous": False},
        )

    assert "Resources must be archived" in exc.value.message
    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@mock_sagemaker_aws_services
def test_create_deployment_create_sagemaker_and_s3_resources_with_expected_tags_from_local(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    expected_tags = [{"Key": "key1", "Value": "value1"}, {"Key": "key2", "Value": "value2"}]

    name = "test-app"
    with mock.patch.dict(os.environ, {}, clear=True):
        sagemaker_deployment_client.create_deployment(
            name=name,
            model_uri=pretrained_model.model_uri,
            config=dict(
                tags={"key1": "value1", "key2": "value2"},
            ),
        )

    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=name)
    endpoint_production_variants = endpoint_description["ProductionVariants"]
    assert len(endpoint_production_variants) == 1
    model_name = endpoint_production_variants[0]["VariantName"]
    description = sagemaker_client.describe_model(ModelName=model_name)

    tags = sagemaker_client.list_tags(ResourceArn=description["ModelArn"])

    # Extra tags exist besides the ones we set, so avoid strict equality
    assert all(tag in tags["Tags"] for tag in expected_tags)


@pytest.mark.parametrize("proxies_enabled", [True, False])
@mock_sagemaker_aws_services
def test_create_deployment_create_sagemaker_and_s3_resources_with_expected_names_and_env_from_local(
    proxies_enabled, pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    expected_model_environment = {
        "MLFLOW_DEPLOYMENT_FLAVOR_NAME": "python_function",
        "SERVING_ENVIRONMENT": "SageMaker",
        "GUNCORN_CMD_ARGS": '"--timeout 60"',
        "DISABLE_NGINX": "true",
    }

    if proxies_enabled:
        proxy_variables = {
            "http_proxy": "http://user:password@proxy.example.net:1234",
            "https_proxy": "https://user:password@proxy.example.net:1234",
            "no_proxy": "localhost",
        }
        expected_model_environment.update(proxy_variables)
        name = "test-app-proxies"
        with mock.patch.dict(os.environ, proxy_variables, clear=True):
            sagemaker_deployment_client.create_deployment(
                name=name,
                model_uri=pretrained_model.model_uri,
                config=dict(
                    env={"DISABLE_NGINX": "true", "GUNCORN_CMD_ARGS": '"--timeout 60"'},
                ),
            )
    else:
        name = "test-app"
        with mock.patch.dict(os.environ, {}, clear=True):
            sagemaker_deployment_client.create_deployment(
                name=name,
                model_uri=pretrained_model.model_uri,
                config=dict(
                    env={"DISABLE_NGINX": "true", "GUNCORN_CMD_ARGS": '"--timeout 60"'},
                ),
            )

    region_name = sagemaker_client.meta.region_name
    s3_client = boto3.client("s3", region_name=region_name)
    default_bucket = mfs._get_default_s3_bucket(region_name)
    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=name)
    endpoint_production_variants = endpoint_description["ProductionVariants"]
    assert len(endpoint_production_variants) == 1
    model_name = endpoint_production_variants[0]["VariantName"]
    assert model_name in [model["ModelName"] for model in sagemaker_client.list_models()["Models"]]
    object_names = [
        entry["Key"] for entry in s3_client.list_objects(Bucket=default_bucket)["Contents"]
    ]
    assert any(model_name in object_name for object_name in object_names)
    assert any(
        name in config["EndpointConfigName"]
        for config in sagemaker_client.list_endpoint_configs()["EndpointConfigs"]
    )
    assert name in [
        endpoint["EndpointName"] for endpoint in sagemaker_client.list_endpoints()["Endpoints"]
    ]
    model_environment = sagemaker_client.describe_model(ModelName=model_name)["PrimaryContainer"][
        "Environment"
    ]

    assert model_environment == expected_model_environment


@pytest.mark.parametrize("proxies_enabled", [True, False])
@mock_sagemaker_aws_services
def test_deploy_cli_creates_sagemaker_and_s3_resources_with_expected_names_and_env_from_local(
    proxies_enabled, pretrained_model, sagemaker_client
):
    region_name = sagemaker_client.meta.region_name
    environment_variables = {"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}
    override_environment_variables = {"DISABLE_NGINX": "true", "GUNCORN_CMD_ARGS": '"--timeout 60"'}
    expected_model_environment = {
        "MLFLOW_DEPLOYMENT_FLAVOR_NAME": "python_function",
        "SERVING_ENVIRONMENT": "SageMaker",
        "GUNCORN_CMD_ARGS": '"--timeout 60"',
        "DISABLE_NGINX": "true",
    }

    if proxies_enabled:
        proxy_variables = {
            "http_proxy": "http://user:password@proxy.example.net:1234",
            "https_proxy": "http://user:password@proxy.example.net:1234",
            "no_proxy": "localhost",
        }
        expected_model_environment.update(proxy_variables)
        app_name = "test-app-proxies"
        create_sagemaker_deployment_through_cli(
            app_name,
            pretrained_model.model_uri,
            region_name,
            {**environment_variables, **proxy_variables},
            config=["env={}".format(json.dumps(override_environment_variables))],
        )
    else:
        proxy_variables = {
            "http_proxy": None,
            "https_proxy": None,
            "no_proxy": None,
        }
        app_name = "test-app"
        create_sagemaker_deployment_through_cli(
            app_name,
            pretrained_model.model_uri,
            region_name,
            {**environment_variables, **proxy_variables},
            config=["env={}".format(json.dumps(override_environment_variables))],
        )

    s3_client = boto3.client("s3", region_name=region_name)
    default_bucket = mfs._get_default_s3_bucket(region_name)
    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=app_name)
    endpoint_production_variants = endpoint_description["ProductionVariants"]
    assert len(endpoint_production_variants) == 1
    model_name = endpoint_production_variants[0]["VariantName"]
    assert model_name in [model["ModelName"] for model in sagemaker_client.list_models()["Models"]]
    object_names = [
        entry["Key"] for entry in s3_client.list_objects(Bucket=default_bucket)["Contents"]
    ]
    assert any(model_name in object_name for object_name in object_names)
    assert any(
        app_name in config["EndpointConfigName"]
        for config in sagemaker_client.list_endpoint_configs()["EndpointConfigs"]
    )
    assert app_name in [
        endpoint["EndpointName"] for endpoint in sagemaker_client.list_endpoints()["Endpoints"]
    ]
    model_environment = sagemaker_client.describe_model(ModelName=model_name)["PrimaryContainer"][
        "Environment"
    ]

    assert model_environment == expected_model_environment


@mock_sagemaker_aws_services
def test_deploy_cli_creates_sagemaker_and_s3_resources_with_expected_tags_from_local(
    pretrained_model, sagemaker_client
):
    expected_tags = [{"Key": "key1", "Value": "value1"}, {"Key": "key2", "Value": "value2"}]
    region_name = sagemaker_client.meta.region_name

    app_name = "test-app"
    create_sagemaker_deployment_through_cli(
        app_name,
        pretrained_model.model_uri,
        region_name,
        env=None,
        config=["tags={}".format(json.dumps({"key1": "value1", "key2": "value2"}))],
    )

    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=app_name)
    endpoint_production_variants = endpoint_description["ProductionVariants"]
    assert len(endpoint_production_variants) == 1
    model_name = endpoint_production_variants[0]["VariantName"]
    description = sagemaker_client.describe_model(ModelName=model_name)

    tags = sagemaker_client.list_tags(ResourceArn=description["ModelArn"])

    # Extra tags exist besides the ones we set, so avoid strict equality
    assert all(tag in tags["Tags"] for tag in expected_tags)


@pytest.mark.parametrize("proxies_enabled", [True, False])
@mock_sagemaker_aws_services
def test_create_deployment_creates_sagemaker_and_s3_resources_with_expected_names_and_env_from_s3(
    proxies_enabled, pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    local_model_path = _download_artifact_from_uri(pretrained_model.model_uri)
    artifact_path = "model"
    region_name = sagemaker_client.meta.region_name
    default_bucket = mfs._get_default_s3_bucket(region_name)
    s3_artifact_repo = S3ArtifactRepository(f"s3://{default_bucket}")
    s3_artifact_repo.log_artifacts(local_model_path, artifact_path=artifact_path)
    model_s3_uri = "s3://{bucket_name}/{artifact_path}".format(
        bucket_name=default_bucket, artifact_path=pretrained_model.model_path
    )
    expected_model_environment = {
        "MLFLOW_DEPLOYMENT_FLAVOR_NAME": "python_function",
        "SERVING_ENVIRONMENT": "SageMaker",
    }

    if proxies_enabled:
        proxy_variables = {
            "http_proxy": "http://user:password@proxy.example.net:1234",
            "https_proxy": "http://user:password@proxy.example.net:1234",
            "no_proxy": "localhost",
        }
        expected_model_environment.update(proxy_variables)
        name = "test-app-proxies"
        with mock.patch.dict(os.environ, proxy_variables, clear=True):
            sagemaker_deployment_client.create_deployment(
                name=name,
                model_uri=model_s3_uri,
            )
    else:
        name = "test-app"
        with mock.patch.dict(os.environ, {}, clear=True):
            sagemaker_deployment_client.create_deployment(
                name=name,
                model_uri=model_s3_uri,
            )

    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=name)
    endpoint_production_variants = endpoint_description["ProductionVariants"]
    assert len(endpoint_production_variants) == 1
    model_name = endpoint_production_variants[0]["VariantName"]
    assert model_name in [model["ModelName"] for model in sagemaker_client.list_models()["Models"]]

    s3_client = boto3.client("s3", region_name=region_name)
    object_names = [
        entry["Key"] for entry in s3_client.list_objects(Bucket=default_bucket)["Contents"]
    ]
    assert any(model_name in object_name for object_name in object_names)
    assert any(
        name in config["EndpointConfigName"]
        for config in sagemaker_client.list_endpoint_configs()["EndpointConfigs"]
    )
    assert name in [
        endpoint["EndpointName"] for endpoint in sagemaker_client.list_endpoints()["Endpoints"]
    ]
    model_environment = sagemaker_client.describe_model(ModelName=model_name)["PrimaryContainer"][
        "Environment"
    ]

    assert model_environment == expected_model_environment


@pytest.mark.parametrize("proxies_enabled", [True, False])
@mock_sagemaker_aws_services
def test_deploy_cli_creates_sagemaker_and_s3_resources_with_expected_names_and_env_from_s3(
    proxies_enabled, pretrained_model, sagemaker_client
):
    local_model_path = _download_artifact_from_uri(pretrained_model.model_uri)
    artifact_path = "model"
    region_name = sagemaker_client.meta.region_name
    default_bucket = mfs._get_default_s3_bucket(region_name)
    s3_artifact_repo = S3ArtifactRepository(f"s3://{default_bucket}")
    s3_artifact_repo.log_artifacts(local_model_path, artifact_path=artifact_path)
    model_s3_uri = "s3://{bucket_name}/{artifact_path}".format(
        bucket_name=default_bucket, artifact_path=pretrained_model.model_path
    )
    environment_variables = {"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}
    expected_model_environment = {
        "MLFLOW_DEPLOYMENT_FLAVOR_NAME": "python_function",
        "SERVING_ENVIRONMENT": "SageMaker",
    }

    if proxies_enabled:
        proxy_variables = {
            "http_proxy": "http://user:password@proxy.example.net:1234",
            "https_proxy": "https://user:password@proxy.example.net:1234",
            "no_proxy": "localhost",
        }
        expected_model_environment.update(proxy_variables)
        app_name = "test-app-proxies"
        create_sagemaker_deployment_through_cli(
            app_name,
            model_s3_uri,
            region_name,
            {**environment_variables, **proxy_variables},
        )
    else:
        proxy_variables = {
            "http_proxy": None,
            "https_proxy": None,
            "no_proxy": None,
        }
        app_name = "test-app"
        create_sagemaker_deployment_through_cli(
            app_name,
            model_s3_uri,
            region_name,
            {**environment_variables, **proxy_variables},
        )

    region_name = sagemaker_client.meta.region_name
    s3_client = boto3.client("s3", region_name=region_name)
    default_bucket = mfs._get_default_s3_bucket(region_name)
    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=app_name)
    endpoint_production_variants = endpoint_description["ProductionVariants"]
    assert len(endpoint_production_variants) == 1
    model_name = endpoint_production_variants[0]["VariantName"]
    assert model_name in [model["ModelName"] for model in sagemaker_client.list_models()["Models"]]
    object_names = [
        entry["Key"] for entry in s3_client.list_objects(Bucket=default_bucket)["Contents"]
    ]
    assert any(model_name in object_name for object_name in object_names)
    assert any(
        app_name in config["EndpointConfigName"]
        for config in sagemaker_client.list_endpoint_configs()["EndpointConfigs"]
    )
    assert app_name in [
        endpoint["EndpointName"] for endpoint in sagemaker_client.list_endpoints()["Endpoints"]
    ]
    model_environment = sagemaker_client.describe_model(ModelName=model_name)["PrimaryContainer"][
        "Environment"
    ]

    assert model_environment == expected_model_environment


@mock_sagemaker_aws_services
def test_create_deployment_with_preexisting_name_throws_exception(
    pretrained_model, sagemaker_deployment_client
):
    name = "test-app"
    sagemaker_deployment_client.create_deployment(
        name=name,
        model_uri=pretrained_model.model_uri,
    )

    with pytest.raises(
        MlflowException, match="an application with the same name already exists"
    ) as exc:
        sagemaker_deployment_client.create_deployment(
            name=name,
            model_uri=pretrained_model.model_uri,
        )

    assert "an application with the same name already exists" in exc.value.message
    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@mock_sagemaker_aws_services
def test_create_deployment_in_sync_mode_waits_for_endpoint_creation_to_complete_before_returning(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    endpoint_creation_latency = 10
    get_sagemaker_backend(sagemaker_client.meta.region_name).set_endpoint_update_latency(
        endpoint_creation_latency
    )

    name = "test-app"
    deployment_start_time = time.time()
    sagemaker_deployment_client.create_deployment(
        name=name,
        model_uri=pretrained_model.model_uri,
        config={"synchronous": True},
    )
    deployment_end_time = time.time()

    assert (deployment_end_time - deployment_start_time) >= endpoint_creation_latency
    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=name)
    assert endpoint_description["EndpointStatus"] == Endpoint.STATUS_IN_SERVICE


@mock_sagemaker_aws_services
def test_create_deployment_in_asynchronous_mode_returns_before_endpoint_creation_completes(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    endpoint_creation_latency = 10
    get_sagemaker_backend(sagemaker_client.meta.region_name).set_endpoint_update_latency(
        endpoint_creation_latency
    )

    name = "test-app"
    deployment_start_time = time.time()
    sagemaker_deployment_client.create_deployment(
        name=name,
        model_uri=pretrained_model.model_uri,
        config={"synchronous": False, "archive": True},
    )
    deployment_end_time = time.time()

    assert (deployment_end_time - deployment_start_time) < endpoint_creation_latency
    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=name)
    assert endpoint_description["EndpointStatus"] == Endpoint.STATUS_CREATING


@mock_sagemaker_aws_services
def test_update_deployment_in_asynchronous_mode_returns_before_endpoint_creation_completes(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    endpoint_update_latency = 10
    get_sagemaker_backend(sagemaker_client.meta.region_name).set_endpoint_update_latency(
        endpoint_update_latency
    )

    name = "test-app"
    sagemaker_deployment_client.create_deployment(
        name=name,
        model_uri=pretrained_model.model_uri,
        config={"synchronous": True},
    )

    update_start_time = time.time()
    sagemaker_deployment_client.update_deployment(
        name=name,
        model_uri=pretrained_model.model_uri,
        config={"mode": mfs.DEPLOYMENT_MODE_REPLACE, "synchronous": False, "archive": True},
    )
    update_end_time = time.time()

    assert (update_end_time - update_start_time) < endpoint_update_latency
    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=name)
    assert endpoint_description["EndpointStatus"] == Endpoint.STATUS_UPDATING


@mock_sagemaker_aws_services
def test_create_deployment_throws_exception_after_endpoint_creation_fails(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    endpoint_creation_latency = 10
    sagemaker_backend = get_sagemaker_backend(sagemaker_client.meta.region_name)
    sagemaker_backend.set_endpoint_update_latency(endpoint_creation_latency)

    boto_caller = botocore.client.BaseClient._make_api_call

    def fail_endpoint_creations(self, operation_name, operation_kwargs):
        """
        Processes all boto3 client operations according to the following rules:
        - If the operation is an endpoint creation, create the endpoint and set its status to
          ``Endpoint.STATUS_FAILED``.
        - Else, execute the client operation as normal
        """
        result = boto_caller(self, operation_name, operation_kwargs)
        if operation_name == "CreateEndpoint":
            endpoint_name = operation_kwargs["EndpointName"]
            sagemaker_backend.set_endpoint_latest_operation(
                endpoint_name=endpoint_name,
                operation=EndpointOperation.create_unsuccessful(
                    latency_seconds=endpoint_creation_latency
                ),
            )
        return result

    with mock.patch(
        "botocore.client.BaseClient._make_api_call", new=fail_endpoint_creations
    ), pytest.raises(MlflowException, match="deployment operation failed") as exc:
        sagemaker_deployment_client.create_deployment(
            name="test-app",
            model_uri=pretrained_model.model_uri,
        )

    assert "deployment operation failed" in exc.value.message
    assert exc.value.error_code == ErrorCode.Name(INTERNAL_ERROR)


@mock_sagemaker_aws_services
def test_create_deployment_in_replace_mode_removes_preexisting_models_from_endpoint(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    name = "test-app"
    sagemaker_deployment_client.create_deployment(
        name=name,
        model_uri=pretrained_model.model_uri,
    )

    sagemaker_deployment_client.update_deployment(
        name=name,
        model_uri=pretrained_model.model_uri,
        config={
            "mode": mfs.DEPLOYMENT_MODE_ADD,
            "archive": True,
            "synchronous": False,
        },
    )

    endpoint_response_before_replacement = sagemaker_client.describe_endpoint(EndpointName=name)
    endpoint_config_name_before_replacement = endpoint_response_before_replacement[
        "EndpointConfigName"
    ]
    endpoint_config_response_before_replacement = sagemaker_client.describe_endpoint_config(
        EndpointConfigName=endpoint_config_name_before_replacement
    )
    production_variants_before_replacement = endpoint_config_response_before_replacement[
        "ProductionVariants"
    ]
    deployed_models_before_replacement = [
        variant["ModelName"] for variant in production_variants_before_replacement
    ]

    sagemaker_deployment_client.update_deployment(
        name=name,
        model_uri=pretrained_model.model_uri,
        config={
            "mode": mfs.DEPLOYMENT_MODE_REPLACE,
            "archive": True,
            "synchronous": False,
        },
    )

    endpoint_response_after_replacement = sagemaker_client.describe_endpoint(EndpointName=name)
    endpoint_config_name_after_replacement = endpoint_response_after_replacement[
        "EndpointConfigName"
    ]
    endpoint_config_response_after_replacement = sagemaker_client.describe_endpoint_config(
        EndpointConfigName=endpoint_config_name_after_replacement
    )
    production_variants_after_replacement = endpoint_config_response_after_replacement[
        "ProductionVariants"
    ]
    deployed_models_after_replacement = [
        variant["ModelName"] for variant in production_variants_after_replacement
    ]
    assert len(deployed_models_after_replacement) == 1
    assert all(
        model_name not in deployed_models_after_replacement
        for model_name in deployed_models_before_replacement
    )


@mock_sagemaker_aws_services
def test_create_deployment_in_add_mode_adds_new_model_to_existing_endpoint(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    name = "test-app"
    sagemaker_deployment_client.create_deployment(
        name=name,
        model_uri=pretrained_model.model_uri,
    )

    sagemaker_deployment_client.update_deployment(
        name=name,
        model_uri=pretrained_model.model_uri,
        config={
            "mode": mfs.DEPLOYMENT_MODE_ADD,
            "archive": True,
            "synchronous": False,
        },
    )
    models_added = 2

    endpoint_response = sagemaker_client.describe_endpoint(EndpointName=name)
    endpoint_config_name = endpoint_response["EndpointConfigName"]
    endpoint_config_response = sagemaker_client.describe_endpoint_config(
        EndpointConfigName=endpoint_config_name
    )
    production_variants = endpoint_config_response["ProductionVariants"]
    assert len(production_variants) == models_added


def test_update_deployment_with_create_mode_raises_exception(
    pretrained_model, sagemaker_deployment_client
):
    with pytest.raises(MlflowException, match="Invalid mode") as exc:
        sagemaker_deployment_client.update_deployment(
            name="invalid mode",
            model_uri=pretrained_model.model_uri,
            config={"mode": mfs.DEPLOYMENT_MODE_CREATE},
        )

    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@mock_sagemaker_aws_services
def test_update_deployment_in_add_mode_adds_new_model_to_existing_endpoint(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    name = "test-app"
    sagemaker_deployment_client.create_deployment(
        name=name,
        model_uri=pretrained_model.model_uri,
    )
    models_added = 1
    for _ in range(11):
        sagemaker_deployment_client.update_deployment(
            name=name,
            model_uri=pretrained_model.model_uri,
            config={
                "mode": mfs.DEPLOYMENT_MODE_ADD,
                "archive": True,
                "synchronous": False,
            },
        )
        models_added += 1

    endpoint_response = sagemaker_client.describe_endpoint(EndpointName=name)
    endpoint_config_name = endpoint_response["EndpointConfigName"]
    endpoint_config_response = sagemaker_client.describe_endpoint_config(
        EndpointConfigName=endpoint_config_name
    )
    production_variants = endpoint_config_response["ProductionVariants"]
    assert len(production_variants) == models_added


@mock_sagemaker_aws_services
def test_update_deployment_in_replace_mode_removes_preexisting_models_from_endpoint(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    name = "test-app"
    sagemaker_deployment_client.create_deployment(
        name=name,
        model_uri=pretrained_model.model_uri,
    )

    for _ in range(11):
        sagemaker_deployment_client.update_deployment(
            name=name,
            model_uri=pretrained_model.model_uri,
            config={
                "mode": mfs.DEPLOYMENT_MODE_ADD,
                "archive": True,
                "synchronous": False,
            },
        )

    endpoint_response_before_replacement = sagemaker_client.describe_endpoint(EndpointName=name)
    endpoint_config_name_before_replacement = endpoint_response_before_replacement[
        "EndpointConfigName"
    ]
    endpoint_config_response_before_replacement = sagemaker_client.describe_endpoint_config(
        EndpointConfigName=endpoint_config_name_before_replacement
    )
    production_variants_before_replacement = endpoint_config_response_before_replacement[
        "ProductionVariants"
    ]
    deployed_models_before_replacement = [
        variant["ModelName"] for variant in production_variants_before_replacement
    ]

    sagemaker_deployment_client.update_deployment(
        name=name,
        model_uri=pretrained_model.model_uri,
        config={
            "mode": mfs.DEPLOYMENT_MODE_REPLACE,
            "archive": True,
            "synchronous": False,
        },
    )

    endpoint_response_after_replacement = sagemaker_client.describe_endpoint(EndpointName=name)
    endpoint_config_name_after_replacement = endpoint_response_after_replacement[
        "EndpointConfigName"
    ]
    endpoint_config_response_after_replacement = sagemaker_client.describe_endpoint_config(
        EndpointConfigName=endpoint_config_name_after_replacement
    )
    production_variants_after_replacement = endpoint_config_response_after_replacement[
        "ProductionVariants"
    ]
    deployed_models_after_replacement = [
        variant["ModelName"] for variant in production_variants_after_replacement
    ]
    assert len(deployed_models_after_replacement) == 1
    assert all(
        model_name not in deployed_models_after_replacement
        for model_name in deployed_models_before_replacement
    )


@mock_sagemaker_aws_services
def test_update_deployment_in_replace_mode_throws_exception_after_endpoint_update_fails(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    endpoint_update_latency = 5
    sagemaker_backend = get_sagemaker_backend(sagemaker_client.meta.region_name)
    sagemaker_backend.set_endpoint_update_latency(endpoint_update_latency)

    name = "test-app"
    sagemaker_deployment_client.create_deployment(
        name=name,
        model_uri=pretrained_model.model_uri,
    )

    boto_caller = botocore.client.BaseClient._make_api_call

    def fail_endpoint_updates(self, operation_name, operation_kwargs):
        """
        Processes all boto3 client operations according to the following rules:
        - If the operation is an endpoint update, update the endpoint and set its status to
          ``Endpoint.STATUS_FAILED``.
        - Else, execute the client operation as normal
        """
        result = boto_caller(self, operation_name, operation_kwargs)
        if operation_name == "UpdateEndpoint":
            endpoint_name = operation_kwargs["EndpointName"]
            sagemaker_backend.set_endpoint_latest_operation(
                endpoint_name=endpoint_name,
                operation=EndpointOperation.update_unsuccessful(
                    latency_seconds=endpoint_update_latency
                ),
            )
        return result

    with mock.patch(
        "botocore.client.BaseClient._make_api_call", new=fail_endpoint_updates
    ), pytest.raises(MlflowException, match="deployment operation failed") as exc:
        sagemaker_deployment_client.update_deployment(
            name=name,
            model_uri=pretrained_model.model_uri,
            config={"mode": mfs.DEPLOYMENT_MODE_REPLACE},
        )
    assert exc.value.error_code == ErrorCode.Name(INTERNAL_ERROR)


@mock_sagemaker_aws_services
def test_update_deployment_waits_for_endpoint_update_completion_before_deleting_resources(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    endpoint_update_latency = 10
    sagemaker_backend = get_sagemaker_backend(sagemaker_client.meta.region_name)
    sagemaker_backend.set_endpoint_update_latency(endpoint_update_latency)

    name = "test-app"
    sagemaker_deployment_client.create_deployment(
        name=name,
        model_uri=pretrained_model.model_uri,
    )
    endpoint_config_name_before_replacement = sagemaker_client.describe_endpoint(EndpointName=name)[
        "EndpointConfigName"
    ]

    boto_caller = botocore.client.BaseClient._make_api_call
    update_start_time = time.time()

    def validate_deletes(self, operation_name, operation_kwargs):
        """
        Processes all boto3 client operations according to the following rules:
        - If the operation deletes an S3 or SageMaker resource, ensure that the deletion was
          initiated after the completion of the endpoint update
        - Else, execute the client operation as normal
        """
        result = boto_caller(self, operation_name, operation_kwargs)
        if "Delete" in operation_name:
            # Confirm that a successful endpoint update occurred prior to the invocation of this
            # delete operation
            endpoint_info = sagemaker_client.describe_endpoint(EndpointName=name)
            assert endpoint_info["EndpointStatus"] == Endpoint.STATUS_IN_SERVICE
            assert endpoint_info["EndpointConfigName"] != endpoint_config_name_before_replacement
            assert time.time() - update_start_time >= endpoint_update_latency
        return result

    with mock.patch("botocore.client.BaseClient._make_api_call", new=validate_deletes):
        sagemaker_deployment_client.update_deployment(
            name=name,
            model_uri=pretrained_model.model_uri,
            config={"mode": mfs.DEPLOYMENT_MODE_REPLACE, "archive": False},
        )


@mock_sagemaker_aws_services
def test_update_deployment_in_replace_mode_with_archiving_does_not_delete_resources(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    region_name = sagemaker_client.meta.region_name
    sagemaker_backend = get_sagemaker_backend(region_name)
    sagemaker_backend.set_endpoint_update_latency(5)

    name = "test-app"
    sagemaker_deployment_client.create_deployment(
        name=name,
        model_uri=pretrained_model.model_uri,
    )

    s3_client = boto3.client("s3", region_name=region_name)
    default_bucket = mfs._get_default_s3_bucket(region_name)
    object_names_before_replacement = [
        entry["Key"] for entry in s3_client.list_objects(Bucket=default_bucket)["Contents"]
    ]
    endpoint_configs_before_replacement = [
        config["EndpointConfigName"]
        for config in sagemaker_client.list_endpoint_configs()["EndpointConfigs"]
    ]
    models_before_replacement = [
        model["ModelName"] for model in sagemaker_client.list_models()["Models"]
    ]

    model_uri = "runs:/{run_id}/{artifact_path}".format(
        run_id=pretrained_model.run_id, artifact_path=pretrained_model.model_path
    )
    sk_model = mlflow.sklearn.load_model(model_uri=model_uri)
    new_artifact_path = "model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=sk_model, artifact_path=new_artifact_path)
        new_model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=new_artifact_path
        )
    sagemaker_deployment_client.update_deployment(
        name=name,
        model_uri=new_model_uri,
        config={"mode": mfs.DEPLOYMENT_MODE_REPLACE, "archive": True, "synchronous": True},
    )

    object_names_after_replacement = [
        entry["Key"] for entry in s3_client.list_objects(Bucket=default_bucket)["Contents"]
    ]
    endpoint_configs_after_replacement = [
        config["EndpointConfigName"]
        for config in sagemaker_client.list_endpoint_configs()["EndpointConfigs"]
    ]
    models_after_replacement = [
        model["ModelName"] for model in sagemaker_client.list_models()["Models"]
    ]
    assert all(
        object_name in object_names_after_replacement
        for object_name in object_names_before_replacement
    )
    assert all(
        endpoint_config in endpoint_configs_after_replacement
        for endpoint_config in endpoint_configs_before_replacement
    )
    assert all(model in models_after_replacement for model in models_before_replacement)


@mock_sagemaker_aws_services
def test_deploy_cli_updates_sagemaker_and_s3_resources_in_replace_mode(
    pretrained_model, sagemaker_client
):
    app_name = "test-app"
    region_name = sagemaker_client.meta.region_name
    create_sagemaker_deployment_through_cli(app_name, pretrained_model.model_uri, region_name)

    result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
        cli_commands,
        [
            "update",
            "--target",
            f"sagemaker:/{region_name}",
            "--name",
            app_name,
            "--model-uri",
            pretrained_model.model_uri,
        ],
    )
    assert result.exit_code == 0

    s3_client = boto3.client("s3", region_name=region_name)
    default_bucket = mfs._get_default_s3_bucket(region_name)
    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=app_name)
    endpoint_production_variants = endpoint_description["ProductionVariants"]
    assert len(endpoint_production_variants) == 1
    model_name = endpoint_production_variants[0]["VariantName"]
    assert model_name in [model["ModelName"] for model in sagemaker_client.list_models()["Models"]]
    object_names = [
        entry["Key"] for entry in s3_client.list_objects(Bucket=default_bucket)["Contents"]
    ]
    assert any(model_name in object_name for object_name in object_names)
    assert any(
        app_name in config["EndpointConfigName"]
        for config in sagemaker_client.list_endpoint_configs()["EndpointConfigs"]
    )
    assert app_name in [
        endpoint["EndpointName"] for endpoint in sagemaker_client.list_endpoints()["Endpoints"]
    ]
    model_environment = sagemaker_client.describe_model(ModelName=model_name)["PrimaryContainer"][
        "Environment"
    ]
    expected_model_environment = {
        "MLFLOW_DEPLOYMENT_FLAVOR_NAME": "python_function",
        "SERVING_ENVIRONMENT": "SageMaker",
    }
    if os.getenv("http_proxy") is not None:
        expected_model_environment.update({"http_proxy": os.environ["http_proxy"]})

    if os.getenv("https_proxy") is not None:
        expected_model_environment.update({"https_proxy": os.environ["https_proxy"]})

    if os.getenv("no_proxy") is not None:
        expected_model_environment.update({"no_proxy": os.environ["no_proxy"]})

    assert model_environment == expected_model_environment


@mock_sagemaker_aws_services
def test_deploy_cli_updates_sagemaker_and_s3_resources_in_add_mode(
    pretrained_model, sagemaker_client
):
    app_name = "test-app"
    region_name = sagemaker_client.meta.region_name
    create_sagemaker_deployment_through_cli(app_name, pretrained_model.model_uri, region_name)

    result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
        cli_commands,
        [
            "update",
            "--target",
            f"sagemaker:/{region_name}",
            "--name",
            app_name,
            "--model-uri",
            pretrained_model.model_uri,
            "--config",
            f"mode={mfs.DEPLOYMENT_MODE_ADD}",
        ],
    )
    assert result.exit_code == 0

    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=app_name)
    endpoint_production_variants = endpoint_description["ProductionVariants"]
    assert len(endpoint_production_variants) == 2


def test_delete_deployment_in_asynchronous_mode_without_archiving_raises_exception(
    sagemaker_deployment_client,
):
    with pytest.raises(MlflowException, match="Resources must be archived") as exc:
        sagemaker_deployment_client.delete_deployment(
            name="dummy", config={"archive": False, "synchronous": False}
        )

    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@mock_sagemaker_aws_services
def test_delete_deployment_synchronous_mode_without_archiving_deletes_all_resources(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    name = "test-app"
    region_name = sagemaker_client.meta.region_name

    sagemaker_deployment_client.create_deployment(
        name=name, model_uri=pretrained_model.model_uri, config={"region_name": region_name}
    )

    sagemaker_deployment_client.delete_deployment(
        name=name, config={"archive": False, "synchronous": True, "region_name": region_name}
    )

    s3_client = boto3.client("s3", region_name=region_name)
    default_bucket = mfs._get_default_s3_bucket(region_name)
    s3_objects = s3_client.list_objects_v2(Bucket=default_bucket)
    endpoints = sagemaker_client.list_endpoints()
    endpoint_configs = sagemaker_client.list_endpoint_configs()
    models = sagemaker_client.list_models()

    assert s3_objects["KeyCount"] == 0
    assert len(endpoints["Endpoints"]) == 0
    assert len(endpoint_configs["EndpointConfigs"]) == 0
    assert len(models["Models"]) == 0


@mock_sagemaker_aws_services
def test_delete_deployment_synchronous_with_archiving_only_deletes_endpoint(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    name = "test-app"
    region_name = sagemaker_client.meta.region_name

    sagemaker_deployment_client.create_deployment(
        name=name, model_uri=pretrained_model.model_uri, config={"region_name": region_name}
    )

    sagemaker_deployment_client.delete_deployment(
        name=name, config={"archive": True, "synchronous": True, "region_name": region_name}
    )

    s3_client = boto3.client("s3", region_name=region_name)
    default_bucket = mfs._get_default_s3_bucket(region_name)
    s3_objects = s3_client.list_objects_v2(Bucket=default_bucket)
    endpoints = sagemaker_client.list_endpoints()
    endpoint_configs = sagemaker_client.list_endpoint_configs()
    models = sagemaker_client.list_models()

    assert s3_objects["KeyCount"] > 0
    assert len(endpoints["Endpoints"]) == 0
    assert len(endpoint_configs["EndpointConfigs"]) > 0
    assert len(models["Models"]) > 0


@mock_sagemaker_aws_services
def test_deploy_cli_deletes_sagemaker_deployment(pretrained_model, sagemaker_client):
    app_name = "test-app"
    region_name = sagemaker_client.meta.region_name
    create_sagemaker_deployment_through_cli(app_name, pretrained_model.model_uri, region_name)

    result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
        cli_commands,
        [
            "delete",
            "--target",
            "sagemaker",
            "--name",
            app_name,
            "--config",
            f"region_name={region_name}",
        ],
    )
    assert result.exit_code == 0

    response = sagemaker_client.list_endpoints()
    assert len(response["Endpoints"]) == 0


@mock_sagemaker_aws_services
def test_get_deployment_successful(pretrained_model, sagemaker_client):
    name = "test-app"
    region_name = sagemaker_client.meta.region_name
    sagemaker_deployment_client = mfs.SageMakerDeploymentClient(f"sagemaker:/{region_name}")
    sagemaker_deployment_client.create_deployment(
        name=name, model_uri=pretrained_model.model_uri, config={"region_name": region_name}
    )

    endpoint_description = sagemaker_deployment_client.get_deployment(name)

    expected_description = sagemaker_client.describe_endpoint(EndpointName=name)
    assert endpoint_description == expected_description


@mock_sagemaker_aws_services
def test_get_deployment_with_assumed_role_arn(
    pretrained_model, sagemaker_client, sagemaker_deployment_client
):
    name = "test-app"
    sagemaker_deployment_client.create_deployment(name=name, model_uri=pretrained_model.model_uri)

    endpoint_description = sagemaker_deployment_client.get_deployment(name)

    expected_description = sagemaker_client.describe_endpoint(EndpointName=name)
    assert endpoint_description == expected_description


@mock_sagemaker_aws_services
def test_get_deployment_non_existent_deployment():
    sagemaker_deployment_client = mfs.SageMakerDeploymentClient("sagemaker:/us-west-2")

    with pytest.raises(MlflowException, match="There was an error while"):
        sagemaker_deployment_client.get_deployment("non-existent app")


@mock_sagemaker_aws_services
def test_deploy_cli_gets_sagemaker_deployment(pretrained_model, sagemaker_client):
    app_name = "test-app"
    region_name = sagemaker_client.meta.region_name
    create_sagemaker_deployment_through_cli(app_name, pretrained_model.model_uri, region_name)

    result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
        cli_commands,
        [
            "get",
            "--target",
            f"sagemaker:/{region_name}",
            "--name",
            app_name,
        ],
    )

    assert result.exit_code == 0


@mock_sagemaker_aws_services
def test_list_deployments_returns_all_endpoints(pretrained_model, sagemaker_client):
    region_name = sagemaker_client.meta.region_name
    sagemaker_deployment_client = mfs.SageMakerDeploymentClient(f"sagemaker:/{region_name}")
    sagemaker_deployment_client.create_deployment(
        name="test-app-1",
        model_uri=pretrained_model.model_uri,
        config={"region_name": region_name},
    )
    sagemaker_deployment_client.create_deployment(
        name="test-app-2",
        model_uri=pretrained_model.model_uri,
        config={"region_name": region_name},
    )

    endpoints = sagemaker_deployment_client.list_deployments()

    assert len(endpoints) == 2
    assert endpoints[0]["EndpointName"] == "test-app-1"
    assert endpoints[1]["EndpointName"] == "test-app-2"


@mock_sagemaker_aws_services
def test_list_deployments_with_assumed_role_arn(pretrained_model, sagemaker_deployment_client):
    sagemaker_deployment_client.create_deployment(
        name="test-app-1",
        model_uri=pretrained_model.model_uri,
    )
    sagemaker_deployment_client.create_deployment(
        name="test-app-2",
        model_uri=pretrained_model.model_uri,
    )

    endpoints = sagemaker_deployment_client.list_deployments()

    assert len(endpoints) == 2
    assert endpoints[0]["EndpointName"] == "test-app-1"
    assert endpoints[1]["EndpointName"] == "test-app-2"


@mock_sagemaker_aws_services
def test_deploy_cli_list_sagemaker_deployments(pretrained_model, sagemaker_client):
    region_name = sagemaker_client.meta.region_name
    create_sagemaker_deployment_through_cli("test-app-1", pretrained_model.model_uri, region_name)
    create_sagemaker_deployment_through_cli("test-app-2", pretrained_model.model_uri, region_name)

    result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
        cli_commands,
        [
            "list",
            "--target",
            f"sagemaker:/{region_name}",
        ],
    )

    assert result.exit_code == 0


@mock_sagemaker_aws_services
def test_predict_with_dataframe_input_output(sagemaker_deployment_client):
    input_df = pd.DataFrame(data=[[1, 2]], columns=["a", "b"])
    output_df = pd.DataFrame({"1": ["2", ".", "3"]})
    boto_caller = botocore.client.BaseClient._make_api_call

    def mock_invoke_endpoint(self, operation_name, operation_kwargs):
        if operation_name == "InvokeEndpoint":
            assert operation_kwargs["Body"] == json.dumps(
                {"dataframe_split": input_df.to_dict(orient="split")}
            )
            output_json = json.dumps({"predictions": output_df.to_dict(orient="records")})
            result = {"Body": BytesIO(bytes(output_json, encoding="utf-8"))}
        else:
            result = boto_caller(self, operation_name, operation_kwargs)
        return result

    with mock.patch("botocore.client.BaseClient._make_api_call", new=mock_invoke_endpoint):
        result = sagemaker_deployment_client.predict("test", input_df).get_predictions()
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, output_df)


@mock_sagemaker_aws_services
def test_predict_with_array_input_output(sagemaker_deployment_client):
    boto_caller = botocore.client.BaseClient._make_api_call

    def mock_invoke_endpoint(self, operation_name, operation_kwargs):
        if operation_name == "InvokeEndpoint":
            assert operation_kwargs["Body"] == json.dumps({"instances": list(range(10))})
            result = {"Body": BytesIO(b'{ "predictions": [1,2,3]}')}
        else:
            result = boto_caller(self, operation_name, operation_kwargs)
        return result

    with mock.patch("botocore.client.BaseClient._make_api_call", new=mock_invoke_endpoint):
        result = sagemaker_deployment_client.predict("test", np.array(range(10))).get_predictions()

        assert isinstance(result, pd.DataFrame)
        assert list(result[0]) == [1, 2, 3]

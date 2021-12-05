import os
import pytest
import time
from collections import namedtuple
from unittest import mock

import boto3
import botocore
import numpy as np
from click.testing import CliRunner
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.sagemaker as mfs
import mlflow.sagemaker.cli as mfscli
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


def get_sagemaker_backend(region_name):
    return mock_sagemaker.backends[region_name]


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

        # Create IAM role to be asssumed (could be in another AWS account)
        iam_client.create_role(RoleName="assumed_role", AssumeRolePolicyDocument=role_policy)
        return fn(*args, **kwargs)

    return mock_wrapper


@mock_sagemaker_aws_services
def test_assume_role_and_get_credentials():
    assumed_role_credentials = mfs._assume_role_and_get_credentials(
        assume_role_arn="arn:aws:iam::123456789012:role/assumed_role"
    )
    assert "aws_access_key_id" in assumed_role_credentials.keys()
    assert "aws_secret_access_key" in assumed_role_credentials.keys()
    assert "aws_session_token" in assumed_role_credentials.keys()
    assert len(assumed_role_credentials["aws_session_token"]) == 356
    assert assumed_role_credentials["aws_session_token"].startswith("FQoGZXIvYXdzE")
    assert len(assumed_role_credentials["aws_access_key_id"]) == 20
    assert assumed_role_credentials["aws_access_key_id"].startswith("ASIA")
    assert len(assumed_role_credentials["aws_secret_access_key"]) == 40


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deployment_with_non_existent_assume_role_arn_raises_exception(pretrained_model):

    match = (
        r"An error occurred \(NoSuchEntity\) when calling the GetRole "
        r"operation: Role non-existent-role-arn not found"
    )
    with pytest.raises(botocore.exceptions.ClientError, match=match):
        mfs.deploy(
            app_name="bad_assume_role_arn",
            model_uri=pretrained_model.model_uri,
            assume_role_arn="arn:aws:iam::123456789012:role/non-existent-role-arn",
        )


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deployment_with_assume_role_arn(pretrained_model, sagemaker_client):
    app_name = "deploy_with_assume_role_arn"
    mfs.deploy(
        app_name=app_name,
        model_uri=pretrained_model.model_uri,
        assume_role_arn="arn:aws:iam::123456789012:role/assumed_role",
    )
    assert app_name in [
        endpoint["EndpointName"] for endpoint in sagemaker_client.list_endpoints()["Endpoints"]
    ]


@pytest.mark.large
def test_deployment_with_unsupported_flavor_raises_exception(pretrained_model):
    unsupported_flavor = "this is not a valid flavor"
    match = "The specified flavor: `this is not a valid flavor` is not supported for deployment"
    with pytest.raises(MlflowException, match=match) as exc:
        mfs.deploy(
            app_name="bad_flavor", model_uri=pretrained_model.model_uri, flavor=unsupported_flavor
        )

    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.large
def test_deployment_with_missing_flavor_raises_exception(pretrained_model):
    missing_flavor = "mleap"
    match = "The specified model does not contain the specified deployment flavor"
    with pytest.raises(MlflowException, match=match) as exc:
        mfs.deploy(
            app_name="missing-flavor", model_uri=pretrained_model.model_uri, flavor=missing_flavor
        )

    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


@pytest.mark.large
def test_deployment_of_model_with_no_supported_flavors_raises_exception(pretrained_model):
    logged_model_path = _download_artifact_from_uri(pretrained_model.model_uri)
    model_config_path = os.path.join(logged_model_path, "MLmodel")
    model_config = Model.load(model_config_path)
    del model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
    model_config.save(path=model_config_path)

    match = "The specified model does not contain any of the supported flavors for deployment"
    with pytest.raises(MlflowException, match=match) as exc:
        mfs.deploy(app_name="missing-flavor", model_uri=logged_model_path, flavor=None)

    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


@pytest.mark.large
def test_validate_deployment_flavor_validates_python_function_flavor_successfully(pretrained_model):
    model_config_path = os.path.join(
        _download_artifact_from_uri(pretrained_model.model_uri), "MLmodel"
    )
    model_config = Model.load(model_config_path)
    mfs._validate_deployment_flavor(model_config=model_config, flavor=mlflow.pyfunc.FLAVOR_NAME)


@pytest.mark.large
def test_get_preferred_deployment_flavor_obtains_valid_flavor_from_model(pretrained_model):
    model_config_path = os.path.join(
        _download_artifact_from_uri(pretrained_model.model_uri), "MLmodel"
    )
    model_config = Model.load(model_config_path)

    selected_flavor = mfs._get_preferred_deployment_flavor(model_config=model_config)

    assert selected_flavor in mfs.SUPPORTED_DEPLOYMENT_FLAVORS
    assert selected_flavor in model_config.flavors


@pytest.mark.large
def test_attempting_to_deploy_in_asynchronous_mode_without_archiving_throws_exception(
    pretrained_model,
):
    with pytest.raises(MlflowException, match="Resources must be archived") as exc:
        mfs.deploy(
            app_name="test-app",
            model_uri=pretrained_model.model_uri,
            mode=mfs.DEPLOYMENT_MODE_CREATE,
            archive=False,
            synchronous=False,
        )

    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_creates_sagemaker_and_s3_resources_with_expected_names_and_env_from_local(
    pretrained_model, sagemaker_client
):
    app_name = "test-app"
    mfs.deploy(
        app_name=app_name, model_uri=pretrained_model.model_uri, mode=mfs.DEPLOYMENT_MODE_CREATE
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
    assert any([model_name in object_name for object_name in object_names])
    assert any(
        [
            app_name in config["EndpointConfigName"]
            for config in sagemaker_client.list_endpoint_configs()["EndpointConfigs"]
        ]
    )
    assert app_name in [
        endpoint["EndpointName"] for endpoint in sagemaker_client.list_endpoints()["Endpoints"]
    ]
    model_environment = sagemaker_client.describe_model(ModelName=model_name)["PrimaryContainer"][
        "Environment"
    ]
    assert model_environment == {
        "MLFLOW_DEPLOYMENT_FLAVOR_NAME": "python_function",
        "SERVING_ENVIRONMENT": "SageMaker",
    }


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_cli_creates_sagemaker_and_s3_resources_with_expected_names_and_env_from_local(
    pretrained_model, sagemaker_client
):
    app_name = "test-app"
    result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
        mfscli.commands,
        [
            "deploy",
            "-a",
            app_name,
            "-m",
            pretrained_model.model_uri,
            "--mode",
            mfs.DEPLOYMENT_MODE_CREATE,
        ],
    )
    assert result.exit_code == 0

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
    assert any([model_name in object_name for object_name in object_names])
    assert any(
        [
            app_name in config["EndpointConfigName"]
            for config in sagemaker_client.list_endpoint_configs()["EndpointConfigs"]
        ]
    )
    assert app_name in [
        endpoint["EndpointName"] for endpoint in sagemaker_client.list_endpoints()["Endpoints"]
    ]
    model_environment = sagemaker_client.describe_model(ModelName=model_name)["PrimaryContainer"][
        "Environment"
    ]
    assert model_environment == {
        "MLFLOW_DEPLOYMENT_FLAVOR_NAME": "python_function",
        "SERVING_ENVIRONMENT": "SageMaker",
    }


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_creates_sagemaker_and_s3_resources_with_expected_names_and_env_from_s3(
    pretrained_model, sagemaker_client
):
    local_model_path = _download_artifact_from_uri(pretrained_model.model_uri)
    artifact_path = "model"
    region_name = sagemaker_client.meta.region_name
    default_bucket = mfs._get_default_s3_bucket(region_name)
    s3_artifact_repo = S3ArtifactRepository("s3://{}".format(default_bucket))
    s3_artifact_repo.log_artifacts(local_model_path, artifact_path=artifact_path)
    model_s3_uri = "s3://{bucket_name}/{artifact_path}".format(
        bucket_name=default_bucket, artifact_path=pretrained_model.model_path
    )

    app_name = "test-app"
    mfs.deploy(app_name=app_name, model_uri=model_s3_uri, mode=mfs.DEPLOYMENT_MODE_CREATE)

    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=app_name)
    endpoint_production_variants = endpoint_description["ProductionVariants"]
    assert len(endpoint_production_variants) == 1
    model_name = endpoint_production_variants[0]["VariantName"]
    assert model_name in [model["ModelName"] for model in sagemaker_client.list_models()["Models"]]

    s3_client = boto3.client("s3", region_name=region_name)
    object_names = [
        entry["Key"] for entry in s3_client.list_objects(Bucket=default_bucket)["Contents"]
    ]
    assert any([model_name in object_name for object_name in object_names])
    assert any(
        [
            app_name in config["EndpointConfigName"]
            for config in sagemaker_client.list_endpoint_configs()["EndpointConfigs"]
        ]
    )
    assert app_name in [
        endpoint["EndpointName"] for endpoint in sagemaker_client.list_endpoints()["Endpoints"]
    ]
    model_environment = sagemaker_client.describe_model(ModelName=model_name)["PrimaryContainer"][
        "Environment"
    ]
    assert model_environment == {
        "MLFLOW_DEPLOYMENT_FLAVOR_NAME": "python_function",
        "SERVING_ENVIRONMENT": "SageMaker",
    }


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_cli_creates_sagemaker_and_s3_resources_with_expected_names_and_env_from_s3(
    pretrained_model, sagemaker_client
):
    local_model_path = _download_artifact_from_uri(pretrained_model.model_uri)
    artifact_path = "model"
    region_name = sagemaker_client.meta.region_name
    default_bucket = mfs._get_default_s3_bucket(region_name)
    s3_artifact_repo = S3ArtifactRepository("s3://{}".format(default_bucket))
    s3_artifact_repo.log_artifacts(local_model_path, artifact_path=artifact_path)
    model_s3_uri = "s3://{bucket_name}/{artifact_path}".format(
        bucket_name=default_bucket, artifact_path=pretrained_model.model_path
    )

    app_name = "test-app"
    result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
        mfscli.commands,
        ["deploy", "-a", app_name, "-m", model_s3_uri, "--mode", mfs.DEPLOYMENT_MODE_CREATE],
    )
    assert result.exit_code == 0

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
    assert any([model_name in object_name for object_name in object_names])
    assert any(
        [
            app_name in config["EndpointConfigName"]
            for config in sagemaker_client.list_endpoint_configs()["EndpointConfigs"]
        ]
    )
    assert app_name in [
        endpoint["EndpointName"] for endpoint in sagemaker_client.list_endpoints()["Endpoints"]
    ]
    model_environment = sagemaker_client.describe_model(ModelName=model_name)["PrimaryContainer"][
        "Environment"
    ]
    assert model_environment == {
        "MLFLOW_DEPLOYMENT_FLAVOR_NAME": "python_function",
        "SERVING_ENVIRONMENT": "SageMaker",
    }


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploying_application_with_preexisting_name_in_create_mode_throws_exception(
    pretrained_model,
):
    app_name = "test-app"
    mfs.deploy(
        app_name=app_name, model_uri=pretrained_model.model_uri, mode=mfs.DEPLOYMENT_MODE_CREATE
    )

    with pytest.raises(
        MlflowException, match="an application with the same name already exists"
    ) as exc:
        mfs.deploy(
            app_name=app_name, model_uri=pretrained_model.model_uri, mode=mfs.DEPLOYMENT_MODE_CREATE
        )

    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_in_synchronous_mode_waits_for_endpoint_creation_to_complete_before_returning(
    pretrained_model, sagemaker_client
):
    endpoint_creation_latency = 10
    get_sagemaker_backend(sagemaker_client.meta.region_name).set_endpoint_update_latency(
        endpoint_creation_latency
    )

    app_name = "test-app"
    deployment_start_time = time.time()
    mfs.deploy(
        app_name=app_name,
        model_uri=pretrained_model.model_uri,
        mode=mfs.DEPLOYMENT_MODE_CREATE,
        synchronous=True,
    )
    deployment_end_time = time.time()

    assert (deployment_end_time - deployment_start_time) >= endpoint_creation_latency
    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=app_name)
    assert endpoint_description["EndpointStatus"] == Endpoint.STATUS_IN_SERVICE


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_create_in_asynchronous_mode_returns_before_endpoint_creation_completes(
    pretrained_model, sagemaker_client
):
    endpoint_creation_latency = 10
    get_sagemaker_backend(sagemaker_client.meta.region_name).set_endpoint_update_latency(
        endpoint_creation_latency
    )

    app_name = "test-app"
    deployment_start_time = time.time()
    mfs.deploy(
        app_name=app_name,
        model_uri=pretrained_model.model_uri,
        mode=mfs.DEPLOYMENT_MODE_CREATE,
        synchronous=False,
        archive=True,
    )
    deployment_end_time = time.time()

    assert (deployment_end_time - deployment_start_time) < endpoint_creation_latency
    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=app_name)
    assert endpoint_description["EndpointStatus"] == Endpoint.STATUS_CREATING


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_replace_in_asynchronous_mode_returns_before_endpoint_creation_completes(
    pretrained_model, sagemaker_client
):
    endpoint_update_latency = 10
    get_sagemaker_backend(sagemaker_client.meta.region_name).set_endpoint_update_latency(
        endpoint_update_latency
    )

    app_name = "test-app"
    mfs.deploy(
        app_name=app_name,
        model_uri=pretrained_model.model_uri,
        mode=mfs.DEPLOYMENT_MODE_CREATE,
        synchronous=True,
    )

    update_start_time = time.time()
    mfs.deploy(
        app_name=app_name,
        model_uri=pretrained_model.model_uri,
        mode=mfs.DEPLOYMENT_MODE_REPLACE,
        synchronous=False,
        archive=True,
    )
    update_end_time = time.time()

    assert (update_end_time - update_start_time) < endpoint_update_latency
    endpoint_description = sagemaker_client.describe_endpoint(EndpointName=app_name)
    assert endpoint_description["EndpointStatus"] == Endpoint.STATUS_UPDATING


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_in_create_mode_throws_exception_after_endpoint_creation_fails(
    pretrained_model, sagemaker_client
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
        mfs.deploy(
            app_name="test-app",
            model_uri=pretrained_model.model_uri,
            mode=mfs.DEPLOYMENT_MODE_CREATE,
        )

    assert exc.value.error_code == ErrorCode.Name(INTERNAL_ERROR)


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_in_add_mode_adds_new_model_to_existing_endpoint(pretrained_model, sagemaker_client):
    app_name = "test-app"
    mfs.deploy(
        app_name=app_name, model_uri=pretrained_model.model_uri, mode=mfs.DEPLOYMENT_MODE_CREATE
    )
    models_added = 1
    for _ in range(11):
        mfs.deploy(
            app_name=app_name,
            model_uri=pretrained_model.model_uri,
            mode=mfs.DEPLOYMENT_MODE_ADD,
            archive=True,
            synchronous=False,
        )
        models_added += 1

    endpoint_response = sagemaker_client.describe_endpoint(EndpointName=app_name)
    endpoint_config_name = endpoint_response["EndpointConfigName"]
    endpoint_config_response = sagemaker_client.describe_endpoint_config(
        EndpointConfigName=endpoint_config_name
    )
    production_variants = endpoint_config_response["ProductionVariants"]
    assert len(production_variants) == models_added


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_in_replace_model_removes_preexisting_models_from_endpoint(
    pretrained_model, sagemaker_client
):
    app_name = "test-app"
    mfs.deploy(
        app_name=app_name, model_uri=pretrained_model.model_uri, mode=mfs.DEPLOYMENT_MODE_ADD
    )

    for _ in range(11):
        mfs.deploy(
            app_name=app_name,
            model_uri=pretrained_model.model_uri,
            mode=mfs.DEPLOYMENT_MODE_ADD,
            archive=True,
            synchronous=False,
        )

    endpoint_response_before_replacement = sagemaker_client.describe_endpoint(EndpointName=app_name)
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

    mfs.deploy(
        app_name=app_name,
        model_uri=pretrained_model.model_uri,
        mode=mfs.DEPLOYMENT_MODE_REPLACE,
        archive=True,
        synchronous=False,
    )

    endpoint_response_after_replacement = sagemaker_client.describe_endpoint(EndpointName=app_name)
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
        [
            model_name not in deployed_models_after_replacement
            for model_name in deployed_models_before_replacement
        ]
    )


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_in_replace_mode_throws_exception_after_endpoint_update_fails(
    pretrained_model, sagemaker_client
):
    endpoint_update_latency = 5
    sagemaker_backend = get_sagemaker_backend(sagemaker_client.meta.region_name)
    sagemaker_backend.set_endpoint_update_latency(endpoint_update_latency)

    app_name = "test-app"
    mfs.deploy(
        app_name=app_name, model_uri=pretrained_model.model_uri, mode=mfs.DEPLOYMENT_MODE_CREATE
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
        mfs.deploy(
            app_name="test-app",
            model_uri=pretrained_model.model_uri,
            mode=mfs.DEPLOYMENT_MODE_REPLACE,
        )
    assert exc.value.error_code == ErrorCode.Name(INTERNAL_ERROR)


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_in_replace_mode_waits_for_endpoint_update_completion_before_deleting_resources(
    pretrained_model, sagemaker_client
):
    endpoint_update_latency = 10
    sagemaker_backend = get_sagemaker_backend(sagemaker_client.meta.region_name)
    sagemaker_backend.set_endpoint_update_latency(endpoint_update_latency)

    app_name = "test-app"
    mfs.deploy(
        app_name=app_name, model_uri=pretrained_model.model_uri, mode=mfs.DEPLOYMENT_MODE_CREATE
    )
    endpoint_config_name_before_replacement = sagemaker_client.describe_endpoint(
        EndpointName=app_name
    )["EndpointConfigName"]

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
            endpoint_info = sagemaker_client.describe_endpoint(EndpointName=app_name)
            assert endpoint_info["EndpointStatus"] == Endpoint.STATUS_IN_SERVICE
            assert endpoint_info["EndpointConfigName"] != endpoint_config_name_before_replacement
            assert time.time() - update_start_time >= endpoint_update_latency
        return result

    with mock.patch("botocore.client.BaseClient._make_api_call", new=validate_deletes):
        mfs.deploy(
            app_name=app_name,
            model_uri=pretrained_model.model_uri,
            mode=mfs.DEPLOYMENT_MODE_REPLACE,
            archive=False,
        )


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_in_replace_mode_with_archiving_does_not_delete_resources(
    pretrained_model, sagemaker_client
):
    region_name = sagemaker_client.meta.region_name
    sagemaker_backend = get_sagemaker_backend(region_name)
    sagemaker_backend.set_endpoint_update_latency(5)

    app_name = "test-app"
    mfs.deploy(
        app_name=app_name, model_uri=pretrained_model.model_uri, mode=mfs.DEPLOYMENT_MODE_CREATE
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
    mfs.deploy(
        app_name=app_name,
        model_uri=new_model_uri,
        mode=mfs.DEPLOYMENT_MODE_REPLACE,
        archive=True,
        synchronous=True,
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
        [
            object_name in object_names_after_replacement
            for object_name in object_names_before_replacement
        ]
    )
    assert all(
        [
            endpoint_config in endpoint_configs_after_replacement
            for endpoint_config in endpoint_configs_before_replacement
        ]
    )
    assert all([model in models_after_replacement for model in models_before_replacement])

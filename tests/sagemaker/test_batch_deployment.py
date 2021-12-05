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
from tests.sagemaker.mock import mock_sagemaker, TransformJob, TransformJobOperation

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

        return fn(*args, **kwargs)

    return mock_wrapper


@pytest.mark.large
def test_batch_deployment_with_unsupported_flavor_raises_exception(pretrained_model):
    unsupported_flavor = "this is not a valid flavor"
    match = "The specified flavor: `this is not a valid flavor` is not supported for deployment"
    with pytest.raises(MlflowException, match=match) as exc:
        mfs.deploy_transform_job(
            job_name="bad_flavor",
            model_uri=pretrained_model.model_uri,
            s3_input_data_type="Some Data Type",
            s3_input_uri="Some Input Uri",
            content_type="Some Content Type",
            s3_output_path="Some Output Path",
            flavor=unsupported_flavor,
        )

    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.large
def test_batch_deployment_with_missing_flavor_raises_exception(pretrained_model):
    missing_flavor = "mleap"
    with pytest.raises(
        MlflowException,
        match="The specified model does not contain the specified deployment flavor",
    ) as exc:
        mfs.deploy_transform_job(
            job_name="missing-flavor",
            model_uri=pretrained_model.model_uri,
            s3_input_data_type="Some Data Type",
            s3_input_uri="Some Input Uri",
            content_type="Some Content Type",
            s3_output_path="Some Output Path",
            flavor=missing_flavor,
        )

    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


@pytest.mark.large
def test_batch_deployment_of_model_with_no_supported_flavors_raises_exception(pretrained_model):
    logged_model_path = _download_artifact_from_uri(pretrained_model.model_uri)
    model_config_path = os.path.join(logged_model_path, "MLmodel")
    model_config = Model.load(model_config_path)
    del model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
    model_config.save(path=model_config_path)

    match = "The specified model does not contain any of the supported flavors for deployment"
    with pytest.raises(MlflowException, match=match) as exc:
        mfs.deploy_transform_job(
            job_name="missing-flavor",
            model_uri=logged_model_path,
            s3_input_data_type="Some Data Type",
            s3_input_uri="Some Input Uri",
            content_type="Some Content Type",
            s3_output_path="Some Output Path",
            flavor=None,
        )

    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


@pytest.mark.large
def test_deploy_sagemaker_transform_job_in_asynchronous_mode_without_archiving_throws_exception(
    pretrained_model,
):
    with pytest.raises(MlflowException, match="Resources must be archived") as exc:
        mfs.deploy_transform_job(
            job_name="test-job",
            model_uri=pretrained_model.model_uri,
            s3_input_data_type="Some Data Type",
            s3_input_uri="Some Input Uri",
            content_type="Some Content Type",
            s3_output_path="Some Output Path",
            archive=False,
            synchronous=False,
        )

    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_creates_sagemaker_transform_job_and_s3_resources_with_expected_names_from_local(
    pretrained_model, sagemaker_client
):
    job_name = "test-job"
    mfs.deploy_transform_job(
        job_name=job_name,
        model_uri=pretrained_model.model_uri,
        s3_input_data_type="Some Data Type",
        s3_input_uri="Some Input Uri",
        content_type="Some Content Type",
        s3_output_path="Some Output Path",
        archive=True,
    )

    region_name = sagemaker_client.meta.region_name
    s3_client = boto3.client("s3", region_name=region_name)
    default_bucket = mfs._get_default_s3_bucket(region_name)
    transform_job_description = sagemaker_client.describe_transform_job(TransformJobName=job_name)
    model_name = transform_job_description["ModelName"]
    assert model_name in [model["ModelName"] for model in sagemaker_client.list_models()["Models"]]
    object_names = [
        entry["Key"] for entry in s3_client.list_objects(Bucket=default_bucket)["Contents"]
    ]
    assert any([model_name in object_name for object_name in object_names])
    assert job_name in [
        transform_job["TransformJobName"]
        for transform_job in sagemaker_client.list_transform_jobs()["TransformJobSummaries"]
    ]


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_cli_creates_sagemaker_transform_job_and_s3_resources_with_expected_names_from_local(
    pretrained_model, sagemaker_client
):
    job_name = "test-job"
    result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
        mfscli.commands,
        [
            "deploy-transform-job",
            "--job-name",
            job_name,
            "--model-uri",
            pretrained_model.model_uri,
            "--input-data-type",
            "Some Data Type",
            "--input-uri",
            "Some Input Uri",
            "--content-type",
            "Some Content Type",
            "--output-path",
            "Some Output Path",
            "--archive",
        ],
    )
    assert result.exit_code == 0

    region_name = sagemaker_client.meta.region_name
    s3_client = boto3.client("s3", region_name=region_name)
    default_bucket = mfs._get_default_s3_bucket(region_name)
    transform_job_description = sagemaker_client.describe_transform_job(TransformJobName=job_name)
    model_name = transform_job_description["ModelName"]
    assert model_name in [model["ModelName"] for model in sagemaker_client.list_models()["Models"]]
    object_names = [
        entry["Key"] for entry in s3_client.list_objects(Bucket=default_bucket)["Contents"]
    ]
    assert any([model_name in object_name for object_name in object_names])
    assert job_name in [
        transform_job["TransformJobName"]
        for transform_job in sagemaker_client.list_transform_jobs()["TransformJobSummaries"]
    ]


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_creates_sagemaker_transform_job_and_s3_resources_with_expected_names_from_s3(
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

    job_name = "test-job"
    mfs.deploy_transform_job(
        job_name=job_name,
        model_uri=model_s3_uri,
        s3_input_data_type="Some Data Type",
        s3_input_uri="Some Input Uri",
        content_type="Some Content Type",
        s3_output_path="Some Output Path",
        archive=True,
    )

    transform_job_description = sagemaker_client.describe_transform_job(TransformJobName=job_name)
    model_name = transform_job_description["ModelName"]
    assert model_name in [model["ModelName"] for model in sagemaker_client.list_models()["Models"]]

    s3_client = boto3.client("s3", region_name=region_name)
    object_names = [
        entry["Key"] for entry in s3_client.list_objects(Bucket=default_bucket)["Contents"]
    ]
    assert any([model_name in object_name for object_name in object_names])
    assert job_name in [
        transform_job["TransformJobName"]
        for transform_job in sagemaker_client.list_transform_jobs()["TransformJobSummaries"]
    ]


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_cli_creates_sagemaker_transform_job_and_s3_resources_with_expected_names_from_s3(
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

    job_name = "test-job"
    result = CliRunner(env={"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}).invoke(
        mfscli.commands,
        [
            "deploy-transform-job",
            "--job-name",
            job_name,
            "--model-uri",
            model_s3_uri,
            "--input-data-type",
            "Some Data Type",
            "--input-uri",
            "Some Input Uri",
            "--content-type",
            "Some Content Type",
            "--output-path",
            "Some Output Path",
            "--archive",
        ],
    )
    assert result.exit_code == 0

    region_name = sagemaker_client.meta.region_name
    s3_client = boto3.client("s3", region_name=region_name)
    default_bucket = mfs._get_default_s3_bucket(region_name)
    transform_job_description = sagemaker_client.describe_transform_job(TransformJobName=job_name)
    model_name = transform_job_description["ModelName"]
    assert model_name in [model["ModelName"] for model in sagemaker_client.list_models()["Models"]]
    object_names = [
        entry["Key"] for entry in s3_client.list_objects(Bucket=default_bucket)["Contents"]
    ]
    assert any([model_name in object_name for object_name in object_names])
    assert job_name in [
        transform_job["TransformJobName"]
        for transform_job in sagemaker_client.list_transform_jobs()["TransformJobSummaries"]
    ]


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploying_sagemaker_transform_job_with_preexisting_name_in_create_mode_throws_exception(
    pretrained_model,
):
    job_name = "test-job"
    mfs.deploy_transform_job(
        job_name=job_name,
        model_uri=pretrained_model.model_uri,
        s3_input_data_type="Some Data Type",
        s3_input_uri="Some Input Uri",
        content_type="Some Content Type",
        s3_output_path="Some Output Path",
    )

    with pytest.raises(
        MlflowException, match="a batch transform job with the same name already exists"
    ) as exc:
        mfs.deploy_transform_job(
            job_name=job_name,
            model_uri=pretrained_model.model_uri,
            s3_input_data_type="Some Data Type",
            s3_input_uri="Some Input Uri",
            content_type="Some Content Type",
            s3_output_path="Some Output Path",
        )

    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_in_synchronous_mode_waits_for_transform_job_creation_to_complete_before_returning(
    pretrained_model, sagemaker_client
):
    transform_job_creation_latency = 10
    get_sagemaker_backend(sagemaker_client.meta.region_name).set_transform_job_update_latency(
        transform_job_creation_latency
    )

    job_name = "test-job"
    deployment_start_time = time.time()
    mfs.deploy_transform_job(
        job_name=job_name,
        model_uri=pretrained_model.model_uri,
        s3_input_data_type="Some Data Type",
        s3_input_uri="Some Input Uri",
        content_type="Some Content Type",
        s3_output_path="Some Output Path",
        synchronous=True,
    )
    deployment_end_time = time.time()

    assert (deployment_end_time - deployment_start_time) >= transform_job_creation_latency
    transform_job_description = sagemaker_client.describe_transform_job(TransformJobName=job_name)
    assert transform_job_description["TransformJobStatus"] == TransformJob.STATUS_COMPLETED


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_create_in_asynchronous_mode_returns_before_transform_job_creation_completes(
    pretrained_model, sagemaker_client
):
    transform_job_creation_latency = 10
    get_sagemaker_backend(sagemaker_client.meta.region_name).set_transform_job_update_latency(
        transform_job_creation_latency
    )

    job_name = "test-job"
    deployment_start_time = time.time()
    mfs.deploy_transform_job(
        job_name=job_name,
        model_uri=pretrained_model.model_uri,
        s3_input_data_type="Some Data Type",
        s3_input_uri="Some Input Uri",
        content_type="Some Content Type",
        s3_output_path="Some Output Path",
        archive=True,
        synchronous=False,
    )
    deployment_end_time = time.time()

    assert (deployment_end_time - deployment_start_time) < transform_job_creation_latency
    transform_job_description = sagemaker_client.describe_transform_job(TransformJobName=job_name)
    assert transform_job_description["TransformJobStatus"] == TransformJob.STATUS_IN_PROGRESS


@pytest.mark.large
@mock_sagemaker_aws_services
def test_deploy_in_throw_exception_after_transform_job_creation_fails(
    pretrained_model, sagemaker_client
):
    transform_job_creation_latency = 10
    sagemaker_backend = get_sagemaker_backend(sagemaker_client.meta.region_name)
    sagemaker_backend.set_transform_job_update_latency(transform_job_creation_latency)

    boto_caller = botocore.client.BaseClient._make_api_call

    def fail_transform_job_creations(self, operation_name, operation_kwargs):
        """
        Processes all boto3 client operations according to the following rules:
        - If the operation is a transform job creation, create the transform job and
          set its status to ``TransformJob.STATUS_FAILED``.
        - Else, execute the client operation as normal
        """
        result = boto_caller(self, operation_name, operation_kwargs)
        if operation_name == "CreateTransformJob":
            transform_job_name = operation_kwargs["TransformJobName"]
            sagemaker_backend.set_transform_job_latest_operation(
                transform_job_name=transform_job_name,
                operation=TransformJobOperation.create_unsuccessful(
                    latency_seconds=transform_job_creation_latency
                ),
            )
        return result

    with mock.patch(
        "botocore.client.BaseClient._make_api_call", new=fail_transform_job_creations
    ), pytest.raises(MlflowException, match="batch transform job failed") as exc:
        mfs.deploy_transform_job(
            job_name="test-job",
            model_uri=pretrained_model.model_uri,
            s3_input_data_type="Some Data Type",
            s3_input_uri="Some Input Uri",
            content_type="Some Content Type",
            s3_output_path="Some Output Path",
        )

    assert exc.value.error_code == ErrorCode.Name(INTERNAL_ERROR)


@pytest.mark.large
@mock_sagemaker_aws_services
def test_attempting_to_terminate_in_asynchronous_mode_without_archiving_throws_exception(
    pretrained_model,
):
    job_name = "test-job"
    mfs.deploy_transform_job(
        job_name=job_name,
        model_uri=pretrained_model.model_uri,
        s3_input_data_type="Some Data Type",
        s3_input_uri="Some Input Uri",
        content_type="Some Content Type",
        s3_output_path="Some Output Path",
    )

    with pytest.raises(MlflowException, match="Resources must be archived") as exc:
        mfs.terminate_transform_job(
            job_name=job_name,
            archive=False,
            synchronous=False,
        )

    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.large
@mock_sagemaker_aws_services
def test_terminate_in_sync_mode_waits_for_transform_job_termination_to_complete_before_returning(
    pretrained_model, sagemaker_client
):
    transform_job_termination_latency = 10
    get_sagemaker_backend(sagemaker_client.meta.region_name).set_transform_job_update_latency(
        transform_job_termination_latency
    )

    job_name = "test-job"
    termination_start_time = time.time()
    mfs.deploy_transform_job(
        job_name=job_name,
        model_uri=pretrained_model.model_uri,
        s3_input_data_type="Some Data Type",
        s3_input_uri="Some Input Uri",
        content_type="Some Content Type",
        s3_output_path="Some Output Path",
        archive=True,
        synchronous=True,
    )

    mfs.terminate_transform_job(job_name=job_name, synchronous=True)
    termination_end_time = time.time()

    assert (termination_end_time - termination_start_time) >= transform_job_termination_latency
    transform_job_description = sagemaker_client.describe_transform_job(TransformJobName=job_name)
    assert transform_job_description["TransformJobStatus"] == TransformJob.STATUS_STOPPED


@pytest.mark.large
@mock_sagemaker_aws_services
def test_terminate_in_asynchronous_mode_returns_before_transform_job_termination_completes(
    pretrained_model, sagemaker_client
):
    transform_job_termination_latency = 10
    get_sagemaker_backend(sagemaker_client.meta.region_name).set_transform_job_update_latency(
        transform_job_termination_latency
    )

    job_name = "test-job"
    termination_start_time = time.time()
    mfs.deploy_transform_job(
        job_name=job_name,
        model_uri=pretrained_model.model_uri,
        s3_input_data_type="Some Data Type",
        s3_input_uri="Some Input Uri",
        content_type="Some Content Type",
        s3_output_path="Some Output Path",
        archive=True,
        synchronous=False,
    )

    mfs.terminate_transform_job(job_name=job_name, archive=True, synchronous=False)
    termination_end_time = time.time()

    assert (termination_end_time - termination_start_time) < transform_job_termination_latency
    transform_job_description = sagemaker_client.describe_transform_job(TransformJobName=job_name)
    assert transform_job_description["TransformJobStatus"] == TransformJob.STATUS_STOPPING

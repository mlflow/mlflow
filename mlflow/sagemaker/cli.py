import json
import os
import tempfile

import click

import mlflow
import mlflow.models.docker_utils
import mlflow.sagemaker
from mlflow.sagemaker import DEFAULT_IMAGE_NAME as IMAGE
from mlflow.utils import cli_args
from mlflow.utils import env_manager as em


@click.group("sagemaker")
def commands():
    """
    Serve models on SageMaker.

    To serve a model associated with a run on a tracking server, set the MLFLOW_TRACKING_URI
    environment variable to the URL of the desired server.
    """


@commands.command("deploy-transform-job")
@click.option("--job-name", "-n", help="Transform job name", required=True)
@cli_args.MODEL_URI
@click.option("--input-data-type", help="Input data type for the transform job", required=True)
@click.option(
    "--input-uri", "-u", help="S3 key name prefix or manifest of the input data", required=True
)
@click.option(
    "--content-type",
    help="The multipurpose internet mail extension (MIME) type of the data",
    required=True,
)
@click.option(
    "--output-path",
    "-o",
    help="The S3 path to store the output results of the Sagemaker transform job",
    required=True,
)
@click.option(
    "--compression-type", default="None", help="The compression type of the transform data"
)
@click.option(
    "--split-type",
    "-s",
    default="Line",
    help="The method to split the transform job's data files into smaller batches",
)
@click.option(
    "--accept",
    "-a",
    default="text/csv",
    help="The multipurpose internet mail extension (MIME) type of the output data",
)
@click.option(
    "--assemble-with",
    default="Line",
    help="The method to assemble the results of the transform job as a single S3 object",
)
@click.option(
    "--input-filter",
    default="$",
    help="A JSONPath expression used to select a portion of the input data for the transform job",
)
@click.option(
    "--output-filter",
    default="$",
    help="A JSONPath expression used to select a portion of the output data from the transform job",
)
@click.option(
    "--join-resource",
    "-j",
    default="None",
    help="The source of the data to join with the transformed data",
)
@click.option("--execution-role-arn", "-e", default=None, help="SageMaker execution role")
@click.option("--bucket", "-b", default=None, help="S3 bucket to store model artifacts")
@click.option("--image-url", "-i", default=None, help="ECR URL for the Docker image")
@click.option(
    "--region-name",
    default="us-west-2",
    help="Name of the AWS region in which to deploy the transform job",
)
@click.option(
    "--instance-type",
    "-t",
    default=mlflow.sagemaker.DEFAULT_SAGEMAKER_INSTANCE_TYPE,
    help="The type of SageMaker ML instance on which to perform the batch transform job."
    " For a list of supported instance types, see"
    " https://aws.amazon.com/sagemaker/pricing/instance-types/.",
)
@click.option(
    "--instance-count",
    "-c",
    default=mlflow.sagemaker.DEFAULT_SAGEMAKER_INSTANCE_COUNT,
    help="The number of SageMaker ML instances on which to perform the batch transform job",
)
@click.option(
    "--vpc-config",
    "-v",
    help="Path to a file containing a JSON-formatted VPC configuration. This"
    " configuration will be used when creating the new SageMaker model associated"
    " with this application. For more information, see"
    " https://docs.aws.amazon.com/sagemaker/latest/dg/API_VpcConfig.html",
)
@click.option(
    "--flavor",
    "-f",
    default=None,
    help=(
        "The name of the flavor to use for deployment. Must be one of the following: "
        f"{mlflow.sagemaker.SUPPORTED_DEPLOYMENT_FLAVORS}. If unspecified, a flavor will be "
        "automatically selected from the model's available flavors."
    ),
)
@click.option(
    "--archive",
    is_flag=True,
    help=(
        "If specified, any SageMaker resources that become inactive after the finished"
        " batch transform job are preserved. These resources may include the associated"
        " SageMaker models and model artifacts. Otherwise, if `--archive` is unspecified,"
        " these resources are deleted. `--archive` must be specified when deploying"
        " asynchronously with `--async`."
    ),
)
@click.option(
    "--async",
    "asynchronous",
    is_flag=True,
    help=(
        "If specified, this command will return immediately after starting the"
        " deployment process. It will not wait for the deployment process to complete."
        " The caller is responsible for monitoring the deployment process via native"
        " SageMaker APIs or the AWS console."
    ),
)
@click.option(
    "--timeout",
    default=1200,
    help=(
        "If the command is executed synchronously, the deployment process will return"
        " after the specified number of seconds if no definitive result (success or"
        " failure) is achieved. Once the function returns, the caller is responsible"
        " for monitoring the health and status of the pending deployment via"
        " native SageMaker APIs or the AWS console. If the command is executed"
        " asynchronously using the `--async` flag, this value is ignored."
    ),
)
def deploy_transform_job(
    job_name,
    model_uri,
    input_data_type,
    input_uri,
    content_type,
    output_path,
    compression_type,
    split_type,
    accept,
    assemble_with,
    input_filter,
    output_filter,
    join_resource,
    execution_role_arn,
    bucket,
    image_url,
    region_name,
    instance_type,
    instance_count,
    vpc_config,
    flavor,
    archive,
    asynchronous,
    timeout,
):
    """
    Deploy model on Sagemaker as a batch transform job. Current active AWS account needs to have
    correct permissions setup.

    By default, unless the ``--async`` flag is specified, this command will block until
    either the batch transform job completes (definitively succeeds or fails) or the specified
    timeout elapses.
    """
    if vpc_config is not None:
        with open(vpc_config) as f:
            vpc_config = json.load(f)

    mlflow.sagemaker.deploy_transform_job(
        job_name=job_name,
        model_uri=model_uri,
        s3_input_data_type=input_data_type,
        s3_input_uri=input_uri,
        content_type=content_type,
        s3_output_path=output_path,
        compression_type=compression_type,
        split_type=split_type,
        accept=accept,
        assemble_with=assemble_with,
        input_filter=input_filter,
        output_filter=output_filter,
        join_resource=join_resource,
        execution_role_arn=execution_role_arn,
        bucket=bucket,
        image_url=image_url,
        region_name=region_name,
        instance_type=instance_type,
        instance_count=instance_count,
        vpc_config=vpc_config,
        flavor=flavor,
        archive=archive,
        synchronous=(not asynchronous),
        timeout_seconds=timeout,
    )


@commands.command("terminate-transform-job")
@click.option("--job-name", "-n", help="Transform job name", required=True)
@click.option(
    "--region-name",
    "-r",
    default="us-west-2",
    help="Name of the AWS region in which the transform job is deployed",
)
@click.option(
    "--archive",
    is_flag=True,
    help=(
        "If specified, resources associated with the application are preserved."
        " These resources may include unused SageMaker models and model artifacts."
        " Otherwise, if `--archive` is unspecified, these resources are deleted."
        " `--archive` must be specified when deleting asynchronously with `--async`."
    ),
)
@click.option(
    "--async",
    "asynchronous",
    is_flag=True,
    help=(
        "If specified, this command will return immediately after starting the"
        " termination process. It will not wait for the termination process to complete."
        " The caller is responsible for monitoring the termination process via native"
        " SageMaker APIs or the AWS console."
    ),
)
@click.option(
    "--timeout",
    default=1200,
    help=(
        "If the command is executed synchronously, the termination process will return"
        " after the specified number of seconds if no definitive result (success or"
        " failure) is achieved. Once the function returns, the caller is responsible"
        " for monitoring the health and status of the pending termination via"
        " native SageMaker APIs or the AWS console. If the command is executed"
        " asynchronously using the `--async` flag, this value is ignored."
    ),
)
def terminate_transform_job(job_name, region_name, archive, asynchronous, timeout):
    """
    Terminate the specified Sagemaker batch transform job. Unless ``--archive`` is specified,
    all SageMaker resources associated with the batch transform job are deleted as well.

    By default, unless the ``--async`` flag is specified, this command will block until
    either the termination process completes (definitively succeeds or fails) or the specified
    timeout elapses.
    """
    mlflow.sagemaker.terminate_transform_job(
        job_name=job_name,
        region_name=region_name,
        archive=archive,
        synchronous=(not asynchronous),
        timeout_seconds=timeout,
    )


@commands.command("push-model")
@click.option("--model-name", "-n", help="Sagemaker model name", required=True)
@cli_args.MODEL_URI
@click.option("--execution-role-arn", "-e", default=None, help="SageMaker execution role")
@click.option("--bucket", "-b", default=None, help="S3 bucket to store model artifacts")
@click.option("--image-url", "-i", default=None, help="ECR URL for the Docker image")
@click.option(
    "--region-name",
    default="us-west-2",
    help="Name of the AWS region in which to push the Sagemaker model",
)
@click.option(
    "--vpc-config",
    "-v",
    help="Path to a file containing a JSON-formatted VPC configuration. This"
    " configuration will be used when creating the new SageMaker model."
    " For more information, see"
    " https://docs.aws.amazon.com/sagemaker/latest/dg/API_VpcConfig.html",
)
@click.option(
    "--flavor",
    "-f",
    default=None,
    help=(
        "The name of the flavor to use for deployment. Must be one of the following:"
        " {supported_flavors}. If unspecified, a flavor will be automatically selected"
        " from the model's available flavors.".format(
            supported_flavors=mlflow.sagemaker.SUPPORTED_DEPLOYMENT_FLAVORS
        )
    ),
)
def push_model_to_sagemaker(
    model_name,
    model_uri,
    execution_role_arn,
    bucket,
    image_url,
    region_name,
    vpc_config,
    flavor,
):
    """
    Push an MLflow model to Sagemaker model registry. Current active AWS account needs to have
    correct permissions setup.
    """
    if vpc_config is not None:
        with open(vpc_config) as f:
            vpc_config = json.load(f)

    mlflow.sagemaker.push_model_to_sagemaker(
        model_name=model_name,
        model_uri=model_uri,
        execution_role_arn=execution_role_arn,
        bucket=bucket,
        image_url=image_url,
        region_name=region_name,
        vpc_config=vpc_config,
        flavor=flavor,
    )


@commands.command("build-and-push-container")
@click.option("--build/--no-build", default=True, help="Build the container if set.")
@click.option("--push/--no-push", default=True, help="Push the container to AWS ECR if set.")
@click.option("--container", "-c", default=IMAGE, help="image name")
@cli_args.ENV_MANAGER
@cli_args.MLFLOW_HOME
def build_and_push_container(build, push, container, env_manager, mlflow_home):
    """
    Build new MLflow Sagemaker image, assign it a name, and push to ECR.

    This function builds an MLflow Docker image.
    The image is built locally and it requires Docker to run.
    The image is pushed to ECR under current active AWS account and to current active AWS region.
    """
    from mlflow.models import docker_utils

    env_manager = env_manager or em.VIRTUALENV
    if not (build or push):
        click.echo("skipping both build and push, have nothing to do!")
    if build:
        sagemaker_image_entrypoint = (
            "import sys; from mlflow.models import container as C; "
            f"C._init(sys.argv[1], '{env_manager}')"
        )

        setup_container = (
            "# Install minimal serving dependencies\n"
            'RUN python -c "from mlflow.models.container import _install_pyfunc_deps;'
            '_install_pyfunc_deps(None, False)"'
        )

        with tempfile.TemporaryDirectory() as tmp:
            docker_utils.generate_dockerfile(
                base_image=mlflow.models.docker_utils.UBUNTU_BASE_IMAGE,
                output_dir=tmp,
                entrypoint=sagemaker_image_entrypoint,
                env_manager=env_manager,
                mlflow_home=os.path.abspath(mlflow_home) if mlflow_home else None,
                model_install_steps=setup_container,
                # Create a conda env or virtualenv at runtime after the model is loaded
                disable_env_creation_at_runtime=False,
            )

            docker_utils.build_image_from_context(tmp, image_name=container)
    if push:
        mlflow.sagemaker.push_image_to_ecr(container)

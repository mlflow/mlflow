from __future__ import print_function

import os
import json

import click

import mlflow
import mlflow.sagemaker
from mlflow.sagemaker import DEFAULT_IMAGE_NAME as IMAGE
from mlflow.utils import cli_args


@click.group("sagemaker")
def commands():
    """
    Serve models on SageMaker.

    To serve a model associated with a run on a tracking server, set the MLFLOW_TRACKING_URI
    environment variable to the URL of the desired server.
    """
    pass


@commands.command("deploy")
@click.option("--app-name", "-a", help="Application name", required=True)
@cli_args.MODEL_PATH
@click.option("--execution-role-arn", "-e", default=None, help="SageMaker execution role")
@click.option("--bucket", "-b", default=None, help="S3 bucket to store model artifacts")
@cli_args.RUN_ID
@click.option("--image-url", "-i", default=None, help="ECR URL for the Docker image")
@click.option("--region-name", default="us-west-2",
              help="Name of the AWS region in which to deploy the application")
@click.option("--mode", default=mlflow.sagemaker.DEPLOYMENT_MODE_CREATE,
              help="The mode in which to deploy the application."
              " Must be one of the following: {mds}".format(
                  mds=", ".join(mlflow.sagemaker.DEPLOYMENT_MODES)))
@click.option("--archive", "-ar", is_flag=True,
              help=("If specified, any SageMaker resources that become inactive (i.e as the"
                    " result of an update in {mode_replace} mode) are preserved."
                    " These resources may include unused SageMaker models and endpoint"
                    " configurations that were associated with a prior version of the application"
                    " endpoint. Otherwise, if `--archive` is unspecified, these resources are"
                    " deleted. `--archive` must be specified when deploying asynchronously with"
                    " `--async`.".format(
                            mode_replace=mlflow.sagemaker.DEPLOYMENT_MODE_REPLACE)))
@click.option("--instance-type", "-t", default=mlflow.sagemaker.DEFAULT_SAGEMAKER_INSTANCE_TYPE,
              help="The type of SageMaker ML instance on which to deploy the model. For a list of"
              " supported instance types, see"
              " https://aws.amazon.com/sagemaker/pricing/instance-types/.")
@click.option("--instance-count", "-c", default=mlflow.sagemaker.DEFAULT_SAGEMAKER_INSTANCE_COUNT,
              help="The number of SageMaker ML instances on which to deploy the model")
@click.option("--vpc-config", "-v",
              help="Path to a file containing a JSON-formatted VPC configuration. This"
              " configuration will be used when creating the new SageMaker model associated"
              " with this application. For more information, see"
              " https://docs.aws.amazon.com/sagemaker/latest/dg/API_VpcConfig.html")
@click.option("--flavor", "-f", default=None,
              help=("The name of the flavor to use for deployment. Must be one of the following:"
                    " {supported_flavors}. If unspecified, a flavor will be automatically selected"
                    " from the model's available flavors.".format(
                        supported_flavors=mlflow.sagemaker.SUPPORTED_DEPLOYMENT_FLAVORS)))
@click.option("--async", "asynchronous", is_flag=True,
              help=("If specified, this command will return immediately after starting the"
                    " deployment process. It will not wait for the deployment process to complete."
                    " The caller is responsible for monitoring the deployment process via native"
                    " SageMaker APIs or the AWS console."))
@click.option("--timeout", default=1200,
              help=("If the command is executed synchronously, the deployment process will return"
                    " after the specified number of seconds if no definitive result (success or"
                    " failure) is achieved. Once the function returns, the caller is responsible"
                    " for monitoring the health and status of the pending deployment via"
                    " native SageMaker APIs or the AWS console. If the command is executed"
                    " asynchronously using the `--async` flag, this value is ignored."))
def deploy(app_name, model_path, execution_role_arn, bucket, run_id, image_url, region_name, mode,
           archive, instance_type, instance_count, vpc_config, flavor, asynchronous, timeout):
    """
    Deploy model on Sagemaker as a REST API endpoint. Current active AWS account needs to have
    correct permissions setup.

    By default, unless the ``--async`` flag is specified, this command will block until
    either the deployment process completes (definitively succeeds or fails) or the specified
    timeout elapses.

    For more information about the input data formats accepted by the deployed REST API endpoint,
    see the following documentation:
    https://www.mlflow.org/docs/latest/models.html#sagemaker-deployment.
    """
    if vpc_config is not None:
        with open(vpc_config, "r") as f:
            vpc_config = json.load(f)

    mlflow.sagemaker.deploy(app_name=app_name, model_path=model_path,
                            execution_role_arn=execution_role_arn, bucket=bucket, run_id=run_id,
                            image_url=image_url, region_name=region_name, mode=mode,
                            archive=archive, instance_type=instance_type,
                            instance_count=instance_count, vpc_config=vpc_config, flavor=flavor,
                            synchronous=(not asynchronous), timeout_seconds=timeout)


@commands.command("list-flavors")
def list_flavors():
    print("Supported model flavors for SageMaker deployment are: {supported_flavors}".format(
        supported_flavors=mlflow.sagemaker.SUPPORTED_DEPLOYMENT_FLAVORS))


@commands.command("delete")
@click.option("--app-name", "-a", help="Application name", required=True)
@click.option("--region-name", "-r", default="us-west-2",
              help="Name of the AWS region in which to deploy the application.")
@click.option("--archive", "-ar", is_flag=True,
              help=("If specified, resources associated with the application are preserved."
                    " These resources may include unused SageMaker models and endpoint"
                    " configurations that were previously associated with the application endpoint."
                    " Otherwise, if `--archive` is unspecified, these resources are deleted."
                    " `--archive` must be specified when deleting asynchronously with `--async`."))
@click.option("--async", "asynchronous", is_flag=True,
              help=("If specified, this command will return immediately after starting the"
                    " deletion process. It will not wait for the deletion process to complete."
                    " The caller is responsible for monitoring the deletion process via native"
                    " SageMaker APIs or the AWS console."))
@click.option("--timeout", default=1200,
              help=("If the command is executed synchronously, the deployment process will return"
                    " after the specified number of seconds if no definitive result (success or"
                    " failure) is achieved. Once the function returns, the caller is responsible"
                    " for monitoring the health and status of the pending deployment via"
                    " native SageMaker APIs or the AWS console. If the command is executed"
                    " asynchronously using the `--async` flag, this value is ignored."))
def delete(app_name, region_name, archive, asynchronous, timeout):
    """
    Delete the specified application. Unless ``--archive`` is specified, all SageMaker resources
    associated with the application are deleted as well.

    By default, unless the ``--async`` flag is specified, this command will block until
    either the deletion process completes (definitively succeeds or fails) or the specified timeout
    elapses.
    """
    mlflow.sagemaker.delete(
        app_name=app_name, region_name=region_name, archive=archive, synchronous=(not asynchronous),
        timeout_seconds=timeout)


@commands.command("run-local")
@cli_args.MODEL_PATH
@cli_args.RUN_ID
@click.option("--port", "-p", default=5000, help="Server port. [default: 5000]")
@click.option("--image", "-i", default=IMAGE, help="Docker image name")
@click.option("--flavor", "-f", default=None,
              help=("The name of the flavor to use for local serving. Must be one of the following:"
                    " {supported_flavors}. If unspecified, a flavor will be automatically selected"
                    " from the model's available flavors.".format(
                        supported_flavors=mlflow.sagemaker.SUPPORTED_DEPLOYMENT_FLAVORS)))
def run_local(model_path, run_id, port, image, flavor):
    """
    Serve model locally running in a Sagemaker-compatible Docker container.
    """
    mlflow.sagemaker.run_local(
        model_path=model_path, run_id=run_id, port=port, image=image, flavor=flavor)


@commands.command("build-and-push-container")
@click.option("--build/--no-build", default=True, help="Build the container if set.")
@click.option("--push/--no-push", default=True, help="Push the container to AWS ECR if set.")
@click.option("--container", "-c", default=IMAGE, help="image name")
@cli_args.MLFLOW_HOME
def build_and_push_container(build, push, container, mlflow_home):
    """
    Build new MLflow Sagemaker image, assign it a name, and push to ECR.

    This function builds an MLflow Docker image.
    The image is built locally and it requires Docker to run.
    The image is pushed to ECR under current active AWS account and to current active AWS region.
    """
    if not (build or push):
        print("skipping both build and push, have nothing to do!")
    if build:
        mlflow.sagemaker.build_image(container,
                                     mlflow_home=os.path.abspath(mlflow_home) if mlflow_home
                                     else None)
    if push:
        mlflow.sagemaker.push_image_to_ecr(container)

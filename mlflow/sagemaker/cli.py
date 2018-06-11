
from __future__ import print_function

import os
from subprocess import Popen, PIPE, STDOUT

import click

import mlflow
import mlflow.sagemaker




@click.group("sagemaker")
def commands():
    """Serve models on SageMaker."""
    pass


@commands.command("deploy")
@click.option("--app-name", "-a", help="Application name", required=True)
@click.option("--model-path", "-m", help="model path", required=True)
@click.option("--execution-role-arn", "-e", help="SageMaker execution role", required=True)
@click.option("--bucket", "-b", help="S3 bucket to store model artifacts", required=True)
@click.option("--run_id", "-r", default=None, help="Run id")
@click.option("--container", "-c", default="mlflow_sage", help="container name")
def deploy(app_name, model_path, execution_role_arn, bucket, run_id=None, container="mlflow_sage"): # noqa
    mlflow.sagemaker.deploy(app_name=app_name, model_path=model_path,
                            execution_role_arn=execution_role_arn, bucket=bucket, run_id=run_id,
                            image=container)


@commands.command("run-local")
@click.option("--model-path", "-m", help="model path", required=True)
@click.option("--run_id", "-r", default=None, help="Run id")
@click.option("--port", "-p", default=5000, help="Server port. [default: 5000]")
@click.option("--container", "-c", default="mlflow_sage", help="container name")
def run_local(model_path, run_id=None, port=5000, container="mlflow_sage"):
    mlflow.sagemaker.run_local(model_path=model_path, run_id=run_id, port=port, image=container)


@commands.command("build-and-push-container")
@click.option("--build/--skip-build", default=True, help="build the container if set")
@click.option("--push/--skip-push", default=True, help="push the container to amazon ecr if set")
@click.option("--container", "-c", default="mlflow_sage", help="container name")
def build_and_push_container(build=False, push=True, container=mlflow.sagemaker.DEFAULT_IMAGE_NAME):
    if not (build or push):
        print("skipping both build nad push, have nothing to do!")
    if build:
        mlflow.sagemaker.build_image(container)
    if push:
        mlflow.sagemaker.push_image_to_ecr(container)



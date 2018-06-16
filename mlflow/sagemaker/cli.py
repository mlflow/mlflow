
from __future__ import print_function

import os
from subprocess import Popen, PIPE, STDOUT

import click

import mlflow
from mlflow import pyfunc

from mlflow.models import Model
from mlflow.tracking import _get_model_log_dir

from . deploy import _deploy, _upload_s3


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
@click.option("--region-name", default="us-west-2", help="region name")
def deploy(app_name, model_path, execution_role_arn, bucket, run_id=None, container="mlflow_sage", region_name="us-west-2"): # noqa
    """ Deploy model on sagemaker.

    :param app_name: Name of the deployed app.
    :param path: Path to the model.
    Either local if no run_id or mlflow-relative if run_id is specified)
    :param execution_role_arn: Amazon execution role with sagemaker rights
    :param bucket: S3 bucket where model artifacts are gonna be stored
    :param run_id: mlflow run id.
    :param container: name of the Docker container to be used.
    :return:
    """
    prefix = model_path
    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)
        prefix = run_id + "/" + prefix
    _check_compatible(model_path)
    model_s3_path = _upload_s3(local_model_path=model_path, bucket=bucket, prefix=prefix)
    print('model_s3_path', model_s3_path)
    _deploy(role=execution_role_arn,
            container_name=container,
            app_name=app_name,
            model_s3_path=model_s3_path,
            run_id=run_id,
            region_name=region_name)


@commands.command("run-local")
@click.option("--model-path", "-m", help="model path", required=True)
@click.option("--run_id", "-r", default=None, help="Run id")
@click.option("--port", "-p", default=5000, help="Server port. [default: 5000]")
@click.option("--container", "-c", default="mlflow_sage", help="container name")
def run_local(model_path, run_id=None, port=5000, container="mlflow_sage"):
    """
    Serve model locally in a sagemaker compatible docker container.
    :param path:  Path to the model.
    Either local if no run_id or mlflow-relative if run_id is specified)
    :param run_id: mlflow run id.
    :param port: local port
    :param container: name of the Docker container to be used.
    :return:
    """
    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)
    _check_compatible(model_path)
    model_path = os.path.abspath(model_path)
    print("launching docker container with path {}".format(model_path))
    proc = Popen(["docker", 
                  "run", 
                  "-v", 
                  "{}:/opt/ml/model/".format(model_path),
                  "-p", 
                  "%d:8080" % port, 
                  "--rm", 
                  container, 
                  "serve"],
                 stdout=PIPE, 
                 stderr=STDOUT, 
                 universal_newlines=True)
    for x in iter(proc.stdout.readline, ""):
        print(x, end='', flush=True)


@commands.command("build-and-push-container")
@click.option("--build", "-b", default=True, help="build the container if set")
@click.option("--push", "-p", default=True, help="push the container to amazon ecr if set")
@click.option("--container", "-c", default="mlflow_sage", help="container name")
def build_and_push_container(build=False, push=True, container="mlflow_sage"):
    """
    (Re)Build docker container and push to to amazon ecr.
    :param build: run docker build if set
    :param push: push to amazon ecr if set
    :param container: name of the container.
    :return:
    """
    if build:
        print("building docker image")
        proc = Popen(["docker", "build", "-t", container, "-f",
                      mlflow._relpath("sagemaker", "container", "Dockerfile"), "."],
                     cwd=os.path.dirname(mlflow._relpath()),
                     stdout=PIPE,
                     stderr=STDOUT,
                     universal_newlines=True)
        for x in iter(proc.stdout.readline, ""):
            print(x, end='', flush=True)
    if push:
        print("")
        print("pushing image to ecr")
        proc = Popen(["bash",
                      mlflow._relpath("sagemaker", "container", "push_image_to_ecr.sh"),
                      container],
                     cwd=os.path.dirname(mlflow._relpath()),
                     stdout=PIPE,
                     stderr=STDOUT,
                     universal_newlines=True)
        for x in iter(proc.stdout.readline, ""):
            print(x, end='', flush=True)


def _check_compatible(path):
    path = os.path.abspath(path)
    servable = Model.load(os.path.join(path, "MLmodel"))
    if pyfunc.FLAVOR_NAME not in servable.flavors:
        raise Exception("Currenlty only supports pyfunc format.")

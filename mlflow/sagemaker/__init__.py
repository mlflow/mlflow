"""
The ``mlflow.sagemaker`` module provides an API for deploying MLflow models to Amazon SageMaker.
"""
from __future__ import print_function

import os
import sys
from subprocess import Popen, PIPE, STDOUT
from six.moves import urllib
import tarfile
import uuid
import shutil
import logging
import time

import base64
import boto3
import yaml
import mlflow
import mlflow.version
from mlflow import pyfunc, mleap
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, INVALID_PARAMETER_VALUE
from mlflow.tracking.utils import _get_model_log_dir
from mlflow.utils import get_unique_resource_id
from mlflow.utils.file_utils import TempDir, _copy_project
from mlflow.utils.logging_utils import eprint
from mlflow.sagemaker.container import SUPPORTED_FLAVORS as SUPPORTED_DEPLOYMENT_FLAVORS
from mlflow.sagemaker.container import DEPLOYMENT_CONFIG_KEY_FLAVOR_NAME

DEFAULT_IMAGE_NAME = "mlflow-pyfunc"

DEPLOYMENT_MODE_ADD = "add"
DEPLOYMENT_MODE_REPLACE = "replace"
DEPLOYMENT_MODE_CREATE = "create"

DEPLOYMENT_MODES = [
    DEPLOYMENT_MODE_CREATE,
    DEPLOYMENT_MODE_ADD,
    DEPLOYMENT_MODE_REPLACE
]

IMAGE_NAME_ENV_VAR = "SAGEMAKER_DEPLOY_IMG_URL"

DEFAULT_BUCKET_NAME_PREFIX = "mlflow-sagemaker"

DEFAULT_SAGEMAKER_INSTANCE_TYPE = "ml.m4.xlarge"
DEFAULT_SAGEMAKER_INSTANCE_COUNT = 1

_DOCKERFILE_TEMPLATE = """
# Build an image that can serve pyfunc model in SageMaker
FROM ubuntu:16.04

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         nginx \
         ca-certificates \
         bzip2 \
         build-essential \
         cmake \
         openjdk-8-jdk \
         git-core \
         maven \
    && rm -rf /var/lib/apt/lists/*

# Download and setup miniconda
RUN curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh >> miniconda.sh
RUN bash ./miniconda.sh -b -p /miniconda; rm ./miniconda.sh;
ENV PATH="/miniconda/bin:${PATH}"
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

RUN conda install gunicorn;\
    conda install gevent;\

%s

# Set up the program in the image
WORKDIR /opt/mlflow

# start mlflow scoring
ENTRYPOINT ["python", "-c", "import sys; from mlflow.sagemaker import container as C; \
C._init(sys.argv[1])"]
"""

_logger = logging.getLogger(__name__)


def _docker_ignore(mlflow_root):
    docker_ignore = os.path.join(mlflow_root, '.dockerignore')

    def strip_slash(x):
        if x.startswith("/"):
            x = x[1:]
        if x.endswith('/'):
            x = x[:-1]
        return x

    if os.path.exists(docker_ignore):
        with open(docker_ignore, "r") as f:
            patterns = [x.strip() for x in f.readlines()]
            patterns = [strip_slash(x)
                        for x in patterns if not x.startswith("#")]

    def ignore(_, names):
        import fnmatch
        res = set()
        for p in patterns:
            res.update(set(fnmatch.filter(names, p)))
        return list(res)

    return ignore


def build_image(name=DEFAULT_IMAGE_NAME, mlflow_home=None):
    """
    Build an MLflow Docker image.
    The image is built locally and it requires Docker to run.

    :param name: Docker image name.
    :param mlflow_home: Path to a local copy of the MLflow GitHub repository. If specified, the
                        image will install MLflow from this directory. Otherwise, it will install
                        MLflow from pip.
    """
    with TempDir() as tmp:
        cwd = tmp.path()
        if mlflow_home:
            mlflow_dir = _copy_project(
                src_path=mlflow_home, dst_path=cwd)
            install_mlflow = (
                "COPY {mlflow_dir} /opt/mlflow\n"
                "RUN pip install /opt/mlflow\n"
                "RUN cd /opt/mlflow/mlflow/java/scoring &&"
                " mvn --batch-mode package -DskipTests &&"
                " mkdir -p /opt/java/jars &&"
                " mv /opt/mlflow/mlflow/java/scoring/target/"
                "mlflow-scoring-*-with-dependencies.jar /opt/java/jars\n"
            ).format(mlflow_dir=mlflow_dir)
        else:
            install_mlflow = (
                "RUN pip install mlflow=={version}\n"
                "RUN mvn --batch-mode dependency:copy"
                " -Dartifact=org.mlflow:mlflow-scoring:{version}:pom"
                " -DoutputDirectory=/opt/java\n"
                "RUN mvn --batch-mode dependency:copy"
                " -Dartifact=org.mlflow:mlflow-scoring:{version}:jar"
                " -DoutputDirectory=/opt/java/jars\n"
                "RUN cd /opt/java && mv mlflow-scoring-{version}.pom pom.xml &&"
                " mvn --batch-mode dependency:copy-dependencies -DoutputDirectory=/opt/java/jars\n"
                "RUN rm /opt/java/pom.xml\n"
            ).format(version=mlflow.version.VERSION)

        with open(os.path.join(cwd, "Dockerfile"), "w") as f:
            f.write(_DOCKERFILE_TEMPLATE % install_mlflow)
        _logger.info("building docker image")
        os.system('find {cwd}/'.format(cwd=cwd))
        proc = Popen(["docker", "build", "-t", name, "-f", "Dockerfile", "."],
                     cwd=cwd,
                     stdout=PIPE,
                     stderr=STDOUT,
                     universal_newlines=True)
        for x in iter(proc.stdout.readline, ""):
            eprint(x, end='')


_full_template = "{account}.dkr.ecr.{region}.amazonaws.com/{image}:{version}"


def push_image_to_ecr(image=DEFAULT_IMAGE_NAME):
    """
    Push local Docker image to AWS ECR.

    The image is pushed under currently active AWS account and to the currently active AWS region.

    :param image: Docker image name.
    """
    _logger.info("Pushing image to ECR")
    client = boto3.client("sts")
    caller_id = client.get_caller_identity()
    account = caller_id['Account']
    my_session = boto3.session.Session()
    region = my_session.region_name or "us-west-2"
    fullname = _full_template.format(account=account, region=region, image=image,
                                     version=mlflow.version.VERSION)
    _logger.info("Pushing docker image %s to %s", image, fullname)
    ecr_client = boto3.client('ecr')
    try:
        ecr_client.describe_repositories(repositoryNames=[image])['repositories']
    except ecr_client.exceptions.RepositoryNotFoundException:
        ecr_client.create_repository(repositoryName=image)
        print("Created new ECR repository: {repository_name}".format(repository_name=image))
    # TODO: it would be nice to translate the docker login, tag and push to python api.
    # x = ecr_client.get_authorization_token()['authorizationData'][0]
    # docker_login_cmd = "docker login -u AWS -p {token} {url}".format(token=x['authorizationToken']
    #                                                                ,url=x['proxyEndpoint'])
    docker_login_cmd = "$(aws ecr get-login --no-include-email)"
    docker_tag_cmd = "docker tag {image} {fullname}".format(
        image=image, fullname=fullname)
    docker_push_cmd = "docker push {}".format(fullname)
    cmd = ";\n".join([docker_login_cmd, docker_tag_cmd, docker_push_cmd])
    os.system(cmd)


def deploy(app_name, model_path, execution_role_arn=None, bucket=None, run_id=None,
           image_url=None, region_name="us-west-2", mode=DEPLOYMENT_MODE_CREATE, archive=False,
           instance_type=DEFAULT_SAGEMAKER_INSTANCE_TYPE,
           instance_count=DEFAULT_SAGEMAKER_INSTANCE_COUNT, vpc_config=None, flavor=None,
           synchronous=True, timeout_seconds=1200):
    """
    Deploy an MLflow model on AWS SageMaker.
    The currently active AWS account must have correct permissions set up.

    This function creates a SageMaker endpoint. For more information about the input data
    formats accepted by this endpoint, see the
    :ref:`MLflow deployment tools documentation <sagemaker_deployment>`.

    :param app_name: Name of the deployed application.
    :param path: Path to the model. Either local if no ``run_id`` or MLflow-relative if ``run_id``
                 is specified.
    :param execution_role_arn: The name of an IAM role granting the SageMaker service permissions to
                               access the specified Docker image and S3 bucket containing MLflow
                               model artifacts. If unspecified, the currently-assumed role will be
                               used. This execution role is passed to the SageMaker service when
                               creating a SageMaker model from the specified MLflow model. It is
                               passed as the ``ExecutionRoleArn`` parameter of the `SageMaker
                               CreateModel API call <https://docs.aws.amazon.com/sagemaker/latest/
                               dg/API_CreateModel.html>`_. This role is *not* assumed for any other
                               call. For more information about SageMaker execution roles for model
                               creation, see
                               https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html.
    :param bucket: S3 bucket where model artifacts will be stored. Defaults to a
                   SageMaker-compatible bucket name.
    :param run_id: MLflow run ID.
    :param image: Name of the Docker image to be used. if not specified, uses a
                  publicly-available pre-built image.
    :param region_name: Name of the AWS region to which to deploy the application.
    :param mode: The mode in which to deploy the application. Must be one of the following:

                 ``mlflow.sagemaker.DEPLOYMENT_MODE_CREATE``
                     Create an application with the specified name and model. This fails if an
                     application of the same name already exists.

                 ``mlflow.sagemaker.DEPLOYMENT_MODE_REPLACE``
                     If an application of the specified name exists, its model(s) is replaced with
                     the specified model. If no such application exists, it is created with the
                     specified name and model.

                 ``mlflow.sagemaker.DEPLOYMENT_MODE_ADD``
                     Add the specified model to a pre-existing application with the specified name,
                     if one exists. If the application does not exist, a new application is created
                     with the specified name and model. NOTE: If the application **already exists**,
                     the specified model is added to the application's corresponding SageMaker
                     endpoint with an initial weight of zero (0). To route traffic to the model,
                     update the application's associated endpoint configuration using either the
                     AWS console or the ``UpdateEndpointWeightsAndCapacities`` function defined in
                     https://docs.aws.amazon.com/sagemaker/latest/dg/API_UpdateEndpointWeightsAndCapacities.html.

    :param archive: If ``True``, any pre-existing SageMaker application resources that become
                    inactive (i.e. as a result of deploying in
                    ``mlflow.sagemaker.DEPLOYMENT_MODE_REPLACE`` mode) are preserved.
                    These resources may include unused SageMaker models and endpoint configurations
                    that were associated with a prior version of the application endpoint. If
                    ``False``, these resources are deleted. In order to use ``archive=False``,
                    ``deploy()`` must be executed synchronously with ``synchronous=True``.
    :param instance_type: The type of SageMaker ML instance on which to deploy the model. For a list
                          of supported instance types, see
                          https://aws.amazon.com/sagemaker/pricing/instance-types/.
    :param instance_count: The number of SageMaker ML instances on which to deploy the model.
    :param vpc_config: A dictionary specifying the VPC configuration to use when creating the
                       new SageMaker model associated with this application. The acceptable values
                       for this parameter are identical to those of the ``VpcConfig`` parameter in
                       the SageMaker boto3 client (https://boto3.readthedocs.io/en/latest/reference/
                       services/sagemaker.html#SageMaker.Client.create_model). For more information,
                       see https://docs.aws.amazon.com/sagemaker/latest/dg/API_VpcConfig.html.

                       Example:

                       >>> import mlflow.sagemaker as mfs
                       >>> vpc_config = {
                       ...                  'SecurityGroupIds': [
                       ...                      'sg-123456abc',
                       ...                  ],
                       ...                  'Subnets': [
                       ...                      'subnet-123456abc',
                       ...                  ]
                       ...              }
                       >>> mfs.deploy(..., vpc_config=vpc_config)

    :param flavor: The name of the flavor of the model to use for deployment. Must be either
                   ``None`` or one of mlflow.sagemaker.SUPPORTED_DEPLOYMENT_FLAVORS. If ``None``,
                   a flavor is automatically selected from the model's available flavors. If the
                   specified flavor is not present or not supported for deployment, an exception
                   will be thrown.
    :param synchronous: If `True`, this function will block until the deployment process succeeds
                        or encounters an irrecoverable failure. If `False`, this function will
                        return immediately after starting the deployment process. It will not wait
                        for the deployment process to complete; in this case, the caller is
                        responsible for monitoring the health and status of the pending deployment
                        via native SageMaker APIs or the AWS console.
    :param timeout_seconds: If `synchronous` is `True`, the deployment process will return after the
                            specified number of seconds if no definitive result (success or failure)
                            is achieved. Once the function returns, the caller is responsible
                            for monitoring the health and status of the pending deployment via
                            native SageMaker APIs or the AWS console. If `synchronous` is False,
                            this parameter is ignored.
    """
    if (not archive) and (not synchronous):
        raise MlflowException(
            message=(
                "Resources must be archived when `deploy()` is executed in non-synchronous mode."
                " Either set `synchronous=True` or `archive=True`."),
            error_code=INVALID_PARAMETER_VALUE)

    if mode not in DEPLOYMENT_MODES:
        raise MlflowException(
                message="`mode` must be one of: {deployment_modes}".format(
                    deployment_modes=",".join(DEPLOYMENT_MODES)),
                error_code=INVALID_PARAMETER_VALUE)

    s3_bucket_prefix = model_path
    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)
        s3_bucket_prefix = os.path.join(run_id, s3_bucket_prefix)

    model_config_path = os.path.join(model_path, "MLmodel")
    if not os.path.exists(model_config_path):
        raise MlflowException(
            message=(
                "Failed to find MLmodel configuration within the specified model's"
                " root directory."),
            error_code=INVALID_PARAMETER_VALUE)
    model_config = Model.load(model_config_path)

    if flavor is None:
        flavor = _get_preferred_deployment_flavor(model_config)
    else:
        _validate_deployment_flavor(model_config, flavor)
    _logger.info("Using the %s flavor for deployment!", flavor)

    sage_client = boto3.client('sagemaker', region_name=region_name)
    s3_client = boto3.client('s3', region_name=region_name)

    endpoint_exists = _find_endpoint(endpoint_name=app_name, sage_client=sage_client) is not None
    if endpoint_exists and mode == DEPLOYMENT_MODE_CREATE:
        raise MlflowException(
                message=(
                    "You are attempting to deploy an application with name: {application_name} in"
                    " '{mode_create}' mode. However, an application with the same name already"
                    " exists. If you want to update this application, deploy in '{mode_add}' or"
                    " '{mode_replace}' mode.".format(
                        application_name=app_name,
                        mode_create=DEPLOYMENT_MODE_CREATE,
                        mode_add=DEPLOYMENT_MODE_ADD,
                        mode_replace=DEPLOYMENT_MODE_REPLACE)),
                error_code=INVALID_PARAMETER_VALUE)

    if not image_url:
        image_url = _get_default_image_url(region_name=region_name)
    if not execution_role_arn:
        execution_role_arn = _get_assumed_role_arn()
    if not bucket:
        _logger.info("No model data bucket specified, using the default bucket")
        bucket = _get_default_s3_bucket(region_name)

    model_s3_path = _upload_s3(local_model_path=model_path,
                               bucket=bucket,
                               prefix=s3_bucket_prefix,
                               region_name=region_name,
                               s3_client=s3_client)
    if endpoint_exists:
        deployment_operation = _update_sagemaker_endpoint(
                endpoint_name=app_name, image_url=image_url, model_s3_path=model_s3_path,
                run_id=run_id, flavor=flavor, instance_type=instance_type,
                instance_count=instance_count, vpc_config=vpc_config, mode=mode,
                role=execution_role_arn, sage_client=sage_client, s3_client=s3_client)
    else:
        deployment_operation = _create_sagemaker_endpoint(
                endpoint_name=app_name, image_url=image_url, model_s3_path=model_s3_path,
                run_id=run_id, flavor=flavor, instance_type=instance_type,
                instance_count=instance_count, vpc_config=vpc_config, role=execution_role_arn,
                sage_client=sage_client)

    if synchronous:
        _logger.info("Waiting for the deployment operation to complete...")
        operation_status = deployment_operation.await_completion(timeout_seconds=timeout_seconds)
        if operation_status.state == _SageMakerOperationStatus.STATE_SUCCEEDED:
            _logger.info("The deployment operation completed successfully with message: \"%s\"",
                         operation_status.message)
        else:
            raise MlflowException(
                "The deployment operation failed with the following error message:"
                " \"{error_message}\"".format(error_message=operation_status.message))
        if not archive:
            deployment_operation.clean_up()


def _get_preferred_deployment_flavor(model_config):
    """
    Obtains the flavor that MLflow would prefer to use when deploying the model.
    If the model does not contain any supported flavors for deployment, an exception
    will be thrown.

    :param model_config: An MLflow model object
    :return: The name of the preferred deployment flavor for the specified model
    """
    if mleap.FLAVOR_NAME in model_config.flavors:
        return mleap.FLAVOR_NAME
    elif pyfunc.FLAVOR_NAME in model_config.flavors:
        return pyfunc.FLAVOR_NAME
    else:
        raise MlflowException(
                message=(
                    "The specified model does not contain any of the supported flavors for"
                    " deployment. The model contains the following flavors: {model_flavors}."
                    " Supported flavors: {supported_flavors}".format(
                        model_flavors=model_config.flavors.keys(),
                        supported_flavors=SUPPORTED_DEPLOYMENT_FLAVORS)),
                error_code=RESOURCE_DOES_NOT_EXIST)


def _validate_deployment_flavor(model_config, flavor):
    """
    Checks that the specified flavor is a supported deployment flavor
    and is contained in the specified model. If one of these conditions
    is not met, an exception is thrown.

    :param model_config: An MLflow Model object
    :param flavor: The deployment flavor to validate
    """
    if flavor not in SUPPORTED_DEPLOYMENT_FLAVORS:
        raise MlflowException(
                message=(
                    "The specified flavor: `{flavor_name}` is not supported for deployment."
                    " Please use one of the supported flavors: {supported_flavor_names}".format(
                        flavor_name=flavor,
                        supported_flavor_names=SUPPORTED_DEPLOYMENT_FLAVORS)),
                error_code=INVALID_PARAMETER_VALUE)
    elif flavor not in model_config.flavors:
        raise MlflowException(
                message=("The specified model does not contain the specified deployment flavor:"
                         " `{flavor_name}`. Please use one of the following deployment flavors"
                         " that the model contains: {model_flavors}".format(
                             flavor_name=flavor, model_flavors=model_config.flavors.keys())),
                error_code=RESOURCE_DOES_NOT_EXIST)


def delete(app_name, region_name="us-west-2", archive=False, synchronous=True, timeout_seconds=300):
    """
    Delete a SageMaker application.

    :param app_name: Name of the deployed application.
    :param region_name: Name of the AWS region in which the application is deployed.
    :param archive: If ``True``, resources associated with the specified application, such
                    as its associated models and endpoint configuration, are preserved.
                    If ``False``, these resources are deleted. In order to use
                    ``archive=False``, ``delete()`` must be executed synchronously with
                    ``synchronous=True``.
    :param synchronous: If `True`, this function blocks until the deletion process succeeds
                        or encounters an irrecoverable failure. If `False`, this function
                        returns immediately after starting the deletion process. It will not wait
                        for the deletion process to complete; in this case, the caller is
                        responsible for monitoring the status of the deletion process via native
                        SageMaker APIs or the AWS console.
    :param timeout_seconds: If `synchronous` is `True`, the deletion process returns after the
                            specified number of seconds if no definitive result (success or failure)
                            is achieved. Once the function returns, the caller is responsible
                            for monitoring the status of the deletion process via native SageMaker
                            APIs or the AWS console. If `synchronous` is False, this parameter
                            is ignored.
    """
    if (not archive) and (not synchronous):
        raise MlflowException(
            message=(
                "Resources must be archived when `deploy()` is executed in non-synchronous mode."
                " Either set `synchronous=True` or `archive=True`."),
            error_code=INVALID_PARAMETER_VALUE)

    s3_client = boto3.client('s3', region_name=region_name)
    sage_client = boto3.client('sagemaker', region_name=region_name)

    endpoint_info = sage_client.describe_endpoint(EndpointName=app_name)
    endpoint_arn = endpoint_info["EndpointArn"]

    sage_client.delete_endpoint(EndpointName=app_name)
    _logger.info("Deleted endpoint with arn: %s", endpoint_arn)

    def status_check_fn():
        endpoint_info = _find_endpoint(endpoint_name=app_name, sage_client=sage_client)
        if endpoint_info is not None:
            return _SageMakerOperationStatus.in_progress(
                "Deletion is still in progress. Current endpoint status: {endpoint_status}".format(
                    endpoint_status=endpoint_info["EndpointStatus"]))
        else:
            return _SageMakerOperationStatus.succeeded(
                    "The SageMaker endpoint was deleted successfully.")

    def cleanup_fn():
        _logger.info("Cleaning up unused resources...")
        config_name = endpoint_info["EndpointConfigName"]
        config_info = sage_client.describe_endpoint_config(
            EndpointConfigName=config_name)
        config_arn = config_info["EndpointConfigArn"]
        sage_client.delete_endpoint_config(EndpointConfigName=config_name)
        _logger.info("Deleted associated endpoint configuration with arn: %s", config_arn)
        for pv in config_info["ProductionVariants"]:
            model_name = pv["ModelName"]
            model_arn = _delete_sagemaker_model(
                model_name, sage_client, s3_client)
            _logger.info("Deleted associated model with arn: %s", model_arn)

    delete_operation = _SageMakerOperation(status_check_fn=status_check_fn, cleanup_fn=cleanup_fn)

    if synchronous:
        _logger.info("Waiting for the delete operation to complete...")
        operation_status = delete_operation.await_completion(timeout_seconds=timeout_seconds)
        if operation_status.state == _SageMakerOperationStatus.STATE_SUCCEEDED:
            _logger.info("The deletion operation completed successfully with message: \"%s\"",
                         operation_status.message)
        else:
            raise MlflowException(
                "The deletion operation failed with the following error message:"
                " \"{error_message}\"".format(error_message=operation_status.message))
        if not archive:
            delete_operation.clean_up()


def run_local(model_path, run_id=None, port=5000, image=DEFAULT_IMAGE_NAME, flavor=None):
    """
    Serve model locally in a SageMaker compatible Docker container.

    :param model_path: path to the model. Either local if no ``run_id`` or MLflow-relative if
                                          ``run_id`` is specified.
    :param run_id: MLflow run ID.
    :param port: Local port.
    :param image: Name of the Docker image to be used.
    :param flavor: The name of the flavor of the model to use for local serving. If ``None``,
                   a flavor is automatically selected from the model's available flavors. If the
                   specified flavor is not present or not supported for deployment, an exception
                   is thrown.
    """
    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)
    model_path = os.path.abspath(model_path)
    model_config_path = os.path.join(model_path, "MLmodel")
    model_config = Model.load(model_config_path)

    if flavor is None:
        flavor = _get_preferred_deployment_flavor(model_config)
    else:
        _validate_deployment_flavor(model_config, flavor)
    print("Using the {selected_flavor} flavor for local serving!".format(selected_flavor=flavor))

    deployment_config = _get_deployment_config(flavor_name=flavor)

    _logger.info("launching docker image with path %s", model_path)
    cmd = ["docker", "run", "-v", "{}:/opt/ml/model/".format(model_path), "-p", "%d:8080" % port]
    for key, value in deployment_config.items():
        cmd += ["-e", "{key}={value}".format(key=key, value=value)]
    cmd += ["--rm", image, "serve"]
    _logger.info('executing: %s', ' '.join(cmd))
    proc = Popen(cmd, stdout=PIPE, stderr=STDOUT, universal_newlines=True)

    def _sigterm_handler(*_):
        _logger.info("received termination signal => killing docker process")
        proc.send_signal(signal.SIGINT)

    import signal
    signal.signal(signal.SIGTERM, _sigterm_handler)
    for x in iter(proc.stdout.readline, ""):
        eprint(x, end='')


def _get_default_image_url(region_name):
    env_img = os.environ.get(IMAGE_NAME_ENV_VAR)
    if env_img:
        return env_img

    ecr_client = boto3.client("ecr", region_name=region_name)
    repository_conf = ecr_client.describe_repositories(
        repositoryNames=[DEFAULT_IMAGE_NAME])['repositories'][0]
    return (repository_conf["repositoryUri"] + ":{version}").format(version=mlflow.version.VERSION)


def _get_account_id():
    sess = boto3.Session()
    sts_client = sess.client("sts")
    identity_info = sts_client.get_caller_identity()
    account_id = identity_info["Account"]
    return account_id


def _get_assumed_role_arn():
    """
    :return: ARN of the user's current IAM role.
    """
    sess = boto3.Session()
    sts_client = sess.client("sts")
    identity_info = sts_client.get_caller_identity()
    sts_arn = identity_info["Arn"]
    role_name = sts_arn.split("/")[1]
    iam_client = sess.client("iam")
    role_response = iam_client.get_role(RoleName=role_name)
    return role_response["Role"]["Arn"]


def _get_default_s3_bucket(region_name):
    # create bucket if it does not exist
    sess = boto3.Session()
    account_id = _get_account_id()
    bucket_name = "{pfx}-{rn}-{aid}".format(pfx=DEFAULT_BUCKET_NAME_PREFIX, rn=region_name,
                                            aid=account_id)
    s3 = sess.client('s3')
    response = s3.list_buckets()
    buckets = [b['Name'] for b in response["Buckets"]]
    if bucket_name not in buckets:
        _logger.info("Default bucket `%s` not found. Creating...", bucket_name)
        bucket_creation_kwargs = {
            'ACL': 'bucket-owner-full-control',
            'Bucket': bucket_name,
        }
        if region_name != "us-east-1":
            # The location constraint is required during bucket creation for all regions
            # outside of us-east-1. This constraint cannot be specified in us-east-1;
            # specifying it in this region results in a failure, so we will only
            # add it if we are deploying outside of us-east-1.
            # See https://docs.aws.amazon.com/cli/latest/reference/s3api/create-bucket.html#examples
            bucket_creation_kwargs['CreateBucketConfiguration'] = {
                'LocationConstraint': region_name
            }
        response = s3.create_bucket(**bucket_creation_kwargs)
        _logger.info("Bucket creation response: %s", response)
    else:
        _logger.info("Default bucket `%s` already exists. Skipping creation.", bucket_name)
    return bucket_name


def _make_tarfile(output_filename, source_dir):
    """
    create a tar.gz from a directory.
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        for f in os.listdir(source_dir):
            tar.add(os.path.join(source_dir, f), arcname=f)


def _upload_s3(local_model_path, bucket, prefix, region_name, s3_client):
    """
    Upload dir to S3 as .tar.gz.
    :param local_model_path: Local path to a dir.
    :param bucket: S3 bucket where to store the data.
    :param prefix: Path within the bucket.
    :param region_name: The AWS region in which to upload data to S3.
    :param s3_client: A boto3 client for S3.
    :return: S3 path of the uploaded artifact.
    """
    sess = boto3.Session(region_name=region_name)
    with TempDir() as tmp:
        model_data_file = tmp.path("model.tar.gz")
        _make_tarfile(model_data_file, local_model_path)
        with open(model_data_file, 'rb') as fobj:
            key = os.path.join(prefix, 'model.tar.gz')
            obj = sess.resource('s3').Bucket(bucket).Object(key)
            obj.upload_fileobj(fobj)
            response = s3_client.put_object_tagging(
                Bucket=bucket,
                Key=key,
                Tagging={'TagSet': [{'Key': 'SageMaker', 'Value': 'true'}, ]}
            )
            _logger.info('tag response: %s', response)
            return '{}/{}/{}'.format(s3_client.meta.endpoint_url, bucket, key)


def _get_deployment_config(flavor_name):
    """
    :return: The deployment configuration as a dictionary
    """
    deployment_config = {DEPLOYMENT_CONFIG_KEY_FLAVOR_NAME: flavor_name}
    return deployment_config


def _get_sagemaker_model_name(endpoint_name):
    return "{en}-model-{uid}".format(en=endpoint_name, uid=get_unique_resource_id())


def _get_sagemaker_config_name(endpoint_name):
    return "{en}-config-{uid}".format(en=endpoint_name, uid=get_unique_resource_id())


def _create_sagemaker_endpoint(endpoint_name, image_url, model_s3_path, run_id, flavor,
                               instance_type, vpc_config, instance_count, role, sage_client):
    """
    :param image_url: URL of the ECR-hosted docker image the model is being deployed into.
    :param model_s3_path: S3 path where we stored the model artifacts.
    :param run_id: Run ID that generated this model.
    :param flavor: The name of the flavor of the model to use for deployment.
    :param instance_type: The type of SageMaker ML instance on which to deploy the model.
    :param instance_count: The number of SageMaker ML instances on which to deploy the model.
    :param vpc_config: A dictionary specifying the VPC configuration to use when creating the
                       new SageMaker model associated with this SageMaker endpoint.
    :param role: SageMaker execution ARN role.
    :param sage_client: A boto3 client for SageMaker.
    """
    _logger.info("Creating new endpoint with name: %s ...", endpoint_name)

    model_name = _get_sagemaker_model_name(endpoint_name)
    model_response = _create_sagemaker_model(model_name=model_name,
                                             model_s3_path=model_s3_path,
                                             flavor=flavor,
                                             vpc_config=vpc_config,
                                             run_id=run_id,
                                             image_url=image_url,
                                             execution_role=role,
                                             sage_client=sage_client)
    _logger.info("Created model with arn: %s", model_response["ModelArn"])

    production_variant = {
        'VariantName': model_name,
        'ModelName': model_name,
        'InitialInstanceCount': instance_count,
        'InstanceType': instance_type,
        'InitialVariantWeight': 1,
    }
    config_name = _get_sagemaker_config_name(endpoint_name)
    endpoint_config_response = sage_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[production_variant],
        Tags=[
            {
                'Key': 'app_name',
                'Value': endpoint_name,
            },
        ],
    )
    _logger.info("Created endpoint configuration with arn: %s",
                 endpoint_config_response["EndpointConfigArn"])

    endpoint_response = sage_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
        Tags=[],
    )
    _logger.info("Created endpoint with arn: %s", endpoint_response["EndpointArn"])

    def status_check_fn():
        endpoint_info = _find_endpoint(endpoint_name=endpoint_name, sage_client=sage_client)

        if endpoint_info is None:
            return _SageMakerOperationStatus.in_progress("Waiting for endpoint to be created...")

        endpoint_status = endpoint_info["EndpointStatus"]
        if endpoint_status == "Creating":
            return _SageMakerOperationStatus.in_progress(
                "Waiting for endpoint to reach the \"InService\" state. Current endpoint status:"
                " \"{endpoint_status}\"".format(endpoint_status=endpoint_status))
        elif endpoint_status == "InService":
            return _SageMakerOperationStatus.succeeded(
                    "The SageMaker endpoint was created successfully.")
        else:
            failure_reason = endpoint_info.get(
                "FailureReason",
                ("An unknown SageMaker failure occurred. Please see the SageMaker console logs for"
                 " more information."))
            return _SageMakerOperationStatus.failed(failure_reason)

    def cleanup_fn():
        pass

    return _SageMakerOperation(status_check_fn=status_check_fn, cleanup_fn=cleanup_fn)


def _update_sagemaker_endpoint(endpoint_name, image_url, model_s3_path, run_id, flavor,
                               instance_type, instance_count, vpc_config, mode, role,
                               sage_client, s3_client):
    """
    :param image_url: URL of the ECR-hosted Docker image the model is being deployed into
    :param model_s3_path: S3 path where we stored the model artifacts
    :param run_id: Run ID that generated this model
    :param flavor: The name of the flavor of the model to use for deployment.
    :param instance_type: The type of SageMaker ML instance on which to deploy the model.
    :param instance_count: The number of SageMaker ML instances on which to deploy the model.
    :param vpc_config: A dictionary specifying the VPC configuration to use when creating the
                       new SageMaker model associated with this SageMaker endpoint.
    :param mode: either mlflow.sagemaker.DEPLOYMENT_MODE_ADD or
                 mlflow.sagemaker.DEPLOYMENT_MODE_REPLACE.
    :param role: SageMaker execution ARN role.
    :param sage_client: A boto3 client for SageMaker.
    :param s3_client: A boto3 client for S3.
    """
    if mode not in [DEPLOYMENT_MODE_ADD, DEPLOYMENT_MODE_REPLACE]:
        msg = "Invalid mode `{md}` for deployment to a pre-existing application".format(
            md=mode)
        raise ValueError(msg)

    endpoint_info = sage_client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_arn = endpoint_info["EndpointArn"]
    deployed_config_name = endpoint_info["EndpointConfigName"]
    deployed_config_info = sage_client.describe_endpoint_config(
        EndpointConfigName=deployed_config_name)
    deployed_config_arn = deployed_config_info["EndpointConfigArn"]
    deployed_production_variants = deployed_config_info["ProductionVariants"]

    _logger.info("Found active endpoint with arn: %s. Updating...", endpoint_arn)

    new_model_name = _get_sagemaker_model_name(endpoint_name)
    new_model_response = _create_sagemaker_model(model_name=new_model_name,
                                                 model_s3_path=model_s3_path,
                                                 flavor=flavor,
                                                 vpc_config=vpc_config,
                                                 run_id=run_id,
                                                 image_url=image_url,
                                                 execution_role=role,
                                                 sage_client=sage_client)
    _logger.info("Created new model with arn: %s", new_model_response["ModelArn"])

    if mode == DEPLOYMENT_MODE_ADD:
        new_model_weight = 0
        production_variants = deployed_production_variants
    elif mode == DEPLOYMENT_MODE_REPLACE:
        new_model_weight = 1
        production_variants = []

    new_production_variant = {
        'VariantName': new_model_name,
        'ModelName': new_model_name,
        'InitialInstanceCount': instance_count,
        'InstanceType': instance_type,
        'InitialVariantWeight': new_model_weight
    }
    production_variants.append(new_production_variant)

    # Create the new endpoint configuration and update the endpoint
    # to adopt the new configuration
    new_config_name = _get_sagemaker_config_name(endpoint_name)
    endpoint_config_response = sage_client.create_endpoint_config(
        EndpointConfigName=new_config_name,
        ProductionVariants=production_variants,
        Tags=[
            {
                'Key': 'app_name',
                'Value': endpoint_name,
            },
        ],
    )
    _logger.info("Created new endpoint configuration with arn: %s",
                 endpoint_config_response["EndpointConfigArn"])

    sage_client.update_endpoint(EndpointName=endpoint_name,
                                EndpointConfigName=new_config_name)
    _logger.info("Updated endpoint with new configuration!")

    operation_start_time = time.time()

    def status_check_fn():
        if time.time() - operation_start_time < 20:
            # Wait at least 20 seconds before checking the status of the update; this ensures
            # that we don't consider the operation to have failed if small delays occur at
            # initialization time
            return _SageMakerOperationStatus.in_progress()

        endpoint_info = sage_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_update_was_rolled_back = (
                endpoint_info["EndpointStatus"] == "InService"
                and endpoint_info["EndpointConfigName"] != new_config_name)
        if endpoint_update_was_rolled_back or endpoint_info["EndpointStatus"] == "Failed":
            failure_reason = endpoint_info.get(
                "FailureReason",
                ("An unknown SageMaker failure occurred. Please see the SageMaker console logs for"
                 " more information."))
            return _SageMakerOperationStatus.failed(failure_reason)
        elif endpoint_info["EndpointStatus"] == "InService":
            return _SageMakerOperationStatus.succeeded(
                    "The SageMaker endpoint was updated successfully.")
        else:
            return _SageMakerOperationStatus.in_progress(
                "The update operation is still in progress. Current endpoint status:"
                " \"{endpoint_status}\"".format(endpoint_status=endpoint_info["EndpointStatus"]))

    def cleanup_fn():
        _logger.info("Cleaning up unused resources...")
        if mode == DEPLOYMENT_MODE_REPLACE:
            for pv in deployed_production_variants:
                deployed_model_arn = _delete_sagemaker_model(model_name=pv["ModelName"],
                                                             sage_client=sage_client,
                                                             s3_client=s3_client)
                _logger.info("Deleted model with arn: %s", deployed_model_arn)

        sage_client.delete_endpoint_config(
            EndpointConfigName=deployed_config_name)
        _logger.info("Deleted endpoint configuration with arn: %s", deployed_config_arn)

    return _SageMakerOperation(status_check_fn=status_check_fn, cleanup_fn=cleanup_fn)


def _create_sagemaker_model(model_name, model_s3_path, flavor, vpc_config, run_id, image_url,
                            execution_role, sage_client):
    """
    :param model_s3_path: S3 path where the model artifacts are stored.
    :param flavor: The name of the flavor of the model.
    :param vpc_config: A dictionary specifying the VPC configuration to use when creating the
                       new SageMaker model associated with this SageMaker endpoint.
    :param run_id: Run ID that generated this model.
    :param image_url: URL of the ECR-hosted Docker image that will serve as the
                      model's container,
    :param execution_role: The ARN of the role that SageMaker will assume when creating the model.
    :param sage_client: A boto3 client for SageMaker.
    :return: AWS response containing metadata associated with the new model.
    """
    create_model_args = {
        "ModelName": model_name,
        "PrimaryContainer": {
            'ContainerHostname': 'mfs-%s' % model_name,
            'Image': image_url,
            'ModelDataUrl': model_s3_path,
            'Environment': _get_deployment_config(flavor_name=flavor),
        },
        "ExecutionRoleArn": execution_role,
        "Tags": [{'Key': 'run_id', 'Value': str(run_id)}],
    }
    if vpc_config is not None:
        create_model_args["VpcConfig"] = vpc_config

    model_response = sage_client.create_model(**create_model_args)
    return model_response


def _delete_sagemaker_model(model_name, sage_client, s3_client):
    """
    :param sage_client: A boto3 client for SageMaker.
    :param s3_client: A boto3 client for S3.
    :return: ARN of the deleted model.
    """
    model_info = sage_client.describe_model(ModelName=model_name)
    model_arn = model_info["ModelArn"]
    model_data_url = model_info["PrimaryContainer"]["ModelDataUrl"]

    # Parse the model data url to obtain a bucket path. The following
    # procedure is safe due to the well-documented structure of the `ModelDataUrl`
    # (see https://docs.aws.amazon.com/sagemaker/latest/dg/API_ContainerDefinition.html)
    parsed_data_url = urllib.parse.urlparse(model_data_url)
    bucket_data_path = parsed_data_url.path.split("/")
    bucket_name = bucket_data_path[1]
    bucket_key = "/".join(bucket_data_path[2:])

    s3_client.delete_object(Bucket=bucket_name,
                            Key=bucket_key)
    sage_client.delete_model(ModelName=model_name)

    return model_arn


def _delete_sagemaker_endpoint_configuration(endpoint_config_name, sage_client):
    """
    :param sage_client: A boto3 client for SageMaker.
    :return: ARN of the deleted endpoint configuration.
    """
    endpoint_config_info = sage_client.describe_endpoint_config(
            EndpointConfigName=endpoint_config_name)
    sage_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    return endpoint_config_info["EndpointConfigArn"]


def _find_endpoint(endpoint_name, sage_client):
    """
    Finds a SageMaker endpoint with the specified name in the caller's AWS account, returning a
    NoneType if the endpoint is not found.

    :param sage_client: A boto3 client for SageMaker.
    :return: If the endpoint exists, a dictionary of endpoint attributes. If the endpoint does not
             exist, `None`.
    """
    endpoints_page = sage_client.list_endpoints(
        MaxResults=100, NameContains=endpoint_name)

    while True:
        for endpoint in endpoints_page["Endpoints"]:
            if endpoint["EndpointName"] == endpoint_name:
                return endpoint

        if "NextToken" in endpoints_page:
            endpoints_page = sage_client.list_endpoints(MaxResults=100,
                                                        NextToken=endpoints_page["NextToken"],
                                                        NameContains=endpoint_name)
        else:
            return None


class _SageMakerOperation:

    def __init__(self, status_check_fn, cleanup_fn):
        self.status_check_fn = status_check_fn
        self.cleanup_fn = cleanup_fn
        self.start_time = time.time()
        self.status = _SageMakerOperationStatus(_SageMakerOperationStatus.STATE_IN_PROGRESS, None)
        self.cleaned_up = False

    def await_completion(self, timeout_seconds):
        iteration = 0
        begin = time.time()
        while (time.time() - begin) < timeout_seconds:
            status = self.status_check_fn()
            if status.state == _SageMakerOperationStatus.STATE_IN_PROGRESS:
                if iteration % 4 == 0:
                    # Log the progress status roughly every 20 seconds
                    _logger.info(status.message)

                time.sleep(5)
                iteration += 1
                continue
            else:
                self.status = status
                return status

        duration_seconds = time.time() - begin
        return _SageMakerOperationStatus.timed_out(duration_seconds)

    def clean_up(self):
        if self.status.state != _SageMakerOperationStatus.STATE_SUCCEEDED:
            raise ValueError(
                "Cannot clean up an operation that has not succeeded! Current operation state:"
                " {operation_state}".format(operation_state=self.status.state))

        if not self.cleaned_up:
            self.cleaned_up = True
        else:
            raise ValueError("`clean_up()` has already been executed for this operation!")

        self.cleanup_fn()


class _SageMakerOperationStatus:

    STATE_SUCCEEDED = "succeeded"
    STATE_FAILED = "failed"
    STATE_IN_PROGRESS = "in progress"
    STATE_TIMED_OUT = "timed_out"

    def __init__(self, state, message):
        self.state = state
        self.message = message

    @classmethod
    def in_progress(cls, message=None):
        if message is None:
            message = "The operation is still in progress."
        return cls(_SageMakerOperationStatus.STATE_IN_PROGRESS, message)

    @classmethod
    def timed_out(cls, duration_seconds):
        return cls(_SageMakerOperationStatus.STATE_TIMED_OUT,
                   "Timed out after waiting {duration_seconds} seconds for the operation to"
                   " complete. This operation may still be in progress. Please check the AWS"
                   " console for more information.".format(duration_seconds=duration_seconds))

    @classmethod
    def failed(cls, message):
        return cls(_SageMakerOperationStatus.STATE_FAILED, message)

    @classmethod
    def succeeded(cls, message):
        return cls(_SageMakerOperationStatus.STATE_SUCCEEDED, message)

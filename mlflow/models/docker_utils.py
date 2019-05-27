import os
from subprocess import Popen, PIPE, STDOUT
import logging

import mlflow
import mlflow.version
from mlflow import pyfunc, mleap
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, INVALID_PARAMETER_VALUE
from mlflow.utils.file_utils import TempDir, _copy_project
from mlflow.utils.logging_utils import eprint

_logger = logging.getLogger(__name__)

DISABLE_ENV_CREATION = "MLFLOW_DISABLE_ENV_CREATION"

SUPPORTED_DEPLOYMENT_FLAVORS = [
    pyfunc.FLAVOR_NAME,
    mleap.FLAVOR_NAME,
]

DEFAULT_IMAGE_NAME = "mlflow-pyfunc"

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
ENV PATH="/miniconda/bin:$PATH"
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

RUN conda install gunicorn;\
    conda install gevent;\

{install_mlflow}
{custom_setup_steps}

# Set up the program in the image
WORKDIR /opt/mlflow

# start mlflow scoring
ENTRYPOINT ["python", "-c", "import sys; from mlflow.sagemaker import container as C; \
C._init(sys.argv[1])"]
"""


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


def _get_mlflow_install_step(dockerfile_context_dir, mlflow_home):
    """
    Get docker build commands for installing MLflow given a Docker context dir and optional source
    directory
    """
    if mlflow_home:
        mlflow_dir = _copy_project(
            src_path=mlflow_home, dst_path=dockerfile_context_dir)
        return (
            "COPY {mlflow_dir} /opt/mlflow\n"
            "RUN pip install /opt/mlflow\n"
        ).format(mlflow_dir=mlflow_dir)
    else:
        return (
            "RUN pip install mlflow=={version}\n"
            "RUN mvn --batch-mode dependency:copy"
            " -Dartifact=org.mlflow:mlflow-scoring:{version}:pom"
            " -DoutputDirectory=/opt/java\n"
            "RUN mvn --batch-mode dependency:copy"
            " -Dartifact=org.mlflow:mlflow-scoring:{version}:jar"
            " -DoutputDirectory=/opt/java/jars\n"
            "RUN cd /opt/java && mv mlflow-scoring-{version}.pom pom.xml &&"
            " mvn --batch-mode dependency:copy-dependencies -DoutputDirectory=/opt/java/jars\n"
            "RUN rm /opt/java/pom.xml"
        ).format(version=mlflow.version.VERSION)


def _build_image(image_name=DEFAULT_IMAGE_NAME, mlflow_home=None, custom_setup_steps_hook=None):
    """
    Build an MLflow Docker image that can be used to serve a
    The image is built locally and it requires Docker to run.

    :param image_name: Docker image name.
    :param mlflow_home: (Optional) Path to a local copy of the MLflow GitHub repository.
                        If specified, the image will install MLflow from this directory.
                        If None, it will install MLflow from pip.
    :param custom_setup_steps_hook: (Optional) Single-argument function that takes the string path
           of a dockerfile context directory and returns a string containing Dockerfile commands to
           run during the image build step.
    """
    mlflow_home = os.path.abspath(mlflow_home) if mlflow_home else None
    with TempDir() as tmp:
        cwd = tmp.path()
        install_mlflow = _get_mlflow_install_step(cwd, mlflow_home)
        custom_setup_steps = custom_setup_steps_hook(cwd) if custom_setup_steps_hook else ""
        with open(os.path.join(cwd, "Dockerfile"), "w") as f:
            f.write(_DOCKERFILE_TEMPLATE.format(
                install_mlflow=install_mlflow, custom_setup_steps=custom_setup_steps))
        _logger.info("Building docker image with name %s", image_name)
        os.system('find {cwd}/'.format(cwd=cwd))
        proc = Popen(["docker", "build", "-t", image_name, "-f", "Dockerfile", "."],
                     cwd=cwd,
                     stdout=PIPE,
                     stderr=STDOUT,
                     universal_newlines=True)
        for x in iter(proc.stdout.readline, ""):
            eprint(x, end='')

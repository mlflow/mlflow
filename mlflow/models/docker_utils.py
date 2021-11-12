import os
from subprocess import Popen, PIPE, STDOUT
import logging

import mlflow
import mlflow.version
from mlflow.utils.file_utils import TempDir, _copy_project
from mlflow.utils.logging_utils import eprint

_logger = logging.getLogger(__name__)

DISABLE_ENV_CREATION = "MLFLOW_DISABLE_ENV_CREATION"

_DOCKERFILE_TEMPLATE = """
# Build an image that can serve mlflow models.
FROM ubuntu:18.04

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
RUN curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh >> miniconda.sh
RUN bash ./miniconda.sh -b -p /miniconda && rm ./miniconda.sh
ENV PATH="/miniconda/bin:$PATH"
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV GUNICORN_CMD_ARGS="--timeout 60 -k gevent"
# Set up the program in the image
WORKDIR /opt/mlflow

{install_mlflow}

{custom_setup_steps}
{entrypoint}
"""


def _get_mlflow_install_step(dockerfile_context_dir, mlflow_home):
    """
    Get docker build commands for installing MLflow given a Docker context dir and optional source
    directory
    """
    if mlflow_home:
        mlflow_dir = _copy_project(src_path=mlflow_home, dst_path=dockerfile_context_dir)
        return (
            "COPY {mlflow_dir} /opt/mlflow\n"
            "RUN pip install /opt/mlflow\n"
            "RUN cd /opt/mlflow/mlflow/java/scoring && "
            "mvn --batch-mode package -DskipTests && "
            "mkdir -p /opt/java/jars && "
            "mv /opt/mlflow/mlflow/java/scoring/target/"
            "mlflow-scoring-*-with-dependencies.jar /opt/java/jars\n"
        ).format(mlflow_dir=mlflow_dir)
    else:
        return (
            "RUN pip install mlflow=={version}\n"
            "RUN mvn "
            " --batch-mode dependency:copy"
            " -Dartifact=org.mlflow:mlflow-scoring:{version}:pom"
            " -DoutputDirectory=/opt/java\n"
            "RUN mvn "
            " --batch-mode dependency:copy"
            " -Dartifact=org.mlflow:mlflow-scoring:{version}:jar"
            " -DoutputDirectory=/opt/java/jars\n"
            "RUN cp /opt/java/mlflow-scoring-{version}.pom /opt/java/pom.xml\n"
            "RUN cd /opt/java && mvn "
            "--batch-mode dependency:copy-dependencies -DoutputDirectory=/opt/java/jars\n"
        ).format(version=mlflow.version.VERSION)


def _build_image(image_name, entrypoint, mlflow_home=None, custom_setup_steps_hook=None):
    """
    Build an MLflow Docker image that can be used to serve a
    The image is built locally and it requires Docker to run.

    :param image_name: Docker image name.
    :param entry_point: String containing ENTRYPOINT directive for docker image
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
            f.write(
                _DOCKERFILE_TEMPLATE.format(
                    install_mlflow=install_mlflow,
                    custom_setup_steps=custom_setup_steps,
                    entrypoint=entrypoint,
                )
            )
        _logger.info("Building docker image with name %s", image_name)
        os.system("find {cwd}/".format(cwd=cwd))
        _build_image_from_context(context_dir=cwd, image_name=image_name)


def _build_image_from_context(context_dir: str, image_name: str):
    import docker

    client = docker.from_env()
    # In Docker < 19, `docker build` doesn't support the `--platform` option
    is_platform_supported = int(client.version()["Version"].split(".")[0]) >= 19
    # Enforcing the AMD64 architecture build for Apple M1 users
    platform_option = ["--platform", "linux/amd64"] if is_platform_supported else []
    commands = [
        "docker",
        "build",
        "-t",
        image_name,
        "-f",
        "Dockerfile",
        *platform_option,
        ".",
    ]
    proc = Popen(commands, cwd=context_dir, stdout=PIPE, stderr=STDOUT, universal_newlines=True)
    for x in iter(proc.stdout.readline, ""):
        eprint(x, end="")

    if proc.wait():
        raise RuntimeError("Docker build failed.")

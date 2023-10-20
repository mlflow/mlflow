import logging
import os
from subprocess import PIPE, STDOUT, Popen
from urllib.parse import urlparse

from mlflow.utils import env_manager as em
from mlflow.utils.file_utils import TempDir, _copy_project
from mlflow.utils.logging_utils import eprint
from mlflow.version import VERSION

_logger = logging.getLogger(__name__)

SETUP_MINICONDA = """
# Setup miniconda
RUN curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh >> miniconda.sh
RUN bash ./miniconda.sh -b -p /miniconda && rm ./miniconda.sh
ENV PATH="/miniconda/bin:$PATH"
"""

SETUP_PYENV_AND_VIRTUALENV = r"""
# Setup pyenv
RUN apt -y update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
RUN git clone \
    --depth 1 \
    --branch $(git ls-remote --tags --sort=v:refname https://github.com/pyenv/pyenv.git | grep -o -E 'v[1-9]+(\.[1-9]+)+$' | tail -1) \
    https://github.com/pyenv/pyenv.git /root/.pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"
RUN apt install -y python3.8 python3.8-distutils
RUN ln -s -f $(which python3.8) /usr/bin/python
RUN wget https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py
RUN python /tmp/get-pip.py
RUN pip install virtualenv
"""


def _get_maven_proxy():
    http_proxy = os.getenv("http_proxy")
    https_proxy = os.getenv("https_proxy")
    if not http_proxy or not https_proxy:
        return ""

    # Expects proxies as either PROTOCOL://{USER}:{PASSWORD}@HOSTNAME:PORT
    # or PROTOCOL://HOSTNAME:PORT
    parsed_http_proxy = urlparse(http_proxy)
    assert parsed_http_proxy.hostname is not None, "Invalid `http_proxy` hostname."
    assert parsed_http_proxy.port is not None, f"Invalid proxy port: {parsed_http_proxy.port}"

    parsed_https_proxy = urlparse(https_proxy)
    assert parsed_https_proxy.hostname is not None, "Invalid `https_proxy` hostname."
    assert parsed_https_proxy.port is not None, f"Invalid proxy port: {parsed_https_proxy.port}"

    maven_proxy_options = (
        "-DproxySet=true",
        f"-Dhttp.proxyHost={parsed_http_proxy.hostname}",
        f"-Dhttp.proxyPort={parsed_http_proxy.port}",
        f"-Dhttps.proxyHost={parsed_https_proxy.hostname}",
        f"-Dhttps.proxyPort={parsed_https_proxy.port}",
        "-Dhttps.nonProxyHosts=repo.maven.apache.org",
    )

    if parsed_http_proxy.username is None or parsed_http_proxy.password is None:
        return " ".join(maven_proxy_options)

    return " ".join(
        (
            *maven_proxy_options,
            f"-Dhttp.proxyUser={parsed_http_proxy.username}",
            f"-Dhttp.proxyPassword={parsed_http_proxy.password}",
        )
    )


_DOCKERFILE_TEMPLATE = """
# Build an image that can serve mlflow models.
FROM ubuntu:20.04

RUN apt-get -y update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y --no-install-recommends \
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

{setup_miniconda}
{setup_pyenv_and_virtualenv}

ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV GUNICORN_CMD_ARGS="--timeout 60 -k gevent"
# Set up the program in the image
WORKDIR /opt/mlflow

{install_mlflow}

{custom_setup_steps}

# granting read/write access and conditional execution authority to all child directories
# and files to allow for deployment to AWS Sagemaker Serverless Endpoints
# (see https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
RUN chmod o+rwX /opt/mlflow/

{entrypoint}
"""


def _get_mlflow_install_step(dockerfile_context_dir, mlflow_home):
    """
    Get docker build commands for installing MLflow given a Docker context dir and optional source
    directory
    """
    maven_proxy = _get_maven_proxy()
    if mlflow_home:
        mlflow_dir = _copy_project(src_path=mlflow_home, dst_path=dockerfile_context_dir)
        return (
            f"COPY {mlflow_dir} /opt/mlflow\n"
            "RUN pip install /opt/mlflow\n"
            "RUN cd /opt/mlflow/mlflow/java/scoring && "
            f"mvn --batch-mode package -DskipTests {maven_proxy} && "
            "mkdir -p /opt/java/jars && "
            "mv /opt/mlflow/mlflow/java/scoring/target/"
            "mlflow-scoring-*-with-dependencies.jar /opt/java/jars\n"
        )
    else:
        return (
            f"RUN pip install mlflow=={VERSION}\n"
            "RUN mvn"
            " --batch-mode dependency:copy"
            f" -Dartifact=org.mlflow:mlflow-scoring:{VERSION}:pom"
            f" -DoutputDirectory=/opt/java {maven_proxy}\n"
            "RUN mvn"
            " --batch-mode dependency:copy"
            f" -Dartifact=org.mlflow:mlflow-scoring:{VERSION}:jar"
            f" -DoutputDirectory=/opt/java/jars {maven_proxy}\n"
            f"RUN cp /opt/java/mlflow-scoring-{VERSION}.pom /opt/java/pom.xml\n"
            "RUN cd /opt/java && mvn "
            "--batch-mode dependency:copy-dependencies "
            f"-DoutputDirectory=/opt/java/jars {maven_proxy}\n"
        )


def _generate_dockerfile_content(
    setup_miniconda, setup_pyenv_and_virtualenv, install_mlflow, custom_setup_steps, entrypoint
):
    """
    Generates a Dockerfile that can be used to build a docker image, that serves ML model
    stored and tracked in MLflow.

    It just takes string parameters containing docker imperatives and has no logic
    whatsoever. It will be more convenient if a more sophisticated function
    with some boolean flags would be called `generate_dockerfile`
    while this function being a backend of sorts for such function.

    :param setup_miniconda: Docker instructions related to set up miniconda. If used at all,
    variable `SETUP_MINICONDA` provides a working template for instructions. Should be either an
    empty string or `SETUP_MINICONDA`-based instructions :param setup_pyenv_and_virtualenv:
    Docker instructions related to set up pyenv and virtualenv. If used at all, variable
    `SETUP_PYENV_AND_VIRTUALENV` provides a working template for instructions. Should be either
    an empty string or `SETUP_PYENV_AND_VIRTUALENV`-based :param install_mlflow: Docker
    instruction for installing MLflow in given Docker context dir and optional source directory
    :param custom_setup_steps: Docker instructions for any customizations in the resulting
    Dockerfile :param entrypoint: String containing ENTRYPOINT directive for docker image
    """
    return _DOCKERFILE_TEMPLATE.format(
        setup_miniconda=setup_miniconda,
        setup_pyenv_and_virtualenv=setup_pyenv_and_virtualenv,
        install_mlflow=install_mlflow,
        custom_setup_steps=custom_setup_steps,
        entrypoint=entrypoint,
    )


def _build_image(
    image_name, entrypoint, env_manager, mlflow_home=None, custom_setup_steps_hook=None
):
    """
    Build an MLflow Docker image that can be used to serve a
    The image is built locally and it requires Docker to run.

    :param image_name: Docker image name.
    :param entry_point: String containing ENTRYPOINT directive for docker image
    :param env_manager: Environment manager to create a model environment for serving.
    :param mlflow_home: (Optional) Path to a local copy of the MLflow GitHub repository.
                        If specified, the image will install MLflow from this directory.
                        If None, it will install MLflow from pip.
    :param custom_setup_steps_hook: (Optional) Single-argument function that takes the string path
           of a dockerfile context directory and returns a string containing Dockerfile commands to
           run during the image build step.
    """
    mlflow_home = os.path.abspath(mlflow_home) if mlflow_home else None

    is_conda = env_manager == em.CONDA
    setup_miniconda = SETUP_MINICONDA if is_conda else ""
    setup_pyenv_and_virtualenv = "" if is_conda else SETUP_PYENV_AND_VIRTUALENV

    with TempDir() as tmp:
        cwd = tmp.path()
        install_mlflow = _get_mlflow_install_step(cwd, mlflow_home)
        custom_setup_steps = custom_setup_steps_hook(cwd) if custom_setup_steps_hook else ""
        with open(os.path.join(cwd, "Dockerfile"), "w") as f:
            f.write(
                _generate_dockerfile_content(
                    setup_miniconda=setup_miniconda,
                    setup_pyenv_and_virtualenv=setup_pyenv_and_virtualenv,
                    install_mlflow=install_mlflow,
                    custom_setup_steps=custom_setup_steps,
                    entrypoint=entrypoint,
                )
            )
        _logger.info("Building docker image with name %s", image_name)
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
    proc = Popen(commands, cwd=context_dir, stdout=PIPE, stderr=STDOUT, text=True)
    for x in iter(proc.stdout.readline, ""):
        eprint(x, end="")

    if proc.wait():
        raise RuntimeError("Docker build failed.")

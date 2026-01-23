import logging
import os
from subprocess import Popen
from typing import Literal
from urllib.parse import urlparse

from mlflow.environment_variables import MLFLOW_DOCKER_OPENJDK_VERSION
from mlflow.utils import env_manager as em
from mlflow.utils.file_utils import _copy_project
from mlflow.version import VERSION

_logger = logging.getLogger(__name__)

UBUNTU_BASE_IMAGE = "ubuntu:22.04"
PYTHON_SLIM_BASE_IMAGE = "python:{version}-slim"


SETUP_PBS_AND_VIRTUALENV = r"""# Setup Python via python-build-standalone (PBS)
ARG PY_VER=3.10.19
ARG PBS_REL=20260114
ARG PBS_BASE=https://github.com/astral-sh/python-build-standalone/releases/download
RUN ARCH=$(uname -m) && \
    PBS_FILE="cpython-${PY_VER}+${PBS_REL}-${ARCH}-unknown-linux-gnu-install_only.tar.gz" && \
    curl -fsSL -o /tmp/python.tar.gz "${PBS_BASE}/${PBS_REL}/${PBS_FILE}" && \
    mkdir -p /opt/python && \
    tar -xzf /tmp/python.tar.gz -C /opt/python --strip-components=1 && \
    rm /tmp/python.tar.gz && \
    ln -s /opt/python/bin/python3 /usr/bin/python && \
    ln -s /opt/python/bin/pip3 /usr/bin/pip
ENV PATH="/opt/python/bin:$PATH"
RUN pip install virtualenv
"""

_DOCKERFILE_TEMPLATE = """# Build an image that can serve mlflow models.
FROM {base_image}

{setup_python_venv}

{setup_java}

WORKDIR /opt/mlflow

{install_mlflow}

{install_model_and_deps}

ENV MLFLOW_DISABLE_ENV_CREATION={disable_env_creation}
ENV ENABLE_MLSERVER={enable_mlserver}

# granting read/write access and conditional execution authority to all child directories
# and files to allow for deployment to AWS Sagemaker Serverless Endpoints
# (see https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
RUN chmod o+rwX /opt/mlflow/

# clean up apt cache to reduce image size
RUN rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["python", "-c", "{entrypoint}"]
"""


SETUP_MINICONDA = """# Setup miniconda
RUN curl --fail -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > miniconda.sh
RUN bash ./miniconda.sh -b -p /miniconda && rm ./miniconda.sh
ENV PATH="/miniconda/bin:$PATH"
# Remove default channels to avoid `CondaToSNonInteractiveError`.
# See https://github.com/mlflow/mlflow/pull/16752 for more details.
RUN conda config --system --remove channels defaults && conda config --system --add channels conda-forge
"""  # noqa: E501


def generate_dockerfile(
    output_dir: str,
    base_image: str,
    model_install_steps: str | None,
    entrypoint: str,
    env_manager: Literal["conda", "local", "virtualenv"] = em.CONDA,
    mlflow_home: str | None = None,
    enable_mlserver: bool = False,
    disable_env_creation_at_runtime: bool = True,
    install_java: bool | None = None,
):
    """
    Generates a Dockerfile that can be used to build a docker image, that serves ML model
    stored and tracked in MLflow.
    """

    setup_java_steps = ""
    setup_python_venv_steps = ""
    install_mlflow_steps = _pip_mlflow_install_step(output_dir, mlflow_home)

    if base_image.startswith("python:"):
        if install_java:
            _logger.warning(
                "`install_java` option is not supported when using python base image, "
                "switch to UBUNTU_BASE_IMAGE to enable java installation."
            )
        setup_python_venv_steps = (
            "RUN apt-get -y update && apt-get install -y --no-install-recommends nginx"
        )

    elif base_image == UBUNTU_BASE_IMAGE:
        setup_python_venv_steps = (
            "RUN apt-get -y update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y "
            "--no-install-recommends wget curl nginx ca-certificates bzip2 build-essential cmake "
            "git-core\n\n"
        )
        setup_python_venv_steps += (
            SETUP_MINICONDA if env_manager == em.CONDA else SETUP_PBS_AND_VIRTUALENV
        )
        if install_java is not False:
            jdk_ver = MLFLOW_DOCKER_OPENJDK_VERSION.get()
            setup_java_steps = (
                "# Setup Java\n"
                f"RUN apt-get install -y --no-install-recommends openjdk-{jdk_ver}-jdk maven\n"
                f"ENV JAVA_HOME=/usr/lib/jvm/java-{jdk_ver}-openjdk-amd64"
            )

    with open(os.path.join(output_dir, "Dockerfile"), "w") as f:
        f.write(
            _DOCKERFILE_TEMPLATE.format(
                base_image=base_image,
                setup_python_venv=setup_python_venv_steps,
                setup_java=setup_java_steps,
                install_mlflow=install_mlflow_steps,
                install_model_and_deps=model_install_steps,
                entrypoint=entrypoint,
                enable_mlserver=enable_mlserver,
                disable_env_creation=disable_env_creation_at_runtime,
            )
        )


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


def _pip_mlflow_install_step(dockerfile_context_dir, mlflow_home):
    """
    Get docker build commands for installing MLflow given a Docker context dir and optional source
    directory
    """
    if mlflow_home:
        mlflow_dir = _copy_project(
            src_path=os.path.abspath(mlflow_home), dst_path=dockerfile_context_dir
        )
        return (
            "# Install MLflow from local source\n"
            f"COPY {mlflow_dir} /opt/mlflow\n"
            "RUN pip install /opt/mlflow"
        )
    else:
        return f"# Install MLflow\nRUN pip install mlflow=={VERSION}"


def build_image_from_context(context_dir: str, image_name: str):
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
    proc = Popen(commands, cwd=context_dir)
    if proc.wait():
        raise RuntimeError("Docker build failed.")

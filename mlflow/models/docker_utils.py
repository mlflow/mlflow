import logging
import os
from subprocess import PIPE, STDOUT, Popen
from urllib.parse import urlparse

from mlflow.utils import env_manager as em
from mlflow.utils.file_utils import _copy_project
from mlflow.utils.logging_utils import eprint
from mlflow.version import VERSION

_logger = logging.getLogger(__name__)

SETUP_MINICONDA = """# Setup miniconda
RUN curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh >> miniconda.sh
RUN bash ./miniconda.sh -b -p /miniconda && rm ./miniconda.sh
ENV PATH="/miniconda/bin:$PATH"
"""

SETUP_PYENV_AND_VIRTUALENV = r"""# Setup pyenv
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
RUN git clone \
    --depth 1 \
    --branch $(git ls-remote --tags --sort=v:refname https://github.com/pyenv/pyenv.git | grep -o -E 'v[1-9]+(\.[1-9]+)+$' | tail -1) \
    https://github.com/pyenv/pyenv.git /root/.pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"
RUN apt install -y python3.8 python3.8-distutils \
    && ln -s -f $(which python3.8) /usr/bin/python \
    && wget https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py \
    && python /tmp/get-pip.py
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

RUN apt-get -y update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y --no-install-recommends \\
    wget curl nginx ca-certificates bzip2 build-essential cmake git-core

{setup_java}

{setup_python_env}

WORKDIR /opt/mlflow

{install_mlflow}

{custom_setup_steps}

ENV MLFLOW_DISABLE_ENV_CREATION={disable_env_creation}
ENV ENABLE_MLSERVER={enable_mlserver}
ENV GUNICORN_CMD_ARGS="--timeout 60 -k gevent"

# granting read/write access and conditional execution authority to all child directories
# and files to allow for deployment to AWS Sagemaker Serverless Endpoints
# (see https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
RUN chmod o+rwX /opt/mlflow/

# clean up apt cache to reduce image size
RUN rm -rf /var/lib/apt/lists/*

{entrypoint}
"""


def _java_mlflow_install_step(mlflow_home):
    maven_proxy = _get_maven_proxy()
    if mlflow_home:
        return (
            "# Install Java mlflow-scoring from local source\n"
            "RUN cd /opt/mlflow/mlflow/java/scoring && "
            f"mvn --batch-mode package -DskipTests {maven_proxy} && "
            "mkdir -p /opt/java/jars && "
            "mv /opt/mlflow/mlflow/java/scoring/target/"
            "mlflow-scoring-*-with-dependencies.jar /opt/java/jars\n"
        )
    else:
        return (
            "# Install Java mlflow-scoring from Maven Central\n"
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


def generate_dockerfile(
    output_dir,
    entrypoint,
    env_manager,
    mlflow_home=None,
    custom_setup_steps=None,
    enable_mlserver=False,
    install_java=False,
    disable_env_creation=False,
):
    """
    Generates a Dockerfile that can be used to build a docker image, that serves ML model
    stored and tracked in MLflow.
    """
    install_mlflow_steps = _pip_mlflow_install_step(output_dir, mlflow_home)
    if install_java:
        install_mlflow_steps += "\n\n" + _java_mlflow_install_step(mlflow_home)

    setup_java_steps = (
        "RUN apt-get install -y --no-install-recommends openjdk-8-jdk maven\n"
        "ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64"
        if install_java
        else ""
    )

    setup_python_env_steps = (
        SETUP_MINICONDA if env_manager == em.CONDA else SETUP_PYENV_AND_VIRTUALENV
    )
    with open(os.path.join(output_dir, "Dockerfile"), "w") as f:
        f.write(
            _DOCKERFILE_TEMPLATE.format(
                setup_java=setup_java_steps,
                setup_python_env=setup_python_env_steps,
                install_mlflow=install_mlflow_steps,
                custom_setup_steps=custom_setup_steps,
                entrypoint=entrypoint,
                enable_mlserver=enable_mlserver,
                disable_env_creation=disable_env_creation,
            )
        )


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
    proc = Popen(commands, cwd=context_dir, stdout=PIPE, stderr=STDOUT, text=True)
    for x in iter(proc.stdout.readline, ""):
        eprint(x, end="")

    if proc.wait():
        raise RuntimeError("Docker build failed.")

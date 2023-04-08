import os
import logging
from importlib.machinery import SourceFileLoader
from setuptools import setup, find_packages, Command

_MLFLOW_SKINNY_ENV_VAR = "MLFLOW_SKINNY"

version = (
    SourceFileLoader("mlflow.version", os.path.join("mlflow", "version.py")).load_module().VERSION
)


# Get a list of all files in the directory to include in our module
def package_files(directory):
    paths = []
    for path, _, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


def is_comment_or_empty(line):
    stripped = line.strip()
    return stripped == "" or stripped.startswith("#")


def remove_comments_and_empty_lines(lines):
    return [line for line in lines if not is_comment_or_empty(line)]


# Prints out a set of paths (relative to the mlflow/ directory) of files in mlflow/server/js/build
# to include in the wheel, e.g. "../mlflow/server/js/build/index.html"
js_files = package_files("mlflow/server/js/build")
models_container_server_files = package_files("mlflow/models/container")
alembic_files = [
    "../mlflow/store/db_migrations/alembic.ini",
    "../mlflow/temporary_db_migrations_for_pre_1_users/alembic.ini",
]
extra_files = [
    "pypi_package_index.json",
    "pyspark/ml/log_model_allowlist.txt",
]
recipes_template_files = package_files("mlflow/recipes/resources")
recipes_files = package_files("mlflow/recipes/cards/templates")


"""
Minimal requirements for the skinny MLflow client which provides a limited
subset of functionality such as: RESTful client functionality for Tracking and
Model Registry, as well as support for Project execution against local backends
and Databricks.
"""
with open(os.path.join("requirements", "skinny-requirements.txt")) as f:
    SKINNY_REQUIREMENTS = remove_comments_and_empty_lines(f.read().splitlines())


"""
These are the core requirements for the complete MLflow platform, which augments
the skinny client functionality with support for running the MLflow Tracking
Server & UI. It also adds project backends such as Docker and Kubernetes among
other capabilities.
"""
with open(os.path.join("requirements", "core-requirements.txt")) as f:
    CORE_REQUIREMENTS = SKINNY_REQUIREMENTS + remove_comments_and_empty_lines(f.read().splitlines())

_is_mlflow_skinny = bool(os.environ.get(_MLFLOW_SKINNY_ENV_VAR))
logging.debug("{} env var is set: {}".format(_MLFLOW_SKINNY_ENV_VAR, _is_mlflow_skinny))


class ListDependencies(Command):
    # `python setup.py <command name>` prints out "running <command name>" by default.
    # This logging message must be hidden by specifying `--quiet` (or `-q`) when piping the output
    # of this command to `pip install`.
    description = "List mlflow dependencies"
    user_options = [
        ("skinny", None, "List mlflow-skinny dependencies"),
    ]

    def initialize_options(self):
        self.skinny = False

    def finalize_options(self):
        pass

    def run(self):
        dependencies = SKINNY_REQUIREMENTS if self.skinny else CORE_REQUIREMENTS
        print("\n".join(dependencies))


MINIMUM_SUPPORTED_PYTHON_VERSION = "3.8"


class MinPythonVersion(Command):
    description = "Print out the minimum supported Python version"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print(MINIMUM_SUPPORTED_PYTHON_VERSION)


setup(
    name="mlflow" if not _is_mlflow_skinny else "mlflow-skinny",
    version=version,
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={
        "mlflow": (
            js_files
            + models_container_server_files
            + alembic_files
            + extra_files
            + recipes_template_files
            + recipes_files
        ),
    }
    if not _is_mlflow_skinny
    # include alembic files to enable usage of the skinny client with SQL databases
    # if users install sqlalchemy and alembic independently
    else {"mlflow": alembic_files + extra_files},
    install_requires=CORE_REQUIREMENTS if not _is_mlflow_skinny else SKINNY_REQUIREMENTS,
    extras_require={
        "extras": [
            # Required to log artifacts and models to HDFS artifact locations
            "pyarrow",
            # Required to sign outgoing request with SigV4 signature
            "requests-auth-aws-sigv4",
            # Required to log artifacts and models to AWS S3 artifact locations
            "boto3",
            # Required to log artifacts and models to GCS artifact locations
            "google-cloud-storage>=1.30.0",
            "azureml-core>=1.2.0",
            # Required to log artifacts to SFTP artifact locations
            "pysftp",
            # Required by the mlflow.projects module, when running projects against
            # a remote Kubernetes cluster
            "kubernetes",
            # Required to serve models through MLServer
            "mlserver>=1.2.0",
            "mlserver-mlflow>=1.2.0",
            "virtualenv",
            # Required for exporting metrics from the MLflow server to Prometheus
            # as part of the MLflow server monitoring add-on
            "prometheus-flask-exporter",
        ],
        "databricks": [
            # Required to write model artifacts to unity catalog locations
            "azure-storage-file-datalake>12",
            "google-cloud-storage>=1.30.0",
            "boto3>1",
        ],
        "sqlserver": ["mlflow-dbstore"],
        "aliyun-oss": ["aliyunstoreplugin"],
    },
    entry_points="""
        [console_scripts]
        mlflow=mlflow.cli:cli

        [mlflow.app]
        basic-auth=mlflow.server.auth:app
    """,
    cmdclass={
        "dependencies": ListDependencies,
        "min_python_version": MinPythonVersion,
    },
    zip_safe=False,
    author="Databricks",
    description="MLflow: A Platform for ML Development and Productionization",
    long_description=open("README.rst").read()
    if not _is_mlflow_skinny
    else open("README_SKINNY.rst").read() + open("README.rst").read(),
    long_description_content_type="text/x-rst",
    license="Apache License 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        f"Programming Language :: Python :: {MINIMUM_SUPPORTED_PYTHON_VERSION}",
    ],
    keywords="ml ai databricks",
    url="https://mlflow.org/",
    python_requires=f">={MINIMUM_SUPPORTED_PYTHON_VERSION}",
    project_urls={
        "Bug Tracker": "https://github.com/mlflow/mlflow/issues",
        "Documentation": "https://mlflow.org/docs/latest/index.html",
        "Source Code": "https://github.com/mlflow/mlflow",
    },
)

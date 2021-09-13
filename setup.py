import os
import logging

from importlib.machinery import SourceFileLoader
from setuptools import setup, find_packages

_MLFLOW_SKINNY_ENV_VAR = "MLFLOW_SKINNY"

version = (
    SourceFileLoader("mlflow.version", os.path.join("mlflow", "version.py")).load_module().VERSION
)


# Get a list of all files in the JS directory to include in our module
def package_files(directory):
    paths = []
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


# Prints out a set of paths (relative to the mlflow/ directory) of files in mlflow/server/js/build
# to include in the wheel, e.g. "../mlflow/server/js/build/index.html"
js_files = package_files("mlflow/server/js/build")
models_container_server_files = package_files("mlflow/models/container")
alembic_files = [
    "../mlflow/store/db_migrations/alembic.ini",
    "../mlflow/temporary_db_migrations_for_pre_1_users/alembic.ini",
]
extra_files = ["ml-package-versions.yml", "pyspark/ml/log_model_allowlist.txt"]

"""
Minimal requirements for the skinny MLflow client which provides a limited
subset of functionality such as: RESTful client functionality for Tracking and
Model Registry, as well as support for Project execution against local backends
and Databricks.
"""
SKINNY_REQUIREMENTS = [
    "click>=7.0",
    "cloudpickle",
    "databricks-cli>=0.8.7",
    "entrypoints",
    "gitpython>=2.1.0",
    "pyyaml>=5.1",
    "protobuf>=3.7.0",
    "pytz",
    "requests>=2.17.3",
    "packaging",
    # Automated dependency detection in MLflow Models relies on
    # `importlib_metadata.packages_distributions` to resolve a module name to its package name
    # (e.g. 'sklearn' -> 'scikit-learn'). importlib_metadata 3.7.0 or newer supports this function:
    # https://github.com/python/importlib_metadata/blob/main/CHANGES.rst#v370
    "importlib_metadata>=3.7.0,!=4.7.0",
]

"""
These are the core requirements for the complete MLflow platform, which augments
the skinny client functionality with support for running the MLflow Tracking
Server & UI. It also adds project backends such as Docker and Kubernetes among
other capabilities.
"""
CORE_REQUIREMENTS = SKINNY_REQUIREMENTS + [
    "alembic<=1.4.1",
    # Required
    "docker>=4.0.0",
    "Flask",
    "gunicorn; platform_system != 'Windows'",
    "numpy",
    "pandas",
    "prometheus-flask-exporter",
    "querystring_parser",
    # Pin sqlparse for: https://github.com/mlflow/mlflow/issues/3433
    "sqlparse>=0.3.1",
    # Required to run the MLflow server against SQL-backed storage
    "sqlalchemy",
    "waitress; platform_system == 'Windows'",
]

_is_mlflow_skinny = bool(os.environ.get(_MLFLOW_SKINNY_ENV_VAR))
logging.debug("{} env var is set: {}".format(_MLFLOW_SKINNY_ENV_VAR, _is_mlflow_skinny))

setup(
    name="mlflow" if not _is_mlflow_skinny else "mlflow-skinny",
    version=version,
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={"mlflow": js_files + models_container_server_files + alembic_files + extra_files}
    if not _is_mlflow_skinny
    # include alembic files to enable usage of the skinny client with SQL databases
    # if users install sqlalchemy, alembic, and sqlparse independently
    else {"mlflow": alembic_files + extra_files},
    install_requires=CORE_REQUIREMENTS if not _is_mlflow_skinny else SKINNY_REQUIREMENTS,
    extras_require={
        "extras": [
            "scikit-learn",
            # Required to log artifacts and models to HDFS artifact locations
            "pyarrow",
            # Required to log artifacts and models to AWS S3 artifact locations
            "boto3",
            "mleap",
            # Required to log artifacts and models to GCS artifact locations
            "google-cloud-storage",
            "azureml-core>=1.2.0",
            # Required to log artifacts to SFTP artifact locations
            "pysftp",
            # Required by the mlflow.projects module, when running projects against
            # a remote Kubernetes cluster
            "kubernetes",
        ],
        "sqlserver": ["mlflow-dbstore"],
        "aliyun-oss": ["aliyunstoreplugin"],
    },
    entry_points="""
        [console_scripts]
        mlflow=mlflow.cli:cli
    """,
    zip_safe=False,
    author="Databricks",
    description="MLflow: A Platform for ML Development and Productionization",
    long_description=open("README.rst").read()
    if not _is_mlflow_skinny
    else open("README_SKINNY.rst").read() + open("README.rst").read(),
    long_description_content_type="text/x-rst",
    license="Apache License 2.0",
    classifiers=["Intended Audience :: Developers", "Programming Language :: Python :: 3.6"],
    keywords="ml ai databricks",
    url="https://mlflow.org/",
    python_requires=">=3.6",
    project_urls={
        "Bug Tracker": "https://github.com/mlflow/mlflow/issues",
        "Documentation": "https://mlflow.org/docs/latest/index.html",
        "Source Code": "https://github.com/mlflow/mlflow",
    },
)

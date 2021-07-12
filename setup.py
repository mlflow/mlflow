import os
import logging

from importlib.machinery import SourceFileLoader
from setuptools import setup, find_packages

_MLFLOW_SKINNY_ENV_VAR = "MLFLOW_SKINNY"

version = (
    SourceFileLoader("mlflux.version", os.path.join("mlflux", "version.py")).load_module().VERSION
)


# Get a list of all files in the JS directory to include in our module
def package_files(directory):
    paths = []
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


# Prints out a set of paths (relative to the mlflux/ directory) of files in mlflux/server/js/build
# to include in the wheel, e.g. "../mlflux/server/js/build/index.html"
js_files = package_files("mlflux/server/js/build")
models_container_server_files = package_files("mlflux/models/container")
alembic_files = [
    "../mlflux/store/db_migrations/alembic.ini",
    "../mlflux/temporary_db_migrations_for_pre_1_users/alembic.ini",
]
extra_files = ["ml-package-versions.yml", "pyspark/ml/log_model_allowlist.txt"]

"""
Minimal requirements for the skinny mlflux client which provides a limited
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
]

"""
These are the core requirements for the complete mlflux platform, which augments
the skinny client functionality with support for running the mlflux Tracking
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
    # Pin sqlparse for: https://github.com/mlflux/mlflux/issues/3433
    "sqlparse>=0.3.1",
    # Required to run the mlflux server against SQL-backed storage
    "sqlalchemy",
    "waitress; platform_system == 'Windows'",
]

_is_mlflow_skinny = bool(os.environ.get(_MLFLOW_SKINNY_ENV_VAR))
logging.debug("{} env var is set: {}".format(_MLFLOW_SKINNY_ENV_VAR, _is_mlflow_skinny))

setup(
    name="mlflux" if not _is_mlflow_skinny else "mlflux-skinny",
    version=version,
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={"mlflux": js_files + models_container_server_files + alembic_files + extra_files}
    if not _is_mlflow_skinny
    # include alembic files to enable usage of the skinny client with SQL databases
    # if users install sqlalchemy, alembic, and sqlparse independently
    else {"mlflux": alembic_files + extra_files},
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
            # Required by the mlflux.projects module, when running projects against
            # a remote Kubernetes cluster
            "kubernetes",
        ],
        "sqlserver": ["mlflux-dbstore"],
        "aliyun-oss": ["aliyunstoreplugin"],
    },
    entry_points="""
        [console_scripts]
        mlflux=mlflux.cli:cli
    """,
    zip_safe=False,
    author="Databricks",
    description="mlflux: A Platform for ML Development and Productionization",
    long_description=open("README.rst").read()
    if not _is_mlflow_skinny
    else open("README_SKINNY.rst").read() + open("README.rst").read(),
    long_description_content_type="text/x-rst",
    license="Apache License 2.0",
    classifiers=["Intended Audience :: Developers", "Programming Language :: Python :: 3.6"],
    keywords="ml ai databricks",
    url="https://mlflux.org/",
    python_requires=">=3.6",
    project_urls={
        "Bug Tracker": "https://github.com/mlflux/mlflux/issues",
        "Documentation": "https://mlflux.org/docs/latest/index.html",
        "Source Code": "https://github.com/mlflux/mlflux",
    },
)

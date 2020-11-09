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

requirements = [
    "click>=7.0",
    "cloudpickle",
    "databricks-cli>=0.8.7",
    "entrypoints",
    "gitpython>=2.1.0",
    "numpy",
    "pandas",
    "python-dateutil",
    "pyyaml",
    "protobuf>=3.6.0",
    "requests>=2.17.3",
    "six>=1.10.0",
    "simplejson",
]

_is_mlflow_skinny = bool(os.environ.get(_MLFLOW_SKINNY_ENV_VAR))
logging.debug("{} env var is set: {}".format(_MLFLOW_SKINNY_ENV_VAR, _is_mlflow_skinny))

if not _is_mlflow_skinny:
    requirements.extend(
        [
            "alembic<=1.4.1",
            # Required
            "azure-storage-blob",
            "cloudpickle",
            "docker>=4.0.0",
            "Flask",
            "gitpython>=2.1.0",
            "gunicorn; platform_system != 'Windows'",
            "prometheus-flask-exporter",
            "querystring_parser",
            # Pin sqlparse for: https://github.com/mlflow/mlflow/issues/3433
            "sqlparse>=0.3.1",
            # Required to run the MLflow server against SQL-backed storage
            "sqlalchemy<=1.3.13",
            "waitress; platform_system == 'Windows'",
        ]
    )

setup(
    name="mlflow",
    version=version,
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={"mlflow": js_files + models_container_server_files + alembic_files},
    install_requires=requirements,
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
    long_description=open("README.rst").read(),
    license="Apache License 2.0",
    classifiers=["Intended Audience :: Developers", "Programming Language :: Python :: 3.6"],
    keywords="ml ai databricks",
    url="https://mlflow.org/",
    python_requires=">=3.5",
    project_urls={
        "Bug Tracker": "https://github.com/mlflow/mlflow/issues",
        "Documentation": "https://mlflow.org/docs/latest/index.html",
        "Source Code": "https://github.com/mlflow/mlflow",
    },
)

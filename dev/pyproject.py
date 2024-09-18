from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from enum import Enum
from pathlib import Path

import toml
import yaml
from packaging.version import Version


class PackageType(Enum):
    SKINNY = "skinny"
    RELEASE = "release"
    DEV = "dev"


SEPARATOR = """
# Package metadata: can't be updated manually, use dev/pyproject.py
# -----------------------------------------------------------------
# Dev tool settings: can be updated manually

"""


def find_duplicates(seq):
    counted = Counter(seq)
    return [item for item, count in counted.items() if count > 1]


def read_requirements(path: Path) -> list[str]:
    lines = (l.strip() for l in path.read_text().splitlines())
    return [l for l in lines if l and not l.startswith("#")]


def read_package_versions_yml():
    with open("mlflow/ml-package-versions.yml") as f:
        return yaml.safe_load(f)


def build(package_type: PackageType) -> None:
    skinny_requirements = read_requirements(Path("requirements", "skinny-requirements.txt"))
    core_requirements = read_requirements(Path("requirements", "core-requirements.txt"))
    gateways_requirements = read_requirements(Path("requirements", "gateway-requirements.txt"))
    package_version = re.search(
        r'^VERSION = "([a-z0-9\.]+)"$', Path("mlflow", "version.py").read_text(), re.MULTILINE
    ).group(1)
    python_version = Path("requirements", "python-version.txt").read_text().strip()
    versions_yaml = read_package_versions_yml()
    langchain_requirements = [
        "langchain>={},<={}".format(
            max(
                Version(versions_yaml["langchain"]["autologging"]["minimum"]),
                Version(versions_yaml["langchain"]["models"]["minimum"]),
            ),
            min(
                Version(versions_yaml["langchain"]["autologging"]["maximum"]),
                Version(versions_yaml["langchain"]["models"]["maximum"]),
            ),
        )
    ]

    if package_type is PackageType.SKINNY:
        dependencies = sorted(skinny_requirements)
    elif package_type is PackageType.RELEASE:
        dependencies = [f"mlflow-skinny=={package_version}"] + sorted(core_requirements)
    else:
        dependencies = sorted(core_requirements + skinny_requirements)

    if dep_duplicates := find_duplicates(dependencies):
        raise RuntimeError(f"Duplicated dependencies are found: {dep_duplicates}")

    package_name = "mlflow-skinny" if package_type is PackageType.SKINNY else "mlflow"
    extra_package_data = (
        []
        if package_type is PackageType.SKINNY
        else ["models/container/**/*", "server/js/build/**/*"]
    )

    data = {
        "build-system": {
            "requires": ["setuptools"],
            "build-backend": "setuptools.build_meta",
        },
        "project": {
            "name": package_name,
            "version": package_version,
            "maintainers": [
                {"name": "Databricks", "email": "mlflow-oss-maintainers@googlegroups.com"}
            ],
            "description": (
                "MLflow is an open source platform for the complete machine learning lifecycle"
            ),
            "readme": "README.rst",
            "license": {
                "file": "LICENSE.txt",
            },
            "keywords": ["mlflow", "ai", "databricks"],
            "classifiers": [
                "Development Status :: 5 - Production/Stable",
                "Intended Audience :: Developers",
                "Intended Audience :: End Users/Desktop",
                "Intended Audience :: Science/Research",
                "Intended Audience :: Information Technology",
                "Topic :: Scientific/Engineering :: Artificial Intelligence",
                "Topic :: Software Development :: Libraries :: Python Modules",
                "License :: OSI Approved :: Apache Software License",
                "Operating System :: OS Independent",
                f"Programming Language :: Python :: {python_version}",
            ],
            "requires-python": f">={python_version}",
            "dependencies": dependencies,
            "optional-dependencies": {
                "extras": [
                    # Required to log artifacts and models to HDFS artifact locations
                    "pyarrow",
                    # Required to sign outgoing request with SigV4 signature
                    "requests-auth-aws-sigv4",
                    # Required to log artifacts and models to AWS S3 artifact locations
                    "boto3",
                    "botocore",
                    # Required to log artifacts and models to GCS artifact locations
                    "google-cloud-storage>=1.30.0",
                    "azureml-core>=1.2.0",
                    # Required to log artifacts to SFTP artifact locations
                    "pysftp",
                    # Required by the mlflow.projects module, when running projects against
                    # a remote Kubernetes cluster
                    "kubernetes",
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
                    "botocore",
                ],
                "mlserver": [
                    # Required to serve models through MLServer
                    "mlserver>=1.2.0,!=1.3.1",
                    "mlserver-mlflow>=1.2.0,!=1.3.1",
                ],
                "gateway": gateways_requirements,
                "genai": gateways_requirements,
                "sqlserver": ["mlflow-dbstore"],
                "aliyun-oss": ["aliyunstoreplugin"],
                "xethub": ["mlflow-xethub"],
                "jfrog": ["mlflow-jfrog-plugin"],
                "langchain": langchain_requirements,
            },
            "urls": {
                "homepage": "https://mlflow.org",
                "issues": "https://github.com/mlflow/mlflow/issues",
                "documentation": "https://mlflow.org/docs/latest/index.html",
                "repository": "https://github.com/mlflow/mlflow",
            },
            "scripts": {
                "mlflow": "mlflow.cli:cli",
            },
            "entry-points": {
                "mlflow.app": {
                    "basic-auth": "mlflow.server.auth:create_app",
                },
                "mlflow.app.client": {
                    "basic-auth": "mlflow.server.auth.client:AuthServiceClient",
                },
                "mlflow.deployments": {
                    "databricks": "mlflow.deployments.databricks",
                    "http": "mlflow.deployments.mlflow",
                    "https": "mlflow.deployments.mlflow",
                    "openai": "mlflow.deployments.openai",
                },
            },
        },
        "tool": {
            "setuptools": {
                "packages": {
                    "find": {
                        "where": ["."],
                        "include": ["mlflow", "mlflow.*"],
                        "exclude": ["tests", "tests.*"],
                    }
                },
                "package-data": {
                    "mlflow": [
                        "store/db_migrations/alembic.ini",
                        "temporary_db_migrations_for_pre_1_users/alembic.ini",
                        "pypi_package_index.json",
                        "pyspark/ml/log_model_allowlist.txt",
                        "server/auth/basic_auth.ini",
                        "server/auth/db/migrations/alembic.ini",
                        "recipes/resources/**/*",
                        "recipes/cards/templates/**/*",
                    ]
                    + extra_package_data
                },
            }
        },
    }

    if package_type in [PackageType.SKINNY, PackageType.RELEASE]:
        out_path = f"pyproject.{package_type.value}.toml"
        with Path(out_path).open("w") as f:
            f.write(toml.dumps(data))
    else:
        out_path = "pyproject.toml"
        original = Path(out_path).read_text().split(SEPARATOR)[1]
        with Path(out_path).open("w") as f:
            f.write(toml.dumps(data))
            f.write(SEPARATOR)
            f.write(original)

    if taplo := shutil.which("taplo"):
        subprocess.check_call([taplo, "fmt", out_path])


def main() -> None:
    if shutil.which("taplo") is None:
        print(
            "taplo is required to generate pyproject.toml. "
            "Please install it by following the instructions at "
            "https://taplo.tamasfe.dev/cli/introduction.html."
        )
        return

    for package_type in PackageType:
        build(package_type)


if __name__ == "__main__":
    main()

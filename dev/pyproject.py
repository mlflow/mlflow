from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import toml

SEPARATOR = """
# Package metadata: can't be updated manually, use dev/pyproject.py
# -----------------------------------------------------------------
# Dev tool settings: can be updated manually

"""


def read_requirements(path: Path) -> list[str]:
    lines = (l.strip() for l in path.read_text().splitlines())
    return [l for l in lines if l and not l.startswith("#")]


def build(skinny: bool) -> None:
    skinny_requirements = read_requirements(Path("requirements", "skinny-requirements.txt"))
    core_requirements = read_requirements(Path("requirements", "core-requirements.txt"))
    gateways_requirements = read_requirements(Path("requirements", "gateway-requirements.txt"))
    version = re.search(
        r'^VERSION = "([a-z0-9\.]+)"$', Path("mlflow", "version.py").read_text(), re.MULTILINE
    ).group(1)
    python_version = Path("requirements", "python-version.txt").read_text().strip()
    data = {
        "build-system": {
            "requires": ["setuptools"],
            "build-backend": "setuptools.build_meta",
        },
        "project": {
            "name": "mlflow" if not skinny else "mlflow-skinny",
            "version": version,
            "maintainers": [
                {"name": "Databricks", "email": "mlflow-oss-maintainers@googlegroups.com "}
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
            "dependencies": sorted(
                skinny_requirements if skinny else skinny_requirements + core_requirements
            ),
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
                    # Required to serve models through MLServer
                    # NOTE: remove the upper version pin once protobuf is no longer pinned in
                    # mlserver. Reference issue: https://github.com/SeldonIO/MLServer/issues/1089
                    "mlserver>=1.2.0,!=1.3.1,<1.4.0",
                    "mlserver-mlflow>=1.2.0,!=1.3.1,<1.4.0",
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
                "gateway": gateways_requirements,
                "genai": gateways_requirements,
                "sqlserver": ["mlflow-dbstore"],
                "aliyun-oss": ["aliyunstoreplugin"],
                "xethub": ["mlflow-xethub"],
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
                        "recipes/resources/*",
                        "recipes/cards/templates/*",
                    ]
                    + ([] if skinny else ["models/container/*", "server/js/build/*"])
                },
            }
        },
    }

    original = Path("pyproject.toml").read_text()
    if SEPARATOR in original:
        original = original.split(SEPARATOR)[1]

    out_path = "pyproject.skinny.toml" if skinny else "pyproject.toml"
    with Path(out_path).open("w") as f:
        f.write(toml.dumps(data))
        f.write(SEPARATOR)
        f.write(original)

    if taplo := shutil.which("taplo"):
        subprocess.run([taplo, "fmt", out_path], check=True)


def main() -> None:
    if shutil.which("taplo") is None:
        print(
            "taplo is required to generate pyproject.toml. "
            "Please install it by following the instructions at "
            "https://taplo.tamasfe.dev/cli/introduction.html."
        )
        return
    build(skinny=False)
    build(skinny=True)


if __name__ == "__main__":
    main()

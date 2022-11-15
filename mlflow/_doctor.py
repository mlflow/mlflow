import os
import json
import pkg_resources
import platform
import click

import mlflow
from mlflow.utils.databricks_utils import get_databricks_runtime


def doctor(mask_envs=False):
    """
    Prints out useful information for debugging issues with MLflow.

    :param mask_envs: If True, mask the environment variable values (e.g. `"MLFLOW_ENV_VAR": "***"`)
                      in the output to prevent leaking sensitive information.

    .. warning::

        - This API should only be used for debugging purposes.
        - The output may contain sensitive information such as a database URI containing a password.

    .. code-block:: python
        :caption: Example

        import mlflow

        with mlflow.start_run():
            mlflow.doctor()

    .. code-block:: text
        :caption: Output

        System information: Linux #58~20.04.1-Ubuntu SMP Thu Oct 13 13:09:46 UTC 2022
        Python version: 3.8.10
        MLflow version: 1.30.0
        MLflow module location: /usr/local/lib/python3.8/site-packages/mlflow/__init__.py
        Tracking URI: sqlite:///mlflow.db
        Registry URI: sqlite:///mlflow.db
        Active experiment ID: 0
        Active run ID: 326d462f79a247669506772952af04ea
        Active run artifact URI: ./mlruns/0/326d462f79a247669506772952af04ea/artifacts
        MLflow environment variables: {
            "MLFLOW_TRACKING_URI": "sqlite:///mlflow.db"
        }
        MLflow dependencies: {
            "click": "8.0.1",
            "cloudpickle": "1.6.0",
            "databricks-cli": "0.14.3",
            "entrypoints": "0.3",
            "gitpython": "3.1.18",
            "pyyaml": "6.0",
            "protobuf": "3.19.0",
            "pytz": "2022.5",
            "requests": "2.28.1",
            "packaging": "21.3",
            "importlib_metadata": "5.0.0",
            "sqlparse": "0.4.3",
            "alembic": "1.8.1",
            "docker": "4.4.4",
            "Flask": "2.2.2",
            "numpy": "1.21.3",
            "scipy": "1.7.1",
            "pandas": "1.3.5",
            "prometheus-flask-exporter": "0.20.3",
            "querystring_parser": "1.2.4",
            "sqlalchemy": "1.4.42",
            "gunicorn": "20.1.0"
        }
    """
    items = [
        ("System information", " ".join((platform.system(), platform.version()))),
        ("Python version", platform.python_version()),
        ("MLflow version", mlflow.__version__),
        ("MLflow module location", mlflow.__file__),
        ("Tracking URI", mlflow.get_tracking_uri()),
        ("Registry URI", mlflow.get_registry_uri()),
    ]

    if (runtime := get_databricks_runtime()) is not None:
        items.append(("Databricks runtime version", runtime))

    active_run = mlflow.active_run()
    if active_run:
        items.extend(
            [
                ("Active experiment ID", active_run.info.experiment_id),
                ("Active run ID", active_run.info.run_id),
                ("Active run artifact URI", active_run.info.artifact_uri),
            ]
        )

    mlflow_envs = {
        k: ("***" if mask_envs else v) for k, v in os.environ.items() if k.startswith("MLFLOW_")
    }
    items.append(("MLflow environment variables", json.dumps(mlflow_envs, indent=4)))
    mlflow_dependencies = {
        r.name: pkg_resources.get_distribution(r.name).version
        for r in pkg_resources.working_set.by_key["mlflow"].requires()
    }
    items.append(("MLflow dependencies", json.dumps(mlflow_dependencies, indent=4)))
    for key, val in items:
        click.echo(f"{key}: {val}")

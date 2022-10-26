import os
import json
import pkg_resources
import platform
import click
import mlflow


def doctor():
    """
    Prints out the current environment's MLflow configuration and dependencies.
    """
    items = [
        ("Platform", platform.platform()),
        ("Python version", platform.python_version()),
        ("MLflow version", mlflow.__version__),
        ("MLflow module location", mlflow.__file__),
        ("Tracking URI", mlflow.get_tracking_uri()),
        ("Registry URI", mlflow.get_registry_uri()),
    ]
    active_run = mlflow.active_run()
    if active_run:
        items.extend(
            [
                ("Active experiment ID", active_run.info.experiment_id),
                ("Active run ID", active_run.info.run_id),
                ("Active run artifact URI", active_run.info.artifact_uri),
            ]
        )

    mlflow_envs = {k: v for k, v in os.environ.items() if k.startswith("MLFLOW_")}
    items.append(("MLflow environment variables", json.dumps(mlflow_envs, indent=2)))
    mlflow_dependencies = dict(
        (r.name, pkg_resources.get_distribution(r.name).version)
        for r in pkg_resources.working_set.by_key["mlflow"].requires()
    )
    items.append(("MLflow dependencies", json.dumps(mlflow_dependencies, indent=2)))
    for key, val in items:
        click.echo(f"{key}: {val}")

"""Provides the MLflow fluent API, allowing management of an active MLflow run.
For example::

    import mlflow
    mlflow.start_run()
    mlflow.log_param("my", "param")
    mlflow.log_metric("score", 100)
    mlflow.end_run()

You can also use syntax like this::

    with mlflow.start_run() as run:
	  ...

which will automatically terminate the run at the end of the block.
"""

import os

# pylint: disable=wrong-import-position
import mlflow.projects as projects  # noqa
import mlflow.tracking as tracking  # noqa
import mlflow.tracking.fluent

ActiveRun = mlflow.tracking.fluent.ActiveRun
log_param = mlflow.tracking.fluent.log_param
log_metric = mlflow.tracking.fluent.log_metric
log_artifacts = mlflow.tracking.fluent.log_artifacts
log_artifact = mlflow.tracking.fluent.log_artifact
active_run = mlflow.tracking.fluent.active_run
start_run = mlflow.tracking.fluent.start_run
end_run = mlflow.tracking.fluent.end_run
get_artifact_uri = mlflow.tracking.fluent.get_artifact_uri
set_tracking_uri = tracking.set_tracking_uri
get_tracking_uri = tracking.get_tracking_uri
create_experiment = mlflow.tracking.fluent.create_experiment


run = projects.run


__all__ = ["ActiveRun", "log_param", "log_metric", "log_artifacts", "log_artifact", "active_run",
           "start_run", "end_run", "get_artifact_uri", "set_tracking_uri", "create_experiment"]

"""
Example of hyperparameter search in MLflow using simple random search.

The run method will evaluate random combinations of parameters in a new MLflow run.

The runs are evaluated based on validation set loss. Test set score is calculated to verify the
results.

Several runs can be run in parallel.
"""

from concurrent.futures import ThreadPoolExecutor

import click
import numpy as np

import mlflow
import mlflow.projects
import mlflow.sklearn
import mlflow.tracking
from mlflow.tracking import MlflowClient

_inf = np.finfo(np.float64).max


@click.command(help="Perform grid search over train (main entry point).")
@click.option("--max-runs", type=click.INT, default=32, help="Maximum number of runs to evaluate.")
@click.option("--max-p", type=click.INT, default=1, help="Maximum number of parallel runs.")
@click.option("--epochs", type=click.INT, default=32, help="Number of epochs")
@click.option("--metric", type=click.STRING, default="rmse", help="Metric to optimize on.")
@click.option("--seed", type=click.INT, default=97531, help="Seed for the random generator")
@click.argument("training_data")
def run(training_data, max_runs, max_p, epochs, metric, seed):
    train_metric = f"train_{metric}"
    val_metric = f"val_{metric}"
    test_metric = f"test_{metric}"
    np.random.seed(seed)
    tracking_client = MlflowClient()

    def new_eval(
        nepochs, experiment_id, null_train_loss=_inf, null_val_loss=_inf, null_test_loss=_inf
    ):
        def eval(params):
            lr, momentum = params
            with mlflow.start_run(nested=True) as child_run:
                p = mlflow.projects.run(
                    run_id=child_run.info.run_id,
                    uri=".",
                    entry_point="train",
                    parameters={
                        "training_data": training_data,
                        "epochs": str(nepochs),
                        "learning_rate": str(lr),
                        "momentum": str(momentum),
                        "seed": str(seed),
                    },
                    experiment_id=experiment_id,
                    synchronous=False,
                )
                succeeded = p.wait()
                mlflow.log_params({"lr": lr, "momentum": momentum})
            if succeeded:
                training_run = tracking_client.get_run(p.run_id)
                metrics = training_run.data.metrics
                # cap the loss at the loss of the null model
                train_loss = min(null_train_loss, metrics[train_metric])
                val_loss = min(null_val_loss, metrics[val_metric])
                test_loss = min(null_test_loss, metrics[test_metric])
            else:
                # run failed => return null loss
                tracking_client.set_terminated(p.run_id, "FAILED")
                train_loss = null_train_loss
                val_loss = null_val_loss
                test_loss = null_test_loss
            mlflow.log_metrics(
                {
                    f"train_{metric}": train_loss,
                    f"val_{metric}": val_loss,
                    f"test_{metric}": test_loss,
                }
            )
            return p.run_id, train_loss, val_loss, test_loss

        return eval

    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id
        _, null_train_loss, null_val_loss, null_test_loss = new_eval(0, experiment_id)((0, 0))
        runs = [(np.random.uniform(1e-5, 1e-1), np.random.uniform(0, 1.0)) for _ in range(max_runs)]
        with ThreadPoolExecutor(max_workers=max_p) as executor:
            _ = executor.map(
                new_eval(epochs, experiment_id, null_train_loss, null_val_loss, null_test_loss),
                runs,
            )

        # find the best run, log its metrics as the final metrics of this run.
        client = MlflowClient()
        runs = client.search_runs(
            [experiment_id], f"tags.mlflow.parentRunId = '{run.info.run_id}' "
        )
        best_val_train = _inf
        best_val_valid = _inf
        best_val_test = _inf
        best_run = None
        for r in runs:
            if r.data.metrics["val_rmse"] < best_val_valid:
                best_run = r
                best_val_train = r.data.metrics["train_rmse"]
                best_val_valid = r.data.metrics["val_rmse"]
                best_val_test = r.data.metrics["test_rmse"]
        mlflow.set_tag("best_run", best_run.info.run_id)
        mlflow.log_metrics(
            {
                f"train_{metric}": best_val_train,
                f"val_{metric}": best_val_valid,
                f"test_{metric}": best_val_test,
            }
        )


if __name__ == "__main__":
    run()

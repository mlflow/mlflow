"""
Example of hyperparameter search in MLflow using simple random search.

The run method will evaluate random combinations of parameters in a new MLflow run.

The runs are evaluated based on validation set loss. Test set score is calculated to verify the
results.

Several runs can be run in parallel.
"""

import math

import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor

import click

import numpy as np

import mlflow
import mlflow.sklearn
import mlflow.tracking
import mlflow.projects


@click.command(help="Perform grid search over train (main entry point).")
@click.option("--max-runs", type=click.INT, default=32,
              help="Maximum number of runs to evaluate.")
@click.option("--max-p", type=click.INT, default=1,
              help="Maximum number of parallel runs.")
@click.option("--epochs", type=click.INT, default=32,
              help="Number of epochs")
@click.option("--metric", type=click.STRING, default="rmse",
              help="Metric to optimize on.")
@click.option("--seed", type=click.INT, default=97531,
              help="Seed for the random generator")
@click.option("--training-experiment-id", type=click.INT, default=-1,
              help="Maximum number of runs to evaluate. Inherit parent;s experiment if == -1.")
@click.argument("training_data")
def run(training_data, max_runs, max_p, epochs, metric, seed, training_experiment_id):
    train_metric = "train_{}".format(metric)
    val_metric = "val_{}".format(metric)
    test_metric = "test_{}".format(metric)
    np.random.seed(seed)
    tracking_client = mlflow.tracking.MlflowClient()

    def new_eval(nepochs,
                 experiment_id,
                 null_train_loss=math.inf,
                 null_val_loss=math.inf,
                 null_test_loss=math.inf):
        def eval(parms):
            lr, momentum = parms
            p = mlflow.projects.run(
                uri=".",
                entry_point="train",
                parameters={
                    "training_data": training_data,
                    "epochs": str(nepochs),
                    "learning_rate": str(lr),
                    "momentum": str(momentum),
                    "seed": str(seed)},
                experiment_id=experiment_id,
                block=False)
            if p.wait():
                training_run = tracking_client.get_run(p.run_id)

                def get_metric(metric_name):
                    return training_run.data.metrics[metric_name].value

                # cap the loss at the loss of the null model
                train_loss = min(null_train_loss, get_metric(train_metric))
                val_loss = min(null_val_loss, get_metric(val_metric))
                test_loss = min(null_test_loss, get_metric(test_metric))
            else:
                # run failed => return null loss
                tracking_client.set_terminated(p.run_id, "FAILED")
                train_loss = null_train_loss
                val_loss = null_val_loss
                test_loss = null_test_loss
            mlflow.log_metric(train_metric, train_loss)
            mlflow.log_metric(val_metric, val_loss)
            mlflow.log_metric(test_metric, test_loss)
            return p.run_id, train_loss, val_loss, test_loss

        return eval

    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id if training_experiment_id == -1 \
            else training_experiment_id
        _, null_train_loss, null_val_loss, null_test_loss = new_eval(0, experiment_id)((0, 0))
        runs = [(np.random.uniform(1e-5, 1e-1), np.random.uniform(0, 1.0)) for _ in range(max_runs)]
        best_train_loss = math.inf
        best_val_loss = math.inf
        best_test_loss = math.inf
        best_run = None
        with ThreadPoolExecutor(max_workers=max_p) as executor:
            result = executor.map(new_eval(epochs,
                                           experiment_id,
                                           null_train_loss,
                                           null_val_loss,
                                           null_test_loss),
                                  runs)
        tmp = tempfile.mkdtemp()
        results_file_path = os.path.join(str(tmp), "results.txt")
        with open(results_file_path, "w") as f:
            for res in result:
                run_id, train_loss, val_loss, test_loss = res
                if val_loss < best_val_loss:
                    best_run = run_id
                    best_train_loss = train_loss
                    best_val_loss = val_loss
                    best_test_loss = test_loss
                f.write("{run_id} {train} {val} {test}\n".format(run_id=run_id,
                                                                 train=train_loss,
                                                                 val=val_loss,
                                                                 test=test_loss))
        mlflow.log_artifact(results_file_path, "training_runs.txt")
        # record which run produced the best results, store it as an artifact
        best_run_path = os.path.join(os.path.join(tmp, "best_run.txt"))
        with open(best_run_path, "w") as f:
            f.write("{run_id} {train} {val} {test}\n".format(run_id=best_run,
                                                             train=best_train_loss,
                                                             val=best_val_loss,
                                                             test=best_test_loss))
        mlflow.log_artifact(best_run_path, "best-run")
        mlflow.log_metric(val_metric, best_val_loss)
        mlflow.log_metric(test_metric, best_test_loss)
        shutil.rmtree(tmp)


if __name__ == '__main__':
    run()

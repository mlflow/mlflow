import click
import math

import os
import shutil
import tempfile

from hyperopt import fmin, hp, tpe, rand

import mlflow.projects


@click.command(help="Perform hyper parameter search with Hyperopt library."
                    "Optimize dl_train target.")
@click.option("--max-runs", type=click.INT, default=10,
              help="Maximum number of runs to evaluate.")
@click.option("--epochs", type=click.INT, default=500,
              help="Number of epochs")
@click.option("--metric", type=click.STRING, default="rmse",
              help="Metric to optimize on.")
@click.option("--algo", type=click.STRING, default="tpe.suggest",
              help="Optimizer algorhitm.")
@click.option("--seed", type=click.INT, default=97531,
              help="Seed for the random generator")
@click.option("--training-experiment-id", type=click.INT, default=-1,
              help="Maximum number of runs to evaluate. Inherit parent;s experiment if == -1.")
@click.argument("training_data")
def train(training_data, max_runs, epochs, metric, algo, seed, training_experiment_id):
    """
    Run hyper param optimization.
    """
    # create random file to store run ids of the training tasks
    tmp = tempfile.mkdtemp()
    results_path = os.path.join(tmp, "results")

    def eval(parms):
        import mlflow.tracking
        active_run = mlflow.active_run()
        experiment_id = active_run.info.experiment_id if training_experiment_id == -1 \
            else training_experiment_id
        lr, beta1, beta2 = parms
        p = mlflow.projects.run(
            uri=".",
            entry_point="dl_train",
            parameters={
                "training_data": training_data,
                "epochs": str(epochs),
                "learning_rate": str(lr),
                "beta1": str(beta1),
                "beta2": str(beta2),
                "seed": seed},
            experiment_id=experiment_id
        )
        store = mlflow.tracking._get_store()
        if not p.wait():
            # at least the null metric should be available
            try:
                metric_val = store.get_metric(p.run_id, metric + "_null")
            except Exception:
                raise Exception("Training run failed.")
        else:
            metric_val = store.get_metric(p.run_id, metric)
            metric_val_null = store.get_metric(p.run_id, metric + "_null")
            # cap loss at the null model to avoid NaNs / Infs
            if metric_val_null.value < metric_val.value:
                metric_val = metric_val_null
        mlflow.log_metric(metric, metric_val.value)
        with open(results_path, "a") as f:
            f.write("{runId} {val}\n".format(runId=p.run_id, val=metric_val.value))
        return metric_val.value

    with mlflow.start_run():
        space = [
            hp.uniform('lr', 1e-3, 1e-1),
            hp.uniform('beta1', .8, 1.0),
            hp.uniform('beta2', .8, 1.0),
        ]
        best = fmin(fn=eval,
                    space=space,
                    algo=tpe.suggest if algo == "tpe.suggest" else rand.suggest,
                    max_evals=max_runs)
        print('best', best)
        best_val = math.inf
        best_run = None
        # we do not have tags yet, for now store list of executed runs as an artifact
        mlflow.log_artifact(results_path, "training_runs")
        with open(results_path) as f:
            for line in f.readlines():
                run_id, str_val = line.split(" ")
                val = float(str_val)
                if val < best_val:
                    best_val = val
                    best_run = run_id
        # record which run produced the best results, store it as a param for now
        best_run_path = os.path.join(os.path.join(tmp, "best_run.txt"))
        with open(best_run_path, "w") as f:
            f.write("{run_id} {val}\n".format(run_id=best_run, val=best_val))
        mlflow.log_artifact(best_run_path, "best-run")
        mlflow.log_metric(metric, best_val)
        shutil.rmtree(tmp)


if __name__ == '__main__':
    train()

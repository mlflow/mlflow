import math

import os
import shutil
import tempfile

import click

import mlflow
import mlflow.sklearn
import mlflow.tracking
import mlflow.projects

"""
Example of hyper param search in MLflow using simple grid search.

The run method will create a grid of pramaters and evaluate each combination in a new mlflow run 
invoking the main entry point.

The runs are evaluated based on validation set loss. Test set score is calculated to verify the 
results. 

Several runs can be run in parallel.  
"""



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
    val_metric = "val_{}".format(metric)
    test_metric = "test_{}".format(metric)

    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id if training_experiment_id == -1 \
            else training_experiment_id

        store = mlflow.tracking._get_store()
        # Evaluate null model first.
        p = mlflow.projects.run(
            uri=".",
            entry_point="main",
            parameters={
                "training_data": training_data,
                "epochs": "0",
                "learning_rate": "0",
                "momentum": "0",
                "seed": str(seed)},
            experiment_id=experiment_id,
            block=True)

        null_val_loss = store.get_metric(p.run_id, val_metric).value
        null_test_loss = store.get_metric(p.run_id, test_metric).value
        nvals = int(math.sqrt(max_runs))
        if nvals <= 1:
            raise Exception("Number of runs must be >= 4")
        momentum_step = .75 / (nvals - 1)
        lr_step = .25 / nvals
        lr_vals = [lr_step * x for x in range(1, nvals + 1)]
        momentum_vals = [momentum_step * x for x in range(nvals)]

        results = []
        best_val_loss = math.inf
        best_test_loss = math.inf
        best_run = None
        grid = [(lr, m) for lr in lr_vals for m in momentum_vals]
        print("grid", grid)
        run_from = 0
        while run_from < len(grid):
            nruns = min(max_p, len(grid) - run_from)
            runs = [
                mlflow.projects.run(
                    uri=".",
                    entry_point="main",
                    parameters={
                        "training_data": training_data,
                        "epochs": str(epochs),
                        "learning_rate": str(grid[run_from + i][0]),
                        "momentum": str(grid[run_from + i][1]),
                        "seed": str(seed)},
                    experiment_id=experiment_id,
                    block=False)
                for i in range(nruns)
            ]
            run_from += nruns
            for p in runs:
                if p.wait():
                    # cap the loss at the loss of the null model
                    val_loss = min(null_val_loss, store.get_metric(p.run_id, val_metric).value)
                    test_loss = min(null_test_loss, store.get_metric(p.run_id, test_metric).value)

                else:
                    # run failed => return null loss
                    val_loss = null_val_loss
                    test_loss = null_test_loss
                mlflow.log_metric(val_metric, val_loss)
                mlflow.log_metric(test_metric, test_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_test_loss = test_loss
                    best_run = p.run_id
                results.append("{runId} {val} {test}".format(runId=p.run_id,
                                                             val=val_loss,
                                                             test=best_test_loss))

        tmp = tempfile.mkdtemp()
        results_file_path = os.path.join(str(tmp), "results.txt")
        with open(results_file_path, "w") as f:
            f.write("\n".join(results))
        mlflow.log_artifact(results_file_path, "training_runs.txt")
        # record which run produced the best results, store it as a param for now
        best_run_path = os.path.join(os.path.join(tmp, "best_run.txt"))
        with open(best_run_path, "w") as f:
            f.write("{run_id} {val} {test}\n".format(run_id=best_run,
                                                     val=best_val_loss,
                                                     test=best_test_loss))
        mlflow.log_artifact(best_run_path, "best-run")
        mlflow.log_metric(val_metric, best_val_loss)
        mlflow.log_metric(test_metric, best_test_loss)
        shutil.rmtree(tmp)


if __name__ == '__main__':
    run()

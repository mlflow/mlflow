import math

import os
import shutil
import tempfile

import click
import GPyOpt

import mlflow
import mlflow.sklearn
import mlflow.tracking
import mlflow.projects

"""
Example of hyper param search in MLflow using GPyOpt.
"""


@click.group()
@click.version_option()
def cli():
    pass


@cli.command()
@click.option("--max-runs", type=click.INT, default=100,
              help="Maximum number of runs to evaluate.")
@click.option("--batch-size", type=click.INT, default=4,
              help="Number of runs to evaluate in a batch")
@click.option("--max-p", type=click.INT, default=4,
              help="Maximum number of parallel runs.")
@click.option("--epochs", type=click.INT, default=500,
              help="Number of epochs")
@click.option("--metric", type=click.STRING, default="rmse",
              help="Metric to optimize on.")
@click.option("--gpy-model", type=click.STRING, default="GP_MCMC",
              help="Optimizer algorhitm.")
@click.option("--gpy-acquisition", type=click.STRING, default="EI_MCMC",
              help="Optimizer algorhitm.")
@click.option("--initial-design", type=click.STRING, default="random",
              help="Optimizer algorhitm.")
@click.option("--seed", type=click.INT, default=97531,
              help="Seed for the random generator")
@click.option("--training-experiment-id", type=click.INT, default=-1,
              help="Maximum number of runs to evaluate. Inherit parent;s experiment if == -1.")
@click.argument("training_data")
def run(training_data, max_runs, batch_size, max_p, epochs, metric, gpy_model, gpy_acquisition,
        initial_design, seed, training_experiment_id):
    bounds = [
        {'name': 'lr', 'type': 'continuous', 'domain': (1e-3, 1e-1)},
        {'name': 'lr_decay', 'type': 'continuous', 'domain': (.8, 1)},
        {'name': 'rho', 'type': 'continuous', 'domain': (.8, 1)},
    ]
    # create random file to store run ids of the training tasks
    tmp = tempfile.mkdtemp()
    results_path = os.path.join(tmp, "results.txt")

    def eval(parms):
        """
        Train keras model with given parameters by invoking mlflow run.

        Notice we store runUuid and resulting metric in a file. We will later use these to pick the
        best run and to log the runUuids of the child runs as an artifact.

        :param parms: Parameters to the train_keras script we optimize over:
                      learning_rate, drop_out_1
        :return: The rmse value logged as the result of the run.
        """
        active_run = mlflow.active_run()
        experiment_id = active_run.info.experiment_id if training_experiment_id == -1 \
            else training_experiment_id
        lr, beta1, beta2 = parms[0]
        # lr, beta1 = parms[0]
        p = mlflow.projects.run(
            uri=".",
            entry_point="dl_train",
            parameters={
                "training_data": training_data,
                "epochs": str(epochs),
                "learning_rate": str(lr),
                "beta1": str(beta1),
                "beta2": str(beta2),
                "seed": str(seed)},
            experiment_id=experiment_id,
            block=False
        )
        store = mlflow.tracking._get_store()
        if not p.wait():
            # at least the null metric shoudl be available
            try:
                metric_val = store.get_metric(p.run_id, metric + "_null")
            except Exception:
                raise Exception("Training run failed.")
        else:
            metric_val = store.get_metric(p.run_id, metric)
            metric_val_null = store.get_metric(p.run_id, metric + "_null")
            # cap loss at the null model to avoid NaNs / Infs, GPyOpt can not handle those.
            # also, get prettier plots this way.
            if metric_val_null.value < metric_val.value:
                metric_val = metric_val_null
        mlflow.log_metric(metric, metric_val.value)
        with open(results_path, "a") as f:
            f.write("{runId} {val}\n".format(runId=p.run_id, val=metric_val.value))
        return metric_val.value

    with mlflow.start_run():
        # null model
        myProblem = GPyOpt.methods.BayesianOptimization(eval,
                                                        bounds,
                                                        batch_size=batch_size,
                                                        num_cores=max_p,
                                                        model_type=gpy_model,
                                                        acquisition_type=gpy_acquisition,
                                                        initial_design_type=initial_design,
                                                        initial_design_numdata=16,
                                                        exact_feval=False)
        myProblem.run_optimization(max_runs)
        import matplotlib
        matplotlib.use('agg')
        from matplotlib import pyplot as plt
        plt.switch_backend('agg')
        acquisition_plot = os.path.join(tmp, "acquisition_plot.png")
        convergence_plot = os.path.join(tmp, "convergence_plot.png")
        myProblem.plot_acquisition(filename=acquisition_plot)
        myProblem.plot_convergence(filename=convergence_plot)
        if os.path.exists(convergence_plot):
            mlflow.log_artifact(convergence_plot, "converegence_plot")
        if os.path.exists(acquisition_plot):
            mlflow.log_artifact(acquisition_plot, "acquisition_plot")
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
    run()

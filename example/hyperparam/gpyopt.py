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

The run method will instantiate and run gpyopt optimizer. Each parameter configuration is 
evaluated in a new mlflow run invoking main entry point with selected parameters.

The runs are evaluated based on validation set loss. Test set score is calculated to verify the 
results.

Several runs can be run in parallel.
"""


@click.command(help="Perform hyper parameter search with GPyOpt library."
                    "Optimize dl_train target.")
@click.option("--max-runs", type=click.INT, default=20,
              help="Maximum number of runs to evaluate.")
@click.option("--batch-size", type=click.INT, default=8,
              help="Number of runs to evaluate in a batch")
@click.option("--max-p", type=click.INT, default=8,
              help="Maximum number of parallel runs.")
@click.option("--epochs", type=click.INT, default=32,
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
        {'name': 'lr', 'type': 'continuous', 'domain': (1e-5, 1e-1)},
        {'name': 'momentum', 'type': 'continuous', 'domain': (0.0, 1.0)},
    ]
    # create random file to store run ids of the training tasks
    tmp = tempfile.mkdtemp()
    results_path = os.path.join(tmp, "results.txt")

    def new_eval(nepochs, experiment_id, valid_null_loss, test_null_loss, return_test_loss=False):
        """
        Create a new eval function

        :param nepochs: Number of epochs to train the model.
        :experiment_id: Experiment id for the training run
        :valid_null_loss: Loss of a null model on the validation dataset
        :test_null_loss: Loss of a null model on the test dataset.
        :return_test_loss: Return both validation and test loss if set.

        :return: new eval function.
        """

        def eval(params):
            """
            Train keras model with given parameters by invoking mlflow run.

            Notice we store runUuid and resulting metric in a file. We will later use these to pick
            the best run and to log the runUuids of the child runs as an artifact. This is a
            temporary workaround until MLflow offers better mechanism of linking runs together.

            :param params: Parameters to the train_keras script we optimize over:
                          learning_rate, drop_out_1
            :return: The metric value evaluated on the validation data.
            """
            lr, momentum = params[0]
            p = mlflow.projects.run(
                uri=".",
                entry_point="main",
                parameters={
                    "training_data": training_data,
                    "epochs": str(nepochs),
                    "learning_rate": str(lr),
                    "momentum": str(momentum),
                    "seed": str(seed)},
                experiment_id=experiment_id,
                block=False
            )
            store = mlflow.tracking._get_store()
            if p.wait():
                # cap the loss at the loss of the null model
                valid_loss = min(valid_null_loss,
                                 store.get_metric(p.run_id, "val_{}".format(metric)).value)
                test_loss = min(test_null_loss,
                                store.get_metric(p.run_id, "test_{}".format(metric)).value)

            else:
                # run failed => return null loss
                valid_loss = valid_null_loss
                test_loss = test_null_loss

            mlflow.log_metric("val_{}".format(metric), valid_loss)
            mlflow.log_metric("test_{}".format(metric), test_loss)
            with open(results_path, "a") as f:
                f.write("{runId} {val} {test}\n".format(runId=p.run_id,
                                                        val=valid_loss,
                                                        test=test_loss))
            if return_test_loss:
                return valid_loss, test_loss
            else:
                return valid_loss

        return eval

    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id if training_experiment_id == -1 \
            else training_experiment_id
        # Evaluate null model first.
        # We use null model (predict everything to the mean) as a reasonable upper bound on loss.
        # We need an upper bound to handle the failed runs (e.g. return NaNs) because GPyOpt can not
        # handle Infs.
        # Allways including a null model in our results is also a good ML practice.
        valid_null_loss, test_null_loss = new_eval(0,
                                                   experiment_id,
                                                   math.inf,
                                                   math.inf,
                                                   True)(params=[[0, 0]])
        myProblem = GPyOpt.methods.BayesianOptimization(new_eval(epochs,
                                                                 experiment_id,
                                                                 valid_null_loss,
                                                                 test_null_loss),
                                                        bounds,
                                                        batch_size=batch_size,
                                                        num_cores=max_p,
                                                        model_type=gpy_model,
                                                        acquisition_type=gpy_acquisition,
                                                        initial_design_type=initial_design,
                                                        initial_design_numdata=max_runs >> 2,
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
        best_val_valid = math.inf
        best_val_test = math.inf
        best_run = None
        # we do not have tags yet, for now store list of executed runs as an artifact
        mlflow.log_artifact(results_path, "training_runs")
        with open(results_path) as f:
            for line in f.readlines():
                run_id, str_val, str_val2 = line.split(" ")
                val = float(str_val)
                if val < best_val_valid:
                    best_val_valid = val
                    best_val_test = float(str_val2)
                    best_run = run_id
        # record which run produced the best results, store it as a param for now
        best_run_path = os.path.join(os.path.join(tmp, "best_run.txt"))
        with open(best_run_path, "w") as f:
            f.write("{run_id} {val}\n".format(run_id=best_run, val=best_val_valid))
        mlflow.log_artifact(best_run_path, "best-run")
        mlflow.log_metric("val_{}".format(metric), best_val_valid)
        mlflow.log_metric("test_{}".format(metric), best_val_test)
        shutil.rmtree(tmp)


if __name__ == '__main__':
    run()

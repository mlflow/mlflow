import math

import os
import shutil
import sys
import tempfile

import GPyOpt

import mlflow
import mlflow.sklearn
import mlflow.tracking
import mlflow.projects

"""
Example of hyper param search in MLflow using GPyOpt.
"""
if __name__ == "__main__":
    max_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    max_p = int(sys.argv[3]) if len(sys.argv) > 3 else batch_size
    max_epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    lr_min = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-5
    lr_max = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-1
    drop_out_1_min = float(sys.argv[7]) if len(sys.argv) > 7 else .001
    drop_out_1_max = float(sys.argv[8]) if len(sys.argv) > 8 else .2
    metric_name = sys.argv[9] if len(sys.argv) > 9 else "rmse"
    gpy_model = sys.argv[10] if len(sys.argv) > 10 else "GP_MCMC"
    gpy_acquisition = sys.argv[11] if len(sys.argv) > 11 else "EI_MCMC"
    initial_design = sys.argv[12] if len(sys.argv) > 12 else "random"
    training_experiment_id = int(sys.argv[13]) if len(sys.argv) > 13 else None
    seed = int(sys.argv[14]) if len(sys.argv) > 14 else 97531

    bounds = [
        {'name': 'lr', 'type': 'continuous', 'domain': (lr_min, lr_max)},
        {'name': 'drop_out_1', 'type': 'continuous', 'domain': (drop_out_1_min, drop_out_1_max)},
    ]
    # create random file to store run ids of the training tasks
    tmp = tempfile.mkdtemp()
    results_path = os.path.join(tmp, "results")

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
        experiment_id = training_experiment_id or active_run.info.experiment_id
        lr, drop1 = parms[0]
        drop2 = 0
        p = mlflow.projects.run(
            uri=".",
            entry_point="dl_train",
            parameters={"epochs": str(max_epochs),
                        "learning_rate": str(lr),
                        'drop_out_1': str(drop1),
                        'drop_out_2': str(drop2),
                        'seed': str(seed)},
            experiment_id=experiment_id
        )
        p.wait()
        store = mlflow.tracking._get_store()
        metric_val = store.get_metric(p.run_id, metric_name)
        mlflow.log_metric(metric_val.key, metric_val.value)
        with open(results_path, "a") as f:
            f.write("{runId} {val}\n".format(runId=active_run.info.run_uuid, val=metric_val.value))
        return metric_val.value


    with mlflow.start_run():
        myProblem = GPyOpt.methods.BayesianOptimization(eval,
                                                        bounds,
                                                        batch_size=batch_size,
                                                        num_cores=max_p,
                                                        model_type=gpy_model,
                                                        acquisition_type=gpy_acquisition,
                                                        initial_design_type=initial_design)
        myProblem.run_optimization(max_runs)
        import matplotlib
        matplotlib.use('agg')
        from matplotlib import pyplot as plt
        plt.switch_backend('agg')
        acquisition_plot = os.path.join(tmp, "acquisition_plot.png")
        convergence_plot = os.path.join(tmp, "convergence_plot.png")
        myProblem.plot_acquisition(filename=acquisition_plot)
        myProblem.plot_convergence(filename=convergence_plot)
        mlflow.log_artifact(convergence_plot, "converegence_plot")
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
        mlflow.log_param("best-run", run_id)
        mlflow.log_metric(metric_name, best_val)
        shutil.rmtree(tmp)




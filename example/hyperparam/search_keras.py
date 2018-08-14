import os
import sys


import GPyOpt

import mlflow
import mlflow.sklearn
import mlflow.tracking
import mlflow.projects

if __name__ == "__main__":
    max_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    max_p = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    max_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    if max_epochs < 50:
        raise Exception("need at least 50 epochs")
    lr_min = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
    lr_max = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-1
    drop_out_1_min = float(sys.argv[6]) if len(sys.argv) > 6 else .001
    drop_out_1_max = float(sys.argv[7]) if len(sys.argv) > 7 else .2
    drop_out_2_min = float(sys.argv[8]) if len(sys.argv) > 8 else .001
    drop_out_2_max = float(sys.argv[9]) if len(sys.argv) > 9 else .2

    metric_name = sys.argv[10] if len(sys.argv) > 10 else "rmse"

    epoch_eps = 1e-3
    bounds = [
        {'name': 'epochs', 'type': 'continuous', 'domain': (epoch_eps, 1)},
        {'name': 'lr', 'type': 'continuous', 'domain': (lr_min, lr_max)},
        {'name': 'drop_out_1', 'type': 'continuous', 'domain': (drop_out_1_min, drop_out_1_max)},
        {'name': 'drop_out_2', 'type': 'continuous', 'domain': (drop_out_2_min, drop_out_2_max)},
    ]
    x = float(max_epochs - 1)/(1 - epoch_eps)
    y = max_epochs - x

    def eval(parms):
        epochs, lr, drop1, drop2 = parms[0]
        p = mlflow.projects.run(
            uri=os.path.abspath(os.path.dirname(__file__)),
            entry_point="dl_train",
            parameters={"epochs": int(x * epochs + y),
                        "learning_rate": lr,
                        'drop_out_1': drop1,
                        'drop_out_2': drop2}
        )
        p.wait()
        store = mlflow.tracking._get_store()
        if metric_name.startswith("-"):
            metric_val = store.get_metric(p.run_id, metric_name[1:])
            mlflow.log_metric(metric_val.key, metric_val.value)
            return -metric_val.value
        else:
            metric_val = store.get_metric(p.run_id, metric_name)
            mlflow.log_metric(metric_val.key, metric_val.value)
            return metric_val.value


    with mlflow.start_run() as active_run:
        mlflow.log_param("max_epochs", max_epochs)
        mlflow.log_param("learning_rate", "(%f, %f)" % (lr_min, lr_max))
        mlflow.log_param("drop_out_1", "(%f, %f)" % (drop_out_1_min, drop_out_1_max))
        mlflow.log_param("drop_out_2", "(%f, %f)" % (drop_out_2_min, drop_out_2_max))
        mlflow.log_param("metric", metric_name)
        myProblem = GPyOpt.methods.BayesianOptimization(eval, bounds,
                                                        batch_size=max_p,
                                                        num_cores=max_p)
        myProblem.run_optimization(max_runs)
        store = mlflow.tracking._get_store()
        store.search_runs(experiment_ids=[active_run.run_info.experiment_id], )


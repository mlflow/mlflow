import os
import sys
import warnings

import GPyOpt
import numpy as np

import mlflow
import mlflow.sklearn
import mlflow.tracking
import mlflow.projects

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    max_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    max_p = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    alpha_min = float(sys.argv[3]) if len(sys.argv) > 3 else 0
    alpha_max = float(sys.argv[4]) if len(sys.argv) > 4 else 1
    l1_min = float(sys.argv[5]) if len(sys.argv) > 5 else 0
    l1_max = float(sys.argv[6]) if len(sys.argv) > 6 else 1
    metric_name = float(sys.argv[7]) if len(sys.argv) > 7 else "rmse"

    bounds = [
        {'name': 'alpha', 'type': 'continuous', 'domain': (alpha_min, alpha_max)},
        {'name': 'l1_ratio', 'type': 'continuous', 'domain': (l1_min, l1_max)},
    ]

    def eval(parms):
        alpha, l1_ratio = parms[0]
        p = mlflow.projects.run(
            uri=os.path.abspath(os.path.dirname(__file__)),
            entry_point="main",
            parameters={"alpha": alpha, "l1_ratio": l1_ratio}
        )
        p.wait()
        store = mlflow.tracking._get_store()
        metric_val = store.get_metric(p.run_id, metric_name)
        # log the final metric to the hyperparam run.
        mlflow.log_metric(metric_val.key, metric_val.value)
        return metric_val.value


    myProblem = GPyOpt.methods.BayesianOptimization(eval, bounds, batch_size=max_p, num_cores=max_p)
    myProblem.run_optimization(max_runs)

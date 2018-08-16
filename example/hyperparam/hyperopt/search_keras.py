import math

import os
import shutil
import sys
import tempfile

from hyperopt import fmin, hp, tpe, rand


import mlflow.projects



if __name__ == "__main__":
    max_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    max_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    lr_min = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    lr_max = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-1
    drop_out_1_min = float(sys.argv[5]) if len(sys.argv) > 5 else .001
    drop_out_1_max = float(sys.argv[6]) if len(sys.argv) > 6 else .2
    metric_name = sys.argv[7] if len(sys.argv) > 7 else "rmse"
    algo = sys.arv[8] if 8 > len(sys.argv) else "tpe.suggest"
    seed = int(sys.argv[9]) if len(sys.argv) > 9 else 97531
    training_experiment_id = int(sys.argv[10]) if len(sys.argv) > 10 else None

    # create random file to store run ids of the training tasks
    tmp = tempfile.mkdtemp()
    results_path = os.path.join(tmp, "results")

    def eval(parms):
        import mlflow
        import mlflow.sklearn
        import mlflow.tracking
        active_run = mlflow.active_run()
        experiment_id = training_experiment_id or active_run.info.experiment_id
        lr, drop1 = parms
        drop2 = 0
        p = mlflow.projects.run(
            uri=".",
            entry_point="dl_train",
            parameters={"epochs": str(max_epochs),
                        "learning_rate": str(lr),
                        'drop_out_1': str(drop1),
                        'drop_out_2': str(drop2),
                        'seed': seed},
            experiment_id=experiment_id
        )
        store = mlflow.tracking._get_store()
        metric_val = store.get_metric(p.run_id, metric_name)
        mlflow.log_metric(metric_val.key, metric_val.value)
        with open(results_path, "a") as f:
            f.write("{runId} {val}\n".format(runId=active_run.info.run_uuid, val=metric_val.value))
        return metric_val.value


    with mlflow.start_run():
        space = [
            hp.uniform('lr', lr_min, lr_max),
            hp.uniform('drop_out_1', drop_out_1_min, drop_out_1_max),
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
        mlflow.log_param("best-run", run_id)
        mlflow.log_metric(metric_name, best_val)
        shutil.rmtree(tmp)




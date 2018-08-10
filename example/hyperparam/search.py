
import os
import re
import sys
import warnings

import GPyOpt

from subprocess import Popen, PIPE, STDOUT

import numpy as np

import mlflow
import mlflow.sklearn
import mlflow.tracking
import mlflow.projects
from mlflow.utils.logging_utils import eprint



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    max_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    max_p = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    alpha_min = float(sys.argv[3]) if len(sys.argv) > 3 else 0
    alpha_max = float(sys.argv[4]) if len(sys.argv) > 4 else 1
    l1_min = float(sys.argv[5]) if len(sys.argv) > 5 else 0
    l1_max = float(sys.argv[6]) if len(sys.argv) > 6 else 1

    bounds = [
        {'name': 'alpha', 'type': 'continuous', 'domain': (alpha_min, alpha_max)},
        {'name': 'l1_ratio', 'type': 'continuous', 'domain': (l1_min, l1_max)},
    ]

    def eval(parms):
        print(parms)
        print(parms.shape)
        alpha, l1_ratio = parms[0]
        cmd = ["mlflow", "run", os.path.abspath(os.path.dirname(__file__))]
        cmd += ["-P", "alpha={}".format(alpha)]
        cmd += ["-P", "l1_ratio={}".format(l1_ratio)]
        print(" ".join(cmd))
        proc = Popen(cmd, stdout=PIPE, stderr=STDOUT, env={
            "PATH": os.environ.get("PATH"),
            "LANG": os.environ.get("LANG")
        },
                     universal_newlines=True, preexec_fn=os.setsid)
        output = []
        for x in iter(proc.stdout.readline, ""):
            m = re.search("=== Run [(]ID '(.*)'[)] (\w+) ===", x)
            if m:
                runId = m.group(1)
                status = m.group(2)
                if status == "succeeded":
                    store = mlflow.tracking._get_store()
                    rmse = store.get_metric(runId, "rmse")
                    mlflow.log_metric(rmse.key, rmse.value)
                    return rmse.value
                else:
                    output_str = "\n".join(output + [x])
                    msg = "evaluation failed, captured output *** \n {} \n ***".format(output_str)
                    raise Exception(msg)
            else:
                output.append(x)
        output_str = "\n".join(output)
        msg = "Evaluation did not fail nor succeed? output *** \n {} \n ***".format(output_str)
        raise Exception(msg)

    myProblem = GPyOpt.methods.BayesianOptimization(eval, bounds, batch_size=max_p, num_cores=max_p)
    myProblem.run_optimization(max_runs)





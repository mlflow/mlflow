from __future__ import print_function

from subprocess import Popen
import os.path
import pytest

# TODO - support for testing jupyter notebooks
# TODO - example specific setup and assertions

def run_an_example(example_dir, example_command, clear_mlruns):
    # construct command to run
    if clear_mlruns:
        command = "cd " + example_dir + " && rm -fr mlruns && " + example_command
    else:
        command = "cd " + example_dir + " && " + example_command

    # example code
    test_run = Popen(command, shell=True)

    # check exit code for return code 0
    test_run.wait()

    # check that example ran successfully
    assert test_run.returncode == 0

    # confirm creation of ./mlruns directory
    assert os.path.isdir(os.path.join(example_dir,'mlruns'))


@pytest.mark.parametrize("directory,command,clear_mlruns",[
    ("examples/sklearn_logistic_regression","python train.py",True),
    ("examples/sklearn_elasticnet_wine","mlflow run . -P alpha=0.5",True),
    ("examples/sklearn_elasticnet_diabetes/linux", "mlflow run .",True),
    ("examples/h2o", "python random_forest.py",True),
    ("examples/hyperparam","mlflow experiments create individual_runs",True),
    ("examples/hyperparam","mlflow experiments create hyper_param_runs",False),
    ("examples/hyperparam","mlflow run -e train --experiment-id 1 .",False),
    ("examples/hyperparam","mlflow run -e random --experiment-id 2  -P training_experiment_id=1 .",False),
    ("examples/hyperparam","mlflow run -e gpyopt --experiment-id 2  -P training_experiment_id=1 .",False),
    ("examples/hyperparam", "mlflow run -e hyperopt --experiment-id 2  -P training_experiment_id=1 .", False),

])
def test_example(directory, command, clear_mlruns):
    run_an_example(directory, command, clear_mlruns)


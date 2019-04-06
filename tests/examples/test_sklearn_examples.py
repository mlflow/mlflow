from __future__ import print_function

from subprocess import Popen, PIPE
import pytest

def test_sklearn_logistic_regression():
    command="cd examples/sklearn_logistic_regression/ && python train.py"
    stdout_response = "Score: 0.6666666666666666\nModel saved in run"

    test_run = Popen(command,shell=True,stdout=PIPE)

    # check exit code for return code 0
    test_run.wait()
    assert test_run.returncode == 0

    #  check for expected output
    out_text = test_run.stdout.read()
    assert out_text[:len(stdout_response)] == stdout_response


def test_sklearn_elasticnet_wine():
    command = "cd examples/sklearn_elasticnet_wine && mlflow run . -P alpha=0.5"
    stdout_response = "Elasticnet model (alpha=0.500000, l1_ratio=0.100000):\n" + \
        "  RMSE: 0.7947931019036529\n" + \
        "  MAE: 0.6189130834228138\n" + \
        "  R2: 0.18411668718221819\n"

    test_run = Popen(command,shell=True,stdout=PIPE)

    # check exit code for return code 0
    test_run.wait()
    assert test_run.returncode == 0

    #  check for expected output
    out_text = test_run.stdout.read()
    print(out_text)
    assert out_text[:len(stdout_response)] == stdout_response


def test_sklearn_elasticnet_diabetes():
    command = "cd examples/sklearn_elasticnet_diabetes/linux && mlflow run ."
    stdout_response = "Elasticnet model (alpha=0.010000, l1_ratio=0.100000):\n" +\
        "  RMSE: 70.90385970638309\n  MAE: 59.63089172068812\n  R2: 0.23986643454764633\n" +\
        "Computing regularization path using the elastic net.\n"

    test_run = Popen(command,shell=True,stdout=PIPE)

    # check exit code for return code 0
    test_run.wait()
    assert test_run.returncode == 0

    #  check for expected output
    out_text = test_run.stdout.read()
    print(out_text)
    assert out_text[:len(stdout_response)] == stdout_response
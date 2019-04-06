from __future__ import print_function

from subprocess import Popen, PIPE
import pytest
import shutil
import os.path





def test_sklearn_logistic_regression():

    command="cd examples/sklearn_logistic_regression/ &&  rm -fr mlruns && python train.py"
    stdout_response = "Score: 0.6666666666666666\nModel saved in run"

    test_run = Popen(command,shell=True,stdout=PIPE)

    # check exit code for return code 0
    test_run.wait()
    assert test_run.returncode == 0

    #  check for expected output
    out_text = test_run.stdout.read()
    assert out_text[:len(stdout_response)] == stdout_response


def test_sklearn_elasticnet_wine():
    # setup conda environment for test
    setup_run = Popen("conda env remove -n tutorial -y",shell=True,stdout=PIPE,stderr=PIPE)  #ignore any errors
    setup_run.wait()
    print(setup_run.stdout.read())
    print(setup_run.stderr.read())

    setup_run = Popen("conda env create -f examples/sklearn_elasticnet_wine/conda.yaml",shell=True,
                      stderr=PIPE, stdout=PIPE)
    setup_run.wait()
    print(setup_run.stdout.read())
    print(setup_run.stderr.read())
    assert setup_run.returncode == 0


    command = "cd examples/sklearn_elasticnet_wine && rm -fr mlruns && . activate tutorial && mlflow run . -P alpha=0.5 --no-conda"
    stdout_response = "Elasticnet model (alpha=0.500000, l1_ratio=0.100000):\n" + \
        "  RMSE: 0.7947931019036529\n" + \
        "  MAE: 0.6189130834228138\n" + \
        "  R2: 0.18411668718221819\n"

    test_run = Popen(command,shell=True,stdout=PIPE)

    # check exit code for return code 0
    test_run.wait()
    assert test_run.returncode == 0

    # #  check for expected output
    # out_text = test_run.stdout.read()
    # print(out_text)
    # assert out_text[:len(stdout_response)] == stdout_response


# def test_sklearn_elasticnet_diabetes():
#     command = "cd examples/sklearn_elasticnet_diabetes/linux  && rm -fr mlruns && mlflow run ."
#     stdout_response = "Elasticnet model (alpha=0.010000, l1_ratio=0.100000):\n" +\
#         "  RMSE: 70.90385970638309\n  MAE: 59.63089172068812\n  R2: 0.23986643454764633\n" +\
#         "Computing regularization path using the elastic net.\n"
#
#     test_run = Popen(command,shell=True,stdout=PIPE)
#
#     # check exit code for return code 0
#     test_run.wait()
#     assert test_run.returncode == 0
#
#     #  check for expected output
#     out_text = test_run.stdout.read()
#     print(out_text)
#     assert out_text[:len(stdout_response)] ==


if __name__ == "__main__":
    test_sklearn_elasticnet_wine()
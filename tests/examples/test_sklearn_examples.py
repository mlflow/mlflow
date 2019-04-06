from __future__ import print_function

from subprocess import Popen, PIPE
import os.path
import pytest
from collections import namedtuple


ExampleToRun = namedtuple("ExampleToRun", ["directory", "command"])


def run_an_example(example_dir, example_command):
    # construct command to run
    command = "cd " + example_dir + " && rm -fr mlruns && " + example_command

    # example code
    test_run = Popen(command, shell=True)

    # check exit code for return code 0
    test_run.wait()

    # check that example ran successfully
    assert test_run.returncode == 0

    # confirm creation of ./mlruns directory
    assert os.path.isdir(os.path.join(example_dir,'mlruns'))


@pytest.fixture()
def setup_for_tests():
    test_list = [ExampleToRun("examples/sklearn_logistic_regression","python train.py"),
                 ExampleToRun("examples/sklearn_elasticnet_wine","mlflow run . -P alpha=0.5"),
                 ExampleToRun("examples/sklearn_elasticnet_diabetes/linux", "mlflow run .")]

    return test_list




def test_examples(setup_for_tests):

    example_list = setup_for_tests

    for example in example_list:
        print("\n\ntesting: ", example.directory)
        run_an_example(example.directory, example.command)








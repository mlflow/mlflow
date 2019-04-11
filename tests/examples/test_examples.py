
import os
import os.path
from tempfile import mkdtemp
import shutil

from mlflow import cli
from mlflow.utils import process
from tests.integration.utils import invoke_cli_runner

from tests.projects.utils import tracking_uri_mock

EXAMPLES_DIR = 'examples'


def test_sklearn_elasticnet_wine(tracking_uri_mock):
    invoke_cli_runner(cli.run, [os.path.join(EXAMPLES_DIR, 'sklearn_elasticnet_wine'),
                                "-P", "alpha=0.5"])


def test_sklearn_elasticnet_diabetes(tracking_uri_mock):
    invoke_cli_runner(cli.run, [os.path.join(EXAMPLES_DIR, 'sklearn_elasticnet_diabetes', 'linux')])


def test_sklearn_logistic_regression():
    tempdir = mkdtemp()
    os.environ['MLFLOW_TRACKING_URI'] = os.path.join(tempdir, 'mlruns')

    try:
        process.exec_cmd(['python', 'train.py'],
                         cwd=os.path.join(EXAMPLES_DIR, 'sklearn_logistic_regression'))
    finally:
        shutil.rmtree(tempdir)
        del os.environ['MLFLOW_TRACKING_URI']

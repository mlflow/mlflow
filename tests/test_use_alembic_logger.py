import uuid

import pytest

import mlflow
from mlflow.pyfunc import PythonModel

def test_verify_alembic_runs():
    from mlflow.pyfunc import PythonModel

    mlflow.set_tracking_uri('sqlite:///mlflowtest.db')
    mlflow.set_experiment("exp_name_example")

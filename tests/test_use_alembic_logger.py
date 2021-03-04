import uuid

import pytest

import mlflow
from mlflow.pyfunc import PythonModel

def test_repro_missing_alembic_config():
    from mlflow.pyfunc import PythonModel

    mlflow.set_tracking_uri('sqlite:///mlflowtest.db')
    mlflow.set_experiment("exp_name_example")

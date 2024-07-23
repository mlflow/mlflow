import pytest

import mlflow
from mlflow.exceptions import MlflowException


class UnpicklableModel(mlflow.pyfunc.PythonModel):
    def __init__(self, path):
        with open(path, "w+") as f:
            pass

        self.not_a_file = f


def test_pyfunc_unpicklable_exception(tmp_path):
    model = UnpicklableModel(tmp_path / "model.pkl")

    with pytest.raises(MlflowException, match="Please audit your class variables"):
        mlflow.pyfunc.save_model(python_model=model, path=tmp_path / "model")

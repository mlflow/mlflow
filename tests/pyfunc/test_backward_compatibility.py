import pytest

import mlflow


@pytest.mark.parametrize("version", ["2.7.1", "2.8.1"])
def test_backward_compatibility(version):
    model = mlflow.pyfunc.load_model(f"tests/resources/pyfunc_models/{version}")
    assert model.predict("MLflow is great!") == "MLflow is great!"

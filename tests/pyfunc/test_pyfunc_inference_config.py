import os
import mlflow
import pytest


@pytest.fixture(scope="module")
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


@pytest.fixture(scope="module")
def inference_config():
    return {
        "use_gpu": True,
        "temperature": 0.9,
        "timeout": 300,
    }


class TestModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input


class InferenceContextModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return context.inference_config


def test_save_with_inference_config(model_path, inference_config):
    model = InferenceContextModel()
    mlflow.pyfunc.save_model(model_path, python_model=model, inference_config=inference_config)

    loaded_model = mlflow.pyfunc.load_model(model_uri=model_path)

    assert loaded_model.inference_config
    assert set(inference_config.keys()) == set(loaded_model.inference_config)
    assert all(loaded_model.inference_config[k] == v for k, v in inference_config.items())
    assert all(loaded_model.inference_config[k] == v for k, v in loaded_model.predict([[0]]))


def test_override_inference_config(model_path, inference_config):
    model = TestModel()
    inference_override = {"timeout": 400}

    mlflow.pyfunc.save_model(model_path, python_model=model, inference_config=inference_config)
    loaded_model = mlflow.pyfunc.load_model(
        model_uri=model_path, inference_config=inference_override
    )

    assert all(loaded_model.inference_config[k] == v for k, v in inference_override.items())


def test_override_inference_config_ignore_invalid(model_path, inference_config):
    model = TestModel()
    inference_override = {"invalid_key": 400}

    mlflow.pyfunc.save_model(model_path, python_model=model, inference_config=inference_config)
    loaded_model = mlflow.pyfunc.load_model(
        model_uri=model_path, inference_config=inference_override
    )

    assert loaded_model.predict([[5]])
    assert all(k not in loaded_model.inference_config for k in inference_override.keys())

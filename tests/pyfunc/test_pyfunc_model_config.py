import os

import pytest

import mlflow
from mlflow.models import Model


@pytest.fixture
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


@pytest.fixture
def model_config():
    return {
        "use_gpu": True,
        "temperature": 0.9,
        "timeout": 300,
    }


def _load_pyfunc(path):
    return TestModel()


class TestModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input


class InferenceContextModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        # This mock class returns the internal inference configuration keys and values available
        return context.model_config.items()


def test_save_with_model_config(model_path, model_config):
    model = InferenceContextModel()
    mlflow.pyfunc.save_model(model_path, python_model=model, model_config=model_config)

    loaded_model = mlflow.pyfunc.load_model(model_uri=model_path)

    assert loaded_model.model_config
    assert set(model_config.keys()) == set(loaded_model.model_config)
    assert all(loaded_model.model_config[k] == v for k, v in model_config.items())
    assert all(loaded_model.model_config[k] == v for k, v in loaded_model.predict([[0]]))


def test_override_model_config(model_path, model_config):
    model = TestModel()
    inference_override = {"timeout": 400}

    mlflow.pyfunc.save_model(model_path, python_model=model, model_config=model_config)
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_path, model_config=inference_override)

    assert all(loaded_model.model_config[k] == v for k, v in inference_override.items())


def test_override_model_config_ignore_invalid(model_path, model_config):
    model = TestModel()
    inference_override = {"invalid_key": 400}

    mlflow.pyfunc.save_model(model_path, python_model=model, model_config=model_config)
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_path, model_config=inference_override)

    assert loaded_model.predict([[5]])
    assert all(k not in loaded_model.model_config for k in inference_override.keys())


def test_pyfunc_without_model_config(model_path, model_config):
    model = TestModel()
    mlflow.pyfunc.save_model(model_path, python_model=model)

    loaded_model = mlflow.pyfunc.load_model(model_uri=model_path, model_config=model_config)

    assert loaded_model.predict([[5]])
    assert not loaded_model.model_config


def test_pyfunc_loader_without_model_config(model_path):
    mlflow.pyfunc.save_model(
        path=model_path,
        data_path=".",
        loader_module=__name__,
        code_path=[__file__],
        mlflow_model=Model(run_id="test", artifact_path="testtest"),
    )

    inference_override = {"invalid_key": 400}
    pyfunc_model = mlflow.pyfunc.load_model(model_path, model_config=inference_override)

    assert not pyfunc_model.model_config


def test_pyfunc_warn_on_model_assignment_no_context(model_path):
    # should warn on self.model assignment in __init__
    # if load_context is not overridden
    class MyModelBad(mlflow.pyfunc.PythonModel):
        def __init__(self, llm):
            self.model = llm

        def predict(self, context, model_input, params=None):
            return model_input

    msg = "It looks like you're trying to save a model"
    with pytest.warns(UserWarning, match=msg):
        MyModelBad(1)


def test_pyfunc_no_warn_on_model_assignment_with_context(model_path):
    # don't warn on model assignment if users override load_context
    class MyModelGood(mlflow.pyfunc.PythonModel):
        def __init__(self):
            self.model = None

        def load_context(self, context):
            self.model = "gpt4"

        def predict(self, context, model_input, params=None):
            return model_input

    with pytest.warns(None) as record:
        MyModelGood()

    # assert no warnings
    assert not record

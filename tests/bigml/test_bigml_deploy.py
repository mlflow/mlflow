import os
import pytest
import json
import pkg_resources
import pandas as pd

from mlflow import pyfunc
from mlflow.bigml import save_model, load_model

from bigml.supervised import SupervisedModel
from bigml.fields import Fields


MODELS_PATH = "../tests/bigml/models"


def res_filename(file):
    return pkg_resources.resource_filename("mlflow", os.path.join(MODELS_PATH, file))


def local_model_check(model, examples, model_path):
    """Generic function to test local model registry, recovery and scoring"""
    local_model = SupervisedModel(model)
    predictions = [local_model.predict(example, full=True) for example in examples]
    save_model(model, path=model_path)
    loaded_model = load_model(model_path)
    loaded_model_predictions = [loaded_model.predict(example, full=True) for example in examples]
    for index, prediction in enumerate(predictions):
        assert prediction == loaded_model_predictions[index]

    # Loading pyfunc model
    pyfunc_loaded = pyfunc.load_model(model_path)
    pyfunc_predictions = pyfunc_loaded.predict(pd.DataFrame.from_records(examples))
    for index, prediction in enumerate(predictions):
        assert pyfunc_predictions[index] == prediction


@pytest.fixture
def diabetes_examples():
    filename = res_filename("logistic_regression.json")
    with open(filename) as handler:
        model_info = json.load(handler)
    fields = Fields(model_info)
    examples = []
    for _ in range(0, 3):
        examples.append(fields.training_data_example())
    return examples


@pytest.fixture
def wines_examples():
    filename = res_filename("linear_regression.json")
    with open(filename) as handler:
        model_info = json.load(handler)
    fields = Fields(model_info)
    examples = []
    for _ in range(0, 3):
        examples.append(fields.training_data_example())
    return examples


@pytest.fixture
def diabetes_logistic():
    filename = res_filename("logistic_regression.json")
    with open(filename) as handler:
        return json.load(handler)


@pytest.fixture
def diabetes_ensemble():
    model_list = []
    filename = res_filename("ensemble.json")
    with open(filename) as handler:
        ensemble = json.load(handler)
        model_list.append(ensemble)
    try:
        for model in ensemble["object"]["models"]:
            filename = model.replace("/", "_")
            with open(res_filename(filename)) as handler:
                model_list.append(json.load(handler))
        return model_list
    except KeyError:
        raise ValueError("This is not a correct ensemble model")


@pytest.fixture
def wines_linear():
    filename = res_filename("linear_regression.json")
    with open(filename) as handler:
        return json.load(handler)


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.mark.large
def test_logistic_save_load(diabetes_logistic, diabetes_examples, model_path):
    local_model_check(diabetes_logistic, diabetes_examples, model_path)


@pytest.mark.large
def test_linear_save_load(wines_linear, wines_examples, model_path):
    local_model_check(wines_linear, wines_examples, model_path)


@pytest.mark.large
def test_ensemble_save_load(diabetes_ensemble, diabetes_examples, model_path):
    local_model_check(diabetes_ensemble, diabetes_examples, model_path)

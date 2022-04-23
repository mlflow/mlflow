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
    filename = res_filename("ensemble.json")
    with open(filename) as handler:
        return json.load(handler)


@pytest.fixture
def wines_linear():
    filename = res_filename("linear_regression.json")
    with open(filename) as handler:
        return json.load(handler)


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.mark.large
def test_model_save_load(diabetes_logistic, diabetes_examples, model_path):
    local_model = SupervisedModel(diabetes_logistic)
    predictions = [local_model.predict(example, full=True) for example in diabetes_examples]
    save_model(diabetes_logistic, path=model_path)
    loaded_model = load_model(model_path)
    loaded_model_predictions = [
        loaded_model.predict(example, full=True) for example in diabetes_examples
    ]
    for index, prediction in enumerate(predictions):
        assert prediction == loaded_model_predictions[index]

    # Loading pyfunc model
    pyfunc_loaded = pyfunc.load_model(model_path)
    pyfunc_predictions = pyfunc_loaded.predict(pd.DataFrame.from_records(diabetes_examples))
    for index, prediction in enumerate(predictions):
        assert pyfunc_predictions[index] == prediction

# pep8: disable=E501


from distutils.version import LooseVersion
import os
import pytest

import keras
from keras.models import Sequential, Model as KerasModel
from keras.layers import Layer, Dense, Input, Concatenate
from keras import backend as K
from keras.optimizers import SGD
import sklearn.datasets as datasets
import pandas as pd
import numpy as np

import mlflow
from mlflow.types import Schema, ColSpec
from mlflow.pyfunc import PyFuncModel
from mlflow.models import Model, infer_signature, ModelSignature

import mlflow.keras
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, infer_signature
from mlflow.models.utils import _read_example
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration
from tests.helper_functions import pyfunc_serve_and_score_model
from tests.helper_functions import score_model_in_sagemaker_docker_container
from tests.helper_functions import set_boto_credentials  # pylint: disable=unused-import
from tests.helper_functions import mock_s3_bucket  # pylint: disable=unused-import
from tests.pyfunc.test_spark import score_model_as_udf
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(tmpdir.strpath, "model")


@pytest.fixture(scope="module")
def data():
    iris = datasets.load_iris()
    data = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
    )
    y = data["target"]
    x = data.drop("target", axis=1)
    return x, y


@pytest.fixture(scope="module")
def single_tensor_input_model(data):
    x, y = data
    model = Sequential()
    model.add(Dense(3, input_dim=4))
    model.add(Dense(1))
    # Use a small learning rate to prevent exploding gradients which may produce
    # infinite prediction values
    lr = 0.001
    kwargs = (
        # `lr` was renamed to `learning_rate` in keras 2.3.0:
        # https://github.com/keras-team/keras/releases/tag/2.3.0
        {"lr": lr}
        if LooseVersion(keras.__version__) < LooseVersion("2.3.0")
        else {"learning_rate": lr}
    )
    model.compile(loss="mean_squared_error", optimizer=SGD(**kwargs))
    model.fit(x.values, y.values)
    return model


@pytest.fixture(scope="module")
def multi_tensor_input_model_list(data):
    x, y = data
    input_a = Input(4,)
    input_b = Input(4,)
    output = Dense(1)(Concatenate()([input_a, input_b]))
    model = KerasModel(inputs=[input_a, input_b], outputs=output)
    lr = 0.001
    kwargs = (
        {"lr": lr}
        if LooseVersion(keras.__version__) < LooseVersion("2.3.0")
        else {"learning_rate": lr}
    )
    model.compile(loss="mean_squared_error", optimizer=SGD(**kwargs))
    model.fit([x.values, x.values], y)
    return model


@pytest.fixture(scope="module")
def multi_tensor_input_model_dict(data):
    x, y = data
    input_a = Input(4,)
    input_b = Input(4,)
    output = Dense(1)(Concatenate()([input_a, input_b]))
    model = KerasModel(inputs={"a": input_a, "b": input_b,}, outputs=output)
    lr = 0.001
    kwargs = (
        {"lr": lr}
        if LooseVersion(keras.__version__) < LooseVersion("2.3.0")
        else {"learning_rate": lr}
    )
    model.compile(loss="mean_squared_error", optimizer=SGD(**kwargs))
    model.fit({"a": x.values, "b": x.values,}, y)
    return model


def test_model_single_tensor_input(single_tensor_input_model, model_path, data):
    x, _ = data
    model_path = os.path.join(model_path, "plain")
    expected = single_tensor_input_model.predict(x.values)
    mlflow.keras.save_model(single_tensor_input_model, model_path)

    # Loading Keras model via PyFunc
    model_loaded = mlflow.pyfunc.load_model(model_path)

    # Calling predict with a dataframe should return a dataframe
    actual = model_loaded.predict(x)
    assert type(actual) == pd.DataFrame
    np.testing.assert_allclose(actual.values, expected, rtol=1e-5)

    # Calling predict with a np array should return a np array
    actual = model_loaded.predict(x.values)
    assert type(actual) == np.ndarray
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

    # Calling predict with a dict should make keras raise an error
    with pytest.raises(Exception):
        model_loaded.predict(
            {"a": [1, 2, 3], "b": [2, 3, 4],}
        )


def test_model_multi_tensor_input_list(multi_tensor_input_model_list, model_path, data):
    x, _ = data
    test_input = [x.values, x.values]

    model_path = os.path.join(model_path, "plain")
    expected = multi_tensor_input_model_list.predict(test_input)
    mlflow.keras.save_model(multi_tensor_input_model_list, model_path)

    # Loading Keras model via PyFunc
    model_loaded = mlflow.pyfunc.load_model(model_path)

    # Calling predict with a list should return a np.ndarray output
    actual = model_loaded.predict(test_input)
    assert type(actual) == np.ndarray
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

    # Calling predict with a non-list input should raise an error
    with pytest.raises(Exception):
        model_loaded.predict({"a": [1], "b": [2]})


def test_model_multi_tensor_input_dict(multi_tensor_input_model_dict, model_path, data):
    x, _ = data
    test_input = {
        "a": x.values,
        "b": x.values,
    }

    # test that input schema works with dictionaries
    # m = Model()
    # input_schema = Schema(
    #     [
    #         ColSpec("float", "a"),
    #         ColSpec("float", "b"),
    #     ]
    # )
    # m.signature = ModelSignature(inputs = input_schema)
    # pyfunc_model = PyFuncModel(model_meta=m, model_impl=multi_tensor_input_model_dict)
    # expected = pyfunc_model.predict(test_input)

    model_path = os.path.join(model_path, "plain")
    expected = multi_tensor_input_model_dict.predict(test_input)
    mlflow.keras.save_model(multi_tensor_input_model_dict, model_path)

    # Loading Keras model via PyFunc
    model_loaded = mlflow.pyfunc.load_model(model_path)

    # Calling predict with a dict should return a np.ndarray output
    actual = model_loaded.predict(test_input)
    assert type(actual) == np.ndarray
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

    # Calling predict with a non-dict input should raise an error
    with pytest.raises(Exception):
        model_loaded.predict([1, 2, 3, 4])

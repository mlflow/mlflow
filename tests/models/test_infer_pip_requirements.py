import logging
import posixpath
import json
from unittest import mock
from contextlib import contextmanager

import importlib_metadata
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
import pytest

import mlflow
from mlflow.pyfunc.scoring_server import CONTENT_TYPE_JSON_SPLIT_ORIENTED, CONTENT_TYPE_JSON
from mlflow.utils.environment import _REQUIREMENTS_FILE_NAME
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from tests.helper_functions import pyfunc_serve_and_score_model, _read_lines


_logger = logging.getLogger(__name__)


def _get_pinned_requirement(package_name):
    return f"{package_name}=={importlib_metadata.version(package_name)}"


def _read_requirements(model_uri):
    reqs_uri = posixpath.join(model_uri, _REQUIREMENTS_FILE_NAME)
    return _read_lines(_download_artifact_from_uri(reqs_uri))


def _is_importable(module_name):
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False
    except Exception:
        _logger.exception("Encountered an unexpected error while importing `%s`", module_name)
        return False


def required_modules(*module_names):
    """
    Skips a test unless the specified modules are all available.
    """

    def decorator(test_func):
        should_skip = not all(map(_is_importable, module_names))
        reason = f"This test requires {module_names}"
        return pytest.mark.skipif(should_skip, reason=reason)(test_func)

    return decorator


# REMOVE THIS FIXTURE BEFORE MERGING THE PR
@pytest.fixture(autouse=True)
def show_inferred_pip_requirements(request):
    import functools

    original = mlflow.infer_pip_requirements

    @functools.wraps(original)
    def patch(*args, **kwargs):
        res = original(*args, **kwargs)

        capture_manager = request.config.pluginmanager.getplugin("capturemanager")
        capture_manager.suspendcapture()
        title = "=" * 20 + " Inferred pip requirements " + "=" * 20
        print("\n" + title)
        print("\n".join(res))
        print("=" * len(title))
        capture_manager.resumecapture()

        return res

    with mock.patch("mlflow.infer_pip_requirements", new=patch):
        yield


@contextmanager
def disable_fallback():
    def side_effect(*args, **kwargs):
        err_msg = (
            "Encountered an unexpected error while inferring pip requirements "
            "(model URI: %s, flavor: %s)"
        )
        if args[0] == err_msg:
            raise Exception("FALLBACK IS DISABLED")

    with mock.patch("mlflow.utils.environment._logger.exception", side_effect=side_effect):
        yield


@required_modules("xgboost")
def test_infer_pip_requirements_xgboost():
    import xgboost as xgb

    X, y = load_iris(return_X_y=True, as_frame=True)
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
    }
    model = xgb.train(params, xgb.DMatrix(X, y))

    with mlflow.start_run(), disable_fallback():
        mlflow.xgboost.log_model(model, artifact_path="model")
        model_uri = mlflow.get_artifact_uri("model")

    reqs = _read_requirements(model_uri)
    assert _get_pinned_requirement("xgboost") in reqs

    data = X.head(3)
    resp = pyfunc_serve_and_score_model(model_uri, data, CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    np.testing.assert_array_equal(json.loads(resp.content), model.predict(xgb.DMatrix(data)))


def _to_pipeline(model):
    return Pipeline([("model", model)])


@pytest.fixture(params=range(2))
def xgb_sklearn_model(request):
    import xgboost as xgb

    model = xgb.XGBClassifier(objective="multi:softmax", n_estimators=10)
    return [model, _to_pipeline(model)][request.param]


@required_modules("xgboost")
def test_infer_pip_requirements_xgboost_sklearn(xgb_sklearn_model):
    model = xgb_sklearn_model
    X, y = load_iris(return_X_y=True, as_frame=True)
    model.fit(X, y)

    with mlflow.start_run(), disable_fallback():
        mlflow.sklearn.log_model(model, artifact_path="model")
        model_uri = mlflow.get_artifact_uri("model")

    reqs = _read_requirements(model_uri)
    assert _get_pinned_requirement("xgboost") in reqs
    assert _get_pinned_requirement("scikit-learn") in reqs

    data = X.head(3)
    resp = pyfunc_serve_and_score_model(model_uri, data, CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    np.testing.assert_array_equal(json.loads(resp.content), model.predict(data))


@required_modules("lightgbm")
def test_infer_pip_requirements_lightgbm():
    import lightgbm as lgb

    X, y = load_iris(return_X_y=True, as_frame=True)
    params = {
        "objective": "multiclass",
        "num_class": 3,
    }
    model = lgb.train(params, lgb.Dataset(X, y), num_boost_round=10)

    with mlflow.start_run(), disable_fallback():
        mlflow.lightgbm.log_model(model, artifact_path="model")
        model_uri = mlflow.get_artifact_uri("model")

    reqs = _read_requirements(model_uri)
    assert _get_pinned_requirement("lightgbm") in reqs

    data = X.head(3)
    resp = pyfunc_serve_and_score_model(model_uri, data, CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    np.testing.assert_array_equal(json.loads(resp.content), model.predict(data))


@pytest.fixture(params=range(2))
def lgb_sklearn_model(request):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(n_estimators=10)
    return [model, _to_pipeline(model)][request.param]


@required_modules("lightgbm")
def test_infer_pip_requirements_lightgbm_sklearn(lgb_sklearn_model):
    model = lgb_sklearn_model
    X, y = load_iris(return_X_y=True, as_frame=True)
    model.fit(X, y)

    with mlflow.start_run(), disable_fallback():
        mlflow.sklearn.log_model(model, artifact_path="model")
        model_uri = mlflow.get_artifact_uri("model")

    reqs = _read_requirements(model_uri)
    assert _get_pinned_requirement("lightgbm") in reqs
    # lightgbm requires scikit-learn so `reqs` should NOT contain scikit-learn:
    # https://github.com/microsoft/LightGBM/blob/v3.2.1/python-package/setup.py#L343
    assert _get_pinned_requirement("scikit-learn") not in reqs

    data = X.head(3)
    resp = pyfunc_serve_and_score_model(model_uri, data, CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    np.testing.assert_array_equal(json.loads(resp.content), model.predict(data))


def _create_catboost_classifier():
    import catboost

    return catboost.CatBoostClassifier(allow_writing_files=False, iterations=10)


@required_modules("catboost")
def test_infer_pip_requirements_catboost():
    model = _create_catboost_classifier()
    X, y = load_iris(return_X_y=True, as_frame=True)
    model.fit(X, y)

    with mlflow.start_run(), disable_fallback():
        mlflow.sklearn.log_model(model, artifact_path="model")
        model_uri = mlflow.get_artifact_uri("model")

    reqs = _read_requirements(model_uri)
    assert _get_pinned_requirement("catboost") in reqs

    data = X.head(3)
    resp = pyfunc_serve_and_score_model(model_uri, data, CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    np.testing.assert_array_equal(json.loads(resp.content), model.predict(data))


@required_modules("catboost")
def test_infer_pip_requirements_catboost_sklearn():
    model = _to_pipeline(_create_catboost_classifier())
    X, y = load_iris(return_X_y=True, as_frame=True)
    model.fit(X, y)

    with mlflow.start_run(), disable_fallback():
        mlflow.sklearn.log_model(model, artifact_path="model")
        model_uri = mlflow.get_artifact_uri("model")

    reqs = _read_requirements(model_uri)
    assert _get_pinned_requirement("catboost") in reqs
    assert _get_pinned_requirement("scikit-learn") in reqs

    data = X.head(3)
    resp = pyfunc_serve_and_score_model(model_uri, data, CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    np.testing.assert_array_equal(json.loads(resp.content), model.predict(data))


@required_modules("keras")
def test_infer_pip_requirements_keras():
    import keras
    import numpy as np

    model = keras.models.Sequential([keras.layers.Dense(1, input_dim=1)])
    model.compile()

    with mlflow.start_run(), disable_fallback():
        mlflow.keras.log_model(
            model,
            artifact_path="model",
            keras_module=keras,
            extra_pip_requirements=["tensorflow==2.4.0"],
        )
        model_uri = mlflow.get_artifact_uri("model")

    reqs = _read_requirements(model_uri)
    assert _get_pinned_requirement("keras") in reqs

    inputs = [0.0, 1.0, 2.0]
    data = json.dumps({"inputs": inputs})
    resp = pyfunc_serve_and_score_model(model_uri, data, CONTENT_TYPE_JSON)
    np.testing.assert_array_equal(json.loads(resp.content), model.predict(inputs))


def _get_tiny_bert_config():
    from transformers import BertConfig

    return BertConfig(
        vocab_size=16,
        hidden_size=2,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=2,
    )


@required_modules("torch")
def test_infer_pip_requirements_pytorch():
    import torch

    model = torch.nn.Linear(1, 1, dtype=torch.double)
    model.eval()

    with mlflow.start_run(), disable_fallback():
        mlflow.pytorch.log_model(model, artifact_path="model")
        model_uri = mlflow.get_artifact_uri("model")

    reqs = _read_requirements(model_uri)
    assert _get_pinned_requirement("torch") in reqs

    inputs = [[0.0], [1.0], [2.0]]
    data = json.dumps({"inputs": inputs})
    resp = pyfunc_serve_and_score_model(model_uri, data, CONTENT_TYPE_JSON)
    np.testing.assert_array_equal(
        json.loads(resp.content), model(torch.tensor(inputs).double()).detach().numpy(),
    )


@required_modules("torch", "torchvision", "transformers")
def test_infer_pip_requirements_pytorch_transformers():
    from transformers import BertModel

    class MyBertModel(BertModel):
        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs).last_hidden_state

    model = MyBertModel(_get_tiny_bert_config())
    model.eval()

    with mlflow.start_run(), disable_fallback():
        mlflow.pytorch.log_model(model, artifact_path="model")
        model_uri = mlflow.get_artifact_uri("model")

    reqs = _read_requirements(model_uri)
    assert _get_pinned_requirement("torch") in reqs
    assert _get_pinned_requirement("transformers") in reqs

    input_ids = model.dummy_inputs["input_ids"]
    data = json.dumps({"inputs": input_ids.tolist()})
    resp = pyfunc_serve_and_score_model(model_uri, data, CONTENT_TYPE_JSON)
    np.testing.assert_array_equal(json.loads(resp.content), model(input_ids).detach().numpy())


@required_modules("tensorflow", "transformers")
def test_infer_pip_requirements_tensorflow_transformers():
    import tensorflow as tf
    from transformers import TFBertModel

    bert = TFBertModel(_get_tiny_bert_config())
    dummy_inputs = bert.dummy_inputs["input_ids"].numpy()
    input_ids = tf.keras.layers.Input(shape=(dummy_inputs.shape[1],), dtype=tf.int32)
    model = tf.keras.Model(inputs=[input_ids], outputs=[bert(input_ids).last_hidden_state])
    model.compile()

    with mlflow.start_run(), disable_fallback():
        mlflow.keras.log_model(model, artifact_path="model", keras_module=tf.keras)
        model_uri = mlflow.get_artifact_uri("model")

    reqs = _read_requirements(model_uri)
    assert _get_pinned_requirement("tensorflow") in reqs

    data = json.dumps({"inputs": dummy_inputs.tolist()})
    resp = pyfunc_serve_and_score_model(model_uri, data, CONTENT_TYPE_JSON)
    np.testing.assert_array_equal(json.loads(resp.content), model.predict(dummy_inputs))

from pathlib import Path
from packaging.version import Version
import os
import shutil
import random
import json
import yaml
import pickle

import pytest
from unittest import mock

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD

# pylint: disable=no-name-in-module
from sklearn import datasets
import pandas as pd
import numpy as np

import mlflow
from mlflow import pyfunc
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow.deployments import PredictionsResponse
from mlflow.models import Model, infer_signature
from mlflow.models.utils import _read_example
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.utils.conda import get_or_create_conda_env
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration

from tests.helper_functions import pyfunc_serve_and_score_model
from tests.helper_functions import (
    _compare_conda_env_requirements,
    _assert_pip_requirements,
    _is_available_on_pypi,
    _is_importable,
    _compare_logged_code_paths,
    assert_array_almost_equal,
    _mlflow_major_version_string,
)
from tests.helper_functions import PROTOBUF_REQUIREMENT
from tests.pyfunc.test_spark import score_model_as_udf
from tests.tensorflow.test_load_saved_tensorflow_estimator import ModelDataInfo


EXTRA_PYFUNC_SERVING_TEST_ARGS = (
    [] if _is_available_on_pypi("tensorflow") else ["--env-manager", "local"]
)
extra_pip_requirements = (
    [PROTOBUF_REQUIREMENT] if Version(tf.__version__) < Version("2.6.0") else []
)


@pytest.fixture(scope="module", autouse=True)
def fix_random_seed():
    SEED = 0
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    if Version(tf.__version__) >= Version("2.0.0"):
        tf.random.set_seed(SEED)
    else:
        tf.set_random_seed(SEED)


@pytest.fixture(scope="module")
def data():
    return datasets.load_iris(as_frame=True, return_X_y=True)


def get_model(data):
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
        if Version(tf.__version__) < Version("2.3.0")
        else {"learning_rate": lr}
    )
    model.compile(loss="mean_squared_error", optimizer=SGD(**kwargs))
    model.fit(x.values, y.values)
    return model


@pytest.fixture(scope="module")
def model(data):
    return get_model(data)


def get_tf_keras_model(data):
    x, y = data
    model = Sequential()
    model.add(Dense(3, input_dim=4))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer=SGD(learning_rate=0.001))
    model.fit(x.values, y.values)
    return model


@pytest.fixture(scope="module")
def tf_keras_model(data):
    return get_tf_keras_model(data)


@pytest.fixture(scope="module")
def predicted(model, data):
    return model.predict(data[0].values)


@pytest.fixture(scope="module")
def custom_layer():
    class MyDense(Layer):
        def __init__(self, output_dim, **kwargs):
            self.output_dim = output_dim
            super().__init__(**kwargs)

        def build(self, input_shape):
            # pylint: disable=attribute-defined-outside-init
            self.kernel = self.add_weight(
                name="kernel",
                shape=(input_shape[1], self.output_dim),
                initializer="uniform",
                trainable=True,
            )
            super().build(input_shape)

        def call(self, inputs):  # pylint: disable=arguments-differ
            return K.dot(inputs, self.kernel)

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.output_dim)

        def get_config(self):
            return {"output_dim": self.output_dim}

    return MyDense


@pytest.fixture(scope="module")
def custom_model(data, custom_layer):
    x, y = data
    model = Sequential()
    model.add(Dense(6, input_dim=4))
    model.add(custom_layer(1))
    model.compile(loss="mean_squared_error", optimizer="SGD")
    model.fit(x.values, y.values, epochs=1)
    return model


@pytest.fixture(scope="module")
def custom_predicted(custom_model, data):
    return custom_model.predict(data[0].values)


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(tmpdir.strpath, "model")


@pytest.fixture
def keras_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["keras", "tensorflow", "pytest"])
    return conda_env


@pytest.mark.parametrize(
    ("build_model", "save_format"),
    [
        (get_model, None),
        (get_tf_keras_model, None),
        (get_tf_keras_model, "h5"),
        (get_tf_keras_model, "tf"),
    ],
)
def test_model_save_load(build_model, save_format, model_path, data):
    x, _ = data
    keras_model = build_model(data)
    if build_model == get_tf_keras_model:
        model_path = os.path.join(model_path, "tf")
    else:
        model_path = os.path.join(model_path, "plain")
    expected = keras_model.predict(x.values)
    kwargs = {"save_format": save_format} if save_format else {}
    mlflow.tensorflow.save_model(keras_model, path=model_path, keras_model_kwargs=kwargs)
    # Loading Keras model
    model_loaded = mlflow.tensorflow.load_model(model_path)
    # When saving as SavedModel, we actually convert the model
    # to a slightly different format, so we cannot assume it is
    # exactly the same.
    if save_format != "tf":
        assert type(keras_model) == type(model_loaded)
    np.testing.assert_allclose(model_loaded.predict(x.values), expected, rtol=1e-5)
    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    np.testing.assert_allclose(pyfunc_loaded.predict(x.values), expected, rtol=1e-5)


def test_pyfunc_serve_and_score(data):
    x, _ = data
    model = get_model(data)
    with mlflow.start_run():
        model_info = mlflow.tensorflow.log_model(model, artifact_path="model")
    expected = model.predict(x.values)
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=pd.DataFrame(x),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=EXTRA_PYFUNC_SERVING_TEST_ARGS,
    )
    actual_scoring_response = (
        PredictionsResponse.from_json(scoring_response.content.decode("utf-8"))
        .get_predictions()
        .values.astype(np.float32)
    )
    np.testing.assert_allclose(actual_scoring_response, expected, rtol=1e-5)


def test_score_model_as_spark_udf(data):
    x, _ = data
    model = get_model(data)
    with mlflow.start_run():
        model_info = mlflow.tensorflow.log_model(model, artifact_path="model")
    expected = model.predict(x.values)
    spark_udf_preds = score_model_as_udf(
        model_uri=model_info.model_uri, pandas_df=pd.DataFrame(x), result_type="float"
    )
    np.testing.assert_allclose(
        np.array(spark_udf_preds), expected.reshape(len(spark_udf_preds)), rtol=1e-5
    )


def test_signature_and_examples_are_saved_correctly(model, data):
    signature_ = infer_signature(data[0].to_numpy(), data[1])
    example_ = data[0].head(3).to_numpy()
    for signature in (None, signature_):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.tensorflow.save_model(
                    model, path=path, signature=signature, input_example=example
                )
                mlflow_model = Model.load(path)
                assert signature == mlflow_model.signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    np.testing.assert_allclose(_read_example(mlflow_model, path), example)


def test_custom_model_save_load(custom_model, custom_layer, data, custom_predicted, model_path):
    x, _ = data
    custom_objects = {"MyDense": custom_layer}
    mlflow.tensorflow.save_model(custom_model, path=model_path, custom_objects=custom_objects)
    # Loading Keras model
    model_loaded = mlflow.tensorflow.load_model(model_path)
    assert all(model_loaded.predict(x.values) == custom_predicted)
    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    assert all(pyfunc_loaded.predict(x.values) == custom_predicted)


@pytest.mark.allow_infer_pip_requirements_fallback
@pytest.mark.skipif(
    Version(tf.__version__) == Version("2.11.1"),
    reason="TensorFlow 2.11.1 has a bug with layers specifying output dimensions",
)
def test_custom_model_save_respects_user_custom_objects(custom_model, custom_layer, model_path):
    class DifferentCustomLayer:
        def __init__(self):
            pass

        def __call__(self):
            pass

    incorrect_custom_objects = {"MyDense": DifferentCustomLayer()}
    correct_custom_objects = {"MyDense": custom_layer}
    mlflow.tensorflow.save_model(
        custom_model, path=model_path, custom_objects=incorrect_custom_objects
    )
    model_loaded = mlflow.tensorflow.load_model(
        model_path, keras_model_kwargs={"custom_objects": correct_custom_objects}
    )
    assert model_loaded is not None
    if Version(tf.__version__) <= Version("2.11.0"):
        with pytest.raises(TypeError, match=r".+"):
            mlflow.tensorflow.load_model(model_path)
    else:
        # TF dev build following the release of 2.11.0 introduced changes to the recursive
        # loading strategy wherein the validation stage of custom objects loaded won't be
        # validated eagerly. This prevents a TypeError from being thrown as in the above
        # expectation catching validation block. The change in logic now permits loading and
        # will not raise an Exception, as validated below.
        incorrect_loaded = mlflow.tensorflow.load_model(model_path)
        assert incorrect_loaded is not None


def test_model_load_from_remote_uri_succeeds(model, model_path, mock_s3_bucket, data, predicted):
    x, _ = data
    mlflow.tensorflow.save_model(model, path=model_path)

    artifact_root = f"s3://{mock_s3_bucket}"
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    model_loaded = mlflow.tensorflow.load_model(model_uri=model_uri)
    assert all(model_loaded.predict(x.values) == predicted)


def test_model_log(model, data, predicted):
    x, _ = data
    # should_start_run tests whether or not calling log_model() automatically starts a run.
    for should_start_run in [False, True]:
        try:
            if should_start_run:
                mlflow.start_run()
            artifact_path = "keras_model"
            model_info = mlflow.tensorflow.log_model(model, artifact_path=artifact_path)
            model_uri = "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )
            assert model_info.model_uri == model_uri

            # Load model
            model_loaded = mlflow.tensorflow.load_model(model_uri=model_uri)
            assert all(model_loaded.predict(x.values) == predicted)

            # Loading pyfunc model
            pyfunc_loaded = mlflow.pyfunc.load_model(model_uri=model_uri)
            assert all(pyfunc_loaded.predict(x.values) == predicted)
        finally:
            mlflow.end_run()


def test_log_model_calls_register_model(model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.tensorflow.log_model(
            model, artifact_path=artifact_path, registered_model_name="AdsModel1"
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )
        mlflow.register_model.assert_called_once_with(
            model_uri, "AdsModel1", await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS
        )


def test_log_model_no_registered_model_name(model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.tensorflow.log_model(model, artifact_path=artifact_path)
        mlflow.register_model.assert_not_called()


def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    model, model_path, keras_custom_env
):
    mlflow.tensorflow.save_model(model, path=model_path, conda_env=keras_custom_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != keras_custom_env

    with open(keras_custom_env) as f:
        keras_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == keras_custom_env_parsed


def test_model_save_accepts_conda_env_as_dict(model, model_path):
    conda_env = dict(mlflow.tensorflow.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.tensorflow.save_model(model, path=model_path, conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


def test_model_save_persists_requirements_in_mlflow_model_directory(
    model, model_path, keras_custom_env
):
    mlflow.tensorflow.save_model(model, path=model_path, conda_env=keras_custom_env)

    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(keras_custom_env, saved_pip_req_path)


def test_log_model_with_pip_requirements(model, tmpdir):
    expected_mlflow_version = _mlflow_major_version_string()
    # Path to a requirements file
    req_file = tmpdir.join("requirements.txt")
    req_file.write("a")
    with mlflow.start_run():
        mlflow.tensorflow.log_model(model, artifact_path="model", pip_requirements=req_file.strpath)
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, "a"], strict=True
        )

    # List of requirements
    with mlflow.start_run():
        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            pip_requirements=[f"-r {req_file.strpath}", "b"],
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, "a", "b"], strict=True
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            pip_requirements=[f"-c {req_file.strpath}", "b"],
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            [expected_mlflow_version, "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )


def test_log_model_with_extra_pip_requirements(model, tmpdir):
    expected_mlflow_version = _mlflow_major_version_string()
    default_reqs = mlflow.tensorflow.get_default_pip_requirements()
    # Path to a requirements file
    req_file = tmpdir.join("requirements.txt")
    req_file.write("a")
    with mlflow.start_run():
        mlflow.tensorflow.log_model(
            model, artifact_path="model", extra_pip_requirements=req_file.strpath
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, *default_reqs, "a"]
        )

    # List of requirements
    with mlflow.start_run():
        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            extra_pip_requirements=[f"-r {req_file.strpath}", "b"],
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, *default_reqs, "a", "b"]
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            extra_pip_requirements=[f"-c {req_file.strpath}", "b"],
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            [expected_mlflow_version, *default_reqs, "b", "-c constraints.txt"],
            ["a"],
        )


def test_model_log_persists_requirements_in_mlflow_model_directory(model, keras_custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.tensorflow.log_model(model, artifact_path=artifact_path, conda_env=keras_custom_env)
        model_path = _download_artifact_from_uri(
            "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )
        )

    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(keras_custom_env, saved_pip_req_path)


def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(model, keras_custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.tensorflow.log_model(model, artifact_path=artifact_path, conda_env=keras_custom_env)
        model_path = _download_artifact_from_uri(
            "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )
        )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != keras_custom_env

    with open(keras_custom_env) as f:
        keras_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == keras_custom_env_parsed


def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    model, model_path
):
    mlflow.tensorflow.save_model(model, path=model_path)
    _assert_pip_requirements(model_path, mlflow.tensorflow.get_default_pip_requirements())


def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(model):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.tensorflow.log_model(model, artifact_path=artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)
    _assert_pip_requirements(model_uri, mlflow.tensorflow.get_default_pip_requirements())


def test_model_load_succeeds_with_missing_data_key_when_data_exists_at_default_path(
    tf_keras_model, model_path, data
):
    """
    This is a backwards compatibility test to ensure that models saved in MLflow version <= 0.8.0
    can be loaded successfully. These models are missing the `data` flavor configuration key.
    """
    mlflow.tensorflow.save_model(
        tf_keras_model, path=model_path, keras_model_kwargs={"save_format": "h5"}
    )
    shutil.move(os.path.join(model_path, "data", "model.h5"), os.path.join(model_path, "model.h5"))
    model_conf_path = os.path.join(model_path, "MLmodel")
    model_conf = Model.load(model_conf_path)
    flavor_conf = model_conf.flavors.get(mlflow.tensorflow.FLAVOR_NAME, None)
    assert flavor_conf is not None
    del flavor_conf["data"]
    model_conf.save(model_conf_path)

    model_loaded = mlflow.tensorflow.load_model(model_path)
    assert all(model_loaded.predict(data[0].values) == tf_keras_model.predict(data[0].values))


@pytest.mark.allow_infer_pip_requirements_fallback
def test_save_model_with_tf_save_format(model_path):
    """Ensures that Keras models can be saved with SavedModel format.

    Using SavedModel format (save_format="tf") requires that the file extension
    is _not_ "h5".
    """
    keras_model = mock.Mock(spec=tf.keras.Model)
    mlflow.tensorflow.save_model(
        keras_model, path=model_path, keras_model_kwargs={"save_format": "tf"}
    )
    _, args, kwargs = keras_model.save.mock_calls[0]
    # Ensure that save_format propagated through
    assert kwargs["save_format"] == "tf"
    # Ensure that the saved model does not have h5 extension
    assert not args[0].endswith(".h5")


def test_save_and_load_model_with_tf_save_format(tf_keras_model, model_path):
    """Ensures that keras models saved with save_format="tf" can be loaded."""
    mlflow.tensorflow.save_model(
        tf_keras_model, path=model_path, keras_model_kwargs={"save_format": "tf"}
    )
    model_conf_path = os.path.join(model_path, "MLmodel")
    model_conf = Model.load(model_conf_path)
    flavor_conf = model_conf.flavors.get(mlflow.tensorflow.FLAVOR_NAME, None)
    assert flavor_conf is not None
    assert flavor_conf.get("save_format") == "tf"
    assert not os.path.exists(
        os.path.join(model_path, "data", "model.h5")
    ), "TF model was saved with HDF5 format; expected SavedModel"
    assert os.path.isdir(
        os.path.join(model_path, "data", "model")
    ), "Expected directory containing saved_model.pb"

    model_loaded = mlflow.tensorflow.load_model(model_path)
    assert tf_keras_model.to_json() == model_loaded.to_json()


def test_load_without_save_format(tf_keras_model, model_path):
    """Ensures that keras models without save_format can still be loaded."""
    mlflow.tensorflow.save_model(
        tf_keras_model, path=model_path, keras_model_kwargs={"save_format": "h5"}
    )
    model_conf_path = os.path.join(model_path, "MLmodel")
    model_conf = Model.load(model_conf_path)
    flavor_conf = model_conf.flavors.get(mlflow.tensorflow.FLAVOR_NAME)
    assert flavor_conf is not None
    del flavor_conf["save_format"]
    model_conf.save(model_conf_path)

    model_loaded = mlflow.tensorflow.load_model(model_path)
    assert tf_keras_model.to_json() == model_loaded.to_json()


# TODO: Remove skipif condition `not Version(tf.__version__).is_devrelease` once
#  https://github.com/huggingface/transformers/issues/22421 is resolved.
@pytest.mark.skipif(
    not (
        _is_importable("transformers")
        and Version(tf.__version__) >= Version("2.6.0")
        and not Version(tf.__version__).is_devrelease
    ),
    reason="This test requires transformers, which is no longer compatible with Keras < 2.6.0, "
    "and transformers is not compatible with Tensorflow dev version, see "
    "https://github.com/huggingface/transformers/issues/22421",
)
def test_pyfunc_serve_and_score_transformers():
    from transformers import BertConfig, TFBertModel  # pylint: disable=import-error

    bert_model = TFBertModel(
        BertConfig(
            vocab_size=16,
            hidden_size=2,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=2,
        )
    )
    dummy_inputs = bert_model.dummy_inputs["input_ids"].numpy()
    input_ids = tf.keras.layers.Input(shape=(dummy_inputs.shape[1],), dtype=tf.int32)
    model = tf.keras.Model(
        inputs=[input_ids], outputs=[bert_model.bert(input_ids).last_hidden_state]
    )
    model.compile()

    with mlflow.start_run():
        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            extra_pip_requirements=extra_pip_requirements,
        )
        model_uri = mlflow.get_artifact_uri("model")

    data = json.dumps({"inputs": dummy_inputs.tolist()})
    resp = pyfunc_serve_and_score_model(
        model_uri,
        data,
        pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=EXTRA_PYFUNC_SERVING_TEST_ARGS,
    )

    scores = PredictionsResponse.from_json(resp.content.decode("utf-8")).get_predictions(
        predictions_format="ndarray"
    )
    assert_array_almost_equal(scores, model.predict(dummy_inputs))


def test_log_model_with_code_paths(model):
    artifact_path = "model"
    with mlflow.start_run(), mock.patch(
        "mlflow.tensorflow._add_code_from_conf_to_system_path"
    ) as add_mock:
        mlflow.tensorflow.log_model(model, artifact_path=artifact_path, code_paths=[__file__])
        model_uri = mlflow.get_artifact_uri(artifact_path)
        _compare_logged_code_paths(__file__, model_uri, mlflow.tensorflow.FLAVOR_NAME)
        mlflow.tensorflow.load_model(model_uri)
        add_mock.assert_called()


def test_virtualenv_subfield_points_to_correct_path(model, model_path):
    mlflow.tensorflow.save_model(model, path=model_path)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    python_env_path = Path(model_path, pyfunc_conf[pyfunc.ENV]["virtualenv"])
    assert python_env_path.exists()
    assert python_env_path.is_file()


def save_or_log_keras_model_by_mlflow128(tmpdir, task_type, save_as_type, save_path=None):
    tf_tests_dir = os.path.dirname(__file__)
    conda_env = get_or_create_conda_env(os.path.join(tf_tests_dir, "mlflow-128-tf-23-env.yaml"))
    output_data_file_path = os.path.join(tmpdir, "output_data.pkl")
    tracking_uri = mlflow.get_tracking_uri()
    exec_py_path = os.path.join(tf_tests_dir, "save_keras_model.py")

    conda_env.execute(
        f"python {exec_py_path} "
        f"--tracking_uri {tracking_uri} "
        f"--task_type {task_type} "
        f"--save_as_type {save_as_type} "
        f"--save_path {save_path if save_path else 'none'}",
    )

    with open(output_data_file_path, "rb") as f:
        inference_df, expected_results_df, run_id = pickle.load(f)
        return ModelDataInfo(
            inference_df=inference_df,
            expected_results_df=expected_results_df,
            raw_results=None,
            raw_df=None,
            run_id=run_id,
        )


def test_load_and_predict_keras_model_saved_by_mlflow128(tmp_path, monkeypatch):
    mlflow.set_tracking_uri(tmp_path.joinpath("mlruns").as_uri())
    monkeypatch.chdir(tmp_path)
    model_data_info = save_or_log_keras_model_by_mlflow128(
        tmp_path, task_type="log_model", save_as_type="keras"
    )

    model_uri = f"runs:/{model_data_info.run_id}/model"

    def load_and_predict(load_model_fn):
        mlflow_model = load_model_fn()
        predictions = mlflow_model.predict(model_data_info.inference_df)
        np.testing.assert_allclose(predictions, model_data_info.expected_results_df)

    load_and_predict(lambda: mlflow.pyfunc.load_model(model_uri))
    load_and_predict(lambda: mlflow.tensorflow.load_model(model_uri))


def test_load_tf_keras_model_with_options(tf_keras_model, model_path):
    mlflow.tensorflow.save_model(tf_keras_model, path=model_path)
    keras_model_kwargs = {
        "compile": False,
        "options": tf.saved_model.LoadOptions(),
    }
    with mock.patch("mlflow.tensorflow._load_keras_model") as mock_load:
        mlflow.tensorflow.load_model(model_path, keras_model_kwargs=keras_model_kwargs)
        mock_load.assert_called_once_with(
            model_path=mock.ANY, keras_module=mock.ANY, save_format=mock.ANY, **keras_model_kwargs
        )


def test_tf_saved_model_model_with_tf_keras_api(tmp_path, monkeypatch):
    mlflow.set_tracking_uri(tmp_path.joinpath("mlruns").as_uri())
    monkeypatch.chdir(tmp_path)
    model_data_info = save_or_log_keras_model_by_mlflow128(
        tmp_path, task_type="log_model", save_as_type="tf1-estimator"
    )

    model_uri = f"runs:/{model_data_info.run_id}/model"
    mlflow_model = mlflow.pyfunc.load_model(model_uri)
    predictions = mlflow_model.predict({"features": model_data_info.inference_df})
    np.testing.assert_allclose(predictions["dense"], model_data_info.expected_results_df)


def test_model_save_load_with_metadata(tf_keras_model, model_path):
    mlflow.tensorflow.save_model(
        tf_keras_model, path=model_path, metadata={"metadata_key": "metadata_value"}
    )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_path)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


def test_model_log_with_metadata(tf_keras_model):
    artifact_path = "model"

    with mlflow.start_run():
        mlflow.tensorflow.log_model(
            tf_keras_model, artifact_path=artifact_path, metadata={"metadata_key": "metadata_value"}
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"

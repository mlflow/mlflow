import sys
import os
import pytest
import mock

from keras.models import Sequential
from keras.layers import Dense
import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import yaml

import tensorflow as tf
import mlflow
import mlflow.keras
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.models import infer_signature, Model
from mlflow.models.utils import _read_example
from mlflow.utils.file_utils import TempDir
from tests.helper_functions import pyfunc_serve_and_score_model
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from pandas.testing import assert_frame_equal

pytestmark = pytest.mark.skipif(
    (sys.version_info < (3, 6)), reason="Tests require Python 3 to run!"
)


@pytest.fixture(scope="module")
def data():
    iris = datasets.load_iris()
    data = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )
    y = data["target"]
    x = data.drop("target", axis=1)
    return x, y


@pytest.fixture(scope="module")
def model(data):
    x, y = data
    model = Sequential()
    model.add(Dense(3, input_dim=4))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="SGD")
    model.fit(x, y)
    return model


@pytest.fixture(scope="module")
def onnx_model(model):
    import onnxmltools

    return onnxmltools.convert_keras(model)


@pytest.fixture(scope="module")
def sklearn_model(data):
    from sklearn.linear_model import LogisticRegression

    x, y = data
    model = LogisticRegression()
    model.fit(x, y)
    return model


@pytest.fixture(scope="module")
def onnx_sklearn_model(sklearn_model):
    import onnxmltools
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = onnxmltools.convert_sklearn(sklearn_model, initial_types=initial_type)
    return onx


@pytest.fixture(scope="module")
def predicted(model, data):
    return model.predict(data[0])


@pytest.fixture(scope="module")
def tf_model_multiple_inputs_float64():
    graph = tf.Graph()
    with graph.as_default():
        t_in1 = tf.placeholder(tf.float64, 10, name="first_input")
        t_in2 = tf.placeholder(tf.float64, 10, name="second_input")
        t_out = tf.multiply(t_in1, t_in2)
        t_out_named = tf.identity(t_out, name="output")
    return graph


@pytest.fixture(scope="module")
def tf_model_multiple_inputs_float32():
    graph = tf.Graph()
    with graph.as_default():
        t_in1 = tf.placeholder(tf.float32, 10, name="first_input")
        t_in2 = tf.placeholder(tf.float32, 10, name="second_input")
        t_out = tf.multiply(t_in1, t_in2)
        t_out_named = tf.identity(t_out, name="output")
    return graph


@pytest.fixture(scope="module")
def onnx_model_multiple_inputs_float64(tf_model_multiple_inputs_float64):
    import tf2onnx

    sess = tf.Session(graph=tf_model_multiple_inputs_float64)

    onnx_graph = tf2onnx.tfonnx.process_tf_graph(
        sess.graph,
        input_names=["first_input:0", "second_input:0",],
        output_names=["output:0"],
    )
    model_proto = onnx_graph.make_model("test")
    return model_proto


@pytest.fixture(scope="module")
def onnx_model_multiple_inputs_float32(tf_model_multiple_inputs_float32):
    import tf2onnx

    sess = tf.Session(graph=tf_model_multiple_inputs_float32)

    onnx_graph = tf2onnx.tfonnx.process_tf_graph(
        sess.graph,
        input_names=["first_input:0", "second_input:0",],
        output_names=["output:0"],
    )
    model_proto = onnx_graph.make_model("test")
    return model_proto


@pytest.fixture(scope="module")
def data_multiple_inputs():
    return pd.DataFrame(
        {"first_input:0": np.random.random(10), "second_input:0": np.random.random(10),}
    )


@pytest.fixture(scope="module")
def predicted_multiple_inputs(data_multiple_inputs):
    return pd.DataFrame(
        data_multiple_inputs["first_input:0"] * data_multiple_inputs["second_input:0"]
    )


@pytest.fixture(scope="module")
def high_dim_model():

    graph = tf.Graph()
    with graph.as_default():
        t_in1 = tf.placeholder(tf.int32, (1, 2, 2), name="first_input")
        t_in2 = tf.placeholder(tf.int32, (1, 2, 2), name="second_input")
        t_out = tf.add(t_in1, t_in2)
        t_out_named = tf.identity(t_out, name="output")

    import tf2onnx

    sess = tf.Session(graph=graph)

    onnx_graph = tf2onnx.tfonnx.process_tf_graph(
        sess.graph,
        input_names=["first_input:0", "second_input:0",],
        output_names=["output:0"],
    )
    model_proto = onnx_graph.make_model("test")
    return model_proto


@pytest.fixture(scope="module")
def data_high_dim_inputs():
    return pd.DataFrame(
        {
            "first_input:0": [np.arange(2 * 2, dtype=np.int32).reshape(2, 2)],
            "second_input:0": [np.arange(2 * 2, dtype=np.int32).reshape(2, 2)],
        },
    )


@pytest.fixture(scope="module")
def predicted_high_dim_inputs(data_high_dim_inputs):
    return pd.DataFrame(
        {
            "output:0": (
                data_high_dim_inputs["first_input:0"]
                + data_high_dim_inputs["second_input:0"]
            )
        },
    )


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(tmpdir.strpath, "model")


@pytest.fixture
def onnx_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(
        conda_env,
        additional_conda_deps=["pytest", "keras"],
        additional_pip_deps=["onnx", "onnxmltools"],
    )
    return conda_env


@pytest.mark.large
def test_cast_float64_to_float32():
    import mlflow.onnx

    df = pd.DataFrame([[1.0, 2.1], [True, False]], columns=["col1", "col2"])
    df["col1"] = df["col1"].astype(np.float64)
    df["col2"] = df["col2"].astype(np.bool)
    df2 = mlflow.onnx._OnnxModelWrapper._cast_float64_to_float32(df, df.columns)
    assert df2["col1"].dtype == np.float32 and df2["col2"].dtype == np.bool


# TODO: Use the default conda environment once MLflow's Travis build supports the onnxruntime
# library
@pytest.mark.large
def test_model_save_load(onnx_model, model_path, onnx_custom_env):
    import onnx
    import mlflow.onnx

    mlflow.onnx.save_model(onnx_model, model_path, conda_env=onnx_custom_env)

    # Loading ONNX model
    onnx.checker.check_model = mock.Mock()
    mlflow.onnx.load_model(model_path)
    assert onnx.checker.check_model.called


@pytest.mark.large
def test_signature_and_examples_are_saved_correctly(onnx_model, data, onnx_custom_env):
    import mlflow.onnx

    model = onnx_model
    signature_ = infer_signature(*data)
    example_ = data[0].head(3)
    for signature in (None, signature_):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.onnx.save_model(
                    model,
                    path=path,
                    conda_env=onnx_custom_env,
                    signature=signature,
                    input_example=example,
                )
                mlflow_model = Model.load(path)
                assert signature == mlflow_model.signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    assert all((_read_example(mlflow_model, path) == example).all())


# TODO: Mark this as large once MLflow's Travis build supports the onnxruntime library
@pytest.mark.release
def test_model_save_load_evaluate_pyfunc_format(
    onnx_model, model_path, data, predicted
):
    import onnx
    import mlflow.onnx

    x, y = data
    mlflow.onnx.save_model(onnx_model, model_path)

    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_pyfunc(model_path)

    actual = np.stack(pyfunc_loaded.predict(x)["dense_2"].values)
    expected = np.stack(predicted)
    np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=1e-05)

    # pyfunc serve
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=x,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    )
    assert np.allclose(
        np.stack(
            pd.read_json(scoring_response.content, orient="records")["dense_2"].values
        ),
        np.stack(predicted),
        rtol=1e-05,
        atol=1e-05,
    )


# TODO: Use the default conda environment once MLflow's Travis build supports the onnxruntime
# library
@pytest.mark.large
def test_model_save_load_multiple_inputs(
    onnx_model_multiple_inputs_float64, model_path, onnx_custom_env
):
    import onnx
    import mlflow.onnx

    mlflow.onnx.save_model(
        onnx_model_multiple_inputs_float64, model_path, conda_env=onnx_custom_env
    )

    # Loading ONNX model
    onnx.checker.check_model = mock.Mock()
    mlflow.onnx.load_model(model_path)
    assert onnx.checker.check_model.called


# TODO: Mark this as large once MLflow's Travis build supports the onnxruntime library
@pytest.mark.release
def test_model_save_load_evaluate_pyfunc_format_multiple_inputs(
    onnx_model_multiple_inputs_float64,
    data_multiple_inputs,
    predicted_multiple_inputs,
    model_path,
):
    import onnx
    import mlflow.onnx

    mlflow.onnx.save_model(onnx_model_multiple_inputs_float64, model_path)

    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_pyfunc(model_path)
    assert np.allclose(
        pyfunc_loaded.predict(data_multiple_inputs).values,
        predicted_multiple_inputs.values,
        rtol=1e-05,
        atol=1e-05,
    )

    # pyfunc serve
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=data_multiple_inputs,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    )
    assert np.allclose(
        pd.read_json(scoring_response.content, orient="records").values,
        predicted_multiple_inputs.values,
        rtol=1e-05,
        atol=1e-05,
    )


# TODO: Remove test, along with explicit casting, when https://github.com/mlflow/mlflow/issues/1286
# is fixed.
# TODO: Mark this as large once MLflow's Travis build supports the onnxruntime library
@pytest.mark.release
def test_pyfunc_representation_of_float32_model_casts_and_evalutes_float64_inputs(
    onnx_model_multiple_inputs_float32,
    model_path,
    data_multiple_inputs,
    predicted_multiple_inputs,
):
    """
    The ``python_function`` representation of an MLflow model with the ONNX flavor
    casts 64-bit floats to 32-bit floats automatically before evaluating, as opposed
    to throwing an unexpected type exception. This behavior is implemented due
    to the issue described in https://github.com/mlflow/mlflow/issues/1286 where
    the JSON representation of a Pandas DataFrame does not always preserve float
    precision (e.g., 32-bit floats may be converted to 64-bit floats when persisting a
    DataFrame as JSON).
    """
    import onnx
    import mlflow.onnx

    mlflow.onnx.save_model(onnx_model_multiple_inputs_float32, model_path)

    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_pyfunc(model_path)
    assert np.allclose(
        pyfunc_loaded.predict(data_multiple_inputs.astype("float64")).values,
        predicted_multiple_inputs.astype("float32").values,
        rtol=1e-05,
        atol=1e-05,
    )
    # with pytest.raises(RuntimeError):
    pyfunc_loaded.predict(data_multiple_inputs.astype("int32"))


@pytest.mark.release
def test_pyfunc_high_dim_models(
    high_dim_model, model_path, data_high_dim_inputs, predicted_high_dim_inputs,
):
    import onnx
    import mlflow.onnx

    mlflow.onnx.save_model(high_dim_model, model_path)
    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_pyfunc(model_path)

    pd.testing.assert_frame_equal(
        pyfunc_loaded.predict(data_high_dim_inputs),  # ["output:0"],
        predicted_high_dim_inputs,  # ["output:0"],
    )


# TODO: Use the default conda environment once MLflow's Travis build supports the onnxruntime
# library
@pytest.mark.large
def test_model_log(onnx_model, onnx_custom_env):
    # pylint: disable=unused-argument

    import onnx
    import mlflow.onnx

    # should_start_run tests whether or not calling log_model() automatically starts a run.
    for should_start_run in [False, True]:
        try:
            if should_start_run:
                mlflow.start_run()
            artifact_path = "onnx_model"
            mlflow.onnx.log_model(
                onnx_model=onnx_model,
                artifact_path=artifact_path,
                conda_env=onnx_custom_env,
            )
            model_uri = "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )

            # Load model
            onnx.checker.check_model = mock.Mock()
            mlflow.onnx.load_model(model_uri)
            assert onnx.checker.check_model.called
        finally:
            mlflow.end_run()


def test_log_model_calls_register_model(onnx_model, onnx_custom_env):
    import mlflow.onnx

    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.onnx.log_model(
            onnx_model=onnx_model,
            artifact_path=artifact_path,
            conda_env=onnx_custom_env,
            registered_model_name="AdsModel1",
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )
        mlflow.register_model.assert_called_once_with(model_uri, "AdsModel1")


def test_log_model_no_registered_model_name(onnx_model, onnx_custom_env):
    import mlflow.onnx

    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.onnx.log_model(
            onnx_model=onnx_model,
            artifact_path=artifact_path,
            conda_env=onnx_custom_env,
        )
        mlflow.register_model.assert_not_called()


# TODO: Mark this as large once MLflow's Travis build supports the onnxruntime library
@pytest.mark.release
def test_model_log_evaluate_pyfunc_format(onnx_model, data, predicted):
    import onnx
    import mlflow.onnx

    x, y = data
    # should_start_run tests whether or not calling log_model() automatically starts a run.
    for should_start_run in [False, True]:
        try:
            if should_start_run:
                mlflow.start_run()
            artifact_path = "onnx_model"
            mlflow.onnx.log_model(onnx_model=onnx_model, artifact_path=artifact_path)
            model_uri = "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )

            # Loading pyfunc model
            pyfunc_loaded = mlflow.pyfunc.load_pyfunc(model_uri=model_uri)

            actual = np.stack(pyfunc_loaded.predict(x)["dense_2"].values)
            expected = np.stack(predicted)
            np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=1e-05)
        finally:
            mlflow.end_run()


@pytest.mark.large
def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    onnx_model, model_path, onnx_custom_env
):
    import mlflow.onnx

    mlflow.onnx.save_model(
        onnx_model=onnx_model, path=model_path, conda_env=onnx_custom_env
    )
    pyfunc_conf = _get_flavor_configuration(
        model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME
    )
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != onnx_custom_env

    with open(onnx_custom_env, "r") as f:
        onnx_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == onnx_custom_env_parsed


# TODO: Mark this as large once MLflow's Travis build supports the onnxruntime library
@pytest.mark.release
def test_model_save_accepts_conda_env_as_dict(onnx_model, model_path):
    import mlflow.onnx

    conda_env = dict(mlflow.onnx.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.onnx.save_model(onnx_model=onnx_model, path=model_path, conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(
        model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME
    )
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


@pytest.mark.large
def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(
    onnx_model, onnx_custom_env
):
    import mlflow.onnx

    artifact_path = "model"
    with mlflow.start_run():
        mlflow.onnx.log_model(
            onnx_model=onnx_model,
            artifact_path=artifact_path,
            conda_env=onnx_custom_env,
        )
        model_path = _download_artifact_from_uri(
            "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )
        )

    pyfunc_conf = _get_flavor_configuration(
        model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME
    )
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != onnx_custom_env

    with open(onnx_custom_env, "r") as f:
        onnx_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == onnx_custom_env_parsed


# TODO: Mark this as large once MLflow's Travis build supports the onnxruntime library
@pytest.mark.release
def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    onnx_model, model_path
):
    import mlflow.onnx

    mlflow.onnx.save_model(onnx_model=onnx_model, path=model_path, conda_env=None)
    pyfunc_conf = _get_flavor_configuration(
        model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME
    )
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.onnx.get_default_conda_env()


# TODO: Mark this as large once MLflow's Travis build supports the onnxruntime library
@pytest.mark.release
def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    onnx_model,
):
    import mlflow.onnx

    artifact_path = "model"
    with mlflow.start_run():
        mlflow.onnx.log_model(
            onnx_model=onnx_model, artifact_path=artifact_path, conda_env=None
        )
        model_path = _download_artifact_from_uri(
            "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )
        )

    pyfunc_conf = _get_flavor_configuration(
        model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME
    )
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.onnx.get_default_conda_env()


# TODO: Mark this as large once MLflow's Travis build supports the onnxruntime library
@pytest.mark.release
def test_pyfunc_predict_supports_models_with_list_outputs(
    onnx_sklearn_model, model_path, data
):
    """
    https://github.com/mlflow/mlflow/issues/2499
    User encountered issue where an sklearn model, converted to onnx, would return a list response.
    The issue resulted in an error because MLflow assumed it would be a numpy array. Therefore,
    the this test validates the service does not receive that error when using such a model.
    """
    import onnx
    import mlflow.onnx
    import skl2onnx

    x, y = data
    mlflow.onnx.save_model(onnx_sklearn_model, model_path)
    wrapper = mlflow.pyfunc.load_model(model_path)
    wrapper.predict(pd.DataFrame(x))

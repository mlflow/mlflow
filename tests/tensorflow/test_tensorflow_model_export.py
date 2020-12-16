# pep8: disable=E501

import collections
import os
import shutil
import pytest
import yaml
import json
from unittest import mock

import numpy as np
import pandas as pd
import pandas.testing
import sklearn.datasets as datasets
import tensorflow as tf

import mlflow
import mlflow.tensorflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow.exceptions import MlflowException
from mlflow import pyfunc
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

from tests.helper_functions import score_model_in_sagemaker_docker_container
from tests.helper_functions import set_boto_credentials  # pylint: disable=unused-import
from tests.helper_functions import mock_s3_bucket  # pylint: disable=unused-import

SavedModelInfo = collections.namedtuple(
    "SavedModelInfo",
    ["path", "meta_graph_tags", "signature_def_key", "inference_df", "expected_results_df"],
)


@pytest.fixture
def saved_tf_iris_model(tmpdir):
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features
    y = iris.target
    trainingFeatures = {}
    for i in range(0, 2):
        # TensorFlow is fickle about feature names, so we remove offending characters
        iris.feature_names[i] = iris.feature_names[i].replace(" ", "")
        iris.feature_names[i] = iris.feature_names[i].replace("(", "")
        iris.feature_names[i] = iris.feature_names[i].replace(")", "")
        trainingFeatures[iris.feature_names[i]] = iris.data[:, i : i + 1]
    tf_feat_cols = []
    feature_names = iris.feature_names[:2]
    # Create TensorFlow-specific numeric columns for input.
    for col in iris.feature_names[:2]:
        tf_feat_cols.append(tf.feature_column.numeric_column(col))
    # Create a training function for the estimator
    input_train = tf.estimator.inputs.numpy_input_fn(
        trainingFeatures, y, shuffle=False, batch_size=1
    )
    estimator = tf.estimator.DNNRegressor(feature_columns=tf_feat_cols, hidden_units=[1])
    # Train the estimator and obtain expected predictions on the training dataset
    estimator.train(input_train, steps=10)
    estimator_preds = np.array([s["predictions"] for s in estimator.predict(input_train)]).ravel()
    estimator_preds_df = pd.DataFrame({"predictions": estimator_preds})

    # Define a function for estimator inference
    feature_spec = {}
    for name in feature_names:
        feature_spec[name] = tf.placeholder("float", name=name, shape=[150])
    receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

    # Save the estimator and its inference function
    saved_estimator_path = str(tmpdir.mkdir("saved_model"))
    saved_estimator_path = estimator.export_savedmodel(saved_estimator_path, receiver_fn).decode(
        "utf-8"
    )
    return SavedModelInfo(
        path=saved_estimator_path,
        meta_graph_tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_key="predict",
        inference_df=pd.DataFrame(data=X, columns=feature_names),
        expected_results_df=estimator_preds_df,
    )


@pytest.fixture
def saved_tf_categorical_model(tmpdir):
    path = os.path.abspath("tests/data/uci-autos-imports-85.data")
    # Order is important for the csv-readers, so we use an OrderedDict here
    defaults = collections.OrderedDict(
        [("body-style", [""]), ("curb-weight", [0.0]), ("highway-mpg", [0.0]), ("price", [0.0])]
    )
    types = collections.OrderedDict((key, type(value[0])) for key, value in defaults.items())
    df = pd.read_csv(path, names=list(types.keys()), dtype=types, na_values="?")
    df = df.dropna()

    # Extract the label from the features dataframe
    y_train = df.pop("price")

    # Create the required input training function
    trainingFeatures = {}
    for i in df:
        trainingFeatures[i] = df[i].values
    input_train = tf.estimator.inputs.numpy_input_fn(
        trainingFeatures, y_train.values, shuffle=False, batch_size=1
    )

    # Create the feature columns required for the DNNRegressor
    body_style_vocab = ["hardtop", "wagon", "sedan", "hatchback", "convertible"]
    body_style = tf.feature_column.categorical_column_with_vocabulary_list(
        key="body-style", vocabulary_list=body_style_vocab
    )
    feature_columns = [
        tf.feature_column.numeric_column(key="curb-weight"),
        tf.feature_column.numeric_column(key="highway-mpg"),
        # Since this is a DNN model, convert categorical columns from sparse to dense.
        # Then, wrap them in an `indicator_column` to create a one-hot vector from the input
        tf.feature_column.indicator_column(body_style),
    ]

    # Build a DNNRegressor, with 2x20-unit hidden layers, with the feature columns
    # defined above as input
    estimator = tf.estimator.DNNRegressor(hidden_units=[20, 20], feature_columns=feature_columns)

    # Train the estimator and obtain expected predictions on the training dataset
    estimator.train(input_fn=input_train, steps=10)
    estimator_preds = np.array([s["predictions"] for s in estimator.predict(input_train)]).ravel()
    estimator_preds_df = pd.DataFrame({"predictions": estimator_preds})

    # Define a function for estimator inference
    feature_spec = {
        "body-style": tf.placeholder("string", name="body-style", shape=[None]),
        "curb-weight": tf.placeholder("float", name="curb-weight", shape=[None]),
        "highway-mpg": tf.placeholder("float", name="highway-mpg", shape=[None]),
    }
    receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

    # Save the estimator and its inference function
    saved_estimator_path = str(tmpdir.mkdir("saved_model"))
    saved_estimator_path = estimator.export_savedmodel(saved_estimator_path, receiver_fn).decode(
        "utf-8"
    )
    return SavedModelInfo(
        path=saved_estimator_path,
        meta_graph_tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_key="predict",
        inference_df=df,
        expected_results_df=estimator_preds_df,
    )


@pytest.fixture
def tf_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_conda_deps=["tensorflow", "pytest"])
    return conda_env


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


def load_and_evaluate(model_path, tf_sess, tf_graph):
    expected_input_keys = ["sepallengthcm", "sepalwidthcm"]
    expected_output_keys = ["predictions"]
    input_length = 10
    signature_def = mlflow.tensorflow.load_model(model_uri=model_path, tf_sess=tf_sess)
    if not tf_sess:
        tf_sess = tf.get_default_session()
        tf_graph = tf.get_default_graph()
    input_signature = signature_def.inputs.items()
    assert len(input_signature) == len(expected_input_keys)
    feed_dict = {}
    for input_key, input_signature in signature_def.inputs.items():
        assert input_key in expected_input_keys
        t_input = tf_graph.get_tensor_by_name(input_signature.name)
        feed_dict[t_input] = np.array(range(input_length), dtype=np.float32)

    output_signature = signature_def.outputs.items()
    assert len(output_signature) == len(expected_output_keys)
    output_tensors = []
    for output_key, output_signature in signature_def.outputs.items():
        assert output_key in expected_output_keys
        t_output = tf_graph.get_tensor_by_name(output_signature.name)
        output_tensors.append(t_output)

    outputs_list = tf_sess.run(output_tensors, feed_dict=feed_dict)
    assert len(outputs_list) == 1
    outputs = outputs_list[0]
    assert len(outputs.ravel()) == input_length


@pytest.mark.large
def test_save_and_load_model_persists_and_restores_model_in_default_graph_context_successfully(
    saved_tf_iris_model, model_path
):
    mlflow.tensorflow.save_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
    )

    tf_graph = tf.Graph()
    tf_sess = tf.Session(graph=tf_graph)
    with tf_graph.as_default():
        signature_def = mlflow.tensorflow.load_model(model_uri=model_path, tf_sess=tf_sess)

        for _, input_signature in signature_def.inputs.items():
            t_input = tf_graph.get_tensor_by_name(input_signature.name)
            assert t_input is not None

        for _, output_signature in signature_def.outputs.items():
            t_output = tf_graph.get_tensor_by_name(output_signature.name)
            assert t_output is not None


@pytest.mark.large
def test_load_model_from_remote_uri_succeeds(saved_tf_iris_model, model_path, mock_s3_bucket):
    mlflow.tensorflow.save_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
    )

    artifact_root = "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    tf_graph = tf.Graph()
    tf_sess = tf.Session(graph=tf_graph)
    with tf_graph.as_default():
        signature_def = mlflow.tensorflow.load_model(model_uri=model_uri, tf_sess=tf_sess)

        for _, input_signature in signature_def.inputs.items():
            t_input = tf_graph.get_tensor_by_name(input_signature.name)
            assert t_input is not None

        for _, output_signature in signature_def.outputs.items():
            t_output = tf_graph.get_tensor_by_name(output_signature.name)
            assert t_output is not None


@pytest.mark.large
def test_save_and_load_model_persists_and_restores_model_in_custom_graph_context_successfully(
    saved_tf_iris_model, model_path
):
    mlflow.tensorflow.save_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
    )

    tf_graph = tf.Graph()
    tf_sess = tf.Session(graph=tf_graph)
    custom_tf_context = tf_graph.device("/cpu:0")
    with custom_tf_context:  # pylint: disable=not-context-manager
        signature_def = mlflow.tensorflow.load_model(model_uri=model_path, tf_sess=tf_sess)

        for _, input_signature in signature_def.inputs.items():
            t_input = tf_graph.get_tensor_by_name(input_signature.name)
            assert t_input is not None

        for _, output_signature in signature_def.outputs.items():
            t_output = tf_graph.get_tensor_by_name(output_signature.name)
            assert t_output is not None


@pytest.mark.large
def test_iris_model_can_be_loaded_and_evaluated_successfully(saved_tf_iris_model, model_path):
    mlflow.tensorflow.save_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
    )

    tf_graph_1 = tf.Graph()
    tf_sess_1 = tf.Session(graph=tf_graph_1)
    with tf_graph_1.as_default():
        load_and_evaluate(model_path=model_path, tf_sess=tf_sess_1, tf_graph=tf_graph_1)

    tf_graph_2 = tf.Graph()
    tf_sess_2 = tf.Session(graph=tf_graph_2)
    with tf_graph_1.device("/cpu:0"):  # pylint: disable=not-context-manager
        load_and_evaluate(model_path=model_path, tf_sess=tf_sess_2, tf_graph=tf_graph_2)


@pytest.mark.large
def test_load_model_session_exists_but_not_passed_in_loads_and_evaluates(
    saved_tf_iris_model, model_path
):
    mlflow.tensorflow.save_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
    )

    tf_graph = tf.Graph()
    tf_sess = tf.Session(graph=tf_graph)
    with tf_sess:
        load_and_evaluate(model_path=model_path, tf_sess=None, tf_graph=None)


@pytest.mark.large
def test_load_model_with_no_default_session_throws_exception(saved_tf_iris_model, model_path):
    mlflow.tensorflow.save_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
    )
    with pytest.raises(MlflowException):
        mlflow.tensorflow.load_model(model_uri=model_path)


@pytest.mark.large
def test_save_model_with_invalid_path_signature_def_or_metagraph_tags_throws_exception(
    saved_tf_iris_model, model_path
):
    with pytest.raises(IOError):
        mlflow.tensorflow.save_model(
            tf_saved_model_dir="not_a_valid_tf_model_dir",
            tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
            tf_signature_def_key=saved_tf_iris_model.signature_def_key,
            path=model_path,
        )

    with pytest.raises(RuntimeError):
        mlflow.tensorflow.save_model(
            tf_saved_model_dir=saved_tf_iris_model.path,
            tf_meta_graph_tags=["bad tags"],
            tf_signature_def_key=saved_tf_iris_model.signature_def_key,
            path=model_path,
        )

    with pytest.raises(MlflowException):
        mlflow.tensorflow.save_model(
            tf_saved_model_dir=saved_tf_iris_model.path,
            tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
            tf_signature_def_key="bad signature",
            path=model_path,
        )

    with pytest.raises(IOError):
        mlflow.tensorflow.save_model(
            tf_saved_model_dir="bad path",
            tf_meta_graph_tags="bad tags",
            tf_signature_def_key="bad signature",
            path=model_path,
        )


@pytest.mark.large
def test_load_model_loads_artifacts_from_specified_model_directory(saved_tf_iris_model, model_path):
    mlflow.tensorflow.save_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
    )

    # Verify that the MLflow model can be loaded even after deleting the TensorFlow `SavedModel`
    # directory that was used to create it, implying that the artifacts were copied to and are
    # loaded from the specified MLflow model path
    shutil.rmtree(saved_tf_iris_model.path)
    with tf.Session(graph=tf.Graph()) as tf_sess:
        mlflow.tensorflow.load_model(model_uri=model_path, tf_sess=tf_sess)


def test_log_model_with_non_keyword_args_fails(saved_tf_iris_model):
    artifact_path = "model"
    with mlflow.start_run():
        with pytest.raises(TypeError):
            mlflow.tensorflow.log_model(
                saved_tf_iris_model.path,
                saved_tf_iris_model.meta_graph_tags,
                saved_tf_iris_model.signature_def_key,
                artifact_path,
            )


@pytest.mark.large
def test_log_and_load_model_persists_and_restores_model_successfully(saved_tf_iris_model):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.tensorflow.log_model(
            tf_saved_model_dir=saved_tf_iris_model.path,
            tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
            tf_signature_def_key=saved_tf_iris_model.signature_def_key,
            artifact_path=artifact_path,
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )

    tf_graph = tf.Graph()
    tf_sess = tf.Session(graph=tf_graph)
    with tf_graph.as_default():
        signature_def = mlflow.tensorflow.load_model(model_uri=model_uri, tf_sess=tf_sess)

        for _, input_signature in signature_def.inputs.items():
            t_input = tf_graph.get_tensor_by_name(input_signature.name)
            assert t_input is not None

        for _, output_signature in signature_def.outputs.items():
            t_output = tf_graph.get_tensor_by_name(output_signature.name)
            assert t_output is not None


def test_log_model_calls_register_model(saved_tf_iris_model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.tensorflow.log_model(
            tf_saved_model_dir=saved_tf_iris_model.path,
            tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
            tf_signature_def_key=saved_tf_iris_model.signature_def_key,
            artifact_path=artifact_path,
            registered_model_name="AdsModel1",
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )
        mlflow.register_model.assert_called_once_with(
            model_uri, "AdsModel1", await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS
        )


def test_log_model_no_registered_model_name(saved_tf_iris_model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        mlflow.tensorflow.log_model(
            tf_saved_model_dir=saved_tf_iris_model.path,
            tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
            tf_signature_def_key=saved_tf_iris_model.signature_def_key,
            artifact_path=artifact_path,
        )
        mlflow.register_model.assert_not_called()


@pytest.mark.large
def test_save_model_persists_specified_conda_env_in_mlflow_model_directory(
    saved_tf_iris_model, model_path, tf_custom_env
):
    mlflow.tensorflow.save_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
        conda_env=tf_custom_env,
    )
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != tf_custom_env

    with open(tf_custom_env, "r") as f:
        tf_custom_env_text = f.read()
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_text = f.read()
    assert saved_conda_env_text == tf_custom_env_text


@pytest.mark.large
def test_save_model_accepts_conda_env_as_dict(saved_tf_iris_model, model_path):
    conda_env = dict(mlflow.tensorflow.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.tensorflow.save_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
        conda_env=conda_env,
    )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


@pytest.mark.large
def test_log_model_persists_specified_conda_env_in_mlflow_model_directory(
    saved_tf_iris_model, tf_custom_env
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.tensorflow.log_model(
            tf_saved_model_dir=saved_tf_iris_model.path,
            tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
            tf_signature_def_key=saved_tf_iris_model.signature_def_key,
            artifact_path=artifact_path,
            conda_env=tf_custom_env,
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != tf_custom_env

    with open(tf_custom_env, "r") as f:
        tf_custom_env_text = f.read()
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_text = f.read()
    assert saved_conda_env_text == tf_custom_env_text


@pytest.mark.large
def test_save_model_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    saved_tf_iris_model, model_path
):
    mlflow.tensorflow.save_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
        conda_env=None,
    )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.tensorflow.get_default_conda_env()


@pytest.mark.large
def test_log_model_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    saved_tf_iris_model, model_path
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.tensorflow.log_model(
            tf_saved_model_dir=saved_tf_iris_model.path,
            tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
            tf_signature_def_key=saved_tf_iris_model.signature_def_key,
            artifact_path=artifact_path,
            conda_env=None,
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.tensorflow.get_default_conda_env()


@pytest.mark.large
def test_iris_data_model_can_be_loaded_and_evaluated_as_pyfunc(saved_tf_iris_model, model_path):
    mlflow.tensorflow.save_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
    )

    pyfunc_wrapper = pyfunc.load_model(model_path)

    # can call predict with a df
    results_df = pyfunc_wrapper.predict(saved_tf_iris_model.inference_df)
    assert isinstance(results_df, pandas.DataFrame)
    pandas.testing.assert_frame_equal(
        results_df, saved_tf_iris_model.expected_results_df, check_less_precise=1
    )

    # can also call predict with a dict
    inp_data = {}
    for col_name in list(saved_tf_iris_model.inference_df):
        inp_data[col_name] = saved_tf_iris_model.inference_df[col_name].values
    results = pyfunc_wrapper.predict(inp_data)
    assert isinstance(results, dict)
    pandas.testing.assert_frame_equal(
        pandas.DataFrame(data=results),
        saved_tf_iris_model.expected_results_df,
        check_less_precise=1,
    )

    # can not call predict with a list
    inp_list = []
    for df_col_name in list(saved_tf_iris_model.inference_df):
        inp_list.append(saved_tf_iris_model.inference_df[df_col_name].values)
    with pytest.raises(TypeError):
        results = pyfunc_wrapper.predict(inp_list)


@pytest.mark.large
def test_categorical_model_can_be_loaded_and_evaluated_as_pyfunc(
    saved_tf_categorical_model, model_path
):
    mlflow.tensorflow.save_model(
        tf_saved_model_dir=saved_tf_categorical_model.path,
        tf_meta_graph_tags=saved_tf_categorical_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_categorical_model.signature_def_key,
        path=model_path,
    )

    pyfunc_wrapper = pyfunc.load_model(model_path)

    # can call predict with a df
    results_df = pyfunc_wrapper.predict(saved_tf_categorical_model.inference_df)
    assert isinstance(results_df, pandas.DataFrame)
    pandas.testing.assert_frame_equal(
        results_df, saved_tf_categorical_model.expected_results_df, check_less_precise=6
    )

    # can also call predict with a dict
    inp_dict = {}
    for df_col_name in list(saved_tf_categorical_model.inference_df):
        inp_dict[df_col_name] = saved_tf_categorical_model.inference_df[df_col_name].values
    results = pyfunc_wrapper.predict(inp_dict)
    assert isinstance(results, dict)
    pandas.testing.assert_frame_equal(
        pandas.DataFrame.from_dict(data=results),
        saved_tf_categorical_model.expected_results_df,
        check_less_precise=6,
    )

    # can not call predict with a list
    inp_list = []
    for df_col_name in list(saved_tf_categorical_model.inference_df):
        inp_list.append(saved_tf_categorical_model.inference_df[df_col_name].values)
    with pytest.raises(TypeError):
        results = pyfunc_wrapper.predict(inp_list)


@pytest.mark.release
def test_model_deployment_with_default_conda_env(saved_tf_iris_model, model_path):
    mlflow.tensorflow.save_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
        conda_env=None,
    )

    scoring_response = score_model_in_sagemaker_docker_container(
        model_uri=model_path,
        data=saved_tf_iris_model.inference_df,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
        flavor=mlflow.pyfunc.FLAVOR_NAME,
    )
    deployed_model_preds = pd.DataFrame(json.loads(scoring_response.content))

    pandas.testing.assert_frame_equal(
        deployed_model_preds,
        saved_tf_iris_model.expected_results_df,
        check_dtype=False,
        check_less_precise=6,
    )

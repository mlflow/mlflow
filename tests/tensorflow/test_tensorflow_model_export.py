# pep8: disable=E501

from __future__ import print_function

import collections
import os
import shutil
import pytest

import numpy as np
import pandas as pd
import pandas.testing
import sklearn.datasets as datasets
import tensorflow as tf

import mlflow
import mlflow.tensorflow
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.tracking.utils import _get_model_log_dir

SavedModelInfo = collections.namedtuple(
        "SavedModelInfo",
        ["path", "meta_graph_tags", "signature_def_key", "inference_df", "expected_results_df"])


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
        trainingFeatures[iris.feature_names[i]] = iris.data[:, i:i+1]
    tf_feat_cols = []
    feature_names = iris.feature_names[:2]
    # Create Tensorflow-specific numeric columns for input.
    for col in iris.feature_names[:2]:
        tf_feat_cols.append(tf.feature_column.numeric_column(col))
    # Create a training function for the estimator
    input_train = tf.estimator.inputs.numpy_input_fn(trainingFeatures,
                                                     y,
                                                     shuffle=False,
                                                     batch_size=1)
    estimator = tf.estimator.DNNRegressor(feature_columns=tf_feat_cols,
                                          hidden_units=[1])
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
    saved_estimator_path = estimator.export_savedmodel(saved_estimator_path,
                                                       receiver_fn).decode("utf-8")
    return SavedModelInfo(path=saved_estimator_path,
                          meta_graph_tags=[tf.saved_model.tag_constants.SERVING],
                          signature_def_key="predict",
                          inference_df=pd.DataFrame(data=X, columns=feature_names),
                          expected_results_df=estimator_preds_df)


@pytest.fixture
def saved_tf_categorical_model(tmpdir):
    path = os.path.abspath("tests/data/uci-autos-imports-85.data")
    # Order is important for the csv-readers, so we use an OrderedDict here
    defaults = collections.OrderedDict([
        ("body-style", [""]),
        ("curb-weight", [0.0]),
        ("highway-mpg", [0.0]),
        ("price", [0.0])
    ])
    types = collections.OrderedDict((key, type(value[0]))
                                    for key, value in defaults.items())
    df = pd.read_csv(path, names=types.keys(), dtype=types, na_values="?")
    df = df.dropna()

    # Extract the label from the features dataframe
    y_train = df.pop("price")

    # Create the required input training function
    trainingFeatures = {}
    for i in df:
        trainingFeatures[i] = df[i].values
    input_train = tf.estimator.inputs.numpy_input_fn(trainingFeatures,
                                                     y_train.values,
                                                     shuffle=False,
                                                     batch_size=1)

    # Create the feature columns required for the DNNRegressor
    body_style_vocab = ["hardtop", "wagon", "sedan", "hatchback", "convertible"]
    body_style = tf.feature_column.categorical_column_with_vocabulary_list(
        key="body-style", vocabulary_list=body_style_vocab)
    feature_columns = [
        tf.feature_column.numeric_column(key="curb-weight"),
        tf.feature_column.numeric_column(key="highway-mpg"),
        # Since this is a DNN model, convert categorical columns from sparse to dense.
        # Then, wrap them in an `indicator_column` to create a one-hot vector from the input
        tf.feature_column.indicator_column(body_style)
    ]

    # Build a DNNRegressor, with 2x20-unit hidden layers, with the feature columns
    # defined above as input
    estimator = tf.estimator.DNNRegressor(
        hidden_units=[20, 20], feature_columns=feature_columns)

    # Train the estimator and obtain expected predictions on the training dataset
    estimator.train(input_fn=input_train, steps=10)
    estimator_preds = np.array([s["predictions"] for s in estimator.predict(input_train)]).ravel()
    estimator_preds_df = pd.DataFrame({"predictions": estimator_preds})

    # Define a function for estimator inference
    feature_spec = {
        "body-style": tf.placeholder("string", name="body-style", shape=[None]),
        "curb-weight": tf.placeholder("float", name="curb-weight", shape=[None]),
        "highway-mpg": tf.placeholder("float", name="highway-mpg", shape=[None])
    }
    receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

    # Save the estimator and its inference function
    saved_estimator_path = str(tmpdir.mkdir("saved_model"))
    saved_estimator_path = estimator.export_savedmodel(saved_estimator_path,
                                                       receiver_fn).decode("utf-8")
    return SavedModelInfo(path=saved_estimator_path,
                          meta_graph_tags=[tf.saved_model.tag_constants.SERVING],
                          signature_def_key="predict",
                          inference_df=df,
                          expected_results_df=estimator_preds_df)


def test_save_and_load_model_persists_and_restores_model_in_default_graph_context_successfully(
        tmpdir, saved_tf_iris_model):
    model_path = os.path.join(str(tmpdir), "model")
    mlflow.tensorflow.save_model(tf_saved_model_dir=saved_tf_iris_model.path,
                                 tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
                                 tf_signature_def_key=saved_tf_iris_model.signature_def_key,
                                 path=model_path)

    tf_graph = tf.Graph()
    tf_sess = tf.Session(graph=tf_graph)
    with tf_graph.as_default():
        signature_def = mlflow.tensorflow.load_model(
                path=model_path, tf_sess=tf_sess)

        for _, input_signature in signature_def.inputs.items():
            t_input = tf_graph.get_tensor_by_name(input_signature.name)
            assert t_input is not None

        for _, output_signature in signature_def.outputs.items():
            t_output = tf_graph.get_tensor_by_name(output_signature.name)
            assert t_output is not None


def test_save_and_load_model_persists_and_restores_model_in_custom_graph_context_successfully(
        tmpdir, saved_tf_iris_model):
    model_path = os.path.join(str(tmpdir), "model")
    mlflow.tensorflow.save_model(tf_saved_model_dir=saved_tf_iris_model.path,
                                 tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
                                 tf_signature_def_key=saved_tf_iris_model.signature_def_key,
                                 path=model_path)

    tf_graph = tf.Graph()
    tf_sess = tf.Session(graph=tf_graph)
    custom_tf_context = tf_graph.device("/cpu:0")
    with custom_tf_context:
        signature_def = mlflow.tensorflow.load_model(path=model_path, tf_sess=tf_sess)

        for _, input_signature in signature_def.inputs.items():
            t_input = tf_graph.get_tensor_by_name(input_signature.name)
            assert t_input is not None

        for _, output_signature in signature_def.outputs.items():
            t_output = tf_graph.get_tensor_by_name(output_signature.name)
            assert t_output is not None


def test_load_model_loads_artifacts_from_specified_model_directory(tmpdir, saved_tf_iris_model):
    model_path = os.path.join(str(tmpdir), "model")
    mlflow.tensorflow.save_model(tf_saved_model_dir=saved_tf_iris_model.path,
                                 tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
                                 tf_signature_def_key=saved_tf_iris_model.signature_def_key,
                                 path=model_path)

    # Verify that the MLflow model can be loaded even after deleting the Tensorflow `SavedModel`
    # directory that was used to create it, implying that the artifacts were copied to and are
    # loaded from the specified MLflow model path
    shutil.rmtree(saved_tf_iris_model.path)
    with tf.Session(graph=tf.Graph()) as tf_sess:
        signature_def = mlflow.tensorflow.load_model(path=model_path, tf_sess=tf_sess)


def test_log_and_load_model_persists_and_restores_model_successfully(saved_tf_iris_model):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.tensorflow.log_model(tf_saved_model_dir=saved_tf_iris_model.path,
                                    tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
                                    tf_signature_def_key=saved_tf_iris_model.signature_def_key,
                                    artifact_path=artifact_path)

        run_id = mlflow.active_run().info.run_uuid

    tf_graph = tf.Graph()
    tf_sess = tf.Session(graph=tf_graph)
    with tf_graph.as_default():
        signature_def = mlflow.tensorflow.load_model(
                path=artifact_path, tf_sess=tf_sess, run_id=run_id)

        for _, input_signature in signature_def.inputs.items():
            t_input = tf_graph.get_tensor_by_name(input_signature.name)
            assert t_input is not None

        for _, output_signature in signature_def.outputs.items():
            t_output = tf_graph.get_tensor_by_name(output_signature.name)
            assert t_output is not None


def test_log_model_persists_conda_environment(tmpdir, saved_tf_iris_model):
    conda_env_path = os.path.join(str(tmpdir), "conda_env.yaml")
    _mlflow_conda_env(path=conda_env_path, additional_conda_deps=["tensorflow"])
    with open(conda_env_path, "r") as f:
        conda_env_text = f.read()

    artifact_path = "model"
    with mlflow.start_run():
        mlflow.tensorflow.log_model(tf_saved_model_dir=saved_tf_iris_model.path,
                                    tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
                                    tf_signature_def_key=saved_tf_iris_model.signature_def_key,
                                    artifact_path=artifact_path,
                                    conda_env=conda_env_path)

        run_id = mlflow.active_run().info.run_uuid

    model_dir = _get_model_log_dir(artifact_path, run_id)
    model_config = Model.load(os.path.join(model_dir, "MLmodel"))
    flavor_config = model_config.flavors.get(pyfunc.FLAVOR_NAME, None)
    assert flavor_config is not None
    pyfunc_env_subpath = flavor_config.get(pyfunc.ENV, None)
    assert pyfunc_env_subpath is not None
    with open(os.path.join(model_dir, pyfunc_env_subpath), "r") as f:
        persisted_env_text = f.read()

    assert persisted_env_text == conda_env_text


def test_iris_data_model_can_be_loaded_and_evaluated_as_pyfunc(tmpdir, saved_tf_iris_model):
    model_path = os.path.join(str(tmpdir), "model")
    mlflow.tensorflow.save_model(tf_saved_model_dir=saved_tf_iris_model.path,
                                 tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
                                 tf_signature_def_key=saved_tf_iris_model.signature_def_key,
                                 path=model_path)

    pyfunc_wrapper = pyfunc.load_pyfunc(model_path)
    results_df = pyfunc_wrapper.predict(saved_tf_iris_model.inference_df)
    assert results_df.equals(saved_tf_iris_model.expected_results_df)


def test_categorical_model_can_be_loaded_and_evaluated_as_pyfunc(
        tmpdir, saved_tf_categorical_model):
    model_path = os.path.join(str(tmpdir), "model")
    mlflow.tensorflow.save_model(tf_saved_model_dir=saved_tf_categorical_model.path,
                                 tf_meta_graph_tags=saved_tf_categorical_model.meta_graph_tags,
                                 tf_signature_def_key=saved_tf_categorical_model.signature_def_key,
                                 path=model_path)

    pyfunc_wrapper = pyfunc.load_pyfunc(model_path)
    results_df = pyfunc_wrapper.predict(saved_tf_categorical_model.inference_df)
    pandas.testing.assert_frame_equal(
        results_df, saved_tf_categorical_model.expected_results_df, check_less_precise=6)

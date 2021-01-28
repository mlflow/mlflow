# pep8: disable=E501

import collections
import os
import shutil
import sys
import pytest
import yaml
import json
import copy
from unittest import mock

import numpy as np
import pandas as pd
import pandas.testing
import tensorflow as tf
import iris_data_utils

import mlflow
import mlflow.tensorflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow.exceptions import MlflowException
from mlflow import pyfunc
from mlflow.models import infer_signature, Model
from mlflow.models.utils import _read_example
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

from tests.helper_functions import score_model_in_sagemaker_docker_container
from tests.helper_functions import set_boto_credentials  # pylint: disable=unused-import
from tests.helper_functions import mock_s3_bucket  # pylint: disable=unused-import

SavedModelInfo = collections.namedtuple(
    "SavedModelInfo",
    [
        "path",
        "meta_graph_tags",
        "signature_def_key",
        "inference_df",
        "expected_results_df",
        "raw_results",
        "raw_df",
    ],
)


@pytest.fixture
def saved_tf_iris_model(tmpdir):
    # Following code from
    # https://github.com/tensorflow/models/blob/master/samples/core/get_started/premade_estimator.py
    train_x, train_y = iris_data_utils.load_data()[0]

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    estimator = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3,
    )

    # Train the Model.
    batch_size = 100
    train_steps = 1000
    estimator.train(
        input_fn=lambda: iris_data_utils.train_input_fn(train_x, train_y, batch_size),
        steps=train_steps,
    )

    # Generate predictions from the model
    predict_x = {
        "SepalLength": [5.1, 5.9, 6.9],
        "SepalWidth": [3.3, 3.0, 3.1],
        "PetalLength": [1.7, 4.2, 5.4],
        "PetalWidth": [0.5, 1.5, 2.1],
    }

    estimator_preds = estimator.predict(
        lambda: iris_data_utils.eval_input_fn(predict_x, None, batch_size)
    )

    # Building a dictionary of the predictions by the estimator.
    if sys.version_info < (3, 0):
        estimator_preds_dict = estimator_preds.next()
    else:
        estimator_preds_dict = next(estimator_preds)
    for row in estimator_preds:
        for key in row.keys():
            estimator_preds_dict[key] = np.vstack((estimator_preds_dict[key], row[key]))

    # Building a pandas DataFrame out of the prediction dictionary.
    estimator_preds_df = copy.deepcopy(estimator_preds_dict)
    for col in estimator_preds_df.keys():
        if all(len(element) == 1 for element in estimator_preds_df[col]):
            estimator_preds_df[col] = estimator_preds_df[col].ravel()
        else:
            estimator_preds_df[col] = estimator_preds_df[col].tolist()

    # Building a DataFrame that contains the names of the flowers predicted.
    estimator_preds_df = pandas.DataFrame.from_dict(data=estimator_preds_df)
    estimator_preds_results = [
        iris_data_utils.SPECIES[id[0]] for id in estimator_preds_dict["class_ids"]
    ]
    estimator_preds_results_df = pd.DataFrame({"predictions": estimator_preds_results})

    # Define a function for estimator inference
    feature_spec = {}
    for name in my_feature_columns:
        feature_spec[name.key] = tf.Variable([], dtype=tf.float64, name=name.key)

    receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

    # Save the estimator and its inference function
    saved_estimator_path = str(tmpdir.mkdir("saved_model"))
    saved_estimator_path = estimator.export_saved_model(saved_estimator_path, receiver_fn).decode(
        "utf-8"
    )
    return SavedModelInfo(
        path=saved_estimator_path,
        meta_graph_tags=["serve"],
        signature_def_key="predict",
        inference_df=pd.DataFrame(
            data=predict_x, columns=[name.key for name in my_feature_columns]
        ),
        expected_results_df=estimator_preds_results_df,
        raw_results=estimator_preds_dict,
        raw_df=estimator_preds_df,
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

    # Build a DNNRegressor, with 20x20-unit hidden layers, with the feature columns
    # defined above as input
    estimator = tf.estimator.DNNRegressor(hidden_units=[20, 20], feature_columns=feature_columns)

    # Train the estimator and obtain expected predictions on the training dataset
    estimator.train(
        input_fn=lambda: iris_data_utils.train_input_fn(trainingFeatures, y_train, 1), steps=10
    )
    estimator_preds = np.array(
        [
            s["predictions"]
            for s in estimator.predict(
                lambda: iris_data_utils.eval_input_fn(trainingFeatures, None, 1)
            )
        ]
    ).ravel()
    estimator_preds_df = pd.DataFrame({"predictions": estimator_preds})

    # Define a function for estimator inference
    feature_spec = {
        "body-style": tf.Variable([], dtype=tf.string, name="body-style"),
        "curb-weight": tf.Variable([], dtype=tf.float64, name="curb-weight"),
        "highway-mpg": tf.Variable([], dtype=tf.float64, name="highway-mpg"),
    }
    receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

    # Save the estimator and its inference function
    saved_estimator_path = str(tmpdir.mkdir("saved_model"))
    saved_estimator_path = estimator.export_saved_model(saved_estimator_path, receiver_fn).decode(
        "utf-8"
    )
    return SavedModelInfo(
        path=saved_estimator_path,
        meta_graph_tags=["serve"],
        signature_def_key="predict",
        inference_df=pd.DataFrame(trainingFeatures),
        expected_results_df=estimator_preds_df,
        raw_results=None,
        raw_df=None,
    )


@pytest.fixture
def tf_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_conda_deps=["tensorflow", "pytest"])
    return conda_env


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


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
    infer = mlflow.tensorflow.load_model(model_uri=model_uri)
    feed_dict = {
        df_column_name: tf.constant(saved_tf_iris_model.inference_df[df_column_name])
        for df_column_name in list(saved_tf_iris_model.inference_df)
    }
    raw_preds = infer(**feed_dict)
    pred_dict = {column_name: raw_preds[column_name].numpy() for column_name in raw_preds.keys()}
    for col in pred_dict:
        assert np.allclose(
            np.array(pred_dict[col], dtype=np.float),
            np.array(saved_tf_iris_model.raw_results[col], dtype=np.float),
        )


@pytest.mark.large
def test_iris_model_can_be_loaded_and_evaluated_successfully(saved_tf_iris_model, model_path):
    mlflow.tensorflow.save_model(
        tf_saved_model_dir=saved_tf_iris_model.path,
        tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
        tf_signature_def_key=saved_tf_iris_model.signature_def_key,
        path=model_path,
    )

    def load_and_evaluate():

        infer = mlflow.tensorflow.load_model(model_uri=model_path)
        feed_dict = {
            df_column_name: tf.constant(saved_tf_iris_model.inference_df[df_column_name])
            for df_column_name in list(saved_tf_iris_model.inference_df)
        }
        raw_preds = infer(**feed_dict)
        pred_dict = {
            column_name: raw_preds[column_name].numpy() for column_name in raw_preds.keys()
        }
        for col in pred_dict:
            assert np.array_equal(pred_dict[col], saved_tf_iris_model.raw_results[col])

    load_and_evaluate()

    with tf.device("/CPU:0"):
        load_and_evaluate()


def test_schema_and_examples_are_save_correctly(saved_tf_iris_model):
    train_x, train_y = iris_data_utils.load_data()[0]
    X = pd.DataFrame(train_x)
    y = pd.Series(train_y)
    for signature in (None, infer_signature(X, y)):
        for example in (None, X.head(3)):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.tensorflow.save_model(
                    tf_saved_model_dir=saved_tf_iris_model.path,
                    tf_meta_graph_tags=saved_tf_iris_model.meta_graph_tags,
                    tf_signature_def_key=saved_tf_iris_model.signature_def_key,
                    path=path,
                    signature=signature,
                    input_example=example,
                )
                mlflow_model = Model.load(path)
                assert signature == mlflow_model.signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    assert all((_read_example(mlflow_model, path) == example).all())


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

    mlflow.tensorflow.load_model(model_uri=model_path)


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

    mlflow.tensorflow.load_model(model_uri=model_uri)


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
    saved_tf_iris_model,
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
    assert isinstance(results_df, pd.DataFrame)
    for key in results_df.keys():
        assert np.array_equal(results_df[key], saved_tf_iris_model.raw_df[key])

    # can also call predict with a dict
    inp_dict = {}
    for df_col_name in list(saved_tf_iris_model.inference_df):
        inp_dict[df_col_name] = saved_tf_iris_model.inference_df[df_col_name].values
    results = pyfunc_wrapper.predict(inp_dict)
    assert isinstance(results, dict)
    for key in results.keys():
        assert np.array_equal(results[key], saved_tf_iris_model.raw_df[key].tolist())

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
    # Precision is less accurate for the categorical model when we load back the saved model.
    pandas.testing.assert_frame_equal(
        results_df, saved_tf_categorical_model.expected_results_df, check_less_precise=3
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
        check_less_precise=3,
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

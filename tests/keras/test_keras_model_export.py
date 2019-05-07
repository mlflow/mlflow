# pep8: disable=E501

from __future__ import print_function

import os
import json
import pytest
from keras.models import Sequential
from keras.layers import Dense
import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import yaml

import mlflow
import mlflow.keras
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.store.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _get_model_log_dir
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from tests.helper_functions import pyfunc_serve_and_score_model
from tests.helper_functions import score_model_in_sagemaker_docker_container
from tests.helper_functions import set_boto_credentials  # pylint: disable=unused-import
from tests.helper_functions import mock_s3_bucket  # pylint: disable=unused-import
from tests.pyfunc.test_spark import score_model_as_udf
from tests.projects.utils import tracking_uri_mock  # pylint: disable=unused-import


@pytest.fixture(scope='module')
def data():
    iris = datasets.load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])
    y = data['target']
    x = data.drop('target', axis=1)
    return x, y


@pytest.fixture(scope='module')
def model(data):
    x, y = data
    model = Sequential()
    model.add(Dense(3, input_dim=4))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='SGD')
    model.fit(x, y)
    return model


@pytest.fixture(scope='module')
def predicted(model, data):
    return model.predict(data[0])


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(tmpdir.strpath, "model")


@pytest.fixture
def keras_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(
        conda_env,
        additional_conda_deps=["keras", "tensorflow", "pytest"])
    return conda_env


@pytest.mark.large
def test_model_save_load(model, model_path, data, predicted):
    x, _ = data
    mlflow.keras.save_model(model, model_path)

    # Loading Keras model
    model_loaded = mlflow.keras.load_model(model_path)
    assert all(model_loaded.predict(x) == predicted)

    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_pyfunc(model_path)
    assert all(pyfunc_loaded.predict(x).values == predicted)

    # pyfunc serve
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=os.path.abspath(model_path),
        data=pd.DataFrame(x),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED)
    assert all(pd.read_json(scoring_response.content, orient="records").values.astype(np.float32)
               == predicted)

    # test spark udf
    spark_udf_preds = score_model_as_udf(model_uri=os.path.abspath(model_path),
                                         pandas_df=pd.DataFrame(x),
                                         result_type="float")
    np.testing.assert_array_almost_equal(
        np.array(spark_udf_preds), predicted.reshape(len(spark_udf_preds)), decimal=4)


@pytest.mark.large
def test_model_load_from_remote_uri_succeeds(model, model_path, mock_s3_bucket, data, predicted):
    x, _ = data
    mlflow.keras.save_model(model, model_path)

    artifact_root = "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    model_loaded = mlflow.keras.load_model(model_uri=model_uri)
    assert all(model_loaded.predict(x) == predicted)


@pytest.mark.large
def test_model_log(tracking_uri_mock, model, data, predicted):  # pylint: disable=unused-argument
    x, _ = data
    # should_start_run tests whether or not calling log_model() automatically starts a run.
    for should_start_run in [False, True]:
        try:
            if should_start_run:
                mlflow.start_run()
            artifact_path = "keras_model"
            mlflow.keras.log_model(model, artifact_path=artifact_path)
            model_uri = "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id,
                artifact_path=artifact_path)

            # Load model
            model_loaded = mlflow.keras.load_model(model_uri=model_uri)
            assert all(model_loaded.predict(x) == predicted)

            # Loading pyfunc model
            pyfunc_loaded = mlflow.pyfunc.load_pyfunc(model_uri=model_uri)
            assert all(pyfunc_loaded.predict(x).values == predicted)
        finally:
            mlflow.end_run()


@pytest.mark.large
def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
        model, model_path, keras_custom_env):
    mlflow.keras.save_model(keras_model=model, path=model_path, conda_env=keras_custom_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != keras_custom_env

    with open(keras_custom_env, "r") as f:
        keras_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == keras_custom_env_parsed


@pytest.mark.large
def test_model_save_accepts_conda_env_as_dict(model, model_path):
    conda_env = dict(mlflow.keras.DEFAULT_CONDA_ENV)
    conda_env["dependencies"].append("pytest")
    mlflow.keras.save_model(keras_model=model, path=model_path, conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


@pytest.mark.large
def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(model, keras_custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.keras.log_model(
            keras_model=model, artifact_path=artifact_path, conda_env=keras_custom_env)
        run_id = mlflow.active_run().info.run_id
    model_path = _get_model_log_dir(artifact_path, run_id)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != keras_custom_env

    with open(keras_custom_env, "r") as f:
        keras_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == keras_custom_env_parsed


@pytest.mark.large
def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
        model, model_path):
    mlflow.keras.save_model(keras_model=model, path=model_path, conda_env=None)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.keras.DEFAULT_CONDA_ENV


@pytest.mark.large
def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
        model):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.keras.log_model(keras_model=model, artifact_path=artifact_path, conda_env=None)
        run_id = mlflow.active_run().info.run_id
    model_path = _get_model_log_dir(artifact_path, run_id)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.keras.DEFAULT_CONDA_ENV


@pytest.mark.large
def test_model_load_succeeds_with_missing_data_key_when_data_exists_at_default_path(
        model, model_path, data, predicted):
    """
    This is a backwards compatibility test to ensure that models saved in MLflow version <= 0.8.0
    can be loaded successfully. These models are missing the `data` flavor configuration key.
    """
    mlflow.keras.save_model(keras_model=model, path=model_path)

    model_conf_path = os.path.join(model_path, "MLmodel")
    model_conf = Model.load(model_conf_path)
    flavor_conf = model_conf.flavors.get(mlflow.keras.FLAVOR_NAME, None)
    assert flavor_conf is not None
    del flavor_conf['data']
    model_conf.save(model_conf_path)

    model_loaded = mlflow.keras.load_model(model_path)
    assert all(model_loaded.predict(data[0]) == predicted)


@pytest.mark.release
def test_sagemaker_docker_model_scoring_with_default_conda_env(model, model_path, data, predicted):
    mlflow.keras.save_model(keras_model=model, path=model_path, conda_env=None)

    scoring_response = score_model_in_sagemaker_docker_container(
        model_uri=model_path,
        data=data[0],
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
        flavor=mlflow.pyfunc.FLAVOR_NAME,
        activity_polling_timeout_seconds=500)
    deployed_model_preds = pd.DataFrame(json.loads(scoring_response.content))

    np.testing.assert_array_almost_equal(
        deployed_model_preds.values,
        predicted,
        decimal=4)

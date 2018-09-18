import os
import pytest
from collections import namedtuple

import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.sagemaker as mfs
from mlflow.models import Model
from mlflow.tracking.utils import _get_model_log_dir

TrainedModel = namedtuple("TrainedModel", ["model_path", "run_id"])


@pytest.fixture
def pretrained_model():
    model_path = "model"
    with mlflow.start_run():
        X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 1, 0])
        lr = LogisticRegression()
        lr.fit(X, y)
        mlflow.sklearn.log_model(lr, model_path)
        run_id = mlflow.active_run().info.run_uuid
        return TrainedModel(model_path, run_id)


def test_deployment_with_unsupported_flavor_throws_value_error(pretrained_model):
    unsupported_flavor = "this is not a valid flavor"
    with pytest.raises(ValueError):
        mfs.deploy(app_name="bad_flavor",
                   model_path=pretrained_model.model_path,
                   run_id=pretrained_model.run_id,
                   flavor=unsupported_flavor)


def test_deployment_with_missing_flavor_throws_value_error(pretrained_model):
    missing_flavor = "mleap"
    with pytest.raises(ValueError):
        mfs.deploy(app_name="missing-flavor",
                   model_path=pretrained_model.model_path,
                   run_id=pretrained_model.run_id,
                   flavor=missing_flavor)


def test_deployment_of_model_with_no_supported_flavors_throws_value_error(pretrained_model):
    logged_model_path = _get_model_log_dir(pretrained_model.model_path, pretrained_model.run_id)
    model_config_path = os.path.join(logged_model_path, "MLmodel")
    model_config = Model.load(model_config_path)
    del model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
    model_config.save(path=model_config_path)

    with pytest.raises(ValueError):
        mfs.deploy(app_name="missing-flavor",
                   model_path=logged_model_path,
                   flavor=None)


def test_validate_deployment_flavor_validates_python_function_flavor_successfully(
        pretrained_model):
    model_config_path = os.path.join(_get_model_log_dir(
        pretrained_model.model_path, pretrained_model.run_id), "MLmodel")
    model_config = Model.load(model_config_path)
    mfs._validate_deployment_flavor(
            model_config=model_config, flavor=mlflow.pyfunc.FLAVOR_NAME)


def test_get_preferred_deployment_flavor_obtains_valid_flavor_from_model(pretrained_model):
    model_config_path = os.path.join(_get_model_log_dir(
        pretrained_model.model_path, pretrained_model.run_id), "MLmodel")
    model_config = Model.load(model_config_path)

    selected_flavor = mfs._get_preferred_deployment_flavor(model_config=model_config)

    assert selected_flavor in mfs.SUPPORTED_DEPLOYMENT_FLAVORS
    assert selected_flavor in model_config.flavors

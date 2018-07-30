"""Sample MLflow integration for h2o."""

from __future__ import absolute_import

import os
import yaml

from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking


def save_model(h2o_model, path, conda_env=None, mlflow_model=Model(), settings=None):
    """
    Save a H2O model to a path on the local file system.

    :param h2o_model: H2O model to be saved.
    :param path: Local path where the model is to be saved.
    :param mlflow_model: MLflow model config this flavor is being added to.
    """
    import h2o

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise Exception("Path '{}' already exists".format(path))
    model_dir = os.path.join(path, "model.h2o")
    os.makedirs(model_dir)

    # Save h2o-model
    h2o_save_location = h2o.save_model(model=h2o_model, path=model_dir, force=True)
    model_file = os.path.basename(h2o_save_location)

    # Save h2o-settings
    if settings is None:
        settings = {}
    settings['full_file'] = h2o_save_location
    settings['model_file'] = model_file
    settings['model_dir'] = model_dir
    with open(os.path.join(model_dir, "h2o.yaml"), 'w') as settings_file:
        yaml.safe_dump(settings, stream=settings_file)

    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.h2o",
                        data="model.h2o", env=conda_env)
    mlflow_model.add_flavor("h2o", saved_model=model_file, h2o_version=h2o.__version__)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def log_model(h2o_model, artifact_path, **kwargs):
    """Log a H2O model as an MLflow artifact for the current run."""
    Model.log(artifact_path=artifact_path, flavor=mlflow.h2o,
              h2o_model=h2o_model, **kwargs)


def _load_model(path, init=False):
    import h2o
    path = os.path.abspath(path)
    with open(os.path.join(path, "h2o.yaml")) as f:
        params = yaml.safe_load(f.read())
    if init:
        h2o.init(**(params["init"] if "init" in params else {}))
        h2o.no_progress()
    return h2o.load_model(os.path.join(path, params['model_file']))


class _H2OModelWrapper:
    def __init__(self, h2o_model):
        self.h2o_model = h2o_model

    def predict(self, dataframe):
        import h2o
        predicted = self.h2o_model.predict(h2o.H2OFrame(dataframe)).as_data_frame()
        predicted.index = dataframe.index
        return predicted


def load_pyfunc(path):
    """
    When loading this model as a pyfunc-model, `h2o.init(...)` will be called.
    Therefore, the right version of h2o(-py) has to be in the environment. The
    arguments given to `h2o.init(...)` can be customized in `model.h2o/h2o.yaml`
    under the key `init`.
    """
    return _H2OModelWrapper(_load_model(path, init=True))


def load_model(path, run_id=None):
    """
    Load a H2O model from a local file (if run_id is None) or a run.

    This function expects there is a h2o instance initialised with
    `h2o.init()`.
    """
    if run_id is not None:
        path = mlflow.tracking._get_model_log_dir(model_name=path, run_id=run_id)
    return _load_model(os.path.join(path, "model.h2o"))

"""Sample MLflow integration for h2o."""

from __future__ import absolute_import

import os
import h2o

from mlflow.utils.file_utils import TempDir
from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking


def save_model(h2o_model, path, conda_env=None, mlflow_model=Model()):
    """
    Save a H2O model to a path on the local file system.

    :param h2o_model: H2O model to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Path to a Conda environment file. If provided, this decribes the environment
           this model should be run it. At minimum, it should specify python, h2o and mlflow
           with appropriate versions.
    :param mlflow_model: MLflow model config this flavor is being added to.
    """
    if os.path.exists(path):
        raise Exception("Path '{}' already exists".format(path))
    os.makedirs(path)
    model_dir = os.path.join(path, "model.h2o")
    h2o_save_location = h2o.save_model(model=h2o_model, path=model_dir, force=True)
    model_file = "model.h2o/"+os.path.basename(h2o_save_location)
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.h2o", data=model_file,  env=conda_env)
    mlflow_model.add_flavor("h2o", saved_model=model_file, h2o_version=h2o.__version__)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def log_model(h2o_model, artifact_path):
    """Log a H2O model as an MLflow artifact for the current run."""

    with TempDir() as tmp:
        local_path = tmp.path("model")
        # TODO: I get active_run_id here but mlflow.tracking.log_output_files has its own way
        run_id = mlflow.tracking.active_run().info.run_uuid
        mlflow_model = Model(artifact_path=artifact_path, run_id=run_id)
        save_model(h2o_model, local_path, mlflow_model=mlflow_model)
        mlflow.tracking.log_artifacts(local_path, artifact_path)


def _load_model(path, init=False, model=None):
    if model is None:
        model = Model.load(os.path.join(path, "MLmodel"))
    assert "h2o" in model.flavors
    params = model.flavors["h2o"]

    if init:
        h2o.init(**(params["init"] if "init" in params else {}))
        h2o.no_progress()
    return h2o.load_model(os.path.join(path, params["saved_model"]))


def load_pyfunc(model, path, data_path):
    h2o_model = _load_model(path, init=True, model=model)

    class ModelWrapper:
        def predict(self, dataframe):
            return h2o_model.predict(h2o.H2OFrame(dataframe)).as_data_frame().values[:, 0]

    return ModelWrapper()


def load_model(path, run_id=None):
    """Load a SciKit-Learn model from a local file (if run_id is None) or a run."""
    if run_id is not None:
        path = mlflow.tracking._get_model_log_dir(model_name=path, run_id=run_id)
    return _load_model(path)

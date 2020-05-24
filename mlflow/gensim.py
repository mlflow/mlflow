"""
The ``mlflow.gensim`` module provides an API for logging and loading Gensim models.
This module exports Gensim models with the following flavors:

Python (native) `pickle <https://scikit-learn.org/stable/modules/model_persistence.html>`_ format
    This is the main flavor that can be loaded back into a gensim model.

:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

from __future__ import absolute_import

import os
import pickle
import yaml
import logging

import mlflow
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.exceptions import MlflowException

FLAVOR_NAME = "gensim"

_logger = logging.getLogger(__name__)


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    import gensim

    return _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            "gensim=={}".format(gensim.__version__),
        ],
        additional_conda_channels=None)


def save_model(gen_model, path, conda_env=None, mlflow_model=Model()):
    """
    Save an Gensim model to a path on the local file system.

    :param gen_model: Gensim model to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this describes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'pip': [
                                    'gensim==3.8.3'
                                ]
                            ]
                        }

    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    """
    import gensim

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))

    model_data_subpath = "model.pkl"
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(path)

    # Save an Gensim model
    _save_model(gen_model=gen_model, output_path=model_data_path)

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.gensim",
                        data=model_data_subpath, env=conda_env_subpath)

    mlflow_model.add_flavor(FLAVOR_NAME, genism_version=gensim.__version__, data=model_data_subpath)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def log_model(gen_model, artifact_path, conda_env=None, registered_model_name=None):
    """
    Log an Gensim model as an MLflow artifact for the current run.

    :param gen_model: gensim model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this describes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'pip': [
                                    'gensim==3.8.3'
                                ]
                            ]
                        }
    :param registered_model_name: (Experimental) If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.


                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        signature = infer_signature(train, model.predict(train))

    """
    Model.log(artifact_path=artifact_path,
              flavor=mlflow.gensim,
              registered_model_name=registered_model_name,
              gen_model=gen_model,
              conda_env=conda_env)


def _load_model_from_local_file(path):
    """
    Load a gensim model saved as an MLflow artifact on the local file system.

    :param path: Local filesystem path to the MLflow Model with the ``gensim`` flavor.
    """
    # TODO: we could validate the gensim version here
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``sklearn`` flavor.
    """
    return _load_model_from_local_file(path)


def _save_model(gen_model, output_path):
    """
    :param gen_model: The gensim model to serialize.
    :param output_path: The file path to which to write the serialized model.
    """
    with open(output_path, "wb") as out:
        pickle.dump(gen_model, out)


def load_model(model_uri):
    """
    Load a gensim model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model, for example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.

    :return: A gensim model.

    .. code-block:: python
        :caption: Example

        import mlflow.gensim
        doc2vec_model = mlflow.gensim.load_model("path/to/your/model/")

        # define tokens to embbed vector
        tokens = ...
        embeddings = doc2vec_model.infer_vector(tokens)
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    gensim_model_artifacts_path = os.path.join(local_model_path, flavor_conf['data'])
    return _load_model_from_local_file(path=gensim_model_artifacts_path)

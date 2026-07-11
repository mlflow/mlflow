"""
The ``mlflow.gensim`` module provides an API for logging and loading Gensim models.
This module exports Gensim models with the following flavors:
Gensim (native) format
    This is the main flavor that can be loaded back into Gensim.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

from __future__ import absolute_import

import os
import yaml
import pickle
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    Optional,
    Union,
)

import mlflow
from mlflow import pyfunc
from mlflow.models import Model, ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.exceptions import MlflowException
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS


FLAVOR_NAME = "gensim"


def get_default_conda_env() -> Optional[Union[Dict[Hashable, Any], List]]:
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    import gensim

    return _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["gensim=={}".format(gensim.__version__), ],
        additional_conda_channels=None,
    )


def save_model(
    gensim_model,
    path: str,
    conda_env: Optional[Union[Dict, str]] = None,
    mlflow_model: Model = Model(),
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
) -> None:
    """
    Save a Gensim model to a path on the local file system.
    :param gensim_model: Gensim model to be saved.
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
                                    'gensim==4.0.1'
                                ]
                            ]
                        }
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:
                      .. code-block:: python
                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    """
    import gensim

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))
    model_data_subpath = "model.gensim"
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(path)

    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    # Save the Gensim model
    gensim_model.save(model_data_path)

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    pyfunc.add_to_model(
        model=mlflow_model,
        loader_module="mlflow.gensim",
        data=model_data_subpath,
        env=conda_env_subpath,
    )

    mlflow_model.add_flavor(
        FLAVOR_NAME, gensim_version=gensim.__version__, data=model_data_subpath,
    )

    mlflow_model.save(os.path.join(path, "MLmodel"))


def log_model(
    gensim_model,
    artifact_path: str,
    conda_env: Optional[Union[Dict, str]] = None,
    registered_model_name: Optional[str] = None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
    await_registration_for: int = DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    **kwargs
) -> None:
    """
    Log a Gensim model as an MLflow artifact for the current run.
    :param gensim_model: Gensim model to be saved.
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
    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:
                      .. code-block:: python
                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.
    """
    Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.gensim,
        registered_model_name=registered_model_name,
        gensim_model=gensim_model,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        **kwargs
    )


def _load_model_from_local_file(path: str) -> Any:
    """
    Load a Gensim model saved as an MLflow artifact on the local file system.
    :param path: Local filesystem path to the MLflow Model with the ``gensim`` flavor.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_pyfunc(path: str) -> Any:
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    :param path: Local filesystem path to the MLflow Model with the ``sklearn`` flavor.
    """
    return _load_model_from_local_file(path)


def _save_model(gensim_model, output_path: str) -> None:
    """
    :param gensim_model: The Gensim model to serialize.
    :param output_path: The file path to which to write the serialized model.
    """
    with open(output_path, "wb") as out:
        pickle.dump(gensim_model, out)


def load_model(model_uri) -> Any:
    """
    Load a Gensim model from a local file or a run.
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
    :return: A Gensim model.
    .. code-block:: python
        :caption: Example
        import mlflow.gensim
        doc2vec_model = mlflow.gensim.load_model("path/to/your/model/")
        # define tokens to embed vector
        tokens = ...
        embeddings = doc2vec_model.infer_vector(tokens)
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    gensim_model_artifacts_path = os.path.join(local_model_path, flavor_conf["data"])
    return _load_model_from_local_file(path=gensim_model_artifacts_path)

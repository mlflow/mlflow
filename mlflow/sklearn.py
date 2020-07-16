"""
The ``mlflow.sklearn`` module provides an API for logging and loading scikit-learn models. This
module exports scikit-learn models with the following flavors:

Python (native) `pickle <https://scikit-learn.org/stable/modules/model_persistence.html>`_ format
    This is the main flavor that can be loaded back into scikit-learn.

:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""
import os
import logging
import pickle
import yaml

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, INTERNAL_ERROR
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration

FLAVOR_NAME = "sklearn"

SERIALIZATION_FORMAT_PICKLE = "pickle"
SERIALIZATION_FORMAT_CLOUDPICKLE = "cloudpickle"

SUPPORTED_SERIALIZATION_FORMATS = [
    SERIALIZATION_FORMAT_PICKLE,
    SERIALIZATION_FORMAT_CLOUDPICKLE
]

_logger = logging.getLogger(__name__)


def get_default_conda_env(include_cloudpickle=False):
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    import sklearn
    pip_deps = None
    if include_cloudpickle:
        import cloudpickle
        pip_deps = ["cloudpickle=={}".format(cloudpickle.__version__)]
    return _mlflow_conda_env(
        additional_conda_deps=[
            "scikit-learn={}".format(sklearn.__version__),
        ],
        additional_pip_deps=pip_deps,
        additional_conda_channels=None
    )


def save_model(sk_model, path, conda_env=None, mlflow_model=None,
               serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
               signature: ModelSignature = None, input_example: ModelInputExample = None):
    """
    Save a scikit-learn model to a path on the local file system.

    :param sk_model: scikit-learn model to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If `None`, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'scikit-learn=0.19.2'
                            ]
                        }

    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    :param serialization_format: The format in which to serialize the model. This should be one of
                                 the formats listed in
                                 ``mlflow.sklearn.SUPPORTED_SERIALIZATION_FORMATS``. The Cloudpickle
                                 format, ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``,
                                 provides better cross-system compatibility by identifying and
                                 packaging code dependencies with the serialized model.

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


    .. code-block:: python
        :caption: Example

        import mlflow.sklearn
        from sklearn.datasets import load_iris
        from sklearn import tree

        iris = load_iris()
        sk_model = tree.DecisionTreeClassifier()
        sk_model = sk_model.fit(iris.data, iris.target)

        # Save the model in cloudpickle format
        # set path to location for persistence
        sk_path_dir_1 = ...
        mlflow.sklearn.save_model(
                sk_model, sk_path_dir_1,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

        # save the model in pickle format
        # set path to location for persistence
        sk_path_dir_2 = ...
        mlflow.sklearn.save_model(sk_model, sk_path_dir_2,
                                  serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
    """
    import sklearn
    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
                message=(
                    "Unrecognized serialization format: {serialization_format}. Please specify one"
                    " of the following supported formats: {supported_formats}.".format(
                        serialization_format=serialization_format,
                        supported_formats=SUPPORTED_SERIALIZATION_FORMATS)),
                error_code=INVALID_PARAMETER_VALUE)

    if os.path.exists(path):
        raise MlflowException(message="Path '{}' already exists".format(path),
                              error_code=RESOURCE_ALREADY_EXISTS)
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    model_data_subpath = "model.pkl"
    _save_model(sk_model=sk_model, output_path=os.path.join(path, model_data_subpath),
                serialization_format=serialization_format)

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env(
            include_cloudpickle=serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE)
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn",
                        model_path=model_data_subpath, env=conda_env_subpath)
    mlflow_model.add_flavor(FLAVOR_NAME,
                            pickled_model=model_data_subpath,
                            sklearn_version=sklearn.__version__,
                            serialization_format=serialization_format)
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def log_model(sk_model, artifact_path, conda_env=None,
              serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE, registered_model_name=None,
              signature: ModelSignature=None, input_example: ModelInputExample=None):
    """
    Log a scikit-learn model as an MLflow artifact for the current run.

    :param sk_model: scikit-learn model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If `None`, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'scikit-learn=0.19.2'
                            ]
                        }

    :param serialization_format: The format in which to serialize the model. This should be one of
                                 the formats listed in
                                 ``mlflow.sklearn.SUPPORTED_SERIALIZATION_FORMATS``. The Cloudpickle
                                 format, ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``,
                                 provides better cross-system compatibility by identifying and
                                 packaging code dependencies with the serialized model.
    :param registered_model_name: (Experimental) If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

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



    .. code-block:: python
        :caption: Example

        import mlflow
        import mlflow.sklearn
        from sklearn.datasets import load_iris
        from sklearn import tree

        iris = load_iris()
        sk_model = tree.DecisionTreeClassifier()
        sk_model = sk_model.fit(iris.data, iris.target)
        # set the artifact_path to location where experiment artifacts will be saved

        #log model params
        mlflow.log_param("criterion", sk_model.criterion)
        mlflow.log_param("splitter", sk_model.splitter)

        # log model
        mlflow.sklearn.log_model(sk_model, "sk_models")
    """
    return Model.log(artifact_path=artifact_path,
                     flavor=mlflow.sklearn,
                     sk_model=sk_model,
                     conda_env=conda_env,
                     serialization_format=serialization_format,
                     registered_model_name=registered_model_name,
                     signature=signature,
                     input_example=input_example)


def _load_model_from_local_file(path, serialization_format):
    """Load a scikit-learn model saved as an MLflow artifact on the local file system.

    :param path: Local filesystem path to the MLflow Model saved with the ``sklearn`` flavor
    :param serialization_format: The format in which the model was serialized. This should be one of
                                 the following: ``mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE`` or
                                 ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``.
    """
    # TODO: we could validate the scikit-learn version here
    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
                message=(
                    "Unrecognized serialization format: {serialization_format}. Please specify one"
                    " of the following supported formats: {supported_formats}.".format(
                        serialization_format=serialization_format,
                        supported_formats=SUPPORTED_SERIALIZATION_FORMATS)),
                error_code=INVALID_PARAMETER_VALUE)
    with open(path, "rb") as f:
        # Models serialized with Cloudpickle cannot necessarily be deserialized using Pickle;
        # That's why we check the serialization format of the model before deserializing
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            return pickle.load(f)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle
            return cloudpickle.load(f)


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``sklearn`` flavor.
    """
    if os.path.isfile(path):
        # Scikit-learn models saved in older versions of MLflow (<= 1.9.1) specify the ``data``
        # field within the pyfunc flavor configuration. For these older models, the ``path``
        # parameter of ``_load_pyfunc()`` refers directly to a serialized scikit-learn model
        # object. In this case, we assume that the serialization format is ``pickle``, since
        # the model loading procedure in older versions of MLflow used ``pickle.load()``.
        serialization_format = SERIALIZATION_FORMAT_PICKLE
    else:
        # In contrast, scikit-learn models saved in versions of MLflow > 1.9.1 do not
        # specify the ``data`` field within the pyfunc flavor configuration. For these newer
        # models, the ``path`` parameter of ``load_pyfunc()`` refers to the top-level MLflow
        # Model directory. In this case, we parse the model path from the MLmodel's pyfunc
        # flavor configuration and attempt to fetch the serialization format from the
        # scikit-learn flavor configuration
        try:
            sklearn_flavor_conf = _get_flavor_configuration(
                model_path=path, flavor_name=FLAVOR_NAME)
            serialization_format = sklearn_flavor_conf.get(
                'serialization_format', SERIALIZATION_FORMAT_PICKLE)
        except MlflowException:
            _logger.warning(
                "Could not find scikit-learn flavor configuration during model loading process."
                " Assuming 'pickle' serialization format.")
            serialization_format = SERIALIZATION_FORMAT_PICKLE

        pyfunc_flavor_conf = _get_flavor_configuration(
            model_path=path, flavor_name=pyfunc.FLAVOR_NAME)
        path = os.path.join(path, pyfunc_flavor_conf['model_path'])

    return _load_model_from_local_file(
        path=path,
        serialization_format=serialization_format)


def _save_model(sk_model, output_path, serialization_format):
    """
    :param sk_model: The scikit-learn model to serialize.
    :param output_path: The file path to which to write the serialized model.
    :param serialization_format: The format in which to serialize the model. This should be one of
                                 the following: ``mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE`` or
                                 ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``.
    """
    with open(output_path, "wb") as out:
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            pickle.dump(sk_model, out)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle
            cloudpickle.dump(sk_model, out)
        else:
            raise MlflowException(
                    message="Unrecognized serialization format: {serialization_format}".format(
                        serialization_format=serialization_format),
                    error_code=INTERNAL_ERROR)


def load_model(model_uri):
    """
    Load a scikit-learn model from a local file or a run.

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

    :return: A scikit-learn model.

    .. code-block:: python
        :caption: Example

        import mlflow.sklearn
        sk_model = mlflow.sklearn.load_model("runs:/96771d893a5e46159d9f3b49bf9013e2/sk_models")

        # use Pandas DataFrame to make predictions
        pandas_df = ...
        predictions = sk_model.predict(pandas_df)
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    sklearn_model_artifacts_path = os.path.join(local_model_path, flavor_conf['pickled_model'])
    serialization_format = flavor_conf.get('serialization_format', SERIALIZATION_FORMAT_PICKLE)
    return _load_model_from_local_file(
        path=sklearn_model_artifacts_path,
        serialization_format=serialization_format)

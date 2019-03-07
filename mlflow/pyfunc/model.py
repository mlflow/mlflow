"""
The ``mlflow.pyfunc.model`` module defines logic for saving and loading custom "python_function"
models with a user-defined ``PythonModel`` subclass.
"""

import os
import tempfile
import shutil
import yaml
from abc import ABCMeta, abstractmethod

import cloudpickle

import mlflow.pyfunc
import mlflow.utils
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_ALREADY_EXISTS
from mlflow.tracking.utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.utils.file_utils import TempDir, _copy_file_or_tree

# `DEFAULT_CONDA_ENV` defines the default Conda environment for models produced by calls to
# `mlflow.pyfunc.save_model()` and `mlflow.pyfunc.log_model()` when a user-defined subclass of
# ``PythonModel`` is provided.
DEFAULT_CONDA_ENV = _mlflow_conda_env(
    additional_conda_deps=None,
    additional_pip_deps=[
        "cloudpickle=={}".format(cloudpickle.__version__),
    ],
    additional_conda_channels=None,
)

CONFIG_KEY_ARTIFACTS = "artifacts"
CONFIG_KEY_ARTIFACT_RELATIVE_PATH = "path"
CONFIG_KEY_ARTIFACT_URI = "uri"
CONFIG_KEY_PYTHON_MODEL = "python_model"
CONFIG_KEY_CLOUDPICKLE_VERSION = "cloudpickle_version"


class PythonModel(object):
    """
    Represents a generic Python model that evaluates inputs and produces API-compatible outputs.
    By subclassing :class:`~PythonModel`, users can create customized MLflow models with the
    "python_function" ("pyfunc") flavor, leveraging custom inference logic and artifact
    dependencies.
    """
    __metaclass__ = ABCMeta

    def load_context(self, context):
        """
        Loads artifacts from the specified :class:`~PythonModelContext` that can be used by
        :func:`~PythonModel.predict` when evaluating inputs. When loading an MLflow model with
        :func:`~load_pyfunc`, this method will be called as soon as the :class:`~PythonModel` is
        constructed.

        The same :class:`~PythonModelContext` will also be available during calls to
        :func:`~PythonModel.predict`, but it may be more efficient to override this method
        and load artifacts from the context at model load time.

        :param context: A :class:`~PythonModelContext` instance containing artifacts that the model
                        can use to perform inference.
        """

    @abstractmethod
    def predict(self, context, model_input):
        """
        Evaluates a pyfunc-compatible input and produces a pyfunc-compatible output.
        For more information about the pyfunc input/output API, see the :ref:`pyfunc-inference-api`.

        :param context: A :class:`~PythonModelContext` instance containing artifacts that the model
                        can use to perform inference.
        :param model_input: A pyfunc-compatible input for the model to evaluate.
        """


class PythonModelContext(object):
    """
    A collection of artifacts that a :class:`~PythonModel` can use when performing inference.
    :class:`~PythonModelContext` objects are created *implicitly* by the
    :func:`save_model() <mlflow.pyfunc.save_model>` and
    :func:`log_model() <mlflow.pyfunc.log_model>` persistence methods, using the contents specified
    by the ``artifacts`` parameter of these methods.
    """

    def __init__(self, artifacts):
        """
        :param artifacts: A dictionary of ``<name, artifact_path>`` entries, where ``artifact_path``
                          is an absolute filesystem path to a given artifact.
        """
        self._artifacts = artifacts

    @property
    def artifacts(self):
        """
        :return: A dictionary containing ``<name, artifact_path>`` entries, where ``artifact_path``
                 is an absolute filesystem path to the artifact.
        """
        return self._artifacts


def _save_model_with_class_artifacts_params(path, python_model, artifacts=None, conda_env=None,
                                            code_paths=None, mlflow_model=Model()):
    """
    :param path: The path to which to save the Python model.
    :param python_model: An instance of a subclass of :class:`~PythonModel`. ``python_model``
                        defines how the model loads artifacts and how it performs inference.
    :param artifacts: A dictionary containing ``<name, artifact_uri>`` entries. Remote artifact URIs
                      will be resolved to absolute filesystem paths, producing a dictionary of
                      ``<name, absolute_path>`` entries. ``python_model`` can reference these
                      resolved entries as the ``artifacts`` property of the ``context`` attribute.
                      If *None*, no artifacts will be added to the model.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :data:`mlflow.pyfunc.DEFAULT_CONDA_ENV`. If `None`, the default
                      :data:`mlflow.pyfunc.DEFAULT_CONDA_ENV` environment will be added to the
                      model.
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files will be *prepended* to the system
                       path before the model is loaded.
    :param mlflow_model: The model configuration to which to add the ``mlflow.pyfunc`` flavor.
    """
    if os.path.exists(path):
        raise MlflowException(
                message="Path '{}' already exists".format(path),
                error_code=RESOURCE_ALREADY_EXISTS)
    os.makedirs(path)

    custom_model_config_kwargs = {
        CONFIG_KEY_CLOUDPICKLE_VERSION: cloudpickle.__version__,
    }
    if isinstance(python_model, PythonModel):
        saved_python_model_subpath = "python_model.pkl"
        with open(os.path.join(path, saved_python_model_subpath), "wb") as out:
            cloudpickle.dump(python_model, out)
        custom_model_config_kwargs[CONFIG_KEY_PYTHON_MODEL] = saved_python_model_subpath
    else:
        raise MlflowException(
                message=("`python_model` must be a subclass of `PythonModel`. Instead, found an"
                         " object of type: {python_model_type}".format(
                             python_model_type=type(python_model))),
                error_code=INVALID_PARAMETER_VALUE)

    if artifacts:
        saved_artifacts_config = {}
        with TempDir() as tmp_artifacts_dir:
            tmp_artifacts_config = {}
            saved_artifacts_dir_subpath = "artifacts"
            for artifact_name, artifact_uri in artifacts.items():
                tmp_artifact_path = _download_artifact_from_uri(
                    artifact_uri=artifact_uri, output_path=tmp_artifacts_dir.path())
                tmp_artifacts_config[artifact_name] = tmp_artifact_path
                saved_artifact_subpath = os.path.join(
                    saved_artifacts_dir_subpath,
                    os.path.relpath(path=tmp_artifact_path, start=tmp_artifacts_dir.path()))
                saved_artifacts_config[artifact_name] = {
                    CONFIG_KEY_ARTIFACT_RELATIVE_PATH: saved_artifact_subpath,
                    CONFIG_KEY_ARTIFACT_URI: artifact_uri,
                }

            shutil.move(tmp_artifacts_dir.path(), os.path.join(path, saved_artifacts_dir_subpath))
        custom_model_config_kwargs[CONFIG_KEY_ARTIFACTS] = saved_artifacts_config

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = DEFAULT_CONDA_ENV
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    saved_code_subpath = None
    if code_paths is not None:
        saved_code_subpath = "code"
        for code_path in code_paths:
            _copy_file_or_tree(src=code_path, dst=path, dst_dir=saved_code_subpath)

    mlflow.pyfunc.add_to_model(model=mlflow_model, loader_module=__name__, code=saved_code_subpath,
                               env=conda_env_subpath, **custom_model_config_kwargs)
    mlflow_model.save(os.path.join(path, 'MLmodel'))


def _load_pyfunc(model_path):
    pyfunc_config = _get_flavor_configuration(
            model_path=model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME)

    python_model_cloudpickle_version = pyfunc_config.get(CONFIG_KEY_CLOUDPICKLE_VERSION, None)
    if python_model_cloudpickle_version is None:
        mlflow.pyfunc._logger.warning(
            "The version of CloudPickle used to save the model could not be found in the MLmodel"
            " configuration")
    elif python_model_cloudpickle_version != cloudpickle.__version__:
        # CloudPickle does not have a well-defined cross-version compatibility policy. Micro version
        # releases have been known to cause incompatibilities. Therefore, we match on the full
        # library version
        mlflow.pyfunc._logger.warning(
            "The version of CloudPickle that was used to save the model, `CloudPickle %s`, differs"
            " from the version of CloudPickle that is currently running, `CloudPickle %s`, and may"
            " be incompatible",
            python_model_cloudpickle_version, cloudpickle.__version__)

    python_model_subpath = pyfunc_config.get(CONFIG_KEY_PYTHON_MODEL, None)
    if python_model_subpath is None:
        raise MlflowException(
            "Python model path was not specified in the model configuration")
    with open(os.path.join(model_path, python_model_subpath), "rb") as f:
        python_model = cloudpickle.load(f)

    # TODO: If the longevity of the temporary directory prior becomes problematic, consider using
    # an alternative solution.
    tmp_artifacts_dir_path = tempfile.mkdtemp(suffix="artifacts")
    artifacts = {}
    for saved_artifact_name, saved_artifact_info in\
            pyfunc_config.get(CONFIG_KEY_ARTIFACTS, {}).items():
        tmp_artifact_path = os.path.join(
                tmp_artifacts_dir_path,
                _copy_file_or_tree(
                    src=os.path.join(
                        model_path, saved_artifact_info[CONFIG_KEY_ARTIFACT_RELATIVE_PATH]),
                    dst=tmp_artifacts_dir_path,
                    dst_dir=saved_artifact_name))
        artifacts[saved_artifact_name] = tmp_artifact_path

    context = PythonModelContext(artifacts=artifacts)
    python_model.load_context(context=context)
    return _PythonModelPyfuncWrapper(python_model=python_model, context=context)


class _PythonModelPyfuncWrapper(object):
    """
    Wrapper class that creates a predict function such that
    predict(model_input: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """

    def __init__(self, python_model, context):
        """
        :param python_model: An instance of a subclass of :class:`~PythonModel`.
        :param context: A :class:`~PythonModelContext` instance containing artifacts that
                        ``python_model`` may use when performing inference.
        """
        self.python_model = python_model
        self.context = context

    def predict(self, model_input):
        return self.python_model.predict(self.context, model_input)

import os
import inspect
import pydoc
import tempfile
import shutil
import sys
import yaml
from abc import ABCMeta, abstractmethod
from distutils.version import StrictVersion

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

DEFAULT_CONDA_ENV = _mlflow_conda_env(
    additional_conda_deps=[
        "cloudpickle=={}".format(cloudpickle.__version__),
    ],
    additional_pip_deps=None,
    additional_conda_channels=None,
)

CONFIG_KEY_ARTIFACTS = "artifacts"
CONFIG_KEY_ARTIFACT_RELATIVE_PATH = "path"
CONFIG_KEY_ARTIFACT_URI = "uri"
CONFIG_KEY_PARAMETERS = "parameters"
CONFIG_KEY_MODEL_CLASS = "model_class"


class PythonModel(object):
    """
    Represents a generic Python model that leverages a collection of *artifacts* and *parameters*
    (Python objects) to evaluate inputs and produce API-compatible outputs. By subclassing
    :class:`~PythonModel`, users can create customized MLflow models with the "python_function"
    ("pyfunc") flavor, leveraging custom inference logic and dependencies.
    """
    __metaclass__ = ABCMeta

    def __init__(self, context):
        """
        :param context: A :class:`~PythonModelContext` instance containing *artifacts* and
                        *parameters* that the model can use to perform inference.
        """
        self.context = context

    @abstractmethod
    def predict(self, model_input):
        """
        Evaluates a pyfunc-compatible input and produces a pyfunc-compatible output.
        For more information about the pyfunc input/output API, see `Inference API`_.

        :param model_input: A pyfunc-compatible input for the model to evaluate.
        """
        pass


class PythonModelContext(object):
    """
    A collection of *artifacts* and *parameters* (Python objects) that a :class:`~PythonModel` can
    use when performing inference. :class:`~PythonModelContext` objects are created *implicitly* by
    the :func:`save_model() <mlflow.pyfunc.save_model>` and
    :func:`log_model() <mlflow.pyfunc.log_model>` persistence methods, using the contents specified
    by the ``artifacts`` and ``parameters`` arguments of these methods.
    """

    def __init__(self, artifacts, parameters, directory_managers=None):
        """
        :param artifacts: A dictionary of ``<name, artifact_path>`` entries, where ``artifact_path``
                          is an absolute filesystem path to a given artifact.
        :param parameters: A dictionary of ``<name, python object>`` entries.
        :param directories: A list of objects managing the lifecycle of directories that the
                            contain the specified artifacts. These objects will be stored
                            as a class attribute, ensuring that their associated directories
                            exist for as long as the :class:`~PythonModelContext` is in scope.
        """
        self._artifacts = artifacts
        self._parameters = parameters
        self._directory_managers = directory_managers

    @property
    def artifacts(self):
        """
        :return: A dictionary containing ``<name, artifact_path>`` entries, where ``artifact_path``
                 is an absolute filesystem path to the artifact.
        """
        return self._artifacts

    @property
    def parameters(self):
        """
        :return: A dictionary containing ``<name, python object>`` entries.
        """
        return self._parameters


def _save_model_with_class_artifacts_params(path, model_class, artifacts=None, parameters=None,
                                            conda_env=None, code_paths=None, mlflow_model=Model()):
    """
    :param path: The path to which to save the Python model.
    :param model_class: A ``type`` object referring to a subclass of
                        :class:`~PythonModel`, or the fully-qualified name of such a subclass.
                        ``model_class`` defines how the model is loaded and how it performs
                        inference.
    :param artifacts: A dictionary containing ``<name, artifact_uri>`` entries. Remote artifact URIs
                      will be resolved to absolute filesystem paths, producing a dictionary of
                      ``<name, absolute_path>`` entries. ``model_class`` can reference these
                      resolved entries as the ``artifacts`` property of the ``context`` attribute.
                      If *None*, no artifacts will be added to the model.
    :param parameters: A dictionary containing ``<name, python object>`` entries. ``python object``
                       may be any Python object that is serializable with CloudPickle.
                       ``model_class`` can reference these resolved entries as the ``parameters``
                       property of the ``context`` attribute. If *None*, no Python object parameters
                       will be added to the model.
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
    if model_class is None:
        raise MlflowException(
            message=("`model_class` must be specified!"),
            error_code=INVALID_PARAMETER_VALUE)

    if os.path.exists(path):
        raise MlflowException(
                message="Path '{}' already exists".format(path),
                error_code=RESOURCE_ALREADY_EXISTS)
    os.makedirs(path)

    custom_model_config_kwargs = {}
    if inspect.isclass(model_class):
        saved_model_class_subpath = "model_class.pkl"
        with open(os.path.join(path, saved_model_class_subpath), "wb") as out:
            cloudpickle.dump(model_class, out)
        custom_model_config_kwargs[CONFIG_KEY_MODEL_CLASS] = saved_model_class_subpath
    else:
        raise MlflowException(
                message=("`model_class` must be a class object. Instead, found an object"
                         " of type: {model_class_type}".format(model_class_type=type(model_class))),
                error_code=INVALID_PARAMETER_VALUE)

    if artifacts is not None and len(artifacts) > 0:
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

            _validate_artifacts(tmp_artifacts_config)
            shutil.move(tmp_artifacts_dir.path(), os.path.join(path, saved_artifacts_dir_subpath))
        custom_model_config_kwargs[CONFIG_KEY_ARTIFACTS] = saved_artifacts_config

    if parameters is not None and len(parameters) > 0:
        saved_parameters_dir_subpath = "parameters"
        os.makedirs(os.path.join(path, saved_parameters_dir_subpath))
        saved_parameters_config = {}
        for parameter_name, parameter_py_obj in parameters.items():
            saved_parameter_subpath = os.path.join(
                saved_parameters_dir_subpath, "{param_name}.pkl".format(param_name=parameter_name))
            with open(os.path.join(path, saved_parameter_subpath), "wb") as out:
                cloudpickle.dump(parameter_py_obj, out)
            saved_parameters_config[parameter_name] = saved_parameter_subpath
        custom_model_config_kwargs[CONFIG_KEY_PARAMETERS] = saved_parameters_config

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


def _save_model_with_loader_module_and_data_path(path, loader_module, data_path=None,
                                                 code_paths=None, conda_env=None,
                                                 mlflow_model=Model()):
    """
    Export model as a generic Python function model.
    :param path: The path to which to save the Python model.
    :param loader_module: The name of the Python module that will be used to load the model
                          from ``data_path``. This module must define a method with the prototype
                          ``_load_pyfunc(data_path)``.
    :param data_path: Path to a file or directory containing model data.
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                      containing file dependencies). These files will be *prepended* to the system
                      path before the model is loaded.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :data:`mlflow.pyfunc.DEFAULT_CONDA_ENV`. If `None`, the default
                      :data:`mlflow.pyfunc.DEFAULT_CONDA_ENV` environment will be added to the
                      model.
    :return: Model configuration containing model info.
    """
    if loader_module is None:
        raise MlflowException(
            message=("`loader_module` must be specified!"),
            error_code=INVALID_PARAMETER_VALUE)

    if os.path.exists(path):
        raise MlflowException(
                message="Path '{}' already exists".format(path),
                error_code=RESOURCE_ALREADY_EXISTS)
    os.makedirs(path)

    code = None
    data = None
    env = None

    if data_path is not None:
        model_file = _copy_file_or_tree(src=data_path, dst=path, dst_dir="data")
        data = model_file

    if code_paths is not None:
        for code_path in code_paths:
            _copy_file_or_tree(src=code_path, dst=path, dst_dir="code")
        code = "code"

    if conda_env is not None:
        shutil.copy(src=conda_env, dst=os.path.join(path, "mlflow_env.yml"))
        env = "mlflow_env.yml"

    mlflow.pyfunc.add_to_model(
        mlflow_model, loader_module=loader_module, code=code, data=data, env=env)
    mlflow_model.save(os.path.join(path, 'MLmodel'))
    return mlflow_model


def _validate_artifacts(artifacts):
    """
    Examines the specified artifacts and verifies that constituent models have compatible
    versions of Python and compatible library dependency versions. If any incompatibilities are
    detected, appropriate warnings are logged.

    :param artifacts: A dictionary containing ``<name, artifact_path>`` entries, where
                      ``artifact_path`` is the absolute filesystem path to the artifact.
    """
    try:
        from conda.resolve import MatchSpec
    except ImportError:
        raise MlflowException(
            "Failed to import the `conda.resolve.MatchSpec` class. Please ensure that the `conda`"
            " package (https://anaconda.org/anaconda/conda) is installed in your current"
            " environment. Note that this package is not automatically included when creating a new"
            " Conda environment via `conda create`; it must be explicitly specified during"
            " environment creation or installed after environment activation via"
            " `conda install conda`.")

    curr_major_py_version = StrictVersion(mlflow.utils.PYTHON_VERSION).version[0]
    curr_cloudpickle_version_spec = MatchSpec("cloudpickle=={curr_cloudpickle_version}".format(
        curr_cloudpickle_version=cloudpickle.__version__))

    models = dict([
        (artifact_name, artifact_path) for artifact_name, artifact_path in artifacts.items()
        if os.path.isdir(artifact_path) and "MLmodel" in os.listdir(artifact_path)])
    model_py_version_data = []
    for model_name, model_path in models.items():
        model_conf = Model.load(os.path.join(model_path, "MLmodel"))
        pyfunc_conf = model_conf.flavors.get(mlflow.pyfunc.FLAVOR_NAME, {})

        model_py_version = pyfunc_conf.get(mlflow.pyfunc.PY_VERSION, None)
        if model_py_version is not None:
            model_py_version_data.append({
                "artifact_name": model_name,
                "model_version": model_py_version,
            })
            model_py_major_version = StrictVersion(model_py_version).version[0]
            if model_py_major_version != curr_major_py_version:
                mlflow.pyfunc._logger.warn(
                    "The artifact with name %s is an MLflow model that was saved with a different"
                    " major version of Python. As a result, your new model may not load or perform"
                    " correctly. Current python version: %s. Model python version: %s",
                    model_name, mlflow.utils.PYTHON_VERSION, model_py_version)

        conda_env_subpath = pyfunc_conf.get(mlflow.pyfunc.ENV, None)
        if conda_env_subpath is not None:
            with open(os.path.join(model_path, conda_env_subpath), "r") as f:
                conda_env = yaml.safe_load(f)

            conda_deps = conda_env.get("dependencies", [])
            pip_deps = dict(enumerate(conda_deps)).get("pip", [])
            # pylint: disable=deprecated-lambda
            cloudpickle_dep_specs = filter(lambda spec: spec.name == "cloudpickle",
                                           [MatchSpec(dep) for dep in conda_deps + pip_deps])
            for cloudpickle_dep_spec in cloudpickle_dep_specs:
                if not curr_cloudpickle_version_spec.match(cloudpickle_dep_spec):
                    mlflow.pyfunc._logger.warn(
                        "The artifact with name %s is an MLflow model that contains a dependency on"
                        " either a different version or a range of versions of the CloudPickle"
                        " library. MLflow model artifacts should depend on *exactly* the same"
                        " version of CloudPickle that is currently installed. As a result, your new"
                        " model may not load or perform correctly. Current CloudPickle version: %s."
                        " Model CloudPickle version: %s",
                        model_name,
                        curr_cloudpickle_version_spec.version,
                        cloudpickle_dep_spec.version)


def _load_pyfunc(model_path):
    pyfunc_config = _get_flavor_configuration(
            model_path=model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME)

    model_class_subpath = pyfunc_config.get(CONFIG_KEY_MODEL_CLASS, None)
    if model_class_subpath is None:
        raise MlflowException(
            "Model class path was not specified in the model configuration")
    with open(os.path.join(model_path, model_class_subpath), "rb") as f:
        model_class = cloudpickle.load(f)

    parameters = {}
    for saved_parameter_name, saved_parameter_path in\
            pyfunc_config.get(CONFIG_KEY_PARAMETERS, {}).items():
        with open(os.path.join(model_path, saved_parameter_path), "rb") as f:
            parameters[saved_parameter_name] = cloudpickle.load(f)

    directory_managers = []
    if sys.version_info >= (3, 2):
        # Create a managed temporary directory that will exist as long as the manager object
        # returned by `tempfile.TemporaryDirectory` is in scope. This directory will be removed
        # some time after the manager object goes out of scope.
        tmp_artifacts_dir_manager = tempfile.TemporaryDirectory(suffix="artifacts")
        directory_managers.append(tmp_artifacts_dir_manager)
        tmp_artifacts_dir_path = tmp_artifacts_dir_manager.name
    else:
        # Because `tempfile.TemporaryDirectory` does not exist prior to Python 3.2, create an
        # unmanaged temporary directory instead. Depending on the system, this directory is likely
        # to be created in "/var" or "/tmp" and will be removed on system reboot.
        # TODO: If the longevity of the temporary directory prior to Python 3.2 becomes problematic,
        # consider using a alternative solution.
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

    context = PythonModelContext(
        artifacts=artifacts, parameters=parameters, directory_managers=directory_managers)
    return model_class(context=context)

import os
import inspect
import pydoc
import shutil
import yaml
from abc import ABCMeta, abstractmethod
from distutils.version import StrictVersion

import cloudpickle

import mlflow.pyfunc
import mlflow.utils
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.utils.file_utils import TempDir, _copy_file_or_tree

DEFAULT_CONDA_ENV = _mlflow_conda_env(
    additional_conda_deps=[
        "cloudpickle={}".format(cloudpickle.__version__),
    ],
    additional_pip_deps=None,
    additional_conda_channels=None,
)

CONFIG_KEY_ARTIFACTS = "artifacts"
CONFIG_KEY_ARTIFACT_RELATIVE_PATH = "path"
CONFIG_KEY_ARTIFACT_URI = "uri"
CONFIG_KEY_PARAMETERS = "parameters"
CONFIG_KEY_MODEL_CLASS = "model_class"
CONFIG_KEY_MODEL_CLASS_PATH = "path"
CONFIG_KEY_MODEL_CLASS_NAME = "name"


class PythonModel(object):
    """
    Represents a generic Python model that leverages a collection of artifacts and parameters 
    (Python objects) to evaluate inputs and produce API-compatible outputs. By subclassing 
    :class:`~PythonModel`, users can create customized MLflow models with the "python_function" 
    ("pyfunc") flavor, leveraging custom inference logic and dependencies. 
    """

    __metaclass__ = ABCMeta

    def __init__(self, context):
        """
        :param context: A :class:`~PythonModelContext`, instance containing artifacts and parameters
                        that the model can use to perform inference.
        """
        self.context = context

    @abstractmethod
    def predict(self, model_input):
        """
        Evaluates a pyfunc-compatible input and produces a pyfunc-compatible output.
        For more information about the pyfunc input/output API, see the `pyfunc flavor
        documentation <https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html>`_.

        :param model_input: A pyfunc-compatible input for the model to evaluate.
        """
        pass


class PythonModelContext(object):
    """
    A collection of artifacts and parameters that a :class:`~PythonModel` can use when performing 
    inference. :class:`~PythonModelContext` objects are created implicitly by the 
    :func:`save_model() <mlflow.pyfunc.save_model>` and 
    :func:`log_model() <mlflow.pyfunc.log_model>` methods, using the contents specified by the 
    ``artifacts`` and ``parameters`` arguments of these methods.
    """

    def __init__(self, artifacts, parameters):
        self._artifacts = artifacts
        self._parameters = parameters

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


def _save_model(path, model_class, artifacts, parameters, conda_env=None, code_paths=None,
               mlflow_model=Model()):
    """
    :param path: The path to which to save the Python model.
    :param model_class: A ``type`` object referring to a subclass of :class:`~PythonModel`, or the
                        fully-qualified name of such a subclass. ``model_class`` defines
                        how the model is loaded and how it performs inference.
    :param artifacts: A dictionary containing ``<name, artifact_uri>`` entries. Remote artifact URIs
                      will be resolved to absolute filesystem paths, producing a dictionary of
                      ``<name, absolute_path>`` entries. ``model_class`` can reference these 
                      resolved entries as the ``artifacts`` property of the ``context`` attribute.
    :param parameters: A dictionary containing ``<name, python object>`` entries. ``python object``
                       may be any Python object that is serializable with CloudPickle.
                       ``model_class`` can reference these resolved entries as the ``parameters``
                       property of the ``context`` attribute.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in ``mlflow.pyfunc.model.DEFAULT_CONDA_ENV``. If `None`, the default
                      ``mlflow.pyfunc.model.DEFAULT_CONDA_ENV`` environment will be added to the
                      model.
    :param code_paths: A list of paths to Python file dependencies that are required by
                       instances of ``model_class``.
    :param mlflow_model: The model configuration to which to add the ``mlflow.pyfunc`` flavor.
    """
    if os.path.exists(path):
        raise MlflowException(
                message="Path '{}' already exists".format(path),
                error_code=INVALID_PARAMETER_VALUE)
    os.makedirs(path)

    saved_artifacts_config = {}
    with TempDir() as tmp_artifacts_dir:
        tmp_artifacts_config = {}
        saved_artifacts_dir_subpath = "artifacts"
        for artifact_name, artifact_uri in artifacts.items():
            tmp_artifact_path = tmp_artifacts_dir.path(saved_artifacts_dir_subpath, artifact_name)
            os.makedirs(tmp_artifact_path)
            tmp_artifact_path = _download_artifact_from_uri(
                artifact_uri=artifact_uri, output_path=tmp_artifact_path)
            tmp_artifacts_config[artifact_name] = tmp_artifact_path
            saved_artifact_subpath = os.path.relpath(
                    path=tmp_artifact_path, start=tmp_artifacts_dir.path())
            saved_artifacts_config[artifact_name] = {
                CONFIG_KEY_ARTIFACT_RELATIVE_PATH: saved_artifact_subpath,
                CONFIG_KEY_ARTIFACT_URI: artifact_uri,
            }

        _validate_artifacts(tmp_artifacts_config)
        shutil.move(tmp_artifacts_dir.path(saved_artifacts_dir_subpath),
                    os.path.join(path, saved_artifacts_dir_subpath))

    saved_parameters_dir_subpath = "parameters"
    os.makedirs(os.path.join(path, saved_parameters_dir_subpath))
    saved_parameters = {}
    for parameter_name, parameter_py_obj in parameters.items():
        saved_parameter_subpath = os.path.join(
            saved_parameters_dir_subpath, "{param_name}.pkl".format(param_name=parameter_name))
        with open(os.path.join(path, saved_parameter_subpath), "wb") as out:
            cloudpickle.dump(parameter_py_obj, out)
        saved_parameters[parameter_name] = saved_parameter_subpath

    saved_model_class = {}
    if inspect.isclass(model_class):
        saved_model_class_subpath = "model_class.pkl"
        with open(os.path.join(path, saved_model_class_subpath), "wb") as out:
            cloudpickle.dump(model_class, out)
        saved_model_class[CONFIG_KEY_MODEL_CLASS_PATH] = saved_model_class_subpath
    else:
        saved_model_class[CONFIG_KEY_MODEL_CLASS_NAME] = model_class

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

    model_kwargs = {
        CONFIG_KEY_ARTIFACTS: saved_artifacts_config,
        CONFIG_KEY_PARAMETERS: saved_parameters,
        CONFIG_KEY_MODEL_CLASS: saved_model_class,
    }
    mlflow.pyfunc.add_to_model(model=mlflow_model, loader_module=__name__, code=saved_code_subpath,
                               env=conda_env_subpath, **model_kwargs)
    mlflow_model.save(os.path.join(path, 'MLmodel'))


def _validate_artifacts(artifacts):
    from conda.resolve import MatchSpec

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
                    "The artifact with name `{artifact_name}` is an MLflow model that was"
                    " saved with a different major version of Python. As a result, your new model"
                    " may not load or perform correctly. Current python version:"
                    " `{curr_py_version}`. Model python version: `{model_py_version}`.".format(
                        artifact_name=model_name,
                        curr_py_version=mlflow.utils.PYTHON_VERSION,
                        model_py_version=model_py_version))


        conda_env_subpath = pyfunc_conf.get(mlflow.pyfunc.ENV, None)
        if conda_env_subpath is not None:
            with open(os.path.join(model_path, conda_env_subpath), "r") as f:
                conda_env = yaml.safe_load(f)

            conda_deps = conda_env.get("dependencies", [])
            pip_deps = dict(enumerate(conda_deps)).get("pip", [])
            cloudpickle_dep_specs = filter(lambda spec : spec.name == "cloudpickle",
                                           [MatchSpec(dep) for dep in conda_deps + pip_deps])
            for cloudpickle_dep_spec in cloudpickle_dep_specs:
                if not curr_cloudpickle_version_spec.match(cloudpickle_dep_spec):
                    mlflow.pyfunc._logger.warn(
                        "The artifact with name `{artifact_name}` is an MLflow model that contains"
                        " a dependency on either a different version or a range of versions of the"
                        " CloudPickle library. MLflow model artifacts should depend on *exactly*"
                        " the same version of CloudPickle that is currently installed. As a result,"
                        " your new model may not load or perform correctly. Current CloudPickle"
                        " version: `{curr_cloudpickle_version}`. Model CloudPickle version:"
                        " `{model_cloudpickle_version}`.".format(
                            artifact_name=model_name,
                            curr_cloudpickle_version=curr_cloudpickle_version_spec.version,
                            model_cloudpickle_version=cloudpickle_dep_spec.version))


def _load_pyfunc(model_path):
    pyfunc_config = _get_flavor_configuration(
            model_path=model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME)

    model_class = pyfunc_config.get(CONFIG_KEY_MODEL_CLASS, {})
    if len(model_class) != 1:
        raise MlflowException(
            message=(
                "Expected `{model_class_config_name}` configuration to contain a single entry, but"
                " multiple entries were found. Model class configuration:"
                " `{model_class_config}`".format(
                    model_class_config_name=CONFIG_KEY_MODEL_CLASS,
                    model_class_config=model_class)))
    if CONFIG_KEY_MODEL_CLASS_PATH in model_class:
        with open(os.path.join(model_path, model_class[CONFIG_KEY_MODEL_CLASS_PATH]), "rb") as f:
            model_class = cloudpickle.load(f)
    elif CONFIG_KEY_MODEL_CLASS_NAME in model_class:
        model_class = pydoc.locate(model_class[CONFIG_KEY_MODEL_CLASS_NAME])
        if model_class is None:
            raise MlflowException(
                "Unable to locate the model class specified by the configuration with name:"
                " `{model_class_name}`".format(model_class_name=model_class))
    else:
        raise MlflowException(
                message=(
                    "Expected `{model_class_config_name}` configuration to contain either a"
                    " `{model_class_path_key}` or `{model_class_name_key}` key, but neither"
                    " was found. Model class configuration: `{model_class_config}`".format(
                        model_class_config_name=CONFIG_KEY_MODEL_CLASS,
                        model_class_path_key=CONFIG_KEY_MODEL_CLASS_PATH,
                        model_class_name_key=CONFIG_KEY_MODEL_CLASS_NAME,
                        model_class_config=model_class)))

    parameters = {}
    for saved_parameter_name, saved_parameter_path in\
            pyfunc_config.get(CONFIG_KEY_PARAMETERS, {}).items():
        with open(os.path.join(model_path, saved_parameter_path), "rb") as f:
            parameters[saved_parameter_name] = cloudpickle.load(f)

    with TempDir() as tmp:
        artifacts = {}
        tmp_artifacts_dir = tmp.path("artifacts")
        for saved_artifact_name, saved_artifact_info in\
                pyfunc_config.get(CONFIG_KEY_ARTIFACTS, {}).items():
            tmp_artifact_path = os.path.join(
                    tmp_artifacts_dir,
                    _copy_file_or_tree(
                        src=os.path.join(
                            model_path, saved_artifact_info[CONFIG_KEY_ARTIFACT_RELATIVE_PATH]),
                        dst=tmp_artifacts_dir,
                        dst_dir=saved_artifact_name))
            artifacts[saved_artifact_name] = tmp_artifact_path

        context = PythonModelContext(artifacts=artifacts, parameters=parameters)
        return model_class(context=context)

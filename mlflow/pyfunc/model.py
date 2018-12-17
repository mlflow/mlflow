import os
import inspect
import yaml
from abc import ABCMeta, abstractmethod

import cloudpickle

import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
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
CONFIG_KEY_PARAMETERS = "parameters"
CONFIG_KEY_MODEL_CLASS = "model_class"
CONFIG_KEY_MODEL_CLASS_PATH = "path"
CONFIG_KEY_MODEL_CLASS_NAME = "name"


class PythonModel(object):

    __metaclass__ = ABCMeta

    def __init__(self, context):
        self.context = context

    @abstractmethod
    def predict(self, model_input):
        pass


class PythonModelContext(object):

    def __init__(self, artifacts, parameters):
        self._artifacts = artifacts
        self._parameters = parameters

    @property
    def artifacts(self):
        return self._artifacts

    @property
    def parameters(self):
        return self._parameters


def save_model(dst_path, artifacts, parameters, model_class, conda_env=None, code_path=None, 
               model=Model()):
    if os.path.exists(dst_path):
        raise MlflowException(
                message="Path '{}' already exists".format(dst_path),
                error_code=INVALID_PARAMETER_VALUE)
    os.makedirs(dst_path)

    # TODO: Resolve artifacts to absolute paths here

    saved_artifacts_dir_subpath = "artifacts"
    saved_artifacts_dir_path = os.path.join(dst_path, saved_artifacts_dir_subpath)
    os.makedirs(saved_artifacts_dir_path)
    saved_artifacts = {}
    for artifact_name, artifact_path in artifacts.items():
        saved_artifact_subpath = os.path.join(
                saved_artifacts_dir_subpath, 
                _copy_file_or_tree(
                    src=artifact_path, dst=saved_artifacts_dir_path, dst_dir=artifact_name))
        saved_artifacts[artifact_name] = saved_artifact_subpath

    saved_parameters_dir_subpath = "parameters"
    os.makedirs(os.path.join(dst_path, saved_parameters_dir_subpath))
    saved_parameters = {}
    for parameter_name, parameter_py_obj in parameters.items():
        saved_parameter_subpath = os.path.join(
            saved_parameters_dir_subpath, "{param_name}.pkl".format(param_name=parameter_name))
        with open(os.path.join(dst_path, saved_parameter_subpath), "wb") as out:
            cloudpickle.dump(parameter_py_obj, out)
        saved_parameters[parameter_name] = saved_parameter_subpath

    saved_model_class = {}
    if inspect.isclass(model_class):
        saved_model_class_subpath = "model_class.pkl"
        with open(os.path.join(dst_path, saved_model_class_subpath), "wb") as out:
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
    with open(os.path.join(dst_path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    code_subpath = None
    if code_path:
        code_subpath = "code"
        for path in code_path:
            _copy_file_or_tree(src=path, dst=dst_path, dst_dir=code_subpath)

    model_kwargs = {
        CONFIG_KEY_ARTIFACTS: saved_artifacts,
        CONFIG_KEY_PARAMETERS: saved_parameters,
        CONFIG_KEY_MODEL_CLASS: saved_model_class,
    }
    mlflow.pyfunc.add_to_model(model=model, loader_module=__name__, code=code_subpath, 
                              env=conda_env_subpath, **model_kwargs)
    model.save(os.path.join(dst_path, 'MLmodel'))


def _resolve_artifact(artifact_path):
    pass


def _validate_artifacts(artifacts):
    pass


def log_model():
    pass




def _load_pyfunc(model_path):
    pyfunc_config = _get_flavor_configuration(
            model_path=model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME)

    model_class = pyfunc_config.get(CONFIG_KEY_MODEL_CLASS, {})
    if len(model_class) != 1:
        raise MlflowException(
            message=(
                "Expected `{model_class_config_name}` configuration to contain a single entry, but"
                " multiple entries were found: {model_class}".format(
                    model_class_config_name=CONFIG_KEY_MODEL_CLASS,
                    model_class=model_class)))
    elif CONFIG_KEY_MODEL_CLASS_PATH in model_class:
        with open(os.path.join(model_path, model_class[CONFIG_KEY_MODEL_CLASS_PATH]), "rb") as f:
            model_class = cloudpickle.load(f)
    elif CONFIG_KEY_MODEL_CLASS_NAME in model_class:
        raise Exception("Unimplemented")
    else:
        raise MlflowException(
                message=(
                    "Expected `{model_class_config_name}` configuration to contain either a"
                    " `{model_class_path_key}` or `{model_class_name_key}` key, but neither"
                    " was found: {model_class}".format(
                        model_class_config_name=CONFIG_KEY_MODEL_CLASS,
                        model_class_path_key=CONFIG_KEY_MODEL_CLASS_PATH,
                        model_class_name_key=CONFIG_KEY_MODEL_CLASS_NAME,
                        model_class=model_class)))

    parameters = {}
    for saved_parameter_name, saved_parameter_path in\
            pyfunc_config.get(CONFIG_KEY_PARAMETERS, {}).items():
        with open(os.path.join(model_path, saved_parameter_path), "rb") as f:
            parameters[saved_parameter_name] = cloudpickle.load(f)

    with TempDir() as tmp:
        artifacts = {}
        tmp_artifacts_dir = tmp.path("artifacts")
        for saved_artifact_name, saved_artifact_path in\
                pyfunc_config.get(CONFIG_KEY_ARTIFACTS, {}).items():
            tmp_artifact_path = os.path.join(
                    tmp_artifacts_dir,
                    _copy_file_or_tree(
                        src=os.path.join(model_path, saved_artifact_path),
                        dst=tmp_artifacts_dir, 
                        dst_dir=saved_artifact_name))
            artifacts[saved_artifact_name] = tmp_artifact_path

        context = PythonModelContext(artifacts=artifacts, parameters=parameters)
        return model_class(context=context)

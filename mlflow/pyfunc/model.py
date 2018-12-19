import os
import inspect
import shutil
import yaml
from abc import ABCMeta, abstractmethod
from distutils.version import StrictVersion

import cloudpickle

import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking import _get_store
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration 
from mlflow.utils.file_utils import TempDir, _copy_file_or_tree
from mlflow.store.artifact_repo import ArtifactRepository

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

    saved_artifacts_config = {}
    with TempDir() as tmp_artifacts_dir:
        tmp_artifact_paths = []
        saved_artifacts_dir_subpath = "artifacts"
        for artifact_name, artifact_uri in artifacts.items():
            tmp_artifact_path = tmp_artifacts_dir.path(saved_artifacts_dir_subpath, artifact_name) 
            os.makedirs(tmp_artifact_path)
            tmp_artifact_path = _resolve_artifact(
                    artifact_src_uri=artifact_uri, artifact_dst_path=tmp_artifact_path)
            tmp_artifact_paths.append(tmp_artifact_path)
            saved_artifact_subpath = os.path.relpath(
                    path=tmp_artifact_path, start=tmp_artifacts_dir.path())
            saved_artifacts_config[artifact_name] = saved_artifact_subpath

        _validate_artifacts(tmp_artifact_paths)
        shutil.move(tmp_artifacts_dir.path(saved_artifacts_dir_subpath), 
                    os.path.join(dst_path, saved_artifacts_dir_subpath))

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
        CONFIG_KEY_ARTIFACTS: saved_artifacts_config,
        CONFIG_KEY_PARAMETERS: saved_parameters,
        CONFIG_KEY_MODEL_CLASS: saved_model_class,
    }
    mlflow.pyfunc.add_to_model(model=model, loader_module=__name__, code=code_subpath, 
                              env=conda_env_subpath, **model_kwargs)
    model.save(os.path.join(dst_path, 'MLmodel'))


def _resolve_artifact(artifact_src_uri, artifact_dst_path):
    artifact_src_dir = os.path.dirname(artifact_src_uri)
    artifact_src_relative_path = os.path.basename(artifact_src_uri)
    artifact_repo = ArtifactRepository.from_artifact_uri(
            artifact_uri=artifact_src_dir, store=_get_store())
    return artifact_repo.download_artifacts(
            artifact_path=artifact_src_relative_path, dst_path=artifact_dst_path)


def _validate_artifacts(artifacts):
    models = dict([
        (artifact_name, artifact_path) for artifact_name, artifact_path in artifacts.items()
        if os.path.isidr(artifact_path) and "MLmodel" in os.listdir(artifact_path)])
    model_py_major_versions = set()
    model_cloudpickle_versions = set()
    for model_name, model_path in model_paths:
        model_conf = Model.load(os.path.join(model_path, "MLmodel"))
        pyfunc_conf = model_conf.flavors.get(mlflow.pyfunc.FLAVOR_NAME, {})
        
        model_py_version = pyfunc_conf.get(mlflow.pyfunc.PY_VERSION, None)
        if model_py_version is not None:
            model_py_major_version = StrictVersion(model_py_major_version).version[0]

            model_py_major_versions.add(model_py_version.version[0])

        conda_env_subpath = pyfunc_conf.get(mlflow.pyfunc.ENV, None)
        if conda_env_subpath is not None:
            try:
                with open(os.path.join(model_path, conda_env_subpath), "r") as f:
                    conda_env = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print("BAD")

            conda_deps = conda_env.get("dependencies", [])
            pip_deps = dict(enumerate(conda_deps)).get("pip", [])
            cloudpickle_versions = 



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

import logging
import os
import yaml
import json

import pandas as pd

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.autologging_utils import safe_patch, autologging_integration
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _PythonEnv,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import _save_example
from mlflow.models import Model, ModelInputExample
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.model_utils import (
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _add_code_from_conf_to_system_path,
    _validate_and_prepare_target_save_path,
)
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

FLAVOR_NAME = "transformers"
_MODEL_SAVE_PATH = "model.transformers"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at a minimum, contains these requirements.
    """
    return [_get_pinned_requirement("transformers")]


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    model,
    path,
    task=None,
    tokenizer=None,
    feature_extractor=None,
    device=-1,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    """
    import transformers

    if task is None:
        MlflowException(message="logging model with out task name is not supported. ",
                        code=INVALID_PARAMETER_VALUE
                        )

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    model_data_subpath = os.path.join(path, _MODEL_SAVE_PATH)
    os.makedirs(model_data_subpath)
    pipeline = transformers.pipeline(task, model=model, tokenizer=tokenizer, feature_extractor=feature_extractor,
                                     device=device)
    pipeline.save_pretrained(model_data_subpath)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.transformers",
        data=model_data_subpath,
        env=_CONDA_ENV_FILE_NAME,
        code=code_dir_subpath,
    )

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        transformers_version=transformers.__version__,
        data=model_data_subpath,
        task=task,
        code=code_dir_subpath
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                model_data_subpath,
                FLAVOR_NAME,
                fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    model,
    artifact_path,
    task=None,
    tokenizer=None,
    feature_extractor=None,
    device=-1,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs,
):
    """
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.transformers,
        registered_model_name=registered_model_name,
        model=model,
        task=task,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        device=device,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        **kwargs,
    )


def _load_model(path, conf):
    import transformers
    try:
        task = conf.get("task")
        path = os.path.abspath(path)
        return transformers.pipeline(task, path)
    except KeyError:
        raise MlflowException(message="Loading model without task name is not supported",
                              code=INVALID_PARAMETER_VALUE)


class _TransformersPipelineWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, dataframe):
        if len(dataframe.columns) != 1:
            raise MlflowException("Shape of input dataframe must be (n_rows, 1column)")

        return pd.DataFrame(
            {"predictions": dataframe.iloc[:, 0].apply(lambda text: self.pipeline(text))}
        )


def _load_pyfunc(path, **kwargs):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.

    :param path: Local filesystem path to the MLflow Model with the ``transformers`` flavor.
    """
    flavor_conf = _get_flavor_configuration(model_path=os.path.dirname(path), flavor_name=FLAVOR_NAME)
    return _TransformersPipelineWrapper(_load_model(path, flavor_conf))


def load_model(model_uri, dst_path=None):
    """

    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    transformers_model_file_path = os.path.join(local_model_path, _MODEL_SAVE_PATH)
    return _load_model(path=transformers_model_file_path, conf=flavor_conf)


@autologging_integration(FLAVOR_NAME)
def autolog(
    task=None,
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    registered_model_name=None,
):  # pylint: disable=unused-argument

    if task is None:
        raise ("Task name is required for logging transformers model", 400)
    import transformers.trainer as transformer_trainer
    from mlflow.transformers._transformers_autolog import patched_train

    safe_patch(FLAVOR_NAME, transformer_trainer.Trainer, "train", patched_train, manage_run=True)

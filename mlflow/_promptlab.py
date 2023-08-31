# loader/gateway_loader_module.py
import os
from string import Template
from typing import List

import pandas as pd
import yaml

import mlflow.gateway
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME, Model
from mlflow.models.utils import _save_example
from mlflow.pyfunc.model import PythonModel
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

mlflow.gateway.set_gateway_uri("databricks")


class _PromptlabModel(PythonModel):
    def __init__(self):
        self.santized_prompt_template = ""
        self.prompt_parameters = {}
        self.python_parameters = {}
        self.model_route = ""

        self.prompt_template = Template(self.santized_prompt_template)

    def predict(self, inputs: pd.DataFrame) -> List[str]:
        results = []
        for idx in inputs.index:
            python_inputs = {
                param.key: inputs["{param.key}"][idx] for param in self.prompt_parameters
            }
            prompt = self.prompt_template.substitute(python_inputs)
            result = mlflow.gateway.query(
                route=self.model_route,
                data={
                    {
                        "prompt": prompt,
                    }.update(self.python_parameters),
                },
            )
            results.append(result["candidates"][0]["text"])
        return results


def save_model(
    promptlab_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature=None,
    input_example=None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    model_data_subpath = "model.pkl"
    model_data_path = os.path.join(path, model_data_subpath)
    _save_model(promptlab_model, model_data_path)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="_promptlab",
        model_path=model_data_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )

    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements(True)
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path, "_promptlab", fallback=default_reqs
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs, pip_requirements, extra_pip_requirements
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


def _save_model(model, path):
    with open(path, "wb") as out:
        import cloudpickle

        cloudpickle.dump(model, out)


def get_default_pip_requirements(include_cloudpickle=False):
    pip_deps = [_get_pinned_requirement("sktime")]
    if include_cloudpickle:
        pip_deps += [_get_pinned_requirement("cloudpickle")]

    return pip_deps


def get_default_conda_env(include_cloudpickle=False):
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements(include_cloudpickle))

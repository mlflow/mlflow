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
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

mlflow.gateway.set_gateway_uri("databricks")


class _PromptlabModel(PythonModel):
    def __init__(self, prompt_template, prompt_parameters, model_parameters, model_route):
        self.prompt_parameters = prompt_parameters
        self.model_parameters = model_parameters
        self.model_route = model_route
        self.prompt_template = Template(prompt_template)

    def predict(self, inputs: pd.DataFrame) -> List[str]:
        results = []
        for idx in inputs.index:
            prompt_parameters_as_dict = {
                param.key: inputs["{param.key}"][idx] for param in self.prompt_parameters
            }
            prompt = self.prompt_template.substitute(prompt_parameters_as_dict)
            model_parameters_as_dict = {param.key: param.value for param in self.model_parameters}
            result = mlflow.gateway.query(
                route=self.model_route,
                data={
                    {
                        "prompt": prompt,
                    }.update(model_parameters_as_dict),
                },
            )
            results.append(result["candidates"][0]["text"])
        return results


def _load_pyfunc(path):
    from mlflow.entities.param import Param

    pyfunc_flavor_conf = _get_flavor_configuration(model_path=path, flavor_name=pyfunc.FLAVOR_NAME)
    parameters_path = os.path.join(path, pyfunc_flavor_conf["parameters_path"])
    with open(parameters_path) as f:
        parameters = yaml.safe_load(f)

        prompt_parameters_as_params = [
            Param(key=key, value=value) for key, value in parameters["prompt_parameters"].items()
        ]
        model_parameters_as_params = [
            Param(key=key, value=value) for key, value in parameters["model_parameters"].items()
        ]

        return _PromptlabModel(
            prompt_template=parameters["prompt_template"],
            prompt_parameters=prompt_parameters_as_params,
            model_parameters=model_parameters_as_params,
            model_route=parameters["model_route"],
        )


def save_model(
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature=None,
    input_example=None,
    pip_requirements=None,
    prompt_template=None,
    prompt_parameters=None,
    model_parameters=None,
    model_route=None,
):
    _validate_env_arguments(conda_env, pip_requirements, None)

    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    parameters_sub_path = "parameters.yaml"
    parameters_path = os.path.join(path, parameters_sub_path)
    # dump prompt_template, prompt_parameters, model_parameters, model_route to parameters_path

    parameters = {
        "prompt_template": prompt_template,
        "prompt_parameters": {param.key: param.value for param in prompt_parameters},
        "model_parameters": {param.key: param.value for param in model_parameters},
        "model_route": model_route,
    }
    with open(parameters_path, "w") as f:
        yaml.safe_dump(parameters, stream=f, default_flow_style=False)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow._promptlab",
        parameters_path=parameters_sub_path,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )

    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements(True)
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path, "mlflow._promptlab", fallback=default_reqs
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs, pip_requirements, None
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


def get_default_pip_requirements(include_cloudpickle=False):
    pip_deps = [_get_pinned_requirement("sktime")]
    if include_cloudpickle:
        pip_deps += [_get_pinned_requirement("cloudpickle")]

    return pip_deps


def get_default_conda_env(include_cloudpickle=False):
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements(include_cloudpickle))

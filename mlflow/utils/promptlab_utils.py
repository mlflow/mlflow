import json
from typing import List

from mlflow.entities import Param


def create_conda_yaml_file(mlflow_version: str) -> str:
    conda_yaml = f"""\
name: mlflow-env
channels:
  - conda-forge
dependencies:
  - pip
  - pip:
    - mlflow[gateway]=={mlflow_version}
"""
    return conda_yaml.strip()


def create_python_env_file() -> str:
    python_yaml = """\
dependencies:
    - -r requirements.txt
    """
    return python_yaml.strip()


def create_requirements_txt_file(mlflow_version: str) -> str:
    requirements_content = f"""\
mlflow[gateway]=={mlflow_version}
    """
    return requirements_content.strip()


def create_model_file(
    run_uuid: str,
    mlflow_version: str,
    prompt_parameters: List[Param],
    model_uuid: str,
    utc_time_created: str,
) -> str:
    inputs_signature = [{"name": param.key, "type": "string"} for param in prompt_parameters]

    outputs_signature = [{"name": "output", "type": "string"}]

    model_file = {
        "artifact_path": "model",
        "flavors": {
            "python_function": {
                "env": {"conda": "conda.yaml", "virtualenv": "python_env.yaml"},
                "loader_module": "gateway_loader_module",
                "code": "loader",
            }
        },
        "mlflow_version": mlflow_version,
        "model_uuid": model_uuid,
        "run_id": run_uuid,
        "utc_time_created": utc_time_created,
        "metadata": {"mlflow_uses_gateway": "true"},
        "saved_input_example_info": {
            "artifact_path": "input_example.json",
            "type": "dataframe",
            "pandas_orient": "split",
        },
        "signature": {
            "inputs": json.dumps(inputs_signature),
            "outputs": json.dumps(outputs_signature),
        },
    }

    model_json = json.dumps(model_file)
    return model_json


def create_input_example_file(prompt_parameters: List[Param]) -> str:
    input_example = {"inputs": [param.value for param in prompt_parameters]}
    input_example_json = json.dumps(input_example)
    return input_example_json


def create_loader_file(prompt_parameters, prompt_template, model_parameters, model_route):
    python_inputs = ",\n\t\t\t\t".join(
        [f"{param.key}=inputs['{param.key}'][idx]" for param in prompt_parameters]
    )

    # Replace {{parameter}} with $parameter in the prompt template
    python_template = prompt_template.replace("{{", "$").replace("}}", "")

    # Escape triple quotes in the prompt template
    sanitized_python_template = python_template.replace('"""', '"""')

    python_parameters = ",\n\t\t\t\t\t".join(
        [f'"{param.key}": {param.value}' for param in model_parameters]
    )

    with open("mlflow/utils/promptlab_loader_module_template.txt") as loader_module_template:
        loader_module_text = loader_module_template.read()
        loader_module_text = loader_module_text.replace(
            "__sanitized_python_template__", sanitized_python_template
        )
        loader_module_text = loader_module_text.replace("__model_route__", model_route)
        loader_module_text = loader_module_text.replace("__python_inputs__", python_inputs)
        loader_module_text = loader_module_text.replace("__python_parameters__", python_parameters)
        return loader_module_text.strip()


def create_eval_results_file(prompt_parameters, model_input, model_output_parameters, model_output):
    columns = [param.key for param in prompt_parameters] + ["prompt", "output"]
    data = [param.value for param in prompt_parameters] + [model_input, model_output]

    updated_columns = columns + [param.key for param in model_output_parameters]
    updated_data = data + [param.value for param in model_output_parameters]

    eval_results = {"columns": updated_columns, "data": [updated_data]}

    eval_results_json = json.dumps(eval_results)
    return eval_results_json

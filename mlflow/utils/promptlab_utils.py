import json
from datetime import datetime
from typing import List

from mlflow.entities import Param


def create_conda_yaml_file(mlflow_version: str) -> str:
    conda_yaml = f"""
    name: mlflow-env
    channels:
      - defaults
    dependencies:
      - pip
      - pip:
        - mlflow[gateway]=={mlflow_version}
    """
    return conda_yaml.strip()


def create_python_env_file() -> str:
    python_yaml = """
    dependencies:
      - -r requirements.txt
    """
    return python_yaml.strip()


def create_requirements_txt_file(mlflow_version: str) -> str:
    requirements_content = f"""
    mlflow[gateway]=={mlflow_version}
    """
    return requirements_content.strip()


def create_model_file(
    run_uuid: str, mlflow_version: str, prompt_parameters: List[Param], model_uuid: str
) -> str:
    utc_time_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

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
    sanitized_python_template = python_template.replace('"""', '\\"""')

    python_parameters = ",\n\t\t\t\t\t".join(
        [f'"{param.key}": {param.value}' for param in model_parameters]
    )

    loader_module_text = f"""
# loader/gateway_loader_module.py
from typing import List, Dict
from string import Template
import pandas as pd
import mlflow.gateway

mlflow.gateway.set_gateway_uri("databricks")

class GatewayModel:
    def __init__(self, model_path):
        self.prompt_template = Template(\"\""
{sanitized_python_template}\n\""")

    def predict(self, inputs: pd.DataFrame) -> List[str]:
        results = []
        for idx in inputs.index:
            prompt = self.prompt_template.substitute(
                {python_inputs}
            )
            result = mlflow.gateway.query(
                route="{model_route}",
                data={{
                    "prompt": prompt,
                    {python_parameters}
                }}
            )
            results.append(result["candidates"][0]["text"])
        return results

def _load_pyfunc(model_path):
    return GatewayModel(model_path)
    """
    return loader_module_text.strip()


def create_eval_results_file(prompt_parameters, model_input, model_output_parameters, model_output):
    columns = [param.key for param in prompt_parameters] + ["prompt", "output"]
    data = [param.value for param in prompt_parameters] + [model_input, model_output]

    updated_columns = columns + [param.key for param in model_output_parameters]
    updated_data = data + [param.value for param in model_output_parameters]

    eval_results = {"columns": updated_columns, "data": [updated_data]}

    eval_results_json = json.dumps(eval_results)
    return eval_results_json

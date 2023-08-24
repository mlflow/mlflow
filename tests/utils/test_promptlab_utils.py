import json

from mlflow.entities import Param
from mlflow.utils.promptlab_utils import (
    create_conda_yaml_file,
    create_eval_results_file,
    create_input_example_file,
    create_loader_file,
    create_model_file,
    create_python_env_file,
    create_requirements_txt_file,
)

mlflow_version = "1.0.0"
run_uuid = "123"
model_uuid = "456"
prompt_parameters = [
    Param(key="question", value="my_question"),
    Param(key="context", value="my_context"),
]
prompt_template = "answer this question: {{question}} using the following context: {{context}}"
model_parameters = [
    Param(key="temperature", value="0.5"),
    Param(key="max_tokens", value="100"),
]
model_route = "gpt4"
model_input = "answer this question: my_question using the following context: my_context"
model_output = "my_answer"
model_output_parameters = [
    Param(key="tokens", value="10"),
    Param(key="latency", value="100"),
]


def test_conda_yaml():
    conda_yaml = create_conda_yaml_file(mlflow_version)
    expected_conda_yaml = """
    name: mlflow-env
    channels:
      - defaults
    dependencies:
      - pip
      - pip:
        - mlflow[gateway]==1.0.0
    """
    assert conda_yaml.strip() == expected_conda_yaml.strip()


def test_python_yaml():
    python_yaml = create_python_env_file()
    expected_python_yaml = """
    dependencies:
      - -r requirements.txt
    """
    assert python_yaml.strip() == expected_python_yaml.strip()


def test_requirements_txt():
    requirements_txt = create_requirements_txt_file(mlflow_version)
    expected_requirements_txt = """
    mlflow[gateway]==1.0.0
    """
    assert requirements_txt.strip() == expected_requirements_txt.strip()


def test_create_model_file():
    utc_time_created = "2023-08-22 18:37:14.724592"
    model_file = create_model_file(
        run_uuid, mlflow_version, prompt_parameters, model_uuid, utc_time_created
    )
    model_json = json.loads(model_file)

    expected_model_json = {
        "artifact_path": "model",
        "flavors": {
            "python_function": {
                "env": {"conda": "conda.yaml", "virtualenv": "python_env.yaml"},
                "loader_module": "gateway_loader_module",
                "code": "loader",
            }
        },
        "mlflow_version": "1.0.0",
        "model_uuid": "456",
        "run_id": "123",
        "utc_time_created": utc_time_created,
        "metadata": {"mlflow_uses_gateway": "true"},
        "saved_input_example_info": {
            "artifact_path": "input_example.json",
            "type": "dataframe",
            "pandas_orient": "split",
        },
        "signature": {
            "inputs": '[{"name": "question", "type": "string"}, '
            '{"name": "context", "type": "string"}]',
            "outputs": '[{"name": "output", "type": "string"}]',
        },
    }

    # Drop the utc_time_created field since it is dynamic
    # and model_uuid since it is random
    model_json_to_compare = {
        k: v for k, v in model_json.items() if k not in ["utc_time_created", "model_uuid"]
    }
    expected_model_to_compare = {
        k: v for k, v in expected_model_json.items() if k not in ["utc_time_created", "model_uuid"]
    }
    assert model_json_to_compare == expected_model_to_compare


def test_loader_module():
    loader_file = create_loader_file(
        prompt_parameters, prompt_template, model_parameters, model_route
    )
    expected_loader_file = """
# loader/gateway_loader_module.py
from typing import List, Dict
from string import Template
import pandas as pd
import mlflow.gateway

mlflow.gateway.set_gateway_uri("databricks")

class GatewayModel:
    def __init__(self, model_path):
        self.prompt_template = Template(\"\"\"
answer this question: $question using the following context: $context
\"\"\")

    def predict(self, inputs: pd.DataFrame) -> List[str]:
        results = []
        for idx in inputs.index:
            prompt = self.prompt_template.substitute(
                question=inputs['question'][idx],
                                context=inputs['context'][idx]
            )
            result = mlflow.gateway.query(
                route="gpt4",
                data={
                    "prompt": prompt,
                    "temperature": 0.5,
                                        "max_tokens": 100
                }
            )
            results.append(result["candidates"][0]["text"])
        return results

def _load_pyfunc(model_path):
    return GatewayModel(model_path)
    """
    generated_lines = loader_file.strip().split("\n")
    expected_lines = expected_loader_file.strip().split("\n")

    assert len(generated_lines) == len(expected_lines)

    for g, e in zip(generated_lines, expected_lines):
        assert g.strip() == e.strip()


def test_eval_results_file():
    eval_results_file = create_eval_results_file(
        prompt_parameters, model_input, model_output_parameters, model_output
    )
    expected_eval_results_json = {
        "columns": ["question", "context", "prompt", "output", "tokens", "latency"],
        "data": [
            [
                "my_question",
                "my_context",
                "answer this question: my_question using the following context: my_context",
                "my_answer",
                "10",
                "100",
            ]
        ],
    }
    assert json.loads(eval_results_file) == expected_eval_results_json


def test_input_example_file():
    input_example_file = create_input_example_file(prompt_parameters)
    expected_input_example_json = {"inputs": ["my_question", "my_context"]}
    assert json.loads(input_example_file) == expected_input_example_json

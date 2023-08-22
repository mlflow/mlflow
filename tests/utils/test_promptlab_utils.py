import json
from mlflow.entities import Param
from mlflow.utils.promptlab_utils import (
    create_model_file,
    create_conda_yaml_file,
    create_python_env_file,
    create_requirements_txt_file,
    create_loader_file,
    create_eval_results_file,
    create_input_example_file,
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
    model_file = create_model_file(run_uuid, mlflow_version, prompt_parameters, model_uuid)
    expected_model_file = """
    { ... } # This contains the expected JSON output 
    """

    model_json = json.loads(model_file)
    expected_model_json = json.loads(expected_model_file)

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
    ... # This contains the expected Python code 
    """
    assert loader_file.strip() == expected_loader_file.strip()


def test_eval_results_file():
    eval_results_file = create_eval_results_file(
        prompt_parameters, model_input, model_output_parameters, model_output
    )
    expected_eval_results_file = """
    { ... } # This contains the expected JSON output
    """
    assert json.loads(eval_results_file) == json.loads(expected_eval_results_file)


def test_input_example_file():
    input_example_file = create_input_example_file(prompt_parameters)
    expected_input_example_file = """
    { ... } # This contains the expected JSON output
    """
    assert json.loads(input_example_file) == json.loads(expected_input_example_file)

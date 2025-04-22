import os
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.models import ModelConfig

dir_path = os.path.dirname(os.path.abspath(__file__))
VALID_CONFIG_PATH = os.path.join(dir_path, "configs/config.yaml")
VALID_CONFIG_PATH_2 = os.path.join(dir_path, "configs/config_2.yaml")


def test_config_not_set():
    with pytest.raises(
        FileNotFoundError, match="Config file is not provided which is needed to load the model."
    ):
        ModelConfig()


def test_config_not_found():
    with pytest.raises(FileNotFoundError, match="Config file 'nonexistent.yaml' not found."):
        ModelConfig(development_config="nonexistent.yaml")


def test_config_invalid_yaml(tmp_path):
    tmp_file = tmp_path / "invalid_config.yaml"
    tmp_file.write_text("invalid_yaml: \n  - this is not valid \n-yaml")
    config = ModelConfig(development_config=str(tmp_file))
    with pytest.raises(MlflowException, match="Error parsing YAML file: "):
        config.get("key")


def test_config_key_not_found():
    config = ModelConfig(development_config=VALID_CONFIG_PATH)
    with pytest.raises(KeyError, match="Key 'key' not found in configuration: "):
        config.get("key")


def test_config_setup_correctly():
    config = ModelConfig(development_config=VALID_CONFIG_PATH)
    assert config.get("llm_parameters").get("temperature") == 0.01


@mock.patch("mlflow.models.model_config.__mlflow_model_config__", new=VALID_CONFIG_PATH)
def test_config_setup_correctly_with_mlflow_langchain():
    config = ModelConfig(development_config="nonexistent.yaml")
    assert config.get("llm_parameters").get("temperature") == 0.01


@mock.patch("mlflow.models.model_config.__mlflow_model_config__", new=VALID_CONFIG_PATH_2)
def test_config_setup_with_mlflow_langchain_path():
    # here the config.yaml has the max_tokens set to 500
    # where as the config_2.yaml has it set to 200.
    # Here we give preference to the __mlflow_model_config__.
    config = ModelConfig(development_config=VALID_CONFIG_PATH)
    assert config.get("llm_parameters").get("max_tokens") == 200


def test_config_development_config_must_be_specified_with_keyword():
    with pytest.raises(TypeError, match="1 positional argument but 2 were given"):
        ModelConfig(VALID_CONFIG_PATH_2)


def test_config_development_config_is_a_dict():
    config = ModelConfig(development_config={"llm_parameters": {"temperature": 0.01}})
    assert config.get("llm_parameters").get("temperature") == 0.01


@mock.patch("mlflow.models.model_config.__mlflow_model_config__", new="")
def test_config_setup_correctly_errors_with_no_config_path():
    with pytest.raises(
        FileNotFoundError, match="Config file is not provided which is needed to load the model."
    ):
        ModelConfig(development_config=VALID_CONFIG_PATH)


def test_config_development_config_to_dict():
    config = ModelConfig(development_config={"llm_parameters": {"temperature": 0.01}})
    assert config.to_dict() == {"llm_parameters": {"temperature": 0.01}}

    config = ModelConfig(development_config=VALID_CONFIG_PATH)
    assert config.to_dict() == {
        "embedding_model_query_instructions": "Represent this sentence for searching "
        "relevant passages:",
        "llm_model": "databricks-dbrx-instruct",
        "llm_prompt_template": "You are a trustful assistant.",
        "retriever_config": {"k": 5, "use_mmr": False},
        "llm_parameters": {"temperature": 0.01, "max_tokens": 500},
        "llm_prompt_template_variables": ["chat_history", "context", "question"],
    }

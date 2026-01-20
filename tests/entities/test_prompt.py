import json

import pytest
from pydantic import ValidationError

from mlflow.entities.model_registry import PromptModelConfig
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.prompt_version import (
    IS_PROMPT_TAG_KEY,
    PROMPT_TEXT_TAG_KEY,
    PromptVersion,
)
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import PROMPT_EXPERIMENT_IDS_TAG_KEY, PROMPT_MODEL_CONFIG_TAG_KEY
from mlflow.prompt.registry_utils import model_version_to_prompt_version
from mlflow.protos.model_registry_pb2 import ModelVersionTag


def test_prompt_initialization():
    prompt = PromptVersion(name="my_prompt", version=1, template="Hello, {{name}}!")
    assert prompt.name == "my_prompt"
    assert prompt.version == 1
    assert prompt.template == "Hello, {{name}}!"
    assert prompt.uri == "prompts:/my_prompt/1"
    # Public property should not return the reserved tag
    assert prompt.tags == {}
    assert prompt._tags[IS_PROMPT_TAG_KEY] == "true"
    assert prompt._tags[PROMPT_TEXT_TAG_KEY] == "Hello, {{name}}!"


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("Hello, {{name}}!", {"name"}),
        ("Hello, {{ title }} {{ name }}!", {"title", "name"}),
        ("Hello, {{ person.name.first }}", {"person.name.first"}),
        ("Hello, {{name1}}", {"name1"}),
        # Invalid variables will be ignored
        ("Hello, {name}", set()),
        ("Hello, {{123name}}", set()),
    ],
)
def test_prompt_variables_extraction(template, expected):
    prompt = PromptVersion(name="test", version=1, template=template)
    assert prompt.variables == expected


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("Hello, {{name}}!", "Hello, {name}!"),
        ("Hello, {{ title }} {{ name }}!", "Hello, {title} {name}!"),
        ("Hello, {{ person.name.first }}", "Hello, {person.name.first}"),
        ("Hello, {{name1}}", "Hello, {name1}"),
        ("Hello, {name}", "Hello, {name}"),
    ],
)
def test_prompt_to_single_brace_format(template, expected):
    prompt = PromptVersion(name="test", version=1, template=template)
    assert prompt.to_single_brace_format() == expected


def test_prompt_format():
    prompt = PromptVersion(name="test", version=1, template="Hello, {{title}} {{name}}!")
    result = prompt.format(title="Ms.", name="Alice")
    assert result == "Hello, Ms. Alice!"

    # By default, missing variables raise an error
    with pytest.raises(MlflowException, match="Missing variables: {'name'}"):
        prompt.format(title="Ms.")

    # Partial formatting
    result = prompt.format(title="Ms.", allow_partial=True)
    assert result.template == "Hello, Ms. {{name}}!"
    assert result.variables == {"name"}

    # Non-string values
    result = prompt.format(title="Ms.", allow_partial=True)
    result = prompt.format(title=1, name=True)
    assert result == "Hello, 1 True!"


@pytest.mark.parametrize(
    ("path_value", "unicode_value", "expected"),
    [
        ("C:\\Users\\test\\file.txt", "test", "Path: C:\\Users\\test\\file.txt, Unicode: test"),
        ("test", "\\u0041\\u0042", "Path: test, Unicode: \\u0041\\u0042"),
        ("line1\nline2", "test", "Path: line1\nline2, Unicode: test"),
        ("test[0-9]+", "test", "Path: test[0-9]+, Unicode: test"),
        (
            "C:\\Users\\test[0-9]+\\file.txt",
            "\\u0041\\u0042",
            "Path: C:\\Users\\test[0-9]+\\file.txt, Unicode: \\u0041\\u0042",
        ),
        ('test"quoted"', "test", 'Path: test"quoted", Unicode: test'),
        ("test(1)", "test", "Path: test(1), Unicode: test"),
        ("$100", "test", "Path: $100, Unicode: test"),
    ],
)
def test_prompt_format_backslash_escape(path_value: str, unicode_value: str, expected: str):
    prompt = PromptVersion(name="test", version=1, template="Path: {{path}}, Unicode: {{unicode}}")
    result = prompt.format(path=path_value, unicode=unicode_value)
    assert result == expected


@pytest.mark.parametrize(
    ("style", "question", "expected_content"),
    [
        ("helpful", "What is C:\\Users\\test?", "What is C:\\Users\\test?"),
        ("helpful", "Unicode: \\u0041\\u0042", "Unicode: \\u0041\\u0042"),
        ("friendly", "Line 1\nLine 2", "Line 1\nLine 2"),
        ("professional", "Pattern: [0-9]+", "Pattern: [0-9]+"),
        (
            "expert",
            "Path: C:\\Users\\test[0-9]+\\file.txt",
            "Path: C:\\Users\\test[0-9]+\\file.txt",
        ),
        ("casual", 'He said "Hello"', 'He said "Hello"'),
    ],
)
def test_prompt_format_chat_backslash_escape(style: str, question: str, expected_content: str):
    chat_template = [
        {"role": "system", "content": "You are a {{style}} assistant."},
        {"role": "user", "content": "{{question}}"},
    ]
    prompt = PromptVersion(name="test", version=1, template=chat_template)

    result = prompt.format(style=style, question=question)
    expected = [
        {"role": "system", "content": f"You are a {style} assistant."},
        {"role": "user", "content": expected_content},
    ]
    assert result == expected


def test_prompt_from_model_version():
    model_config_json = json.dumps({"model_name": "gpt-5", "temperature": 0.7, "max_tokens": 1000})
    model_version = ModelVersion(
        name="my-prompt",
        version=1,
        description="test",
        creation_timestamp=123,
        tags=[
            ModelVersionTag(key=IS_PROMPT_TAG_KEY, value="true"),
            ModelVersionTag(key=PROMPT_TEXT_TAG_KEY, value="Hello, {{name}}!"),
            ModelVersionTag(key=PROMPT_MODEL_CONFIG_TAG_KEY, value=model_config_json),
        ],
        aliases=["alias"],
    )

    prompt = model_version_to_prompt_version(model_version)
    assert prompt.name == "my-prompt"
    assert prompt.version == 1
    assert prompt.description == "test"
    assert prompt.creation_timestamp == 123
    assert prompt.template == "Hello, {{name}}!"
    assert prompt.model_config == {"model_name": "gpt-5", "temperature": 0.7, "max_tokens": 1000}
    assert prompt.tags == {}
    assert prompt.aliases == ["alias"]

    invalid_model_version = ModelVersion(
        name="my-prompt",
        version=1,
        creation_timestamp=123,
        # Missing the is_prompt tag
        tags=[ModelVersionTag(key=PROMPT_TEXT_TAG_KEY, value="Hello, {{name}}!")],
    )

    with pytest.raises(MlflowException, match="Name `my-prompt` is registered as a model"):
        model_version_to_prompt_version(invalid_model_version)

    invalid_model_version = ModelVersion(
        name="my-prompt",
        version=1,
        creation_timestamp=123,
        # Missing the prompt text tag
        tags=[ModelVersionTag(key=IS_PROMPT_TAG_KEY, value="true")],
    )

    with pytest.raises(MlflowException, match="Prompt `my-prompt` does not contain a prompt text"):
        model_version_to_prompt_version(invalid_model_version)


def test_model_version_to_prompt_version_merges_prompt_tags():
    model_version = ModelVersion(
        name="my-prompt",
        version=1,
        description="test",
        creation_timestamp=123,
        tags=[
            ModelVersionTag(key=IS_PROMPT_TAG_KEY, value="true"),
            ModelVersionTag(key=PROMPT_TEXT_TAG_KEY, value="Hello, {{name}}!"),
        ],
    )

    prompt_tags = {
        PROMPT_EXPERIMENT_IDS_TAG_KEY: ",exp1,exp2,",
    }

    prompt = model_version_to_prompt_version(model_version, prompt_tags=prompt_tags)

    assert prompt._tags[PROMPT_EXPERIMENT_IDS_TAG_KEY] == ",exp1,exp2,"
    assert prompt._tags[IS_PROMPT_TAG_KEY] == "true"
    assert prompt._tags[PROMPT_TEXT_TAG_KEY] == "Hello, {{name}}!"


def test_prompt_with_model_config_dict():
    model_config = {
        "model_name": "gpt-5",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 1000,
    }
    prompt = PromptVersion(
        name="my_prompt",
        version=1,
        template="Hello, {{name}}!",
        model_config=model_config,
    )
    assert prompt.model_config == model_config

    # Test prompt without model_config
    prompt_without_config = PromptVersion(name="my_prompt", version=2, template="Hello, {{name}}!")
    assert prompt_without_config.model_config is None


def test_prompt_with_model_config_instance():
    config = PromptModelConfig(model_name="gpt-5", temperature=0.7, max_tokens=1000)
    prompt = PromptVersion(
        name="my_prompt",
        version=1,
        template="Hello, {{name}}!",
        model_config=config,
    )
    # Should be stored as dict
    assert prompt.model_config == {"model_name": "gpt-5", "temperature": 0.7, "max_tokens": 1000}


def test_prompt_with_model_config_instance_and_extra_params():
    config = PromptModelConfig(
        model_name="gpt-5",
        temperature=0.5,
        extra_params={"anthropic_version": "2023-06-01", "custom": "value"},
    )
    prompt = PromptVersion(
        name="my_prompt",
        version=1,
        template="Hello, {{name}}!",
        model_config=config,
    )
    # extra_params should be merged at top level
    assert prompt.model_config == {
        "model_name": "gpt-5",
        "temperature": 0.5,
        "anthropic_version": "2023-06-01",
        "custom": "value",
    }


def test_prompt_model_config_instance_validates():
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        PromptModelConfig(temperature=-0.5)

    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        PromptModelConfig(max_tokens=0)

    # This should not raise during PromptVersion construction
    config = PromptModelConfig(model_name="gpt-5", temperature=0.7)
    PromptVersion(name="test", version=1, template="{{x}}", model_config=config)


def test_prompt_model_config_basic():
    config = PromptModelConfig(model_name="gpt-5", temperature=0.7, max_tokens=1000)
    assert config.model_name == "gpt-5"
    assert config.temperature == 0.7
    assert config.max_tokens == 1000
    assert config.top_p is None
    assert config.extra_params == {}


def test_prompt_model_config_with_extra_params():
    config = PromptModelConfig(
        model_name="gpt-5",
        temperature=0.7,
        extra_params={"anthropic_version": "2023-06-01", "custom_param": "value"},
    )
    assert config.model_name == "gpt-5"
    assert config.extra_params == {"anthropic_version": "2023-06-01", "custom_param": "value"}


def test_prompt_model_config_to_dict():
    config = PromptModelConfig(model_name="gpt-5", temperature=0.7, max_tokens=1000)
    config_dict = config.to_dict()
    assert config_dict == {"model_name": "gpt-5", "temperature": 0.7, "max_tokens": 1000}
    # None values should not be included
    assert "top_p" not in config_dict
    assert "top_k" not in config_dict


def test_prompt_model_config_to_dict_with_extra_params():
    config = PromptModelConfig(
        model_name="gpt-5",
        temperature=0.7,
        extra_params={"custom_param": "value", "another_param": 123},
    )
    config_dict = config.to_dict()
    # extra_params should be merged at top level
    assert config_dict == {
        "model_name": "gpt-5",
        "temperature": 0.7,
        "custom_param": "value",
        "another_param": 123,
    }
    assert "extra_params" not in config_dict


def test_prompt_model_config_from_dict():
    config_dict = {"model_name": "gpt-5", "temperature": 0.7, "max_tokens": 1000}
    config = PromptModelConfig.from_dict(config_dict)
    assert config.model_name == "gpt-5"
    assert config.temperature == 0.7
    assert config.max_tokens == 1000


def test_prompt_model_config_from_dict_with_unknown_fields():
    config_dict = {
        "model_name": "gpt-5",
        "temperature": 0.7,
        "custom_param": "value",
        "another_param": 123,
    }
    config = PromptModelConfig.from_dict(config_dict)
    assert config.model_name == "gpt-5"
    assert config.temperature == 0.7
    assert config.extra_params == {"custom_param": "value", "another_param": 123}


@pytest.mark.parametrize(
    ("field", "value", "error_match"),
    [
        ("temperature", "invalid", r"Input should be a valid number"),
        ("temperature", -0.5, r"Input should be greater than or equal to 0"),
        ("max_tokens", 100.5, r"Input should be a valid integer"),
        ("max_tokens", 0, r"Input should be greater than 0"),
        ("top_p", "invalid", r"Input should be a valid number"),
        ("top_p", 1.5, r"Input should be less than or equal to 1"),
        ("top_k", 10.5, r"Input should be a valid integer"),
        ("top_k", 0, r"Input should be greater than 0"),
        ("frequency_penalty", "invalid", r"Input should be a valid number"),
        ("presence_penalty", "invalid", r"Input should be a valid number"),
        ("stop_sequences", "not a list", r"Input should be a valid list"),
        ("stop_sequences", ["valid", 123], r"Input should be a valid string"),
        ("extra_params", "not a dict", r"Input should be a valid dictionary"),
    ],
)
def test_prompt_model_config_validation(field, value, error_match):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=error_match):
        PromptModelConfig(**{field: value})


def test_prompt_model_config_empty():
    config = PromptModelConfig()
    assert config.model_name is None
    assert config.temperature is None
    assert config.max_tokens is None
    assert config.to_dict() == {}

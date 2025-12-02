import pytest

from mlflow.entities.model_registry import PromptModelConfig
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.prompt_version import (
    IS_PROMPT_TAG_KEY,
    PROMPT_TEXT_TAG_KEY,
    PromptVersion,
)
from mlflow.exceptions import MlflowException
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
    model_version = ModelVersion(
        name="my-prompt",
        version=1,
        description="test",
        creation_timestamp=123,
        tags=[
            ModelVersionTag(key=IS_PROMPT_TAG_KEY, value="true"),
            ModelVersionTag(key=PROMPT_TEXT_TAG_KEY, value="Hello, {{name}}!"),
        ],
        aliases=["alias"],
    )

    prompt = model_version_to_prompt_version(model_version)
    assert prompt.name == "my-prompt"
    assert prompt.version == 1
    assert prompt.description == "test"
    assert prompt.creation_timestamp == 123
    assert prompt.template == "Hello, {{name}}!"
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


def test_prompt_with_model_config():
    model_config = {
        "model_name": "gpt-4",
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
    assert prompt.model_config["model_name"] == "gpt-4"
    assert prompt.model_config["temperature"] == 0.7

    # Test prompt without model_config
    prompt_without_config = PromptVersion(
        name="my_prompt", version=2, template="Hello, {{name}}!"
    )
    assert prompt_without_config.model_config is None


def test_prompt_model_config_with_chat_template():
    model_config = {
        "model_name": "gpt-4",
        "temperature": 0.5,
    }
    chat_template = [
        {"role": "system", "content": "You are a {{style}} assistant."},
        {"role": "user", "content": "{{question}}"},
    ]
    prompt = PromptVersion(
        name="chat_prompt",
        version=1,
        template=chat_template,
        model_config=model_config,
    )
    assert prompt.model_config == model_config
    assert prompt.is_text_prompt is False
    assert prompt.template == chat_template


def test_prompt_model_config_preserved_in_partial_format():
    model_config = {
        "model_name": "gpt-4",
        "temperature": 0.8,
    }
    prompt = PromptVersion(
        name="test",
        version=1,
        template="Hello, {{title}} {{name}}!",
        model_config=model_config,
    )

    # Partial formatting should preserve model_config
    result = prompt.format(title="Ms.", allow_partial=True)
    assert isinstance(result, PromptVersion)
    assert result.model_config == model_config
    assert result.model_config["temperature"] == 0.8
    assert result.template == "Hello, Ms. {{name}}!"


def test_prompt_with_model_config_instance():
    config = PromptModelConfig(model_name="gpt-4", temperature=0.7, max_tokens=1000)
    prompt = PromptVersion(
        name="my_prompt",
        version=1,
        template="Hello, {{name}}!",
        model_config=config,
    )
    # Should be stored as dict
    assert prompt.model_config == {"model_name": "gpt-4", "temperature": 0.7, "max_tokens": 1000}


def test_prompt_with_model_config_instance_and_extra_params():
    config = PromptModelConfig(
        model_name="claude-3",
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
        "model_name": "claude-3",
        "temperature": 0.5,
        "anthropic_version": "2023-06-01",
        "custom": "value",
    }


def test_prompt_model_config_instance_validates():
    with pytest.raises(ValueError, match="temperature must be non-negative"):
        PromptModelConfig(temperature=-0.5)

    with pytest.raises(ValueError, match="max_tokens must be positive"):
        PromptModelConfig(max_tokens=0)

    # This should not raise during PromptVersion construction
    config = PromptModelConfig(model_name="gpt-4", temperature=0.7)
    PromptVersion(name="test", version=1, template="{{x}}", model_config=config)


# PromptModelConfig tests


def test_prompt_model_config_basic():
    config = PromptModelConfig(model_name="gpt-4", temperature=0.7, max_tokens=1000)
    assert config.model_name == "gpt-4"
    assert config.temperature == 0.7
    assert config.max_tokens == 1000
    assert config.top_p is None
    assert config.extra_params == {}


def test_prompt_model_config_with_all_fields():
    config = PromptModelConfig(
        model_name="claude-3-opus",
        temperature=0.5,
        max_tokens=2000,
        top_p=0.9,
        top_k=50,
        frequency_penalty=0.2,
        presence_penalty=0.1,
        stop_sequences=["END", "STOP"],
    )
    assert config.model_name == "claude-3-opus"
    assert config.temperature == 0.5
    assert config.max_tokens == 2000
    assert config.top_p == 0.9
    assert config.top_k == 50
    assert config.frequency_penalty == 0.2
    assert config.presence_penalty == 0.1
    assert config.stop_sequences == ["END", "STOP"]


def test_prompt_model_config_with_extra_params():
    config = PromptModelConfig(
        model_name="gpt-4",
        temperature=0.7,
        extra_params={"anthropic_version": "2023-06-01", "custom_param": "value"},
    )
    assert config.model_name == "gpt-4"
    assert config.extra_params == {"anthropic_version": "2023-06-01", "custom_param": "value"}


def test_prompt_model_config_to_dict():
    config = PromptModelConfig(model_name="gpt-4", temperature=0.7, max_tokens=1000)
    config_dict = config.to_dict()
    assert config_dict == {"model_name": "gpt-4", "temperature": 0.7, "max_tokens": 1000}
    # None values should not be included
    assert "top_p" not in config_dict
    assert "top_k" not in config_dict


def test_prompt_model_config_to_dict_with_extra_params():
    config = PromptModelConfig(
        model_name="gpt-4",
        temperature=0.7,
        extra_params={"custom_param": "value", "another_param": 123},
    )
    config_dict = config.to_dict()
    # extra_params should be merged at top level
    assert config_dict == {
        "model_name": "gpt-4",
        "temperature": 0.7,
        "custom_param": "value",
        "another_param": 123,
    }
    assert "extra_params" not in config_dict


def test_prompt_model_config_from_dict():
    config_dict = {"model_name": "gpt-4", "temperature": 0.7, "max_tokens": 1000}
    config = PromptModelConfig.from_dict(config_dict)
    assert config.model_name == "gpt-4"
    assert config.temperature == 0.7
    assert config.max_tokens == 1000


def test_prompt_model_config_from_dict_with_unknown_fields():
    config_dict = {
        "model_name": "gpt-4",
        "temperature": 0.7,
        "custom_param": "value",
        "another_param": 123,
    }
    config = PromptModelConfig.from_dict(config_dict)
    assert config.model_name == "gpt-4"
    assert config.temperature == 0.7
    assert config.extra_params == {"custom_param": "value", "another_param": 123}


def test_prompt_model_config_roundtrip():
    original = PromptModelConfig(
        model_name="claude-3",
        temperature=0.5,
        max_tokens=2000,
        extra_params={"version": "v1", "cache": True},
    )
    config_dict = original.to_dict()
    reconstructed = PromptModelConfig.from_dict(config_dict)
    assert reconstructed.model_name == original.model_name
    assert reconstructed.temperature == original.temperature
    assert reconstructed.max_tokens == original.max_tokens
    assert reconstructed.extra_params == original.extra_params


def test_prompt_model_config_validation_temperature():
    with pytest.raises(TypeError, match="temperature must be a number"):
        PromptModelConfig(temperature="invalid")

    with pytest.raises(ValueError, match="temperature must be non-negative"):
        PromptModelConfig(temperature=-0.5)

    # Valid temperatures
    PromptModelConfig(temperature=0.0)
    PromptModelConfig(temperature=1.5)


def test_prompt_model_config_validation_max_tokens():
    with pytest.raises(TypeError, match="max_tokens must be an integer"):
        PromptModelConfig(max_tokens=100.5)

    with pytest.raises(ValueError, match="max_tokens must be positive"):
        PromptModelConfig(max_tokens=0)

    with pytest.raises(ValueError, match="max_tokens must be positive"):
        PromptModelConfig(max_tokens=-10)

    # Valid max_tokens
    PromptModelConfig(max_tokens=1)
    PromptModelConfig(max_tokens=1000)


def test_prompt_model_config_validation_top_p():
    with pytest.raises(TypeError, match="top_p must be a number"):
        PromptModelConfig(top_p="invalid")

    with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
        PromptModelConfig(top_p=-0.1)

    with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
        PromptModelConfig(top_p=1.5)

    # Valid top_p
    PromptModelConfig(top_p=0.0)
    PromptModelConfig(top_p=0.5)
    PromptModelConfig(top_p=1.0)


def test_prompt_model_config_validation_top_k():
    with pytest.raises(TypeError, match="top_k must be an integer"):
        PromptModelConfig(top_k=10.5)

    with pytest.raises(ValueError, match="top_k must be positive"):
        PromptModelConfig(top_k=0)

    with pytest.raises(ValueError, match="top_k must be positive"):
        PromptModelConfig(top_k=-5)

    # Valid top_k
    PromptModelConfig(top_k=1)
    PromptModelConfig(top_k=100)


def test_prompt_model_config_validation_penalties():
    with pytest.raises(TypeError, match="frequency_penalty must be a number"):
        PromptModelConfig(frequency_penalty="invalid")

    with pytest.raises(TypeError, match="presence_penalty must be a number"):
        PromptModelConfig(presence_penalty="invalid")

    # Valid penalties (can be negative)
    PromptModelConfig(frequency_penalty=-2.0)
    PromptModelConfig(frequency_penalty=0.0)
    PromptModelConfig(frequency_penalty=2.0)
    PromptModelConfig(presence_penalty=-1.5)
    PromptModelConfig(presence_penalty=1.5)


def test_prompt_model_config_validation_stop_sequences():
    with pytest.raises(TypeError, match="stop_sequences must be a list"):
        PromptModelConfig(stop_sequences="not a list")

    with pytest.raises(TypeError, match="All stop_sequences must be strings"):
        PromptModelConfig(stop_sequences=["valid", 123, "also valid"])

    # Valid stop_sequences
    PromptModelConfig(stop_sequences=[])
    PromptModelConfig(stop_sequences=["END"])
    PromptModelConfig(stop_sequences=["END", "STOP", "\n"])


def test_prompt_model_config_validation_extra_params():
    with pytest.raises(TypeError, match="extra_params must be a dict"):
        PromptModelConfig(extra_params="not a dict")

    with pytest.raises(TypeError, match="extra_params must be a dict"):
        PromptModelConfig(extra_params=["list", "items"])

    # Valid extra_params
    PromptModelConfig(extra_params={})
    PromptModelConfig(extra_params={"key": "value", "number": 123, "nested": {"a": 1}})


def test_prompt_model_config_empty():
    config = PromptModelConfig()
    assert config.model_name is None
    assert config.temperature is None
    assert config.max_tokens is None
    assert config.to_dict() == {}

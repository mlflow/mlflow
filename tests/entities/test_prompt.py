import pytest

from mlflow.entities.model_registry.prompt import IS_PROMPT_TAG_KEY, PROMPT_TEXT_TAG_KEY, Prompt
from mlflow.exceptions import MlflowException


def test_prompt_initialization():
    prompt = Prompt(name="my_prompt", version=1, template="Hello, {{name}}!")
    assert prompt.name == "my_prompt"
    assert prompt.version == 1
    assert prompt.template == "Hello, {{name}}!"
    # Public property should not return the reserved tag
    assert prompt.tags == {}
    assert prompt._tags[IS_PROMPT_TAG_KEY] == "true"
    assert prompt._tags[PROMPT_TEXT_TAG_KEY] == "Hello, {{name}}!"


def test_prompt_variables_extraction():
    prompt = Prompt(name="test", version=1, template="Hello, {{first_name}} {{last_name}}!")
    assert prompt.variables == {"first_name", "last_name"}


def test_prompt_format():
    prompt = Prompt(name="test", version=1, template="Hello, {{title}} {{name}}!")
    result = prompt.format(title="Ms.", name="Alice")
    assert result == "Hello, Ms. Alice!"

    # By default, missing variables raise an error
    with pytest.raises(MlflowException, match="Missing variables: {'name'}"):
        prompt.format(title="Ms.")

    # Partial formatting
    result = prompt.format(title="Ms.", allow_partial=True)
    assert result.template == "Hello, Ms. {{name}}!"
    assert result.variables == {"name"}


def test_prompt_with_spaces():
    # Prompt template should handle spaces in variable names
    prompt = Prompt(name="test", version=1, template="Hello, {{ title }} {{  name  }}!")
    result = prompt.format(title="Ms.", name="Alice")
    assert result == "Hello, Ms. Alice!"
    assert prompt.variables == {"title", "name"}

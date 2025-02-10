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
    prompt = Prompt(name="test", version=1, template=template)
    assert prompt.variables == expected


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

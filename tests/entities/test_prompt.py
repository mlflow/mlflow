import pytest

from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.prompt import IS_PROMPT_TAG_KEY, PROMPT_TEXT_TAG_KEY, Prompt
from mlflow.exceptions import MlflowException
from mlflow.protos.model_registry_pb2 import ModelVersionTag


def test_prompt_initialization():
    prompt = Prompt(name="my_prompt", version=1, template="Hello, {{name}}!")
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
    prompt = Prompt(name="test", version=1, template=template)
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
    prompt = Prompt(name="test", version=1, template=template)
    assert prompt.to_single_brace_format() == expected


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

    # Non-string values
    result = prompt.format(title="Ms.", allow_partial=True)
    result = prompt.format(title=1, name=True)
    assert result == "Hello, 1 True!"


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

    prompt = Prompt.from_model_version(model_version)
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
        Prompt.from_model_version(invalid_model_version)

    invalid_model_version = ModelVersion(
        name="my-prompt",
        version=1,
        creation_timestamp=123,
        # Missing the prompt text tag
        tags=[ModelVersionTag(key=IS_PROMPT_TAG_KEY, value="true")],
    )

    with pytest.raises(MlflowException, match="Prompt `my-prompt` does not contain a prompt text"):
        Prompt.from_model_version(invalid_model_version)

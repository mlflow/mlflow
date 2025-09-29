import warnings
from contextlib import contextmanager

import pytest
from pydantic import BaseModel

import mlflow
from mlflow.genai.prompts.utils import format_prompt


@contextmanager
def no_future_warning():
    with warnings.catch_warnings():
        # Translate future warning into an exception
        warnings.simplefilter("error", FutureWarning)
        yield


def test_suppress_prompt_api_migration_warning():
    with no_future_warning():
        mlflow.genai.register_prompt("test_prompt", "test_template")
        mlflow.genai.search_prompts()
        mlflow.genai.load_prompt("prompts:/test_prompt/1")
        mlflow.genai.set_prompt_alias("test_prompt", "test_alias", 1)
        mlflow.genai.delete_prompt_alias("test_prompt", "test_alias")


def test_prompt_api_migration_warning():
    with pytest.warns(FutureWarning, match="The `mlflow.register_prompt` API is"):
        mlflow.register_prompt("test_prompt", "test_template")

    with pytest.warns(FutureWarning, match="The `mlflow.search_prompts` API is"):
        mlflow.search_prompts()

    with pytest.warns(FutureWarning, match="The `mlflow.load_prompt` API is"):
        mlflow.load_prompt("prompts:/test_prompt/1")

    with pytest.warns(FutureWarning, match="The `mlflow.set_prompt_alias` API is"):
        mlflow.set_prompt_alias("test_prompt", "test_alias", 1)

    with pytest.warns(FutureWarning, match="The `mlflow.delete_prompt_alias` API is"):
        mlflow.delete_prompt_alias("test_prompt", "test_alias")


def test_register_chat_prompt_with_messages():
    """Test registering chat prompts with list of message dictionaries."""
    chat_template = [
        {"role": "system", "content": "You are a {{style}} assistant."},
        {"role": "user", "content": "{{question}}"},
    ]

    prompt = mlflow.genai.register_prompt(
        name="test_chat", template=chat_template, commit_message="Test chat prompt"
    )

    not prompt.is_text_prompt
    assert prompt.template == chat_template
    assert prompt.commit_message == "Test chat prompt"


def test_register_prompt_with_pydantic_response_format():
    """Test registering prompts with Pydantic response format."""

    class ResponseSchema(BaseModel):
        answer: str
        confidence: float

    prompt = mlflow.genai.register_prompt(
        name="test_response",
        template="What is {{question}}?",
        response_format=ResponseSchema,
    )

    expected_schema = ResponseSchema.model_json_schema()
    assert prompt.response_format == expected_schema


def test_register_prompt_with_dict_response_format():
    """Test registering prompts with dictionary response format."""
    response_format = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
    }

    prompt = mlflow.genai.register_prompt(
        name="test_dict_response",
        template="What is {{question}}?",
        response_format=response_format,
    )

    assert prompt.response_format == response_format


def test_register_prompt_error_handling_invalid_chat_format():
    """Test error handling for invalid chat message formats."""
    invalid_template = [{"content": "Hello"}]  # Missing role

    with pytest.raises(ValueError, match="Template must be a list of dicts with role and content"):
        mlflow.genai.register_prompt(name="test_invalid", template=invalid_template)


def test_register_and_load_chat_prompt_integration():
    """Test that registered chat prompts can be loaded and formatted correctly."""
    chat_template = [
        {"role": "system", "content": "You are a {{style}} assistant."},
        {"role": "user", "content": "{{question}}"},
    ]

    mlflow.genai.register_prompt(name="test_integration", template=chat_template)

    loaded_prompt = mlflow.genai.load_prompt("test_integration", version=1)

    assert not loaded_prompt.is_text_prompt
    assert loaded_prompt.template == chat_template

    # Test formatting
    formatted = loaded_prompt.format(style="helpful", question="How are you?")
    expected = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How are you?"},
    ]
    assert formatted == expected


def test_register_text_prompt_backward_compatibility():
    """Test that text prompt registration continues to work as before."""
    prompt = mlflow.genai.register_prompt(
        name="test_text_backward",
        template="Hello {{name}}!",
        commit_message="Test backward compatibility",
    )

    assert prompt.is_text_prompt
    assert prompt.template == "Hello {{name}}!"
    assert prompt.commit_message == "Test backward compatibility"


def test_register_prompt_with_tags():
    """Test registering prompts with custom tags."""
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "{{question}}"},
    ]

    prompt = mlflow.genai.register_prompt(
        name="test_with_tags",
        template=chat_template,
        tags={"author": "test_user", "model": "gpt-4"},
    )

    assert prompt.tags["author"] == "test_user"
    assert prompt.tags["model"] == "gpt-4"


def test_register_prompt_with_complex_response_format():
    """Test registering prompts with complex Pydantic response format."""

    class ComplexResponse(BaseModel):
        summary: str
        key_points: list[str]
        confidence: float
        metadata: dict[str, str] = {}

    chat_template = [
        {"role": "system", "content": "You are a data analyst."},
        {"role": "user", "content": "Analyze this data: {{data}}"},
    ]

    prompt = mlflow.genai.register_prompt(
        name="test_complex_response",
        template=chat_template,
        response_format=ComplexResponse,
    )

    expected_schema = ComplexResponse.model_json_schema()
    assert prompt.response_format == expected_schema
    assert "properties" in prompt.response_format
    assert "summary" in prompt.response_format["properties"]
    assert "key_points" in prompt.response_format["properties"]
    assert "confidence" in prompt.response_format["properties"]
    assert "metadata" in prompt.response_format["properties"]


def test_register_prompt_with_none_response_format():
    """Test registering prompts with None response format."""
    prompt = mlflow.genai.register_prompt(
        name="test_none_response", template="Hello {{name}}!", response_format=None
    )

    assert prompt.response_format is None


def test_register_prompt_with_empty_chat_template():
    """Test registering prompts with empty chat template list."""
    # Empty list should be treated as text prompt
    prompt = mlflow.genai.register_prompt(name="test_empty_chat", template=[])

    assert prompt.is_text_prompt
    assert prompt.template == "[]"  # Empty list serialized as string


def test_register_prompt_with_single_message_chat():
    """Test registering prompts with single message chat template."""
    chat_template = [{"role": "user", "content": "Hello {{name}}!"}]

    prompt = mlflow.genai.register_prompt(name="test_single_message", template=chat_template)

    assert prompt.template == chat_template
    assert prompt.variables == {"name"}


def test_register_prompt_with_multiple_variables_in_chat():
    """Test registering prompts with multiple variables in chat messages."""
    chat_template = [
        {
            "role": "system",
            "content": "You are a {{style}} assistant named {{name}}.",
        },
        {"role": "user", "content": "{{greeting}}! {{question}}"},
        {
            "role": "assistant",
            "content": "I understand you're asking about {{topic}}.",
        },
    ]

    prompt = mlflow.genai.register_prompt(name="test_multiple_variables", template=chat_template)

    expected_variables = {"style", "name", "greeting", "question", "topic"}
    assert prompt.variables == expected_variables


def test_register_prompt_with_mixed_content_types():
    """Test registering prompts with mixed content types in chat messages."""
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello {{name}}!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
    ]

    prompt = mlflow.genai.register_prompt(name="test_mixed_content", template=chat_template)

    assert prompt.template == chat_template
    assert prompt.variables == {"name"}


def test_register_prompt_with_nested_variables():
    """Test registering prompts with nested variable names."""
    chat_template = [
        {
            "role": "system",
            "content": "You are a {{user.preferences.style}} assistant.",
        },
        {
            "role": "user",
            "content": "Hello {{user.name}}! {{user.preferences.greeting}}",
        },
    ]

    prompt = mlflow.genai.register_prompt(name="test_nested_variables", template=chat_template)

    expected_variables = {
        "user.preferences.style",
        "user.name",
        "user.preferences.greeting",
    }
    assert prompt.variables == expected_variables


@pytest.mark.parametrize(
    ("prompt_template", "values", "expected"),
    [
        # Test with Unicode escape-like sequences
        (
            "User input: {{ user_text }}",
            {"user_text": r"Path is C:\users\john"},
            r"User input: Path is C:\users\john",
        ),
        # Test with newlines and tabs
        (
            "Data: {{ data }}",
            {"data": "Line1\\nLine2\\tTabbed"},
            "Data: Line1\\nLine2\\tTabbed",
        ),
        # Test with multiple variables
        (
            "Path: {{ path }}, Command: {{ cmd }}",
            {"path": r"C:\temp", "cmd": r"echo \u0041"},
            r"Path: C:\temp, Command: echo \u0041",
        ),
    ],
)
def test_format_prompt_with_backslashes(
    prompt_template: str, values: dict[str, str], expected: str
):
    """Test that format_prompt correctly handles values containing backslashes."""
    result = format_prompt(prompt_template, **values)
    assert result == expected

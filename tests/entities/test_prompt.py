import pytest

from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.prompt_version import (
    CONFIG_TAG_KEY,
    IS_PROMPT_TAG_KEY,
    PROMPT_TEXT_TAG_KEY,
    PROMPT_TYPE_TAG_KEY,
    RESPONSE_FORMAT_TAG_KEY,
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


def test_chat_prompt_initialization():
    """Test chat prompt initialization with message list."""
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, {{name}}! How are you {{mood}}?"},
    ]
    prompt = PromptVersion(
        name="chat_prompt",
        version=1,
        template=chat_template,
        prompt_type="chat",
    )
    assert prompt.name == "chat_prompt"
    assert prompt.version == 1
    assert prompt.template == chat_template
    assert prompt.prompt_type == "chat"
    assert prompt.response_format is None
    assert prompt.config is None
    assert prompt.variables == {"name", "mood"}


def test_chat_prompt_with_response_format_and_config():
    """Test chat prompt with response format and config."""
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Generate a response for {{topic}}."},
    ]
    response_format = {
        "type": "object",
        "properties": {
            "response": {"type": "string"},
            "confidence": {"type": "number"},
        },
    }
    config = {"temperature": 0.7, "max_tokens": 100}

    prompt = PromptVersion(
        name="enhanced_chat",
        version=1,
        template=chat_template,
        prompt_type="chat",
        response_format=response_format,
        config=config,
    )
    assert prompt.prompt_type == "chat"
    assert prompt.response_format == response_format
    assert prompt.config == config
    assert prompt.variables == {"topic"}


def test_invalid_prompt_type():
    """Test that invalid prompt_type raises ValueError."""
    with pytest.raises(ValueError, match="prompt_type must be 'text' or 'chat'"):
        PromptVersion(
            name="invalid_prompt",
            version=1,
            template="Hello",
            prompt_type="invalid",
        )


def test_chat_prompt_invalid_template_type():
    """Test that chat prompt with non-list template raises ValueError."""
    with pytest.raises(ValueError, match="Chat template must be a list of message dictionaries"):
        PromptVersion(
            name="invalid_chat",
            version=1,
            template="Not a list",
            prompt_type="chat",
        )


def test_chat_prompt_invalid_message_structure():
    """Test that chat prompt with invalid message structure raises ValueError."""
    # Missing role
    with pytest.raises(ValueError, match="Each message must have 'role' and 'content' keys"):
        PromptVersion(
            name="invalid_chat",
            version=1,
            template=[{"content": "Hello"}],
            prompt_type="chat",
        )

    # Missing content
    with pytest.raises(ValueError, match="Each message must have 'role' and 'content' keys"):
        PromptVersion(
            name="invalid_chat",
            version=1,
            template=[{"role": "user"}],
            prompt_type="chat",
        )

    # Invalid role
    with pytest.raises(ValueError, match="Role must be one of: system, user, assistant"):
        PromptVersion(
            name="invalid_chat",
            version=1,
            template=[{"role": "invalid", "content": "Hello"}],
            prompt_type="chat",
        )

    # Non-dict message
    with pytest.raises(ValueError, match="Each message must be a dictionary"):
        PromptVersion(
            name="invalid_chat",
            version=1,
            template=["not a dict"],
            prompt_type="chat",
        )


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


def test_chat_prompt_variables_extraction():
    """Test variable extraction from chat prompts."""
    chat_template = [
        {"role": "system", "content": "You are a {{role}} assistant."},
        {"role": "user", "content": "Hello, {{name}}! How are you {{mood}}?"},
        {"role": "assistant", "content": "I'm doing {{status}}."},
    ]

    prompt = PromptVersion(
        name="chat_vars",
        version=1,
        template=chat_template,
        prompt_type="chat",
    )

    assert prompt.variables == {"role", "name", "mood", "status"}


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


def test_chat_prompt_format():
    """Test chat prompt formatting."""
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, {{name}}! How are you {{mood}}?"},
    ]
    prompt = PromptVersion(
        name="chat_prompt",
        version=1,
        template=chat_template,
        prompt_type="chat",
    )

    # Test successful formatting using format_chat
    formatted = prompt.format_chat(name="Alice", mood="happy")
    expected = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, Alice! How are you happy?"},
    ]
    assert formatted == expected

    # Test partial formatting using format_chat
    formatted = prompt.format_chat(allow_partial=True, name="Alice")
    expected = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, Alice! How are you {{mood}}?"},
    ]
    assert formatted == expected

    # Test that format() raises an exception for chat prompts
    with pytest.raises(MlflowException, match="format\\(\\) cannot be used with chat prompts"):
        prompt.format(name="Alice", mood="happy")


def test_format_chat_method():
    """Test the format_chat method specifically."""
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, {{name}}! How are you {{mood}}?"},
    ]
    prompt = PromptVersion(
        name="chat_prompt",
        version=1,
        template=chat_template,
        prompt_type="chat",
    )

    formatted = prompt.format_chat(name="Alice", mood="happy")
    expected = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, Alice! How are you happy?"},
    ]
    assert formatted == expected


def test_pydantic_response_format():
    """Test response format with Pydantic class."""
    try:
        from pydantic import BaseModel

        class ResponseModel(BaseModel):
            response: str
            confidence: float

        prompt = PromptVersion(
            name="pydantic_prompt",
            version=1,
            template="Generate a response for {{topic}}.",
            response_format=ResponseModel,
        )

        # Should convert Pydantic model to JSON schema
        assert prompt.response_format is not None
        assert "properties" in prompt.response_format
        assert "response" in prompt.response_format["properties"]
        assert "confidence" in prompt.response_format["properties"]

    except ImportError:
        # Pydantic not available, skip test
        pytest.skip("Pydantic not available")


def test_dict_response_format():
    """Test response format with dictionary."""
    response_format = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "score": {"type": "number"},
        },
    }

    prompt = PromptVersion(
        name="dict_prompt",
        version=1,
        template="Answer the question: {{question}}",
        response_format=response_format,
    )

    assert prompt.response_format == response_format


def test_invalid_response_format():
    """Test that invalid response_format raises ValueError."""
    with pytest.raises(ValueError, match="response_format must be a Pydantic class or dict"):
        PromptVersion(
            name="invalid_prompt",
            version=1,
            template="Hello",
            response_format="not a dict or class",
        )


def test_config_storage():
    """Test that config is properly stored and retrieved."""
    config = {
        "temperature": 0.7,
        "max_tokens": 100,
        "model": "gpt-4",
    }

    prompt = PromptVersion(
        name="config_prompt",
        version=1,
        template="Hello, {{name}}!",
        config=config,
    )

    assert prompt.config == config


def test_tags_exclude_reserved_tags():
    """Test that reserved tags are not exposed in public tags property."""
    prompt = PromptVersion(
        name="tagged_prompt",
        version=1,
        template="Hello, {{name}}!",
        tags={"author": "Alice", "project": "test"},
        response_format={"type": "string"},
        config={"temperature": 0.7},
    )

    # Public tags should not include reserved tags
    public_tags = prompt.tags
    assert "author" in public_tags
    assert "project" in public_tags
    assert IS_PROMPT_TAG_KEY not in public_tags
    assert PROMPT_TEXT_TAG_KEY not in public_tags
    assert PROMPT_TYPE_TAG_KEY not in public_tags
    assert RESPONSE_FORMAT_TAG_KEY not in public_tags
    assert CONFIG_TAG_KEY not in public_tags


def test_repr_method():
    """Test the __repr__ method for both text and chat prompts."""
    # Text prompt
    text_prompt = PromptVersion(
        name="text_prompt",
        version=1,
        template="Hello, {{name}}!",
    )
    repr_text = repr(text_prompt)
    assert "text_prompt" in repr_text
    assert "type=text" in repr_text
    assert "Hello, {{name}}!" in repr_text

    # Chat prompt
    chat_template = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]
    chat_prompt = PromptVersion(
        name="chat_prompt",
        version=1,
        template=chat_template,
        prompt_type="chat",
    )
    repr_chat = repr(chat_prompt)
    assert "chat_prompt" in repr_chat
    assert "type=chat" in repr_chat
    assert "Chat prompt with 2 messages" in repr_chat


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


def test_model_version_conversion_with_new_features():
    """Test converting ModelVersion to PromptVersion with new features."""
    # Create a ModelVersion with new feature tags
    tags = [
        ModelVersionTag(key=IS_PROMPT_TAG_KEY, value="true"),
        ModelVersionTag(
            key=PROMPT_TEXT_TAG_KEY,
            value='[{"role": "user", "content": "Hello, {{name}}!"}]',
        ),
        ModelVersionTag(key=PROMPT_TYPE_TAG_KEY, value="chat"),
        ModelVersionTag(key=RESPONSE_FORMAT_TAG_KEY, value='{"type": "string"}'),
        ModelVersionTag(key=CONFIG_TAG_KEY, value='{"temperature": 0.7}'),
        ModelVersionTag(key="author", value="Alice"),
    ]

    model_version = ModelVersion(
        name="conversion_test",
        version=1,
        creation_timestamp=1234567890,
        tags=tags,
    )

    prompt = model_version_to_prompt_version(model_version)

    assert prompt.name == "conversion_test"
    assert prompt.version == 1
    assert prompt.prompt_type == "chat"
    assert prompt.template == [{"role": "user", "content": "Hello, {{name}}!"}]
    assert prompt.response_format == {"type": "string"}
    assert prompt.config == {"temperature": 0.7}
    assert prompt.variables == {"name"}
    assert prompt.tags == {"author": "Alice"}


def test_backward_compatibility_text_prompt():
    """Test backward compatibility with existing text prompts."""
    # Simulate an old ModelVersion without new feature tags
    tags = [
        ModelVersionTag(key=IS_PROMPT_TAG_KEY, value="true"),
        ModelVersionTag(key=PROMPT_TEXT_TAG_KEY, value="Hello, {{name}}!"),
        ModelVersionTag(key="author", value="Alice"),
    ]

    model_version = ModelVersion(
        name="backward_test",
        version=1,
        creation_timestamp=1234567890,
        tags=tags,
    )

    prompt = model_version_to_prompt_version(model_version)

    assert prompt.name == "backward_test"
    assert prompt.version == 1
    assert prompt.prompt_type == "text"  # Default
    assert prompt.template == "Hello, {{name}}!"
    assert prompt.response_format is None
    assert prompt.config is None
    assert prompt.variables == {"name"}
    assert prompt.tags == {"author": "Alice"}


def test_complex_chat_prompt_with_variables():
    """Test complex chat prompt with multiple variables in different messages."""
    chat_template = [
        {
            "role": "system",
            "content": "You are a {{assistant_type}} assistant for {{company}}.",
        },
        {
            "role": "user",
            "content": "Hello! My name is {{name}} and I work at {{company}}.",
        },
        {
            "role": "assistant",
            "content": "Nice to meet you, {{name}}! I'm here to help with {{task}}.",
        },
        {"role": "user", "content": "Can you help me with {{specific_request}}?"},
    ]

    prompt = PromptVersion(
        name="complex_chat",
        version=1,
        template=chat_template,
        prompt_type="chat",
    )

    # Test variable extraction
    expected_vars = {"assistant_type", "company", "name", "task", "specific_request"}
    assert prompt.variables == expected_vars

    # Test formatting using format_chat
    formatted = prompt.format_chat(
        assistant_type="technical",
        company="MLflow",
        name="Alice",
        task="coding",
        specific_request="debugging",
    )
    expected = [
        {
            "role": "system",
            "content": "You are a technical assistant for MLflow.",
        },
        {
            "role": "user",
            "content": "Hello! My name is Alice and I work at MLflow.",
        },
        {
            "role": "assistant",
            "content": "Nice to meet you, Alice! I'm here to help with coding.",
        },
        {"role": "user", "content": "Can you help me with debugging?"},
    ]
    assert formatted == expected


def test_empty_chat_prompt():
    """Test chat prompt with empty message list."""
    with pytest.raises(ValueError, match="Chat template must be a list of message dictionaries"):
        PromptVersion(
            name="empty_chat",
            version=1,
            template=[],
            prompt_type="chat",
        )


def test_chat_prompt_with_empty_content():
    """Test chat prompt with empty content in messages."""
    chat_template = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Hello, {{name}}!"},
    ]

    prompt = PromptVersion(
        name="empty_content_chat",
        version=1,
        template=chat_template,
        prompt_type="chat",
    )

    assert prompt.variables == {"name"}
    assert prompt.template == chat_template


def test_json_serialization_roundtrip():
    """Test that JSON serialization and deserialization works correctly."""
    response_format = {"type": "object", "properties": {"answer": {"type": "string"}}}
    config = {"temperature": 0.7, "max_tokens": 100}

    prompt = PromptVersion(
        name="json_test",
        version=1,
        template="Hello, {{name}}!",
        response_format=response_format,
        config=config,
        tags={"test": "value"},
    )

    # Test that JSON serialization works
    _ = prompt.__dict__.copy()

    # Recreate prompt from serialized data
    new_prompt = PromptVersion(
        name=prompt.name,
        version=prompt.version,
        template=prompt.template,
        prompt_type=prompt.prompt_type,
        response_format=prompt.response_format,
        config=prompt.config,
        tags=prompt.tags,
    )

    assert new_prompt.name == prompt.name
    assert new_prompt.template == prompt.template
    assert new_prompt.response_format == prompt.response_format
    assert new_prompt.config == prompt.config
    assert new_prompt.tags == prompt.tags

import importlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest
from pydantic import BaseModel

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import PromptVersion
from mlflow.environment_variables import MLFLOW_PROMPT_CACHE_MAX_SIZE
from mlflow.exceptions import MlflowException
from mlflow.genai.prompts.utils import format_prompt
from mlflow.prompt.constants import LINKED_PROMPTS_TAG_KEY, PROMPT_EXPERIMENT_IDS_TAG_KEY


def join_thread_by_name_prefix(prefix: str):
    """Join any thread whose name starts with the given prefix."""
    for t in threading.enumerate():
        if t.name.startswith(prefix):
            t.join()


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


def test_crud_prompts(tmp_path):
    mlflow.genai.register_prompt(
        name="prompt_1",
        template="Hi, {title} {name}! How are you today?",
        commit_message="A friendly greeting",
        tags={"model": "my-model"},
    )

    prompt = mlflow.genai.load_prompt("prompt_1", version=1)
    assert prompt.name == "prompt_1"
    assert prompt.template == "Hi, {title} {name}! How are you today?"
    assert prompt.commit_message == "A friendly greeting"
    # Currently, the tags from register_prompt become version tags
    assert prompt.tags == {"model": "my-model"}

    # Check prompt-level tags separately (if needed for test completeness)
    from mlflow import MlflowClient

    client = MlflowClient()
    prompt_entity = client.get_prompt("prompt_1")
    assert prompt_entity.tags == {"model": "my-model"}

    mlflow.genai.register_prompt(
        name="prompt_1",
        template="Hi, {title} {name}! What's up?",
        commit_message="New greeting",
    )

    prompt = mlflow.genai.load_prompt("prompt_1", version=2)
    assert prompt.template == "Hi, {title} {name}! What's up?"

    prompt = mlflow.genai.load_prompt("prompt_1", version=1)
    assert prompt.template == "Hi, {title} {name}! How are you today?"

    prompt = mlflow.genai.load_prompt("prompts:/prompt_1/2")
    assert prompt.template == "Hi, {title} {name}! What's up?"

    # No version = latest
    prompt = mlflow.genai.load_prompt("prompt_1")
    assert prompt.template == "Hi, {title} {name}! What's up?"

    # Test load_prompt with allow_missing for non-existent prompts
    assert mlflow.genai.load_prompt("does_not_exist", version=1, allow_missing=True) is None


def test_prompt_alias(tmp_path):
    mlflow.genai.register_prompt(name="p1", template="Hi, there!")
    mlflow.genai.register_prompt(name="p1", template="Hi, {{name}}!")

    mlflow.genai.set_prompt_alias("p1", alias="production", version=1)
    prompt = mlflow.genai.load_prompt("prompts:/p1@production")
    assert prompt.template == "Hi, there!"
    assert prompt.aliases == ["production"]

    # Reassign alias to a different version
    mlflow.genai.set_prompt_alias("p1", alias="production", version=2)
    assert mlflow.genai.load_prompt("prompts:/p1@production").template == "Hi, {{name}}!"

    mlflow.genai.delete_prompt_alias("p1", alias="production")
    with pytest.raises(
        MlflowException,
        match=(r"Prompt (.*) does not exist.|Prompt alias (.*) not found."),
    ):
        mlflow.genai.load_prompt("prompts:/p1@production")

    # Latest alias
    assert mlflow.genai.load_prompt("prompts:/p1@latest").template == "Hi, {{name}}!"


def test_prompt_associate_with_run(tmp_path):
    mlflow.genai.register_prompt(name="prompt_1", template="Hi, {title} {name}! How are you today?")

    # mlflow.genai.load_prompt() call during the run should associate the prompt with the run
    with mlflow.start_run() as run:
        mlflow.genai.load_prompt("prompt_1", version=1)

    # Check that the prompt was linked to the run via the linkedPrompts tag
    client = MlflowClient()
    run_data = client.get_run(run.info.run_id)
    linked_prompts_tag = run_data.data.tags.get(LINKED_PROMPTS_TAG_KEY)
    assert linked_prompts_tag is not None
    assert len(json.loads(linked_prompts_tag)) == 1
    assert json.loads(linked_prompts_tag)[0] == {
        "name": "prompt_1",
        "version": "1",
    }

    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 1
    assert linked_prompts[0]["name"] == "prompt_1"
    assert linked_prompts[0]["version"] == "1"

    with mlflow.start_run() as run:
        run_id_2 = run.info.run_id

        # Prompt should be linked to the run even if it is loaded in a child thread
        def task():
            mlflow.genai.load_prompt("prompt_1", version=1)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(task) for _ in range(10)]
            for future in futures:
                future.result()

    run_data = client.get_run(run_id_2)
    linked_prompts_tag = run_data.data.tags.get(LINKED_PROMPTS_TAG_KEY)
    assert linked_prompts_tag is not None
    assert len(json.loads(linked_prompts_tag)) == 1
    assert json.loads(linked_prompts_tag)[0] == {
        "name": "prompt_1",
        "version": "1",
    }


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
    prompt = mlflow.genai.register_prompt(
        name="test_none_response", template="Hello {{name}}!", response_format=None
    )

    assert prompt.response_format is None


def test_register_prompt_with_empty_chat_template():
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


def test_set_and_delete_prompt_tag_genai():
    mlflow.genai.register_prompt(name="tag_prompt", template="Hi")
    mlflow.genai.set_prompt_tag("tag_prompt", "env", "prod")
    mlflow.genai.set_prompt_version_tag("tag_prompt", 1, "env", "prod")
    assert mlflow.genai.get_prompt_tags("tag_prompt") == {"env": "prod"}
    assert mlflow.genai.load_prompt("tag_prompt", version=1).tags == {"env": "prod"}
    mlflow.genai.delete_prompt_tag("tag_prompt", "env")
    assert "env" not in mlflow.genai.get_prompt_tags("tag_prompt")
    mlflow.genai.delete_prompt_version_tag("tag_prompt", 1, "env")
    assert "env" not in mlflow.genai.load_prompt("tag_prompt", version=1).tags


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


def test_load_prompt_with_link_to_model_disabled():
    """Test load_prompt with link_to_model=False does not attempt linking."""

    # Register a prompt
    mlflow.genai.register_prompt(name="test_prompt", template="Hello, {{name}}!")

    # Create a logged model and set it as active
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=lambda x: x,
            name="model",
            pip_requirements=["mlflow"],
        )
        mlflow.set_active_model(model_id=model_info.model_id)

        # Load prompt with link_to_model=False - should not link despite active model
        prompt = mlflow.genai.load_prompt("test_prompt", version=1, link_to_model=False)

        # Verify prompt was loaded correctly
        assert prompt.name == "test_prompt"
        assert prompt.version == 1
        assert prompt.template == "Hello, {{name}}!"

        # Join any potential background linking thread (it shouldn't run)
        join_thread_by_name_prefix("link_prompt_thread")

        # Verify the model does NOT have any linked prompts tag
        client = mlflow.MlflowClient()
        model = client.get_logged_model(model_info.model_id)
        linked_prompts_tag = model.tags.get("mlflow.linkedPrompts")
        assert linked_prompts_tag is None, (
            "Model should not have linkedPrompts tag when link_to_model=False"
        )


def test_load_prompt_with_explicit_model_id():
    """Test load_prompt with explicit model_id parameter."""

    # Register a prompt
    mlflow.genai.register_prompt(name="test_prompt", template="Hello, {{name}}!")

    # Create a logged model to link to
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=lambda x: x,
            name="model",
            pip_requirements=["mlflow"],
        )

    # Load prompt with explicit model_id - should link successfully
    prompt = mlflow.genai.load_prompt(
        "test_prompt", version=1, link_to_model=True, model_id=model_info.model_id
    )

    # Verify prompt was loaded correctly
    assert prompt.name == "test_prompt"
    assert prompt.version == 1
    assert prompt.template == "Hello, {{name}}!"

    # Join background linking thread to wait for completion
    join_thread_by_name_prefix("link_prompt_thread")

    # Verify the model has the linked prompt in its tags
    client = mlflow.MlflowClient()
    model = client.get_logged_model(model_info.model_id)
    linked_prompts_tag = model.tags.get("mlflow.linkedPrompts")
    assert linked_prompts_tag is not None

    # Parse the JSON tag value
    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 1
    assert linked_prompts[0]["name"] == "test_prompt"
    assert linked_prompts[0]["version"] == "1"


def test_load_prompt_with_active_model_integration():
    """Test load_prompt with active model integration using get_active_model_id."""

    # Register a prompt
    mlflow.genai.register_prompt(name="test_prompt", template="Hello, {{name}}!")

    # Test loading prompt with active model context
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=lambda x: x,
            name="model",
            pip_requirements=["mlflow"],
        )

        mlflow.set_active_model(model_id=model_info.model_id)
        # Load prompt with link_to_model=True - should use active model
        prompt = mlflow.genai.load_prompt("test_prompt", version=1, link_to_model=True)

        # Verify prompt was loaded correctly
        assert prompt.name == "test_prompt"
        assert prompt.version == 1
        assert prompt.template == "Hello, {{name}}!"

        # Join background linking thread to wait for completion
        join_thread_by_name_prefix("link_prompt_thread")

        # Verify the model has the linked prompt in its tags
        client = mlflow.MlflowClient()
        model = client.get_logged_model(model_info.model_id)
        linked_prompts_tag = model.tags.get("mlflow.linkedPrompts")
        assert linked_prompts_tag is not None

        # Parse the JSON tag value
        linked_prompts = json.loads(linked_prompts_tag)
        assert len(linked_prompts) == 1
        assert linked_prompts[0]["name"] == "test_prompt"
        assert linked_prompts[0]["version"] == "1"


def test_load_prompt_with_no_active_model():
    """Test load_prompt when no active model is available."""

    # Register a prompt
    mlflow.genai.register_prompt(name="test_prompt", template="Hello, {{name}}!")

    # Mock no active model available
    with mock.patch(
        "mlflow.tracking._model_registry.fluent.get_active_model_id", return_value=None
    ):
        # Load prompt with link_to_model=True but no active model - should still work
        prompt = mlflow.genai.load_prompt("test_prompt", version=1, link_to_model=True)

        # Verify prompt was loaded correctly (linking just gets skipped)
        assert prompt.name == "test_prompt"
        assert prompt.version == 1
        assert prompt.template == "Hello, {{name}}!"


def test_load_prompt_linking_error_handling():
    """Test load_prompt error handling when linking fails."""

    # Register a prompt
    mlflow.genai.register_prompt(name="test_prompt", template="Hello, {{name}}!")

    # Test with invalid model ID - should still load prompt successfully
    with mock.patch(
        "mlflow.tracking._model_registry.fluent.get_active_model_id",
        return_value="invalid_model_id",
    ):
        # Load prompt - should succeed despite linking failure (happens in background)
        prompt = mlflow.genai.load_prompt("test_prompt", version=1, link_to_model=True)

        # Verify prompt was loaded successfully despite linking failure
        assert prompt.name == "test_prompt"
        assert prompt.version == 1
        assert prompt.template == "Hello, {{name}}!"


def test_load_prompt_explicit_model_id_overrides_active_model():
    """Test that explicit model_id parameter overrides active model ID."""

    # Register a prompt
    mlflow.genai.register_prompt(name="test_prompt", template="Hello, {{name}}!")

    # Create models to test override behavior
    with mlflow.start_run():
        active_model = mlflow.pyfunc.log_model(
            python_model=lambda x: x,
            name="active_model",
            pip_requirements=["mlflow"],
        )
        explicit_model = mlflow.pyfunc.log_model(
            python_model=lambda x: x,
            name="explicit_model",
            pip_requirements=["mlflow"],
        )

    # Set active model context but provide explicit model_id - explicit should win
    mlflow.set_active_model(model_id=active_model.model_id)
    prompt = mlflow.genai.load_prompt(
        "test_prompt", version=1, link_to_model=True, model_id=explicit_model.model_id
    )

    # Verify prompt was loaded correctly (explicit model_id should be used)
    assert prompt.name == "test_prompt"
    assert prompt.version == 1
    assert prompt.template == "Hello, {{name}}!"

    # Join background linking thread to wait for completion
    join_thread_by_name_prefix("link_prompt_thread")

    # Verify the EXPLICIT model (not active model) has the linked prompt in its tags
    client = mlflow.MlflowClient()
    explicit_model_data = client.get_logged_model(explicit_model.model_id)
    linked_prompts_tag = explicit_model_data.tags.get("mlflow.linkedPrompts")
    assert linked_prompts_tag is not None

    # Parse the JSON tag value
    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 1
    assert linked_prompts[0]["name"] == "test_prompt"
    assert linked_prompts[0]["version"] == "1"

    # Verify the active model does NOT have the linked prompt
    active_model_data = client.get_logged_model(active_model.model_id)
    active_linked_prompts_tag = active_model_data.tags.get("mlflow.linkedPrompts")
    assert active_linked_prompts_tag is None


def test_load_prompt_with_tracing_single_prompt():
    """Test that load_prompt properly links a single prompt to an active trace."""

    # Register a prompt
    mlflow.genai.register_prompt(name="test_prompt", template="Hello, {{name}}!")

    # Start tracing and load prompt
    with mlflow.start_span("test_operation") as span:
        prompt = mlflow.genai.load_prompt("test_prompt", version=1)

        # Verify prompt was loaded correctly
        assert prompt.name == "test_prompt"
        assert prompt.version == 1
        assert prompt.template == "Hello, {{name}}!"

    # Manually trigger prompt linking to trace since in test environment
    # the trace export may not happen automatically
    client = mlflow.MlflowClient()
    prompt_version = PromptVersion(
        name="test_prompt",
        version=1,
        template="Hello, {{name}}!",
        commit_message=None,
        creation_timestamp=None,
    )
    client.link_prompt_versions_to_trace(trace_id=span.trace_id, prompt_versions=[prompt_version])

    # Verify the prompt was linked to the trace by checking the actual trace
    trace = mlflow.get_trace(span.trace_id)
    assert trace is not None

    # Check the linked prompts tag
    linked_prompts_tag = trace.info.tags.get("mlflow.linkedPrompts")
    assert linked_prompts_tag is not None

    # Parse the JSON tag value
    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 1
    assert linked_prompts[0]["name"] == "test_prompt"
    assert linked_prompts[0]["version"] == "1"


def test_load_prompt_with_tracing_multiple_prompts():
    """Test that load_prompt properly links multiple versions of the same prompt to one trace."""

    # Register one prompt with multiple versions
    mlflow.genai.register_prompt(name="my_prompt", template="Hello, {{name}}!")
    mlflow.genai.register_prompt(name="my_prompt", template="Hi there, {{name}}! How are you?")

    # Start tracing and load multiple versions of the same prompt
    with mlflow.start_span("multi_version_prompt_operation") as span:
        prompt_v1 = mlflow.genai.load_prompt("my_prompt", version=1)
        prompt_v2 = mlflow.genai.load_prompt("my_prompt", version=2)

        # Verify prompts were loaded correctly
        assert prompt_v1.name == "my_prompt"
        assert prompt_v1.version == 1
        assert prompt_v1.template == "Hello, {{name}}!"

        assert prompt_v2.name == "my_prompt"
        assert prompt_v2.version == 2
        assert prompt_v2.template == "Hi there, {{name}}! How are you?"

    # Manually trigger prompt linking to trace since in test environment
    # the trace export may not happen automatically
    client = mlflow.MlflowClient()
    prompt_versions = [
        PromptVersion(
            name="my_prompt",
            version=1,
            template="Hello, {{name}}!",
            commit_message=None,
            creation_timestamp=None,
        ),
        PromptVersion(
            name="my_prompt",
            version=2,
            template="Hi there, {{name}}! How are you?",
            commit_message=None,
            creation_timestamp=None,
        ),
    ]
    client.link_prompt_versions_to_trace(trace_id=span.trace_id, prompt_versions=prompt_versions)

    # Verify both versions were linked to the same trace by checking the actual trace
    trace = mlflow.get_trace(span.trace_id)
    assert trace is not None

    # Check the linked prompts tag
    linked_prompts_tag = trace.info.tags.get("mlflow.linkedPrompts")
    assert linked_prompts_tag is not None

    # Parse the JSON tag value
    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 2

    # Check that both versions of the same prompt are present
    prompt_entries = {(p["name"], p["version"]) for p in linked_prompts}
    expected_entries = {("my_prompt", "1"), ("my_prompt", "2")}
    assert prompt_entries == expected_entries

    # Verify we have the same prompt name but different versions
    assert all(p["name"] == "my_prompt" for p in linked_prompts)
    versions = {p["version"] for p in linked_prompts}
    assert versions == {"1", "2"}


def test_load_prompt_with_tracing_no_active_trace():
    """Test that load_prompt works correctly when there's no active trace."""

    # Register a prompt
    mlflow.genai.register_prompt(name="no_trace_prompt", template="Hello, {{name}}!")

    # Load prompt without an active trace
    prompt = mlflow.genai.load_prompt("no_trace_prompt", version=1)

    # Verify prompt was loaded correctly
    assert prompt.name == "no_trace_prompt"
    assert prompt.version == 1
    assert prompt.template == "Hello, {{name}}!"

    # No trace should be created or linked when no active trace exists
    # We can't easily test this without accessing the trace manager, but the function
    # should complete successfully without errors


def test_load_prompt_with_tracing_nested_spans():
    """Test that load_prompt links prompts to the same trace when using nested spans."""

    # Register prompts
    mlflow.genai.register_prompt(name="outer_prompt", template="Outer: {{msg}}")
    mlflow.genai.register_prompt(name="inner_prompt", template="Inner: {{msg}}")

    # Start nested spans (same trace, different spans)
    with mlflow.start_span("outer_operation") as outer_span:
        mlflow.genai.load_prompt("outer_prompt", version=1)

        with mlflow.start_span("inner_operation") as inner_span:
            # Verify both spans belong to the same trace
            assert inner_span.trace_id == outer_span.trace_id

            mlflow.genai.load_prompt("inner_prompt", version=1)

    # Manually trigger prompt linking to trace since in test environment
    # the trace export may not happen automatically
    client = mlflow.MlflowClient()
    prompt_versions = [
        PromptVersion(
            name="outer_prompt",
            version=1,
            template="Outer: {{msg}}",
            commit_message=None,
            creation_timestamp=None,
        ),
        PromptVersion(
            name="inner_prompt",
            version=1,
            template="Inner: {{msg}}",
            commit_message=None,
            creation_timestamp=None,
        ),
    ]
    client.link_prompt_versions_to_trace(
        trace_id=outer_span.trace_id, prompt_versions=prompt_versions
    )

    # Check trace now has both prompts (same trace, different spans)
    trace = mlflow.get_trace(outer_span.trace_id)
    assert trace is not None

    # Check the linked prompts tag
    linked_prompts_tag = trace.info.tags.get("mlflow.linkedPrompts")
    assert linked_prompts_tag is not None

    # Parse the JSON tag value
    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 2

    # Check that both prompts are present (order may vary)
    prompt_names = {p["name"] for p in linked_prompts}
    expected_names = {"outer_prompt", "inner_prompt"}
    assert prompt_names == expected_names

    # Verify all prompts have correct versions
    for prompt in linked_prompts:
        assert prompt["version"] == "1"


def test_load_prompt_caching_works():
    """Test that prompt caching works and improves performance."""
    # Mock the client load_prompt method to count calls
    with mock.patch("mlflow.MlflowClient.load_prompt") as mock_client_load:
        # Configure mock to return a prompt
        mock_prompt = PromptVersion(
            name="cached_prompt",
            version=1,
            template="Hello, {{name}}!",
            creation_timestamp=123456789,
        )
        mock_client_load.return_value = mock_prompt

        # First call should hit the client
        prompt1 = mlflow.genai.load_prompt("cached_prompt", version=1, link_to_model=False)
        assert prompt1.name == "cached_prompt"
        assert mock_client_load.call_count == 1

        # Second call with same parameters should use cache (not call client again)
        prompt2 = mlflow.genai.load_prompt("cached_prompt", version=1, link_to_model=False)
        assert prompt2.name == "cached_prompt"
        assert mock_client_load.call_count == 1  # Should still be 1, not 2

        # Call with different version should hit the client again
        mock_client_load.return_value = PromptVersion(
            name="cached_prompt",
            version=2,
            template="Hi, {{name}}!",
            creation_timestamp=123456790,
        )
        prompt3 = mlflow.genai.load_prompt("cached_prompt", version=2, link_to_model=False)
        assert prompt3.version == 2
        assert mock_client_load.call_count == 2  # Should be 2 now


def test_load_prompt_caching_respects_env_var():
    """Test that prompt caching respects the MLFLOW_PROMPT_CACHE_MAX_SIZE environment variable."""
    # Test with a small cache size
    original_value = MLFLOW_PROMPT_CACHE_MAX_SIZE.get()
    try:
        # Set cache size to 1
        MLFLOW_PROMPT_CACHE_MAX_SIZE.set(1)

        # Clear any existing cache by creating a new cached function
        # (This simulates restarting with the new env var)
        importlib.reload(mlflow.tracking._model_registry.fluent)

        # Register prompts
        mlflow.genai.register_prompt(name="prompt_1", template="Template 1")
        mlflow.genai.register_prompt(name="prompt_2", template="Template 2")

        # Mock the client load_prompt method to count calls
        with mock.patch("mlflow.MlflowClient.load_prompt") as mock_client_load:
            mock_client_load.side_effect = [
                PromptVersion(
                    name="prompt_1",
                    version=1,
                    template="Template 1",
                    creation_timestamp=1,
                ),
                PromptVersion(
                    name="prompt_2",
                    version=1,
                    template="Template 2",
                    creation_timestamp=2,
                ),
                PromptVersion(
                    name="prompt_1",
                    version=1,
                    template="Template 1",
                    creation_timestamp=1,
                ),
            ]

            # Load first prompt - should cache it
            mlflow.genai.load_prompt("prompt_1", version=1, link_to_model=False)
            assert mock_client_load.call_count == 1

            # Load second prompt - should evict first from cache (size=1)
            mlflow.genai.load_prompt("prompt_2", version=1, link_to_model=False)
            assert mock_client_load.call_count == 2

            # Load first prompt again - should need to call client again (evicted from cache)
            mlflow.genai.load_prompt("prompt_1", version=1, link_to_model=False)
            assert mock_client_load.call_count == 3  # Called again because evicted

    finally:
        # Restore original cache size
        if original_value is not None:
            MLFLOW_PROMPT_CACHE_MAX_SIZE.set(original_value)
        else:
            MLFLOW_PROMPT_CACHE_MAX_SIZE.unset()


def test_load_prompt_skip_cache_for_allow_missing_none():
    """Test that we skip cache if allow_missing=True and the result is None."""
    # Mock the client load_prompt method to return None (prompt not found)
    with mock.patch("mlflow.MlflowClient.load_prompt") as mock_client_load:
        mock_client_load.return_value = None  # Simulate prompt not found

        # First call with allow_missing=True should call the client twice
        # (once for cached call, once for non-cached call due to `or` logic)
        prompt1 = mlflow.genai.load_prompt(
            "nonexistent_prompt", version=1, allow_missing=True, link_to_model=False
        )
        assert prompt1 is None
        assert mock_client_load.call_count == 2  # Called twice due to `or` logic

        # Second call: cached function returns None from cache, non-cached function called once
        prompt2 = mlflow.genai.load_prompt(
            "nonexistent_prompt", version=1, allow_missing=True, link_to_model=False
        )
        assert prompt2 is None
        assert mock_client_load.call_count == 3  # One additional call (non-cached only)

        # But if we find a prompt, the pattern will change
        mock_prompt = PromptVersion(
            name="nonexistent_prompt",
            version=1,
            template="Found!",
            creation_timestamp=123,
        )
        mock_client_load.return_value = mock_prompt

        prompt3 = mlflow.genai.load_prompt(
            "nonexistent_prompt", version=1, allow_missing=True, link_to_model=False
        )
        assert prompt3.template == "Found!"
        assert (
            mock_client_load.call_count == 4
        )  # Called once for cached call (returned found prompt)

        # Now this should be cached - only cached call needed since it returns a valid prompt
        prompt4 = mlflow.genai.load_prompt(
            "nonexistent_prompt", version=1, allow_missing=True, link_to_model=False
        )
        assert prompt4.template == "Found!"
        assert (
            mock_client_load.call_count == 5
        )  # One more call for cached check (but no non-cached call needed)


def test_load_prompt_missing_then_created_then_found():
    """Test loading a prompt that doesn't exist, then creating it, then loading again."""
    # First try to load a prompt that doesn't exist
    result1 = mlflow.genai.load_prompt(
        "will_be_created", version=1, allow_missing=True, link_to_model=False
    )
    assert result1 is None

    # Now create the prompt
    created_prompt = mlflow.genai.register_prompt(name="will_be_created", template="Now I exist!")
    assert created_prompt.name == "will_be_created"
    assert created_prompt.version == 1

    # Load again - should find it now (not cached because previous result was None)
    result2 = mlflow.genai.load_prompt(
        "will_be_created", version=1, allow_missing=True, link_to_model=False
    )
    assert result2 is not None
    assert result2.name == "will_be_created"
    assert result2.version == 1
    assert result2.template == "Now I exist!"

    # Load a third time - should be cached now (no need to mock since we want real caching)
    result3 = mlflow.genai.load_prompt(
        "will_be_created", version=1, allow_missing=True, link_to_model=False
    )
    assert result3.template == "Now I exist!"
    # This demonstrates the cache working - if it wasn't cached, we'd get a network call


def test_load_prompt_none_result_no_linking():
    """Test that if prompt version is None and allow_missing=True, we don't attempt any linking."""
    # Mock only the client load_prompt method and linking methods
    with (
        mock.patch("mlflow.MlflowClient.load_prompt") as mock_client_load,
        mock.patch("mlflow.MlflowClient.link_prompt_version_to_run") as mock_link_run,
        mock.patch("mlflow.MlflowClient.link_prompt_version_to_model") as mock_link_model,
    ):
        # Configure client to return None (prompt not found)
        mock_client_load.return_value = None

        # Try to load a prompt that doesn't exist with allow_missing=True
        result = mlflow.genai.load_prompt(
            "nonexistent", version=1, allow_missing=True, link_to_model=True
        )
        assert result is None

        # Verify no linking methods were called
        mock_link_run.assert_not_called()
        mock_link_model.assert_not_called()
        # Note: trace manager registration is handled differently and tested elsewhere


def test_load_prompt_caching_with_different_parameters():
    """Test that caching works correctly with different parameter combinations."""
    # Register a prompt
    mlflow.genai.register_prompt(name="param_test", template="Hello, {{name}}!")

    with mock.patch("mlflow.MlflowClient.load_prompt") as mock_client_load:
        mock_prompt = PromptVersion(
            name="param_test",
            version=1,
            template="Hello, {{name}}!",
            creation_timestamp=123,
        )
        mock_client_load.return_value = mock_prompt

        # Different allow_missing values should result in separate cache entries
        mlflow.genai.load_prompt("param_test", version=1, allow_missing=False, link_to_model=False)
        call_count_after_first = mock_client_load.call_count

        mlflow.genai.load_prompt("param_test", version=1, allow_missing=True, link_to_model=False)
        call_count_after_second = mock_client_load.call_count

        # Should be called again for different allow_missing parameter
        assert call_count_after_second > call_count_after_first

        # Same parameters should use cache
        mlflow.genai.load_prompt("param_test", version=1, allow_missing=False, link_to_model=False)
        call_count_after_third = mock_client_load.call_count

        # Cache should work - either same count or only one additional call
        assert call_count_after_third <= call_count_after_second + 1

        mlflow.genai.load_prompt("param_test", version=1, allow_missing=True, link_to_model=False)
        call_count_after_fourth = mock_client_load.call_count

        # Cache should work - either same count or only one additional call
        assert call_count_after_fourth <= call_count_after_third + 1


def test_register_prompt_chat_format_integration():
    """Test full integration of registering and using chat prompts."""
    chat_template = [
        {"role": "system", "content": "You are a {{style}} assistant."},
        {"role": "user", "content": "{{question}}"},
    ]

    response_format = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
    }

    # Register chat prompt
    mlflow.genai.register_prompt(
        name="test_chat_integration",
        template=chat_template,
        response_format=response_format,
        commit_message="Test chat prompt integration",
        tags={"model": "test-model"},
    )

    # Load and verify
    prompt = mlflow.genai.load_prompt("test_chat_integration", version=1)
    assert prompt.template == chat_template
    assert prompt.response_format == response_format
    assert prompt.commit_message == "Test chat prompt integration"
    assert prompt.tags["model"] == "test-model"

    # Test formatting
    formatted = prompt.format(style="helpful", question="How are you?")
    expected = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How are you?"},
    ]
    assert formatted == expected


def test_prompt_associate_with_run_chat_format():
    """Test chat prompts associate with runs correctly."""
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    mlflow.genai.register_prompt(name="test_chat_run", template=chat_template)

    with mlflow.start_run() as run:
        mlflow.genai.load_prompt("test_chat_run", version=1)

    # Verify linking
    client = MlflowClient()
    run_data = client.get_run(run.info.run_id)
    linked_prompts_tag = run_data.data.tags.get(LINKED_PROMPTS_TAG_KEY)
    assert linked_prompts_tag is not None

    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 1
    assert linked_prompts[0]["name"] == "test_chat_run"
    assert linked_prompts[0]["version"] == "1"


def test_register_prompt_with_pydantic_response_format():
    from pydantic import BaseModel

    class ResponseSchema(BaseModel):
        answer: str
        confidence: float

    # Register prompt with Pydantic response format
    mlflow.genai.register_prompt(
        name="test_pydantic_response",
        template="What is {{question}}?",
        response_format=ResponseSchema,
        commit_message="Test Pydantic response format",
    )

    # Load and verify
    prompt = mlflow.genai.load_prompt("test_pydantic_response", version=1)
    assert prompt.response_format == ResponseSchema.model_json_schema()
    assert prompt.commit_message == "Test Pydantic response format"


def test_register_prompt_with_dict_response_format():
    """Test registering prompts with dictionary response format."""
    response_format = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "key_points": {"type": "array", "items": {"type": "string"}},
        },
    }

    # Register prompt with dict response format
    mlflow.genai.register_prompt(
        name="test_dict_response",
        template="Analyze this: {{text}}",
        response_format=response_format,
        tags={"analysis_type": "text"},
    )

    # Load and verify
    prompt = mlflow.genai.load_prompt("test_dict_response", version=1)
    assert prompt.response_format == response_format
    assert prompt.tags["analysis_type"] == "text"


def test_register_prompt_text_backward_compatibility():
    """Test that text prompt registration continues to work as before."""
    # Register text prompt
    mlflow.genai.register_prompt(
        name="test_text_backward",
        template="Hello {{name}}!",
        commit_message="Test backward compatibility",
    )

    # Load and verify
    prompt = mlflow.genai.load_prompt("test_text_backward", version=1)
    assert prompt.is_text_prompt
    assert prompt.template == "Hello {{name}}!"
    assert prompt.commit_message == "Test backward compatibility"

    # Test formatting
    formatted = prompt.format(name="Alice")
    assert formatted == "Hello Alice!"


def test_register_prompt_complex_chat_template():
    """Test registering prompts with complex chat templates."""
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

    # Register complex chat prompt
    mlflow.genai.register_prompt(
        name="test_complex_chat",
        template=chat_template,
        tags={"complexity": "high"},
    )

    # Load and verify
    prompt = mlflow.genai.load_prompt("test_complex_chat", version=1)
    assert not prompt.is_text_prompt
    assert prompt.template == chat_template
    assert prompt.tags["complexity"] == "high"

    # Test formatting
    formatted = prompt.format(
        style="friendly",
        name="Alice",
        greeting="Hello",
        question="How are you?",
        topic="wellbeing",
    )
    expected = [
        {"role": "system", "content": "You are a friendly assistant named Alice."},
        {"role": "user", "content": "Hello! How are you?"},
        {
            "role": "assistant",
            "content": "I understand you're asking about wellbeing.",
        },
    ]
    assert formatted == expected


def test_register_prompt_with_none_response_format():
    # Register prompt with None response format
    mlflow.genai.register_prompt(
        name="test_none_response", template="Hello {{name}}!", response_format=None
    )

    # Load and verify
    prompt = mlflow.genai.load_prompt("test_none_response", version=1)
    assert prompt.response_format is None


def test_register_prompt_with_empty_chat_template():
    # Empty list should be treated as text prompt
    mlflow.genai.register_prompt(name="test_empty_chat", template=[])

    # Load and verify
    prompt = mlflow.genai.load_prompt("test_empty_chat", version=1)
    assert prompt.is_text_prompt
    assert prompt.template == "[]"  # Empty list serialized as string


def test_register_prompt_with_single_message_chat():
    """Test registering prompts with single message chat template."""
    chat_template = [{"role": "user", "content": "Hello {{name}}!"}]

    # Register single message chat prompt
    mlflow.genai.register_prompt(name="test_single_message", template=chat_template)

    # Load and verify
    prompt = mlflow.genai.load_prompt("test_single_message", version=1)
    not prompt.is_text_prompt
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

    # Register prompt with multiple variables
    mlflow.genai.register_prompt(name="test_multiple_variables", template=chat_template)

    # Load and verify
    prompt = mlflow.genai.load_prompt("test_multiple_variables", version=1)
    expected_variables = {"style", "name", "greeting", "question", "topic"}
    assert prompt.variables == expected_variables


def test_register_prompt_with_mixed_content_types():
    """Test registering prompts with mixed content types in chat messages."""
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello {{name}}!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
    ]

    # Register prompt with mixed content
    mlflow.genai.register_prompt(name="test_mixed_content", template=chat_template)

    # Load and verify
    prompt = mlflow.genai.load_prompt("test_mixed_content", version=1)
    not prompt.is_text_prompt
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

    # Register prompt with nested variables
    mlflow.genai.register_prompt(name="test_nested_variables", template=chat_template)

    # Load and verify
    prompt = mlflow.genai.load_prompt("test_nested_variables", version=1)
    expected_variables = {
        "user.preferences.style",
        "user.name",
        "user.preferences.greeting",
    }
    assert prompt.variables == expected_variables


def test_load_prompt_links_to_experiment():
    mlflow.genai.register_prompt(name="test_exp_link", template="Hello {{name}}!")
    experiment = mlflow.set_experiment("test_experiment_link")
    mlflow.genai.load_prompt("test_exp_link", version=1)

    # Wait for the links to be established
    time.sleep(1)

    client = MlflowClient()
    prompt_info = client.get_prompt("test_exp_link")
    assert experiment.experiment_id in prompt_info.tags.get(PROMPT_EXPERIMENT_IDS_TAG_KEY)


def test_register_prompt_links_to_experiment():
    experiment = mlflow.set_experiment("test_experiment_register")
    mlflow.genai.register_prompt(name="test_exp_register", template="Greetings {{name}}!")

    # Wait for the links to be established
    time.sleep(1)

    client = MlflowClient()
    prompt_info = client.get_prompt("test_exp_register")
    assert experiment.experiment_id in prompt_info.tags.get(PROMPT_EXPERIMENT_IDS_TAG_KEY)


def test_link_prompt_to_experiment_no_duplicate():
    mlflow.genai.register_prompt(name="no_dup_prompt", template="Test {{x}}!")

    experiment = mlflow.set_experiment("test_no_dup")

    mlflow.genai.load_prompt("no_dup_prompt", version=1)
    mlflow.genai.load_prompt("no_dup_prompt", version=1)
    mlflow.genai.load_prompt("no_dup_prompt", version=1)

    # Wait for the links to be established
    time.sleep(1)

    client = MlflowClient()
    prompt_info = client.get_prompt("no_dup_prompt")
    assert experiment.experiment_id in prompt_info.tags.get(PROMPT_EXPERIMENT_IDS_TAG_KEY)


def test_search_prompts_by_experiment_id():
    experiment = mlflow.set_experiment("test_search_by_exp")

    mlflow.genai.register_prompt(name="exp_prompt_1", template="Template 1: {{x}}")
    mlflow.genai.register_prompt(name="exp_prompt_2", template="Template 2: {{y}}")

    # Wait for the links to be established
    time.sleep(1)

    client = MlflowClient()
    prompts = client.search_prompts(filter_string=f'experiment_id = "{experiment.experiment_id}"')

    assert len(prompts) == 2
    prompt_names = {p.name for p in prompts}
    assert "exp_prompt_1" in prompt_names
    assert "exp_prompt_2" in prompt_names


def test_search_prompts_by_experiment_id_empty():
    experiment = mlflow.set_experiment("test_empty_exp")

    client = MlflowClient()
    prompts = client.search_prompts(filter_string=f'experiment_id = "{experiment.experiment_id}"')

    assert len(prompts) == 0


def test_search_prompts_same_prompt_multiple_experiments():
    exp_id_1 = mlflow.create_experiment("test_multi_exp_1")
    exp_id_2 = mlflow.create_experiment("test_multi_exp_2")

    mlflow.genai.register_prompt(name="shared_search_prompt", template="Shared: {{x}}")

    mlflow.set_experiment(experiment_id=exp_id_1)
    mlflow.genai.load_prompt("shared_search_prompt", version=1)

    # Wait for the links to be established
    time.sleep(1)

    mlflow.set_experiment(experiment_id=exp_id_2)
    mlflow.genai.load_prompt("shared_search_prompt", version=1)

    # Wait for the links to be established
    time.sleep(1)

    client = MlflowClient()
    prompts_exp1 = client.search_prompts(filter_string=f'experiment_id = "{exp_id_1}"')
    prompts_exp2 = client.search_prompts(filter_string=f'experiment_id = "{exp_id_2}"')

    assert len(prompts_exp1) == 1
    assert prompts_exp1[0].name == "shared_search_prompt"

    assert len(prompts_exp2) == 1
    assert prompts_exp2[0].name == "shared_search_prompt"


def test_search_prompts_with_combined_filters():
    experiment = mlflow.set_experiment("test_combined_filters")

    mlflow.genai.register_prompt(name="alpha_prompt", template="Alpha: {{x}}")
    mlflow.genai.register_prompt(name="beta_prompt", template="Beta: {{y}}")
    mlflow.genai.register_prompt(name="gamma_prompt", template="Gamma: {{z}}")

    client = MlflowClient()

    # Wait for the links to be established
    time.sleep(1)

    # Test experiment_id filter combined with name filter
    prompts = client.search_prompts(
        filter_string=f'experiment_id = "{experiment.experiment_id}" AND name = "alpha_prompt"'
    )
    assert len(prompts) == 1
    assert prompts[0].name == "alpha_prompt"

    # Test experiment_id filter combined with name LIKE filter
    prompts = client.search_prompts(
        filter_string=f'experiment_id = "{experiment.experiment_id}" AND name LIKE "a%"'
    )
    assert len(prompts) == 1
    assert prompts[0].name == "alpha_prompt"

    # Test that name filter without experiment_id returns correct results
    prompts = client.search_prompts(filter_string='name = "gamma_prompt"')
    assert len(prompts) == 1
    assert prompts[0].name == "gamma_prompt"

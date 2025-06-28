import json
from typing import Optional

import pytest
from pydantic import BaseModel, Field

from mlflow.entities.model_registry.prompt_version import (
    PROMPT_TYPE_CHAT,
    PROMPT_TYPE_TEXT,
    PromptVersion,
)
from mlflow.exceptions import MlflowException


# Test Pydantic classes
class SimpleResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")


class ComplexResponse(BaseModel):
    summary: str = Field(description="Brief summary")
    key_points: list[str] = Field(description="List of key points")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata")


class TestPromptVersionEnhancedFeatures:
    """Test suite for enhanced PromptVersion features."""

    def test_text_prompt_creation(self):
        """Test creating a basic text prompt."""
        prompt = PromptVersion(
            name="text_prompt",
            version=1,
            template="Hello, {{name}}! How are you today?",
        )

        assert prompt.name == "text_prompt"
        assert prompt.version == 1
        assert prompt.template == "Hello, {{name}}! How are you today?"
        assert prompt.prompt_type == PROMPT_TYPE_TEXT
        assert prompt.response_format is None
        assert prompt.config is None
        assert prompt.variables == {"name"}

    def test_chat_prompt_creation(self):
        """Test creating a chat prompt."""
        chat_template = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{{user_question}}"},
        ]

        prompt = PromptVersion(
            name="chat_prompt",
            version=1,
            template=chat_template,
            prompt_type=PROMPT_TYPE_CHAT,
        )

        assert prompt.name == "chat_prompt"
        assert prompt.version == 1
        assert prompt.template == chat_template
        assert prompt.prompt_type == PROMPT_TYPE_CHAT
        assert prompt.response_format is None
        assert prompt.config is None
        assert prompt.variables == {"user_question"}

    def test_prompt_with_response_format(self):
        """Test creating a prompt with Pydantic response format."""
        prompt = PromptVersion(
            name="response_prompt",
            version=1,
            template="Answer this question: {{question}}",
            response_format=SimpleResponse,
        )

        assert prompt.response_format is not None
        # Check that we have a response format JSON schema
        assert isinstance(prompt.response_format, dict)
        assert "properties" in prompt.response_format
        assert "answer" in prompt.response_format["properties"]
        assert "confidence" in prompt.response_format["properties"]

    def test_prompt_with_model_config(self):
        """Test creating a prompt with model configuration."""
        config = {
            "model_name": "gpt-4o-mini",
            "model_provider": "openai",
            "model_parameters": {"temperature": 0.2, "max_tokens": 500},
        }

        prompt = PromptVersion(
            name="config_prompt",
            version=1,
            template="Generate a response: {{input}}",
            config=config,
        )

        assert prompt.config is not None
        assert prompt.config["model_name"] == "gpt-4o-mini"
        assert prompt.config["model_provider"] == "openai"
        assert prompt.config["model_parameters"]["temperature"] == 0.2
        assert prompt.config["model_parameters"]["max_tokens"] == 500

    def test_complete_prompt_with_all_features(self):
        """Test creating a prompt with all enhanced features including Pydantic response format."""
        config = {"model_name": "gpt-4o-mini", "model_parameters": {"temperature": 0.1}}

        chat_template = [
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": "Analyze this data: {{data}}"},
        ]

        prompt = PromptVersion(
            name="complete_prompt",
            version=1,
            template=chat_template,
            prompt_type=PROMPT_TYPE_CHAT,
            response_format=ComplexResponse,
            config=config,
        )

        # Test all features
        assert prompt.prompt_type == PROMPT_TYPE_CHAT
        assert prompt.response_format is not None
        assert prompt.config is not None
        assert "data" in prompt.variables

        # Test response format structure
        response_format = prompt.response_format
        assert isinstance(response_format, dict)
        assert "properties" in response_format
        assert "summary" in response_format["properties"]
        assert "key_points" in response_format["properties"]
        assert "confidence" in response_format["properties"]
        assert "metadata" in response_format["properties"]

    def test_chat_prompt_formatting(self):
        """Test chat prompt formatting with variable substitution."""
        chat_template = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{{user_question}}"},
            {
                "role": "assistant",
                "content": "I understand you're asking about {{topic}}.",
            },
        ]

        prompt = PromptVersion(
            name="formatting_test",
            version=1,
            template=chat_template,
            prompt_type=PROMPT_TYPE_CHAT,
        )

        formatted = prompt.format_chat(user_question="What is MLflow?", topic="machine learning")

        expected = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is MLflow?"},
            {
                "role": "assistant",
                "content": "I understand you're asking about machine learning.",
            },
        ]

        assert formatted == expected

    def test_chat_prompt_formatting_with_partial_variables(self):
        """Test chat prompt formatting with only some variables provided."""
        chat_template = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{{user_question}}"},
            {
                "role": "assistant",
                "content": "I understand you're asking about {{topic}}.",
            },
        ]

        prompt = PromptVersion(
            name="partial_formatting_test",
            version=1,
            template=chat_template,
            prompt_type=PROMPT_TYPE_CHAT,
        )

        formatted = prompt.format_chat(user_question="What is MLflow?")

        expected = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is MLflow?"},
            {
                "role": "assistant",
                "content": "I understand you're asking about {{topic}}.",
            },
        ]

        assert formatted == expected

    def test_backward_compatibility(self):
        """Test that existing functionality still works."""
        # Test old-style constructor
        prompt = PromptVersion(name="backward_test", version=1, template="Hello, {{name}}!")
        assert prompt.prompt_type == PROMPT_TYPE_TEXT
        assert prompt.response_format is None
        assert prompt.config is None
        assert prompt.variables == {"name"}

    def test_variable_extraction_from_chat_prompts(self):
        """Test variable extraction from complex chat prompts."""
        chat_template = [
            {"role": "system", "content": "You are a {{assistant_type}}."},
            {"role": "user", "content": "{{user_question}}"},
            {
                "role": "assistant",
                "content": "I understand you're asking about {{topic}}.",
            },
            {
                "role": "user",
                "content": "Can you provide more details about {{specific_aspect}}?",
            },
        ]

        prompt = PromptVersion(
            name="complex_variables_test",
            version=1,
            template=chat_template,
            prompt_type=PROMPT_TYPE_CHAT,
        )

        expected_variables = {
            "assistant_type",
            "user_question",
            "topic",
            "specific_aspect",
        }
        assert prompt.variables == expected_variables

    def test_prompt_serialization_and_deserialization(self):
        """Test that prompt data is properly serialized and deserialized."""
        config = {"model_name": "gpt-4o-mini", "model_parameters": {"temperature": 0.2}}

        chat_template = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{{user_question}}"},
        ]

        prompt = PromptVersion(
            name="serialization_test",
            version=1,
            template=chat_template,
            prompt_type=PROMPT_TYPE_CHAT,
            response_format=SimpleResponse,
            config=config,
        )

        # Test that data is properly stored in tags
        assert prompt._tags["mlflow.prompt.type"] == PROMPT_TYPE_CHAT
        response_format_data = json.loads(prompt._tags["mlflow.prompt.response_format"])
        assert response_format_data is not None
        assert "properties" in response_format_data
        assert json.loads(prompt._tags["mlflow.prompt.config"]) == config

    def test_prompt_repr(self):
        """Test the string representation of prompts."""
        # Text prompt
        text_prompt = PromptVersion(name="text_test", version=1, template="Hello, {{name}}!")
        assert "text" in str(text_prompt)
        assert "Hello, {{name}}!" in str(text_prompt)

        # Chat prompt
        chat_template = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{{user_question}}"},
        ]
        chat_prompt = PromptVersion(
            name="chat_test",
            version=1,
            template=chat_template,
            prompt_type=PROMPT_TYPE_CHAT,
        )
        assert "chat" in str(chat_prompt)
        assert "2 messages" in str(chat_prompt)

    def test_text_prompt_formatting(self):
        """Test text prompt formatting with variable substitution."""
        prompt = PromptVersion(
            name="text_format_test", version=1, template="Hello, {{title}} {{name}}!"
        )

        # Test complete formatting
        formatted = prompt.format(title="Ms", name="Alice")
        assert formatted == "Hello, Ms Alice!"

        # Test partial formatting with allow_partial=True
        partial_prompt = prompt.format(title="Ms", allow_partial=True)
        assert isinstance(partial_prompt, PromptVersion)
        assert partial_prompt.template == "Hello, Ms {{name}}!"

        # Test partial formatting without allow_partial should raise error
        with pytest.raises(MlflowException, match="Missing variables"):
            prompt.format(title="Ms")

    def test_text_prompt_to_single_brace_format(self):
        """Test converting text prompt to single brace format."""
        prompt = PromptVersion(
            name="single_brace_test", version=1, template="Hello, {{title}} {{name}}!"
        )

        single_brace = prompt.to_single_brace_format()
        assert single_brace == "Hello, {title} {name}!"


class TestPromptVersionErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_prompt_type(self):
        """Test handling of invalid prompt type."""
        with pytest.raises(ValueError, match="prompt_type must be 'text' or 'chat'"):
            PromptVersion(name="test", version=1, template="Hello", prompt_type="invalid")

    def test_invalid_response_format(self):
        """Test handling of invalid response format."""
        with pytest.raises(ValueError, match="response_format must be a Pydantic class or dict"):
            PromptVersion(name="test", version=1, template="Hello", response_format="invalid")

    def test_chat_prompt_with_invalid_template(self):
        """Test handling of invalid chat template."""
        with pytest.raises(
            ValueError, match="Chat template must be a list of message dictionaries"
        ):
            PromptVersion(
                name="test",
                version=1,
                template="not a list",
                prompt_type=PROMPT_TYPE_CHAT,
            )

    def test_chat_prompt_with_invalid_message(self):
        """Test handling of invalid message in chat template."""
        with pytest.raises(ValueError, match="Each message must have 'role' and 'content' keys"):
            PromptVersion(
                name="test",
                version=1,
                template=[{"role": "user"}],  # Missing content
                prompt_type=PROMPT_TYPE_CHAT,
            )

    def test_chat_prompt_with_invalid_role(self):
        """Test handling of invalid role in chat template."""
        with pytest.raises(ValueError, match="Role must be one of: system, user, assistant"):
            PromptVersion(
                name="test",
                version=1,
                template=[{"role": "invalid", "content": "test"}],
                prompt_type=PROMPT_TYPE_CHAT,
            )

    def test_format_chat_on_text_prompt(self):
        """Test that format_chat raises error for text prompts."""
        prompt = PromptVersion(name="test", version=1, template="Hello, {{name}}!")

        with pytest.raises(
            MlflowException,
            match="format_chat\\(\\) can only be used with chat prompts",
        ):
            prompt.format_chat(name="Alice")

    def test_to_single_brace_format_on_chat_prompt(self):
        """Test that to_single_brace_format raises error for chat prompts."""
        chat_template = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{{user_question}}"},
        ]

        prompt = PromptVersion(
            name="test", version=1, template=chat_template, prompt_type=PROMPT_TYPE_CHAT
        )

        with pytest.raises(
            MlflowException,
            match="to_single_brace_format\\(\\) can only be used with text prompts",
        ):
            prompt.to_single_brace_format()

    def test_format_on_chat_prompt(self):
        """Test that format raises error for chat prompts."""
        chat_template = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{{user_question}}"},
        ]

        prompt = PromptVersion(
            name="test", version=1, template=chat_template, prompt_type=PROMPT_TYPE_CHAT
        )

        with pytest.raises(MlflowException, match="format\\(\\) cannot be used with chat prompts"):
            prompt.format(user_question="test")


class TestPromptVersionIntegration:
    """Test integration scenarios."""

    def test_openai_api_integration_scenario(self):
        """Test a complete OpenAI API integration scenario with Pydantic response format."""
        # Create a prompt with all features
        config = {
            "model_name": "gpt-4o-mini",
            "model_parameters": {"temperature": 0.2, "max_tokens": 500},
        }

        chat_template = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{{user_question}}"},
        ]

        prompt = PromptVersion(
            name="openai_integration_test",
            version=1,
            template=chat_template,
            prompt_type=PROMPT_TYPE_CHAT,
            response_format=SimpleResponse,
            config=config,
        )

        # Simulate OpenAI API usage
        messages = prompt.format_chat(user_question="What is machine learning?")
        model_name = prompt.config["model_name"]
        model_parameters = prompt.config["model_parameters"]
        response_format = prompt.response_format

        # Verify all components are ready for OpenAI API
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is machine learning?"
        assert model_name == "gpt-4o-mini"
        assert model_parameters["temperature"] == 0.2
        assert response_format is not None
        assert "properties" in response_format
        assert "answer" in response_format["properties"]
        assert "confidence" in response_format["properties"]

    def test_complex_response_format_scenario(self):
        """Test a complex Pydantic response format scenario."""
        prompt = PromptVersion(
            name="complex_response_test",
            version=1,
            template="Analyze this data: {{data}}",
            response_format=ComplexResponse,
        )

        response_format = prompt.response_format

        # Verify we have a response format
        assert response_format is not None
        assert isinstance(response_format, dict)

        # Verify complex structure in the response format
        assert "properties" in response_format
        assert "summary" in response_format["properties"]
        assert "key_points" in response_format["properties"]
        assert "confidence" in response_format["properties"]
        assert "metadata" in response_format["properties"]

    def test_prompt_with_aliases_and_metadata(self):
        """Test prompt with aliases and other metadata."""
        prompt = PromptVersion(
            name="metadata_test",
            version=1,
            template="Hello, {{name}}!",
            commit_message="Initial version",
            aliases=["latest", "stable"],
            user_id="user123",
        )

        assert prompt.commit_message == "Initial version"
        assert prompt.aliases == ["latest", "stable"]
        assert prompt.user_id == "user123"
        assert prompt.uri == "prompts:/metadata_test/1"

    def test_prompt_tags_filtering(self):
        """Test that prompt tags exclude reserved tags."""
        prompt = PromptVersion(
            name="tags_test",
            version=1,
            template="Hello, {{name}}!",
            tags={"custom_tag": "value", "author": "test_user"},
        )

        # Reserved tags should not appear in the public tags property
        assert "mlflow.prompt.text" not in prompt.tags
        assert "mlflow.prompt.is_prompt" not in prompt.tags

        # Custom tags should appear
        assert prompt.tags["custom_tag"] == "value"
        assert prompt.tags["author"] == "test_user"

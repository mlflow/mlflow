import json

from mlflow.entities.model_registry.prompt import Prompt
from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.prompt.constants import (
    PROMPT_MODEL_CONFIG_TAG_KEY,
    PROMPT_TYPE_TAG_KEY,
    PROMPT_TYPE_TEXT,
    RESPONSE_FORMAT_TAG_KEY,
)
from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
    Prompt as ProtoPrompt,
)
from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
    PromptTag as ProtoPromptTag,
)
from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
    PromptVersion as ProtoPromptVersion,
)
from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
    PromptVersionTag as ProtoPromptVersionTag,
)
from mlflow.store._unity_catalog.registry.utils import (
    mlflow_prompt_to_proto,
    mlflow_tags_to_proto,
    mlflow_tags_to_proto_version_tags,
    proto_info_to_mlflow_prompt_info,
    proto_to_mlflow_prompt,
    proto_to_mlflow_tags,
    proto_version_tags_to_mlflow_tags,
)


def test_proto_to_mlflow_tags():
    # Test with empty tags
    assert proto_to_mlflow_tags([]) == {}

    # Test with tags
    proto_tags = [
        ProtoPromptTag(key="key1", value="value1"),
        ProtoPromptTag(key="key2", value="value2"),
    ]
    expected = {"key1": "value1", "key2": "value2"}
    assert proto_to_mlflow_tags(proto_tags) == expected

    # Test with None
    assert proto_to_mlflow_tags(None) == {}


def test_mlflow_tags_to_proto():
    # Test with empty tags
    assert mlflow_tags_to_proto({}) == []

    # Test with tags
    tags = {"key1": "value1", "key2": "value2"}
    proto_tags = mlflow_tags_to_proto(tags)
    assert len(proto_tags) == 2
    assert all(isinstance(tag, ProtoPromptTag) for tag in proto_tags)
    assert {tag.key: tag.value for tag in proto_tags} == tags

    # Test with None
    assert mlflow_tags_to_proto(None) == []


def test_proto_info_to_mlflow_prompt_info():
    # Create test proto info
    proto_info = ProtoPrompt(
        name="test_prompt",
        description="Test prompt description",
        tags=[
            ProtoPromptTag(key="key1", value="value1"),
            ProtoPromptTag(key="key2", value="value2"),
        ],
    )

    # Test without prompt tags
    prompt_info = proto_info_to_mlflow_prompt_info(proto_info)
    assert isinstance(prompt_info, Prompt)
    assert prompt_info.name == "test_prompt"
    assert prompt_info.description == "Test prompt description"
    assert prompt_info.tags == {"key1": "value1", "key2": "value2"}

    # Test with additional prompt tags
    prompt_tags = {"tag1": "value1", "tag2": "value2"}
    prompt_info = proto_info_to_mlflow_prompt_info(proto_info, prompt_tags)
    expected_tags = {
        "key1": "value1",
        "key2": "value2",
        "tag1": "value1",
        "tag2": "value2",
    }
    assert prompt_info.tags == expected_tags


def test_proto_to_mlflow_prompt():
    # Test with version tags including response_format and model_config
    proto_version = ProtoPromptVersion()
    proto_version.name = "test_prompt"
    proto_version.version = "1"
    proto_version.template = json.dumps("Hello {{name}}!")
    proto_version.description = "Test description"

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "test_schema",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        },
    }

    model_config = {
        "model_name": "databricks-meta-llama-3-1-70b-instruct",
        "max_tokens": 100,
        "temperature": 0.7,
    }

    # Add version tags including both response_format and model_config
    proto_version.tags.extend(
        [
            ProtoPromptVersionTag(key="env", value="production"),
            ProtoPromptVersionTag(key="author", value="alice"),
            ProtoPromptVersionTag(key=PROMPT_TYPE_TAG_KEY, value=PROMPT_TYPE_TEXT),
            ProtoPromptVersionTag(
                key=RESPONSE_FORMAT_TAG_KEY,
                value=json.dumps(response_format),
            ),
            ProtoPromptVersionTag(
                key=PROMPT_MODEL_CONFIG_TAG_KEY,
                value=json.dumps(model_config),
            ),
        ]
    )

    result = proto_to_mlflow_prompt(proto_version)

    # Verify response_format and model_config are extracted
    assert result.template == "Hello {{name}}!"
    assert result.response_format == response_format
    assert result.model_config == model_config

    # Verify user tags remain and _mlflow tags are filtered out
    expected_tags = {"env": "production", "author": "alice"}
    assert result.tags == expected_tags
    assert RESPONSE_FORMAT_TAG_KEY not in result.tags
    assert PROMPT_MODEL_CONFIG_TAG_KEY not in result.tags
    assert PROMPT_TYPE_TAG_KEY not in result.tags

    # Test with no tags
    proto_no_tags = ProtoPromptVersion()
    proto_no_tags.name = "no_tags_prompt"
    proto_no_tags.version = "2"
    proto_no_tags.template = json.dumps("Simple template")

    result_no_tags = proto_to_mlflow_prompt(proto_no_tags)
    assert result_no_tags.tags == {}
    assert result_no_tags.response_format is None
    assert result_no_tags.model_config is None


def test_mlflow_prompt_to_proto():
    # Create test prompt (skip timestamp for simplicity)
    prompt = PromptVersion(
        name="test_prompt",
        version=1,
        template="Hello {{name}}!",
        commit_message="Test prompt",
        tags={"key1": "value1", "key2": "value2"},
        aliases=["production"],
    )

    # Convert to proto
    proto_version = mlflow_prompt_to_proto(prompt)

    # Verify conversion
    assert isinstance(proto_version, ProtoPromptVersion)
    assert proto_version.name == "test_prompt"
    assert proto_version.version == "1"
    assert proto_version.template == "Hello {{name}}!"
    assert proto_version.description == "Test prompt"
    tags_dict = {tag.key: tag.value for tag in proto_version.tags}
    assert tags_dict == {"key1": "value1", "key2": "value2"}

    # Test with empty fields
    prompt = PromptVersion(name="test_prompt", version=1, template="Hello {{name}}!")
    proto_version = mlflow_prompt_to_proto(prompt)
    assert len(proto_version.tags) == 0


def test_proto_version_tags_to_mlflow_tags():
    # Test with empty tags
    assert proto_version_tags_to_mlflow_tags([]) == {}

    # Test with version tags
    proto_tags = [
        ProtoPromptVersionTag(key="key1", value="value1"),
        ProtoPromptVersionTag(key="key2", value="value2"),
    ]
    expected = {"key1": "value1", "key2": "value2"}
    assert proto_version_tags_to_mlflow_tags(proto_tags) == expected

    # Test with None
    assert proto_version_tags_to_mlflow_tags(None) == {}


def test_mlflow_tags_to_proto_version_tags():
    # Test with empty tags
    assert mlflow_tags_to_proto_version_tags({}) == []

    # Test with tags
    tags = {"key1": "value1", "key2": "value2"}
    proto_tags = mlflow_tags_to_proto_version_tags(tags)
    assert len(proto_tags) == 2
    assert all(isinstance(tag, ProtoPromptVersionTag) for tag in proto_tags)
    assert {tag.key: tag.value for tag in proto_tags} == tags

    # Test with None
    assert mlflow_tags_to_proto_version_tags(None) == []

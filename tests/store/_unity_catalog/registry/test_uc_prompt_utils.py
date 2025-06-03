from mlflow.entities.model_registry.prompt import Prompt
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
from mlflow.store._unity_catalog.registry.prompt_info import PromptInfo
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
    assert isinstance(prompt_info, PromptInfo)
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
    # Create test proto version using proper message initialization
    proto_version = ProtoPromptVersion()
    proto_version.name = "test_prompt"
    proto_version.version = "1"
    proto_version.template = "Hello {{name}}!"
    proto_version.description = "Test prompt"
    # Skip timestamp for now as it's a complex protobuf type

    # Add version tags (use PromptVersionTag for prompt versions)
    tag1 = ProtoPromptVersionTag()
    tag1.key = "key1"
    tag1.value = "value1"
    tag2 = ProtoPromptVersionTag()
    tag2.key = "key2"
    tag2.value = "value2"
    proto_version.tags.extend([tag1, tag2])

    # Test without prompt tags
    prompt = proto_to_mlflow_prompt(proto_version)
    assert isinstance(prompt, Prompt)
    assert prompt.name == "test_prompt"
    assert prompt.version == 1
    assert prompt.template == "Hello {{name}}!"
    assert prompt.commit_message == "Test prompt"
    assert prompt.version_metadata == {"key1": "value1", "key2": "value2"}
    assert prompt.tags == {}  # No prompt-level tags
    assert prompt.aliases == []

    # Test with prompt tags
    prompt_tags = {"prompt_tag1": "value1", "prompt_tag2": "value2"}
    prompt = proto_to_mlflow_prompt(proto_version, prompt_tags)
    # Should have prompt-level tags
    assert prompt.tags == prompt_tags
    # Version metadata should still be there
    assert prompt.version_metadata == {"key1": "value1", "key2": "value2"}


def test_mlflow_prompt_to_proto():
    # Create test prompt (skip timestamp for simplicity)
    prompt = Prompt(
        name="test_prompt",
        version=1,
        template="Hello {{name}}!",
        commit_message="Test prompt",
        version_metadata={"key1": "value1", "key2": "value2"},
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
    prompt = Prompt(name="test_prompt", version=1, template="Hello {{name}}!")
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

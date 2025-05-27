import pytest

from mlflow.entities.model_registry.prompt import Prompt
from mlflow.protos.databricks_uc_registry_prompts_pb2 import (
    PromptVersion as ProtoPromptVersion,
    PromptTag as ProtoPromptTag,
    PromptVersionTag as ProtoPromptVersionTag,
)
from mlflow.store._unity_catalog.registry.utils import (
    proto_to_mlflow_tags,
    mlflow_tags_to_proto,
    proto_to_mlflow_prompt,
    mlflow_prompt_to_proto,
)

def test_proto_to_mlflow_tags():
    # Test with empty tags
    assert proto_to_mlflow_tags([]) == {}
    
    # Test with tags
    proto_tags = [
        ProtoPromptVersionTag(key="key1", value="value1"),
        ProtoPromptVersionTag(key="key2", value="value2")
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
    assert all(isinstance(tag, ProtoPromptVersionTag) for tag in proto_tags)
    assert {tag.key: tag.value for tag in proto_tags} == tags
    
    # Test with None
    assert mlflow_tags_to_proto(None) == []

def test_proto_to_mlflow_prompt():
    # Create test proto version
    proto_version = ProtoPromptVersion(
        name="test_prompt",
        version=1,
        template="Hello {{name}}!",
        description="Test prompt",
        creation_timestamp=123456789,
        tags=[
            ProtoPromptVersionTag(key="key1", value="value1"),
            ProtoPromptVersionTag(key="key2", value="value2")
        ]
    )
    proto_version.aliases.append(ProtoPromptVersion.PromptAlias(alias="production"))
    
    # Test without prompt tags
    prompt = proto_to_mlflow_prompt(proto_version)
    assert isinstance(prompt, Prompt)
    assert prompt.name == "test_prompt"
    assert prompt.version == 1
    assert prompt.template == "Hello {{name}}!"
    assert prompt.commit_message == "Test prompt"
    assert prompt.creation_timestamp == 123456789
    assert prompt.version_metadata == {"key1": "value1", "key2": "value2"}
    assert prompt.prompt_tags == None
    assert prompt.aliases == ["production"]
    
    # Test with prompt tags
    prompt_tags = {"tag1": "value1", "tag2": "value2"}
    prompt = proto_to_mlflow_prompt(proto_version, prompt_tags)
    assert prompt.prompt_tags == prompt_tags

def test_mlflow_prompt_to_proto():
    # Create test prompt
    prompt = Prompt(
        name="test_prompt",
        version=1,
        template="Hello {{name}}!",
        commit_message="Test prompt",
        creation_timestamp=123456789,
        version_metadata={"key1": "value1", "key2": "value2"},
        prompt_tags={"tag1": "value1", "tag2": "value2"},
        aliases=["production"]
    )
    
    # Convert to proto
    proto_version = mlflow_prompt_to_proto(prompt)
    
    # Verify conversion
    assert isinstance(proto_version, ProtoPromptVersion)
    assert proto_version.name == "test_prompt"
    assert proto_version.version == 1
    assert proto_version.template == "Hello {{name}}!"
    assert proto_version.description == "Test prompt"
    assert proto_version.creation_timestamp == 123456789
    assert {tag.key: tag.value for tag in proto_version.tags} == {"key1": "value1", "key2": "value2"}
    assert [alias.alias for alias in proto_version.aliases] == ["production"]
    
    # Test with empty fields
    prompt = Prompt(
        name="test_prompt",
        version=1,
        template="Hello {{name}}!"
    )
    proto_version = mlflow_prompt_to_proto(prompt)
    assert not proto_version.tags
    assert not proto_version.aliases 
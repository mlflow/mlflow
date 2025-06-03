"""
Utility functions for converting between Unity Catalog proto and MLflow entities.
"""

from typing import Optional

from mlflow.entities.model_registry.prompt import Prompt
from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
    PromptAlias as ProtoPromptAlias,
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


def proto_to_mlflow_tags(proto_tags: list[ProtoPromptTag]) -> dict[str, str]:
    """Convert proto prompt tags to MLflow tags dictionary."""
    return {tag.key: tag.value for tag in proto_tags} if proto_tags else {}


def mlflow_tags_to_proto(tags: dict[str, str]) -> list[ProtoPromptTag]:
    """Convert MLflow tags dictionary to proto prompt tags."""
    return [ProtoPromptTag(key=k, value=v) for k, v in tags.items()] if tags else []


def proto_version_tags_to_mlflow_tags(proto_tags: list[ProtoPromptVersionTag]) -> dict[str, str]:
    """Convert proto prompt version tags to MLflow tags dictionary."""
    return {tag.key: tag.value for tag in proto_tags} if proto_tags else {}


def mlflow_tags_to_proto_version_tags(tags: dict[str, str]) -> list[ProtoPromptVersionTag]:
    """Convert MLflow tags dictionary to proto prompt version tags."""
    return [ProtoPromptVersionTag(key=k, value=v) for k, v in tags.items()] if tags else []


def proto_info_to_mlflow_prompt_info(
    proto_info,  # Prompt type from protobuf
    prompt_tags: Optional[dict[str, str]] = None,
) -> PromptInfo:
    """Convert proto Prompt to MLflow PromptInfo entity.

    Prompt doesn't have template or version fields.
    This is used for create_prompt and search_prompts responses.
    """
    tags = proto_to_mlflow_tags(proto_info.tags) if proto_info.tags else {}
    if prompt_tags:
        tags.update(prompt_tags)

    return PromptInfo(
        name=proto_info.name,
        description=proto_info.description,
        tags=tags,
    )


def proto_to_mlflow_prompt(
    proto_version,  # PromptVersion type from protobuf
    prompt_tags: Optional[dict[str, str]] = None,
) -> Prompt:
    """Convert proto PromptVersion to MLflow prompt entity.

    PromptVersion has template and version fields.
    This is used for get_prompt_version responses.
    """
    # Extract version tags
    version_tags = (
        proto_version_tags_to_mlflow_tags(proto_version.tags) if proto_version.tags else {}
    )

    # Extract aliases
    aliases = []
    if hasattr(proto_version, "aliases") and proto_version.aliases:
        aliases = [alias.alias for alias in proto_version.aliases]

    return Prompt(
        name=proto_version.name,
        version=int(proto_version.version),
        template=proto_version.template,
        commit_message=proto_version.description,
        creation_timestamp=proto_version.creation_timestamp,
        version_metadata=version_tags,
        prompt_tags=prompt_tags,
        aliases=aliases,
    )


def mlflow_prompt_to_proto(prompt: Prompt) -> ProtoPromptVersion:
    """Convert MLflow prompt entity to proto prompt version."""
    proto_version = ProtoPromptVersion()
    proto_version.name = prompt.name
    proto_version.version = str(prompt.version)
    proto_version.template = prompt.template
    if prompt.commit_message:
        proto_version.description = prompt.commit_message
    if prompt.creation_timestamp:
        proto_version.creation_timestamp = prompt.creation_timestamp

    # Add version tags
    if prompt.version_metadata:
        proto_version.tags.extend(mlflow_tags_to_proto_version_tags(prompt.version_metadata))

    # Add aliases
    if prompt.aliases:
        for alias in prompt.aliases:
            alias_proto = ProtoPromptAlias()
            alias_proto.alias = alias
            alias_proto.version = str(prompt.version)
            proto_version.aliases.append(alias_proto)

    return proto_version

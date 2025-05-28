"""
Utility functions for converting between Unity Catalog proto and MLflow entities.
"""

from typing import Dict, List, Optional

from mlflow.entities.model_registry.prompt import Prompt
from mlflow.entities.model_registry.prompt_info import PromptInfo
from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
    PromptVersionInfo as ProtoPromptVersion,
    PromptTag as ProtoPromptTag,
)

def proto_to_mlflow_tags(proto_tags: List[ProtoPromptTag]) -> Dict[str, str]:
    """Convert proto tags to MLflow tags dictionary."""
    return {tag.key: tag.value for tag in proto_tags} if proto_tags else {}

def mlflow_tags_to_proto(tags: Dict[str, str]) -> List[ProtoPromptTag]:
    """Convert MLflow tags dictionary to proto tags."""
    return [ProtoPromptTag(key=k, value=v) for k, v in tags.items()] if tags else []

def proto_info_to_mlflow_prompt_info(
    proto_info,  # PromptInfo type from protobuf
    prompt_tags: Optional[Dict[str, str]] = None
) -> PromptInfo:
    """Convert proto PromptInfo to MLflow PromptInfo entity.
    
    PromptInfo doesn't have template or version fields.
    This is used for create_prompt and search_prompts responses.
    """
    return PromptInfo(
        name=proto_info.name,
        description=proto_info.description,
        creation_timestamp=proto_info.creation_timestamp,
        tags=proto_to_mlflow_tags(proto_info.tags),
    )

def proto_to_mlflow_prompt(
    proto_version,  # PromptVersionInfo type from protobuf
    prompt_tags: Optional[Dict[str, str]] = None
) -> Prompt:
    """Convert proto PromptVersionInfo to MLflow prompt entity.
    
    PromptVersionInfo has template and version fields.
    This is used for get_prompt_version responses.
    """
    return Prompt(
        name=proto_version.name,
        version=int(proto_version.version) if proto_version.version else None,
        template=proto_version.template or "",
        commit_message=proto_version.description,
        creation_timestamp=proto_version.creation_timestamp,
        version_metadata=proto_to_mlflow_tags(proto_version.tags),
        prompt_tags=prompt_tags or {},
    )

def mlflow_prompt_to_proto(prompt: Prompt) -> ProtoPromptVersion:
    """Convert MLflow prompt entity to proto prompt version."""
    proto_version = ProtoPromptVersion(
        name=prompt.name,
        version=prompt.version,
        template=prompt.template,
        description=prompt.commit_message,
        creation_timestamp=prompt.creation_timestamp,
    )
    
    # Add version tags
    if prompt.version_metadata:
        proto_version.tags.extend(mlflow_tags_to_proto(prompt.version_metadata))
    
    # Add aliases
    if prompt.aliases:
        for alias in prompt.aliases:
            proto_version.aliases.append(ProtoPromptVersion.PromptAlias(alias=alias))
    
    return proto_version 
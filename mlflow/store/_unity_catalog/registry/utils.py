"""
Utility functions for converting between Unity Catalog proto and MLflow entities.
"""

from typing import Dict, List, Optional

from mlflow.entities.model_registry.prompt import Prompt
from mlflow.protos.databricks_uc_registry_prompts_pb2 import (
    PromptVersion as ProtoPromptVersion,
    PromptTag as ProtoPromptTag,
    PromptVersionTag as ProtoPromptVersionTag,
)

def proto_to_mlflow_tags(proto_tags: List[ProtoPromptVersionTag]) -> Dict[str, str]:
    """Convert proto tags to MLflow tags dictionary."""
    return {tag.key: tag.value for tag in proto_tags} if proto_tags else {}

def mlflow_tags_to_proto(tags: Dict[str, str]) -> List[ProtoPromptVersionTag]:
    """Convert MLflow tags dictionary to proto tags."""
    return [ProtoPromptVersionTag(key=k, value=v) for k, v in tags.items()] if tags else []

def proto_to_mlflow_prompt(
    proto_version: ProtoPromptVersion,
    prompt_tags: Optional[Dict[str, str]] = None
) -> Prompt:
    """Convert proto prompt version to MLflow prompt entity."""
    return Prompt(
        name=proto_version.name,
        version=proto_version.version,
        template=proto_version.template,
        commit_message=proto_version.description,
        creation_timestamp=proto_version.creation_timestamp,
        version_metadata=proto_to_mlflow_tags(proto_version.tags),
        prompt_tags=prompt_tags,
        aliases=[alias.alias for alias in proto_version.aliases]
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
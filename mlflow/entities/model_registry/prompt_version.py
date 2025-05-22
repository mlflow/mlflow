"""
MLflow entity for Unity Catalog Prompt Version.
"""
from typing import Dict, Optional

class PromptVersion:
    """
    MLflow entity for Unity Catalog Prompt Version.
    """
    
    def __init__(
        self,
        name: str,
        version: str,
        creation_timestamp: Optional[int] = None,
        last_updated_timestamp: Optional[int] = None,
        description: Optional[str] = None,
        template: str = None,
        aliases: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize a PromptVersion.
        
        Args:
            name: Full three tier UC name of the prompt
            version: Version number
            creation_timestamp: Timestamp recorded when this version was created
            last_updated_timestamp: Timestamp recorded when metadata was last updated
            description: Description of this version
            template: The prompt template text
            aliases: Dictionary of alias name to version number
            tags: Dictionary of version tags
        """
        self._name = name
        self._version = version
        self._creation_timestamp = creation_timestamp
        self._last_updated_timestamp = last_updated_timestamp
        self._description = description
        self._template = template
        self._aliases = aliases or {}
        self._tags = tags or {}

    @property
    def name(self) -> str:
        """Get the prompt name."""
        return self._name

    @property
    def version(self) -> str:
        """Get the version number."""
        return self._version

    @property
    def creation_timestamp(self) -> Optional[int]:
        """Get the creation timestamp."""
        return self._creation_timestamp

    @property
    def last_updated_timestamp(self) -> Optional[int]:
        """Get the last updated timestamp."""
        return self._last_updated_timestamp

    @property
    def description(self) -> Optional[str]:
        """Get the description."""
        return self._description

    @property
    def template(self) -> Optional[str]:
        """Get the template text."""
        return self._template

    @property
    def aliases(self) -> Dict[str, str]:
        """Get the aliases dictionary."""
        return self._aliases

    @property
    def tags(self) -> Dict[str, str]:
        """Get the tags dictionary."""
        return self._tags

    @classmethod
    def from_proto(cls, proto):
        """
        Create a PromptVersion from protocol buffer.
        
        Args:
            proto: Protocol buffer message
            
        Returns:
            PromptVersion object
        """
        aliases = {}
        if proto.aliases:
            aliases = {a.alias: a.version for a in proto.aliases}
            
        tags = {}
        if proto.tags:
            tags = {t.key: t.value for t in proto.tags}
            
        creation_ts = None
        if proto.creation_timestamp:
            creation_ts = int(proto.creation_timestamp.seconds * 1000)
            
        last_updated_ts = None
        if proto.last_updated_timestamp:
            last_updated_ts = int(proto.last_updated_timestamp.seconds * 1000)
            
        return cls(
            name=proto.name,
            version=proto.version,
            creation_timestamp=creation_ts,
            last_updated_timestamp=last_updated_ts,
            description=proto.description,
            template=proto.template,
            aliases=aliases,
            tags=tags,
        )

    def to_proto(self):
        """
        Convert to protocol buffer message.
        
        Returns:
            Proto message
        """
        from mlflow.protos import uc_prompt_pb2

        proto = uc_prompt_pb2.PromptVersion()
        proto.name = self.name
        proto.version = self.version
        
        if self.description:
            proto.description = self.description
            
        if self.template:
            proto.template = self.template
            
        for alias, version in self.aliases.items():
            proto_alias = proto.aliases.add()
            proto_alias.alias = alias
            proto_alias.version = version
            
        for key, value in self.tags.items():
            proto_tag = proto.tags.add()
            proto_tag.key = key
            proto_tag.value = value
            
        return proto

    def __repr__(self):
        return f"<PromptVersion: name={self.name}, version={self.version}>" 
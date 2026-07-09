from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from mlflow.entities._mlflow_object import _MlflowObject

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SkillStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DELETED = "deleted"


VALID_STATUS_TRANSITIONS: dict[SkillStatus, set[SkillStatus]] = {
    SkillStatus.DRAFT: {SkillStatus.ACTIVE, SkillStatus.DELETED},
    SkillStatus.ACTIVE: {SkillStatus.DRAFT, SkillStatus.DEPRECATED},
    SkillStatus.DEPRECATED: {SkillStatus.ACTIVE, SkillStatus.DELETED},
}


class SkillSourceType(str, Enum):
    GIT = "git"
    OCI = "oci"
    ZIP = "zip"
    MLFLOW = "mlflow"


# ---------------------------------------------------------------------------
# Alias types (frozen — immutable value objects)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillAlias:
    name: str
    alias: str
    version: str


@dataclass(frozen=True)
class SubagentAlias:
    name: str
    alias: str
    version: str


@dataclass(frozen=True)
class HookAlias:
    name: str
    alias: str
    version: str


@dataclass(frozen=True)
class SkillBundleAlias:
    name: str
    alias: str
    version: str


# ---------------------------------------------------------------------------
# Tag types (frozen — immutable key-value pairs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillTag:
    key: str
    value: str


@dataclass(frozen=True)
class SkillVersionTag:
    key: str
    value: str


@dataclass(frozen=True)
class SubagentTag:
    key: str
    value: str


@dataclass(frozen=True)
class SubagentVersionTag:
    key: str
    value: str


@dataclass(frozen=True)
class HookTag:
    key: str
    value: str


@dataclass(frozen=True)
class HookVersionTag:
    key: str
    value: str


@dataclass(frozen=True)
class SkillBundleTag:
    key: str
    value: str


@dataclass(frozen=True)
class SkillBundleVersionTag:
    key: str
    value: str


# ---------------------------------------------------------------------------
# Member reference types (frozen — used in SkillBundleVersion)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillMemberRef:
    name: str
    version: str
    member_subpath: str | None = None


@dataclass(frozen=True)
class SubagentMemberRef:
    name: str
    version: str
    member_subpath: str | None = None


@dataclass(frozen=True)
class HookMemberRef:
    name: str
    version: str
    member_subpath: str | None = None


@dataclass(frozen=True)
class McpServerMemberRef:
    name: str
    version: str


# ---------------------------------------------------------------------------
# Top-level entities
# ---------------------------------------------------------------------------


@dataclass
class Skill(_MlflowObject):
    name: str
    display_name: str | None = None
    description: str | None = None
    workspace: str | None = None
    status: SkillStatus | None = None
    tags: dict[str, str] = field(default_factory=dict)
    aliases: list[SkillAlias] = field(default_factory=list)
    latest_version: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def to_proto(self):
        raise NotImplementedError

    @classmethod
    def from_proto(cls, proto):
        raise NotImplementedError


@dataclass
class Subagent(_MlflowObject):
    name: str
    display_name: str | None = None
    description: str | None = None
    workspace: str | None = None
    status: SkillStatus | None = None
    tags: dict[str, str] = field(default_factory=dict)
    aliases: list[SubagentAlias] = field(default_factory=list)
    latest_version: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def to_proto(self):
        raise NotImplementedError

    @classmethod
    def from_proto(cls, proto):
        raise NotImplementedError


@dataclass
class Hook(_MlflowObject):
    name: str
    display_name: str | None = None
    description: str | None = None
    workspace: str | None = None
    status: SkillStatus | None = None
    tags: dict[str, str] = field(default_factory=dict)
    aliases: list[HookAlias] = field(default_factory=list)
    latest_version: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def to_proto(self):
        raise NotImplementedError

    @classmethod
    def from_proto(cls, proto):
        raise NotImplementedError


@dataclass
class SkillBundle(_MlflowObject):
    name: str
    display_name: str | None = None
    description: str | None = None
    workspace: str | None = None
    status: SkillStatus | None = None
    tags: dict[str, str] = field(default_factory=dict)
    aliases: list[SkillBundleAlias] = field(default_factory=list)
    latest_version: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def to_proto(self):
        raise NotImplementedError

    @classmethod
    def from_proto(cls, proto):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Versioned entities
# ---------------------------------------------------------------------------


@dataclass
class SkillVersion(_MlflowObject):
    name: str
    version: str
    display_name: str | None = None
    source_type: SkillSourceType | None = None
    source: str | None = None
    subpath: str | None = None
    status: SkillStatus = SkillStatus.DRAFT
    content_digest: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    workspace: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def to_proto(self):
        raise NotImplementedError

    @classmethod
    def from_proto(cls, proto):
        raise NotImplementedError


@dataclass
class SubagentVersion(_MlflowObject):
    name: str
    version: str
    display_name: str | None = None
    source_type: SkillSourceType | None = None
    source: str | None = None
    subpath: str | None = None
    status: SkillStatus = SkillStatus.DRAFT
    content_digest: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    workspace: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def to_proto(self):
        raise NotImplementedError

    @classmethod
    def from_proto(cls, proto):
        raise NotImplementedError


@dataclass
class HookVersion(_MlflowObject):
    name: str
    version: str
    display_name: str | None = None
    source_type: SkillSourceType | None = None
    source: str | None = None
    subpath: str | None = None
    status: SkillStatus = SkillStatus.DRAFT
    content_digest: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    workspace: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def to_proto(self):
        raise NotImplementedError

    @classmethod
    def from_proto(cls, proto):
        raise NotImplementedError


@dataclass
class SkillBundleVersion(_MlflowObject):
    name: str
    version: str
    display_name: str | None = None
    source_type: SkillSourceType | None = None
    source: str | None = None
    subpath: str | None = None
    status: SkillStatus = SkillStatus.DRAFT
    content_digest: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    skills: list[SkillMemberRef] = field(default_factory=list)
    subagents: list[SubagentMemberRef] = field(default_factory=list)
    hooks: list[HookMemberRef] = field(default_factory=list)
    mcp_servers: list[McpServerMemberRef] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    workspace: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    def to_proto(self):
        raise NotImplementedError

    @classmethod
    def from_proto(cls, proto):
        raise NotImplementedError

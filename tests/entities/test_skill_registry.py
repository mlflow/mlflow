from __future__ import annotations

import pytest

from mlflow.entities.skill_registry import (
    VALID_STATUS_TRANSITIONS,
    Hook,
    HookAlias,
    HookMemberRef,
    HookTag,
    HookVersion,
    HookVersionTag,
    McpServerMemberRef,
    Skill,
    SkillAlias,
    SkillBundle,
    SkillBundleAlias,
    SkillBundleTag,
    SkillBundleVersion,
    SkillBundleVersionTag,
    SkillMemberRef,
    SkillSourceType,
    SkillStatus,
    SkillTag,
    SkillVersion,
    SkillVersionTag,
    Subagent,
    SubagentAlias,
    SubagentMemberRef,
    SubagentTag,
    SubagentVersion,
    SubagentVersionTag,
)


class TestSkillStatusEnum:
    def test_values(self):
        assert SkillStatus.DRAFT == "draft"
        assert SkillStatus.ACTIVE == "active"
        assert SkillStatus.DEPRECATED == "deprecated"
        assert SkillStatus.DELETED == "deleted"

    def test_members(self):
        assert set(SkillStatus) == {
            SkillStatus.DRAFT,
            SkillStatus.ACTIVE,
            SkillStatus.DEPRECATED,
            SkillStatus.DELETED,
        }

    def test_str_enum(self):
        assert isinstance(SkillStatus.DRAFT, str)
        assert f"status={SkillStatus.ACTIVE}" == "status=active"


class TestSkillSourceTypeEnum:
    def test_values(self):
        assert SkillSourceType.GIT == "git"
        assert SkillSourceType.OCI == "oci"
        assert SkillSourceType.ZIP == "zip"
        assert SkillSourceType.MLFLOW == "mlflow"

    def test_members(self):
        assert set(SkillSourceType) == {
            SkillSourceType.GIT,
            SkillSourceType.OCI,
            SkillSourceType.ZIP,
            SkillSourceType.MLFLOW,
        }


class TestValidStatusTransitions:
    def test_draft_transitions(self):
        assert VALID_STATUS_TRANSITIONS[SkillStatus.DRAFT] == {
            SkillStatus.ACTIVE,
            SkillStatus.DELETED,
        }

    def test_active_transitions(self):
        assert VALID_STATUS_TRANSITIONS[SkillStatus.ACTIVE] == {
            SkillStatus.DRAFT,
            SkillStatus.DEPRECATED,
        }

    def test_deprecated_transitions(self):
        assert VALID_STATUS_TRANSITIONS[SkillStatus.DEPRECATED] == {
            SkillStatus.ACTIVE,
            SkillStatus.DELETED,
        }

    def test_deleted_is_terminal(self):
        assert SkillStatus.DELETED not in VALID_STATUS_TRANSITIONS

    def test_active_cannot_directly_delete(self):
        assert SkillStatus.DELETED not in VALID_STATUS_TRANSITIONS[SkillStatus.ACTIVE]


# ---------------------------------------------------------------------------
# Top-level entities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "alias_cls"),
    [
        (Skill, SkillAlias),
        (Subagent, SubagentAlias),
        (Hook, HookAlias),
        (SkillBundle, SkillBundleAlias),
    ],
)
class TestTopLevelEntities:
    def test_minimal_construction(self, cls, alias_cls):
        entity = cls(name="code-review")
        assert entity.name == "code-review"
        assert entity.display_name is None
        assert entity.description is None
        assert entity.workspace is None
        assert entity.status is None
        assert entity.tags == {}
        assert entity.aliases == []
        assert entity.latest_version is None
        assert entity.created_by is None
        assert entity.last_updated_by is None
        assert entity.creation_timestamp is None
        assert entity.last_updated_timestamp is None

    def test_full_construction(self, cls, alias_cls):
        alias = alias_cls(name="code-review", alias="production", version="1.0.0")
        entity = cls(
            name="code-review",
            display_name="Code Review",
            description="Reviews pull requests",
            workspace="default",
            status=SkillStatus.ACTIVE,
            tags={"team": "platform"},
            aliases=[alias],
            latest_version="1.0.0",
            created_by="user@example.com",
            last_updated_by="user@example.com",
            creation_timestamp=1000000,
            last_updated_timestamp=2000000,
        )
        assert entity.name == "code-review"
        assert entity.display_name == "Code Review"
        assert entity.description == "Reviews pull requests"
        assert entity.workspace == "default"
        assert entity.status == SkillStatus.ACTIVE
        assert entity.tags == {"team": "platform"}
        assert entity.aliases == [alias]
        assert entity.latest_version == "1.0.0"
        assert entity.created_by == "user@example.com"
        assert entity.creation_timestamp == 1000000


# ---------------------------------------------------------------------------
# Versioned entities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [SkillVersion, SubagentVersion, HookVersion],
)
class TestVersionedEntities:
    def test_minimal_construction(self, cls):
        v = cls(name="code-review", version="1.0.0")
        assert v.name == "code-review"
        assert v.version == "1.0.0"
        assert v.display_name is None
        assert v.source_type is None
        assert v.source is None
        assert v.subpath is None
        assert v.status == SkillStatus.DRAFT
        assert v.content_digest is None
        assert v.tags == {}
        assert v.aliases == []
        assert v.workspace is None

    def test_full_construction(self, cls):
        v = cls(
            name="code-review",
            version="1.0.0",
            display_name="Code Review v1",
            source_type=SkillSourceType.GIT,
            source="https://github.com/org/repo@v1.0.0",
            subpath="skills/code-review",
            status=SkillStatus.ACTIVE,
            content_digest="sha256:abc123",
            tags={"approved": "true"},
            aliases=["production"],
            workspace="default",
            created_by="user@example.com",
            last_updated_by="user@example.com",
            creation_timestamp=1000000,
            last_updated_timestamp=2000000,
        )
        assert v.source_type == SkillSourceType.GIT
        assert v.source == "https://github.com/org/repo@v1.0.0"
        assert v.subpath == "skills/code-review"
        assert v.content_digest == "sha256:abc123"
        assert v.aliases == ["production"]


class TestSkillBundleVersion:
    def test_minimal_construction(self):
        v = SkillBundleVersion(name="pr-workflow", version="1.0.0")
        assert v.name == "pr-workflow"
        assert v.version == "1.0.0"
        assert v.status == SkillStatus.DRAFT
        assert v.skills == []
        assert v.subagents == []
        assert v.hooks == []
        assert v.mcp_servers == []

    def test_with_member_refs(self):
        v = SkillBundleVersion(
            name="pr-workflow",
            version="1.0.0",
            skills=[
                SkillMemberRef(name="code-review", version="1.0.0"),
                SkillMemberRef(name="linter", version="2.0.0", member_subpath="skills/linter"),
            ],
            subagents=[
                SubagentMemberRef(name="security-auditor", version="1.0.0"),
            ],
            hooks=[
                HookMemberRef(name="pre-commit", version="1.0.0"),
            ],
            mcp_servers=[
                McpServerMemberRef(name="jira", version="1.0.0"),
            ],
        )
        assert len(v.skills) == 2
        assert len(v.subagents) == 1
        assert len(v.hooks) == 1
        assert len(v.mcp_servers) == 1

    def test_monolithic_bundle_with_source(self):
        v = SkillBundleVersion(
            name="pr-workflow",
            version="1.0.0",
            source_type=SkillSourceType.OCI,
            source="ghcr.io/org/pr-workflow:1.0.0",
            content_digest="sha256:def456",
            skills=[
                SkillMemberRef(
                    name="code-review",
                    version="1.0.0",
                    member_subpath="skills/code-review",
                ),
            ],
        )
        assert v.source_type == SkillSourceType.OCI
        assert v.skills[0].member_subpath == "skills/code-review"


# ---------------------------------------------------------------------------
# Alias types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [SkillAlias, SubagentAlias, HookAlias, SkillBundleAlias],
)
class TestAliasTypes:
    def test_construction(self, cls):
        alias = cls(name="code-review", alias="production", version="1.0.0")
        assert alias.name == "code-review"
        assert alias.alias == "production"
        assert alias.version == "1.0.0"

    def test_frozen(self, cls):
        alias = cls(name="code-review", alias="production", version="1.0.0")
        with pytest.raises(AttributeError, match="cannot assign"):
            alias.version = "2.0.0"


# ---------------------------------------------------------------------------
# Tag types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        SkillTag,
        SkillVersionTag,
        SubagentTag,
        SubagentVersionTag,
        HookTag,
        HookVersionTag,
        SkillBundleTag,
        SkillBundleVersionTag,
    ],
)
class TestTagTypes:
    def test_construction(self, cls):
        tag = cls(key="team", value="platform")
        assert tag.key == "team"
        assert tag.value == "platform"

    def test_frozen(self, cls):
        tag = cls(key="team", value="platform")
        with pytest.raises(AttributeError, match="cannot assign"):
            tag.key = "other"


# ---------------------------------------------------------------------------
# Member reference types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [SkillMemberRef, SubagentMemberRef, HookMemberRef],
)
class TestMemberRefWithSubpath:
    def test_without_subpath(self, cls):
        ref = cls(name="code-review", version="1.0.0")
        assert ref.name == "code-review"
        assert ref.version == "1.0.0"
        assert ref.member_subpath is None

    def test_with_subpath(self, cls):
        ref = cls(name="code-review", version="1.0.0", member_subpath="skills/code-review")
        assert ref.member_subpath == "skills/code-review"

    def test_frozen(self, cls):
        ref = cls(name="code-review", version="1.0.0")
        with pytest.raises(AttributeError, match="cannot assign"):
            ref.name = "other"


class TestMcpServerMemberRef:
    def test_construction(self):
        ref = McpServerMemberRef(name="jira", version="1.0.0")
        assert ref.name == "jira"
        assert ref.version == "1.0.0"

    def test_no_subpath_field(self):
        assert not hasattr(McpServerMemberRef(name="jira", version="1.0.0"), "member_subpath")

    def test_frozen(self):
        ref = McpServerMemberRef(name="jira", version="1.0.0")
        with pytest.raises(AttributeError, match="cannot assign"):
            ref.name = "other"


# ---------------------------------------------------------------------------
# Default factory isolation
# ---------------------------------------------------------------------------


class TestDefaultFactoryIsolation:
    def test_tags_not_shared(self):
        a = Skill(name="a")
        b = Skill(name="b")
        a.tags["key"] = "value"
        assert b.tags == {}

    def test_aliases_not_shared(self):
        a = SkillVersion(name="a", version="1.0.0")
        b = SkillVersion(name="b", version="1.0.0")
        a.aliases.append("prod")
        assert b.aliases == []

    def test_member_lists_not_shared(self):
        a = SkillBundleVersion(name="a", version="1.0.0")
        b = SkillBundleVersion(name="b", version="1.0.0")
        a.skills.append(SkillMemberRef(name="x", version="1.0.0"))
        assert b.skills == []

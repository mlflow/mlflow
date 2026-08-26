import pytest

from mlflow.entities.agent_plugin_version import AgentPluginVersion
from mlflow.entities.skill import SkillStatus
from mlflow.entities.skill_source import GitSource, SkillSourceType
from mlflow.exceptions import MlflowException
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


def test_agent_plugin_version_defaults():
    apv = AgentPluginVersion(name="pr-workflow", version="1.0.0")
    assert apv.organization == ""
    assert apv.plugin_json == {}
    assert apv.status == SkillStatus.ACTIVE
    assert apv.skills == []
    assert apv.aliases == []
    assert apv.workspace == resolve_entity_workspace_name(None)


def test_agent_plugin_version_roundtrip():
    apv = AgentPluginVersion(
        name="pr-workflow",
        version="1.0.0",
        organization="acme",
        plugin_json={"name": "pr-workflow", "version": "1.0.0"},
        source=GitSource(url="https://github.com/acme/plugins", ref="v1"),
        source_type=SkillSourceType.GIT,
        skills=["skills:/code-review/1", "skills:/security-scan@production"],
        aliases=["production"],
    )
    data = apv.to_dict()
    assert data["source_type"] == "git"
    assert data["skills"] == ["skills:/code-review/1", "skills:/security-scan@production"]
    restored = AgentPluginVersion.from_dict(data)
    assert restored == apv


def test_agent_plugin_version_from_dict_missing_field():
    with pytest.raises(MlflowException, match="missing required field"):
        AgentPluginVersion.from_dict({"name": "pr-workflow"})

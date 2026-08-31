import pytest

from mlflow.entities.agent_plugin import AgentPlugin
from mlflow.entities.skill import SkillStatus
from mlflow.exceptions import MlflowException
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


def test_agent_plugin_defaults():
    plugin = AgentPlugin(name="pr-workflow")
    assert plugin.organization == ""
    assert plugin.aliases == {}
    assert plugin.latest_version is None
    assert plugin.workspace == resolve_entity_workspace_name(None)


def test_agent_plugin_from_dict():
    data = {
        "name": "pr-workflow",
        "organization": "acme",
        "status": "active",
        "tags": {"team": "platform"},
        "aliases": {"production": "1.0.0"},
        "latest_version": "1.2.0",
    }
    restored = AgentPlugin.from_dict(data)
    assert restored.name == "pr-workflow"
    assert restored.organization == "acme"
    assert restored.status == SkillStatus.ACTIVE
    assert restored.aliases == {"production": "1.0.0"}
    assert restored.latest_version == "1.2.0"


def test_agent_plugin_from_dict_missing_name():
    with pytest.raises(MlflowException, match="missing required field"):
        AgentPlugin.from_dict({"organization": "acme"})

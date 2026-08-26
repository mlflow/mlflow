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


def test_agent_plugin_to_from_dict_roundtrip():
    plugin = AgentPlugin(
        name="pr-workflow",
        organization="acme",
        status=SkillStatus.ACTIVE,
        tags={"team": "platform"},
        aliases={"production": "1.0.0"},
        latest_version="1.2.0",
    )
    data = plugin.to_dict()
    assert data["status"] == "active"
    assert data["aliases"] == {"production": "1.0.0"}
    assert data["latest_version"] == "1.2.0"
    restored = AgentPlugin.from_dict(data)
    assert restored == plugin


def test_agent_plugin_from_dict_missing_name():
    with pytest.raises(MlflowException, match="missing required field"):
        AgentPlugin.from_dict({"organization": "acme"})

import pytest

from mlflow.entities.skill import Skill, SkillStatus
from mlflow.exceptions import MlflowException
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


def test_skill_status_str():
    assert str(SkillStatus.ACTIVE) == "active"
    assert SkillStatus("draft") == SkillStatus.DRAFT


def test_skill_defaults_and_workspace_resolution():
    skill = Skill(name="code-review")
    assert skill.organization == ""
    assert skill.status is None
    assert skill.tags == {}
    assert skill.aliases == {}
    assert skill.latest_version is None
    assert skill.workspace == resolve_entity_workspace_name(None)


def test_skill_to_from_dict_roundtrip():
    skill = Skill(
        name="code-review",
        organization="acme",
        description="Reviews PRs",
        status=SkillStatus.ACTIVE,
        tags={"team": "platform"},
        aliases={"production": 1},
        latest_version=2,
    )
    data = skill.to_dict()
    assert data["status"] == "active"
    assert data["aliases"] == {"production": 1}
    restored = Skill.from_dict(data)
    assert restored.name == "code-review"
    assert restored.organization == "acme"
    assert restored.status == SkillStatus.ACTIVE
    assert restored.aliases == {"production": 1}
    assert restored.latest_version == 2


def test_skill_from_dict_requires_dict():
    with pytest.raises(MlflowException, match="expected a dictionary"):
        Skill.from_dict("nope")


def test_skill_from_dict_missing_name():
    with pytest.raises(MlflowException, match="missing required field"):
        Skill.from_dict({"organization": "acme"})

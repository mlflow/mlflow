import pytest

from mlflow.entities.skill import SkillStatus
from mlflow.entities.skill_source import GitSource, OCISource, SkillSourceType, ZipSource
from mlflow.entities.skill_version import SkillVersion
from mlflow.exceptions import MlflowException
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


def test_skill_version_defaults():
    sv = SkillVersion(name="code-review", version=1)
    assert sv.organization == ""
    assert sv.status == SkillStatus.ACTIVE
    assert sv.aliases == []
    assert sv.workspace == resolve_entity_workspace_name(None)


def test_skill_version_typed_source_roundtrip():
    sv = SkillVersion(
        name="code-review",
        version=2,
        source=GitSource(url="https://github.com/acme/skills", ref="v2"),
        source_type=SkillSourceType.GIT,
        digest="a" * 64,
        tags={"scan": "clean"},
        aliases=["production"],
    )
    data = sv.to_dict()
    assert data["source"] == {"url": "https://github.com/acme/skills", "ref": "v2"}
    assert data["source_type"] == "git"
    assert data["status"] == "active"
    restored = SkillVersion.from_dict(data)
    assert restored == sv


@pytest.mark.parametrize(
    ("source", "source_type"),
    [
        (OCISource(image="ghcr.io/acme/plugin:v1"), SkillSourceType.OCI),
        (ZipSource(url="https://example.com/a.zip", subpath="x"), SkillSourceType.ZIP),
    ],
)
def test_skill_version_oci_zip_source_roundtrip(source, source_type):
    sv = SkillVersion(name="code-review", version=4, source=source, source_type=source_type)
    restored = SkillVersion.from_dict(sv.to_dict())
    assert restored.source == source
    assert restored.source_type == source_type


def test_skill_version_string_source_roundtrip():
    sv = SkillVersion(
        name="code-review",
        version=3,
        source="artifacts:/skills/code-review/3",
        source_type=SkillSourceType.MLFLOW,
    )
    restored = SkillVersion.from_dict(sv.to_dict())
    assert restored.source == "artifacts:/skills/code-review/3"
    assert restored.source_type == SkillSourceType.MLFLOW


def test_skill_version_from_dict_invalid_source_type_raises_mlflow_exception():
    with pytest.raises(MlflowException, match="Failed to parse SkillVersion response"):
        SkillVersion.from_dict({"name": "code-review", "version": 5, "source_type": "not-a-type"})


def test_skill_version_from_dict_missing_field():
    with pytest.raises(MlflowException, match="missing required field"):
        SkillVersion.from_dict({"name": "code-review"})

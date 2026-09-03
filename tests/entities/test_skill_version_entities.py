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


def test_skill_version_from_dict_flat_git_source():
    data = {
        "name": "code-review",
        "version": 2,
        "source_type": "git",
        "source": "https://github.com/acme/skills",
        "ref": "v2",
        "subpath": "code-review",
        "status": "active",
        "aliases": ["production"],
        "tags": {"scan": "clean"},
    }
    sv = SkillVersion.from_dict(data)
    assert sv.source == GitSource(
        url="https://github.com/acme/skills", ref="v2", subpath="code-review"
    )
    assert sv.source_type == SkillSourceType.GIT
    assert sv.status == SkillStatus.ACTIVE
    assert sv.aliases == ["production"]


@pytest.mark.parametrize(
    ("source_type", "source", "ref", "subpath", "expected"),
    [
        (
            "oci",
            "ghcr.io/acme/plugin:v1",
            None,
            "x",
            OCISource(image="ghcr.io/acme/plugin:v1", subpath="x"),
        ),
        (
            "zip",
            "https://example.com/a.zip",
            None,
            "x",
            ZipSource(url="https://example.com/a.zip", subpath="x"),
        ),
        (
            "mlflow",
            "artifacts:/skills/code-review/3",
            None,
            None,
            "artifacts:/skills/code-review/3",
        ),
    ],
)
def test_skill_version_from_dict_flat_sources(source_type, source, ref, subpath, expected):
    data = {
        "name": "code-review",
        "version": 3,
        "source_type": source_type,
        "source": source,
        "ref": ref,
        "subpath": subpath,
    }
    assert SkillVersion.from_dict(data).source == expected


def test_skill_version_from_dict_invalid_source_type_raises_mlflow_exception():
    with pytest.raises(MlflowException, match="Failed to parse SkillVersion response"):
        SkillVersion.from_dict({"name": "code-review", "version": 5, "source_type": "not-a-type"})


def test_skill_version_from_dict_missing_field():
    with pytest.raises(MlflowException, match="missing required field"):
        SkillVersion.from_dict({"name": "code-review"})

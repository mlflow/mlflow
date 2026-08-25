import pytest

from mlflow.entities.skill_source import (
    GitSource,
    OCISource,
    SkillSourceType,
    ZipSource,
    source_from_dict,
    source_to_dict,
)
from mlflow.exceptions import MlflowException


def test_source_type_str():
    assert str(SkillSourceType.GIT) == "git"
    assert SkillSourceType("oci") == SkillSourceType.OCI


def test_git_source_roundtrip_drops_none():
    src = GitSource(url="https://github.com/acme/skills", ref="v1", subpath="code-review")
    assert src.to_dict() == {
        "url": "https://github.com/acme/skills",
        "ref": "v1",
        "subpath": "code-review",
    }
    assert GitSource.from_dict(src.to_dict()) == src
    assert GitSource(url="u").to_dict() == {"url": "u"}


def test_oci_and_zip_source_roundtrip():
    oci = OCISource(image="ghcr.io/acme/plugin:v1")
    assert OCISource.from_dict(oci.to_dict()) == oci
    zip_src = ZipSource(url="https://example.com/a.zip", subpath="x")
    assert ZipSource.from_dict(zip_src.to_dict()) == zip_src


def test_from_dict_missing_required_key():
    with pytest.raises(MlflowException, match="url"):
        GitSource.from_dict({"ref": "v1"})


def test_source_to_from_dict_helpers():
    assert source_to_dict(None) is None
    assert source_to_dict("artifacts:/x") == "artifacts:/x"
    git = GitSource(url="u", ref="r")
    assert source_to_dict(git) == {"url": "u", "ref": "r"}
    assert source_from_dict({"url": "u", "ref": "r"}, SkillSourceType.GIT) == git
    assert source_from_dict("artifacts:/x", SkillSourceType.MLFLOW) == "artifacts:/x"
    assert source_from_dict(None, None) is None

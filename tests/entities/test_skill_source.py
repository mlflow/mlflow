import pytest

from mlflow.entities.skill_source import (
    GitSource,
    OCISource,
    SkillSourceType,
    ZipSource,
    build_source,
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


def test_build_source_reconstructs_typed_sources():
    assert build_source(SkillSourceType.GIT, "https://x/y", "v1", "sub") == GitSource(
        url="https://x/y", ref="v1", subpath="sub"
    )
    assert build_source(SkillSourceType.OCI, "ghcr.io/a/b:v1", None, "sub") == OCISource(
        image="ghcr.io/a/b:v1", subpath="sub"
    )
    assert build_source(SkillSourceType.ZIP, "https://x/a.zip", None, "sub") == ZipSource(
        url="https://x/a.zip", subpath="sub"
    )
    assert (
        build_source(SkillSourceType.MLFLOW, "artifacts:/skills/x/3", None, None)
        == "artifacts:/skills/x/3"
    )
    assert build_source(None, None, None, None) is None

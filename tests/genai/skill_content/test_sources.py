from pathlib import Path

import pytest

from mlflow.entities.skill_source import GitSource, OCISource, SkillSourceType, ZipSource
from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.sources import (
    ResolvedSource,
    is_local_path,
    resolve_source_type,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("./skills/review", True),
        ("skills/review", True),
        ("/abs/skills", True),
        ("C:\\skills\\review", True),
        ("https://example.com/repo.git", False),
        ("oci://ghcr.io/acme/skills:v1", False),
        ("git@github.com:acme/skills.git", False),
        ("mlflow-artifacts:/skills/review/abc", False),
        ("runs:/abc/skill", False),
    ],
)
def test_is_local_path(source, expected):
    assert is_local_path(source) is expected


@pytest.mark.parametrize(
    ("source", "expected_type", "expected_value"),
    [
        (
            "https://github.com/acme/skills.git",
            SkillSourceType.GIT,
            "https://github.com/acme/skills.git",
        ),
        ("git@github.com:acme/skills.git", SkillSourceType.GIT, "git@github.com:acme/skills.git"),
        ("git://host/repo", SkillSourceType.GIT, "git://host/repo"),
        ("ssh://git@host/repo", SkillSourceType.GIT, "ssh://git@host/repo"),
        ("oci://ghcr.io/acme/skills:v1", SkillSourceType.OCI, "ghcr.io/acme/skills:v1"),
        (
            "https://example.com/skills.zip?x=1",
            SkillSourceType.ZIP,
            "https://example.com/skills.zip?x=1",
        ),
        (
            "mlflow-artifacts:/skills/review/t1",
            SkillSourceType.MLFLOW,
            "mlflow-artifacts:/skills/review/t1",
        ),
        ("runs:/run1/skill", SkillSourceType.MLFLOW, "runs:/run1/skill"),
    ],
)
def test_resolve_source_type_infers_from_string(source, expected_type, expected_value):
    resolved = resolve_source_type(source)
    assert resolved.source_type == expected_type
    assert resolved.source == expected_value
    assert resolved.is_local is False


def test_resolve_source_type_typed_sources():
    git = resolve_source_type(GitSource(url="https://h/r", ref="v1", subpath="skills/a/"))
    assert git == ResolvedSource(SkillSourceType.GIT, "https://h/r", ref="v1", subpath="skills/a")

    oci = resolve_source_type(OCISource(image="oci://ghcr.io/acme/p:v1", subpath="s"))
    assert (oci.source_type, oci.source, oci.subpath) == (
        SkillSourceType.OCI,
        "ghcr.io/acme/p:v1",
        "s",
    )

    zipped = resolve_source_type(ZipSource(url="https://h/a"))
    assert (zipped.source_type, zipped.source, zipped.subpath) == (
        SkillSourceType.ZIP,
        "https://h/a",
        None,
    )


def test_resolve_source_type_typed_source_rejects_extra_kwargs():
    with pytest.raises(MlflowException, match="must not be passed alongside a typed source"):
        resolve_source_type(GitSource(url="https://h/r"), ref="main")


def test_resolve_source_type_local_path(tmp_path):
    resolved = resolve_source_type(str(tmp_path), subpath="a//b")
    assert resolved.is_local is True
    assert resolved.source_type == SkillSourceType.MLFLOW
    assert Path(resolved.source) == tmp_path.resolve()
    assert resolved.subpath == "a/b"
    with pytest.raises(MlflowException, match="'ref' applies to Git sources only"):
        resolve_source_type(str(tmp_path), ref="main")


@pytest.mark.parametrize(
    "source",
    ["https://example.com/acme/skills", "https://example.com/archive.tar.gz", "oci://"],
)
def test_resolve_source_type_ambiguous(source):
    with pytest.raises(MlflowException, match="Cannot infer|must include an image reference"):
        resolve_source_type(source)


def test_resolve_source_type_ref_only_for_git():
    with pytest.raises(MlflowException, match="'ref' applies to Git sources only"):
        resolve_source_type("https://example.com/skills.zip", ref="v1")
    resolved = resolve_source_type("https://example.com/skills.git", ref="v1", subpath="x")
    assert (resolved.ref, resolved.subpath) == ("v1", "x")


@pytest.mark.parametrize("source", ["", "   ", None, 5])
def test_resolve_source_type_rejects_empty(source):
    with pytest.raises(MlflowException, match="non-empty string"):
        resolve_source_type(source)

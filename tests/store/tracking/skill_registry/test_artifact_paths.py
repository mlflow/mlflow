import re

import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.tracking.skill_registry.artifact_paths import (
    SkillArtifactIdentity,
    artifact_path_from_uri,
    new_skill_upload_path,
    owned_skill_upload_path,
    parse_skill_upload_path,
    to_artifact_uri,
    validate_referenced_mlflow_source,
)

_TOKEN = "0123456789abcdef0123456789abcdef"


@pytest.mark.parametrize(
    ("organization", "prefix"),
    [("", "skills/reviewer/"), ("acme", "skills/@acme/reviewer/")],
)
def test_new_skill_upload_path_shape(organization, prefix):
    path = new_skill_upload_path("reviewer", organization)
    assert path.startswith(prefix)
    assert re.fullmatch(r"[0-9a-f]{32}", path.removeprefix(prefix))


def test_new_skill_upload_path_is_unique_per_call():
    paths = {new_skill_upload_path("reviewer", "acme") for _ in range(1000)}
    assert len(paths) == 1000


@pytest.mark.parametrize(("name", "organization"), [("Bad_Name", ""), ("reviewer", "../acme")])
def test_new_skill_upload_path_rejects_invalid_identity(name, organization):
    with pytest.raises(MlflowException, match="Invalid"):
        new_skill_upload_path(name, organization)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("mlflow-artifacts:/skills/reviewer/t", "skills/reviewer/t"),
        ("mlflow-artifacts:/skills/reviewer/t/", "skills/reviewer/t"),
        ("mlflow-artifacts:/", None),
        ("https://example.com/skills/reviewer/t", None),
        ("skills/reviewer/t", None),
        (None, None),
    ],
)
def test_artifact_path_from_uri(source, expected):
    assert artifact_path_from_uri(source) == expected


def test_owned_skill_upload_path_accepts_only_the_upload_shape():
    path = f"skills/@acme/reviewer/{_TOKEN}"
    owned = owned_skill_upload_path(
        name="reviewer",
        organization="acme",
        source_type="mlflow",
        source=to_artifact_uri(path),
        subpath=None,
    )
    assert owned == path


@pytest.mark.parametrize(
    "overrides",
    [
        # An external pointer is never MLflow content.
        {"source_type": "git", "source": "https://example.com/skills.git"},
        # A reference into an agent plugin's package tree belongs to the plugin version.
        {"source": f"mlflow-artifacts:/agent-plugins/@acme/toolkit/{_TOKEN}", "subpath": "a"},
        # Even a path under the skill's own prefix is only a reference when it has a subpath.
        {"subpath": "nested"},
        # Another identity's upload.
        {"source": f"mlflow-artifacts:/skills/@acme/other/{_TOKEN}"},
        {"source": f"mlflow-artifacts:/skills/reviewer/{_TOKEN}"},
        # Too short (the identity prefix itself) or too long.
        {"source": "mlflow-artifacts:/skills/@acme/reviewer"},
        {"source": f"mlflow-artifacts:/skills/@acme/reviewer/{_TOKEN}/extra"},
        # Not a server-generated token.
        {"source": "mlflow-artifacts:/skills/@acme/reviewer/not-a-token"},
        {"source": "mlflow-artifacts:/skills/@acme/reviewer/.."},
        {"source": None},
        {"source_type": None},
    ],
)
def test_owned_skill_upload_path_rejects_everything_else(overrides):
    fields = {
        "name": "reviewer",
        "organization": "acme",
        "source_type": "mlflow",
        "source": f"mlflow-artifacts:/skills/@acme/reviewer/{_TOKEN}",
        "subpath": None,
    } | overrides
    assert owned_skill_upload_path(**fields) is None


@pytest.mark.parametrize(
    ("source", "subpath", "expected"),
    [
        (
            f"mlflow-artifacts:/agent-plugins/@acme/toolkit/{_TOKEN}",
            "skills/reviewer/",
            (f"mlflow-artifacts:/agent-plugins/@acme/toolkit/{_TOKEN}", "skills/reviewer"),
        ),
        (
            f"mlflow-artifacts:/agent-plugins/tool.kit/{_TOKEN}/",
            "reviewer",
            (f"mlflow-artifacts:/agent-plugins/tool.kit/{_TOKEN}", "reviewer"),
        ),
    ],
)
def test_validate_referenced_mlflow_source(source, subpath, expected):
    canonical = validate_referenced_mlflow_source(source, subpath)
    assert canonical == expected
    # A validated reference is never mistaken for content the skill owns.
    assert (
        owned_skill_upload_path(
            name="reviewer",
            organization="acme",
            source_type="mlflow",
            source=canonical[0],
            subpath=canonical[1],
        )
        is None
    )


@pytest.mark.parametrize(
    ("source", "subpath", "message"),
    [
        ("https://example.com/plugin", "a", "must be an 'mlflow-artifacts:/' URI"),
        (None, "a", "must be an 'mlflow-artifacts:/' URI"),
        (f"mlflow-artifacts:/skills/reviewer/{_TOKEN}", "a", "must point into 'agent-plugins/'"),
        (f"mlflow-artifacts:/agent-plugins/{_TOKEN}", "a", "must have the form"),
        (f"mlflow-artifacts:/agent-plugins/a/b/c/{_TOKEN}", "a", "must have the form"),
        (f"mlflow-artifacts:/agent-plugins/@/toolkit/{_TOKEN}", "a", "Invalid|empty organization"),
        (f"mlflow-artifacts:/agent-plugins/Tool_Kit/{_TOKEN}", "a", "Invalid agent plugin name"),
        ("mlflow-artifacts:/agent-plugins/toolkit/not-a-token", "a", "valid upload token"),
        (f"mlflow-artifacts:/agent-plugins/../toolkit/{_TOKEN}", "a", "Invalid"),
        (f"mlflow-artifacts:/agent-plugins/toolkit/{_TOKEN}", None, "requires a 'subpath'"),
        (f"mlflow-artifacts:/agent-plugins/toolkit/{_TOKEN}", "../escape", "subpath|Subpath"),
    ],
)
def test_validate_referenced_mlflow_source_rejects(source, subpath, message):
    with pytest.raises(MlflowException, match=message):
        validate_referenced_mlflow_source(source, subpath)


@pytest.mark.parametrize(
    ("artifact_path", "expected"),
    [
        # A token path, its contents, and every ancestor a listing can address.
        (f"skills/@acme/reviewer/{_TOKEN}/SKILL.md", SkillArtifactIdentity("acme", "reviewer")),
        (
            f"skills/@acme/reviewer/{_TOKEN}/scripts/run.py",
            SkillArtifactIdentity("acme", "reviewer"),
        ),
        (f"skills/@acme/reviewer/{_TOKEN}", SkillArtifactIdentity("acme", "reviewer")),
        ("skills/@acme/reviewer", SkillArtifactIdentity("acme", "reviewer")),
        ("skills/@acme/reviewer/", SkillArtifactIdentity("acme", "reviewer")),
        ("/skills/@acme/reviewer", SkillArtifactIdentity("acme", "reviewer")),
        (f"skills/reviewer/{_TOKEN}/SKILL.md", SkillArtifactIdentity("", "reviewer")),
        ("skills/reviewer", SkillArtifactIdentity("", "reviewer")),
        # Paths that name no single skill.
        ("skills", None),
        ("skills/", None),
        ("skills/@acme", None),
        ("skills/@/reviewer", None),
        # Outside the upload root, or a malformed identity.
        (f"agent-plugins/@acme/toolkit/{_TOKEN}", None),
        ("0/run/artifacts/model.pkl", None),
        ("skillsX/reviewer", None),
        ("skills/Reviewer", None),
        ("skills/@Acme/reviewer", None),
        ("skills/../reviewer", None),
        # Non-canonical or traversal segments, raw or percent-encoded, fail closed rather
        # than resolve to the identity the path started with.
        (f"skills/@acme/reviewer/../../@victim/secret/{_TOKEN}", None),
        ("skills/@acme/reviewer/./../../@victim/secret", None),
        ("skills/@acme/reviewer//../../@victim/secret", None),
        ("skills/@acme/reviewer/..", None),
        ("skills/@acme/reviewer/.", None),
        ("skills/@acme/reviewer/%2e%2e/%2e%2e/@victim/secret", None),
        ("skills/@acme/reviewer/%2E%2E/%2E%2E/@victim/secret", None),
        ("skills/@acme/reviewer/.%2e/.%2e/@victim/secret", None),
        ("skills/@acme/reviewer/%252e%252e/%252e%252e/@victim/secret", None),
        ("skills/@acme/reviewer%2f..%2f..%2f@victim/secret", None),
        # Encoded characters resolve to the identity the artifact handlers serve.
        (f"skills/%40acme/reviewer/{_TOKEN}", SkillArtifactIdentity("acme", "reviewer")),
        (f"skills/%2540acme/reviewer/{_TOKEN}", SkillArtifactIdentity("acme", "reviewer")),
        (
            f"skills/@acme/reviewer/{_TOKEN}/notes%20v2.md",
            SkillArtifactIdentity("acme", "reviewer"),
        ),
        ("skills%2F%40acme%2Freviewer", SkillArtifactIdentity("acme", "reviewer")),
        ("", None),
        (None, None),
    ],
)
def test_parse_skill_upload_path(artifact_path, expected):
    assert parse_skill_upload_path(artifact_path) == expected


def test_parse_skill_upload_path_round_trips_new_paths():
    path = new_skill_upload_path("reviewer", "acme")
    assert parse_skill_upload_path(path) == SkillArtifactIdentity("acme", "reviewer")
    assert parse_skill_upload_path(new_skill_upload_path("linter")) == SkillArtifactIdentity(
        "", "linter"
    )

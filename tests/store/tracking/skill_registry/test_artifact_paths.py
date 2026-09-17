import re

import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.tracking.skill_registry.artifact_paths import (
    artifact_path_from_uri,
    new_skill_upload_path,
    owned_skill_upload_path,
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

# The Skill SDK against a running MLflow server.
#
# tests/store/tracking/skill_registry/test_rest_mixin.py covers every SDK method in process,
# through a test client. test_skills_pull.py drives registration and pull against a server
# process. These tests run the rest of the SDK surface the same way, over real HTTP, so a
# mismatch between what the client sends and what the deployed app reads is caught for every
# method.

import uuid

import pytest

import mlflow
from mlflow.entities.skill import SkillStatus
from mlflow.entities.skill_source import GitSource, OCISource
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from tests.tracking.integration_test_utils import _init_server


@pytest.fixture(scope="module")
def tracking_server(tmp_path_factory):
    root = tmp_path_factory.mktemp("server")
    with _init_server(
        backend_uri=f"sqlite:///{root / 'mlflow.db'}",
        root_artifact_uri=(root / "artifacts").as_uri(),
    ) as url:
        yield url


@pytest.fixture
def client(tracking_server):
    mlflow.set_tracking_uri(tracking_server)
    try:
        yield MlflowClient()
    finally:
        mlflow.set_tracking_uri(None)


@pytest.fixture
def org():
    # The module-scoped server keeps every registration, so each test uses its own organization.
    return f"sdk-{uuid.uuid4().hex[:8]}"


def test_skill_parent_crud_and_search(client, org):
    identity = {"name": "review", "organization": org}
    created = client.create_skill(**identity, description="Reviews code")
    assert (created.name, created.organization, created.description) == (
        "review",
        org,
        "Reviews code",
    )
    assert created.latest_version is None
    assert client.get_skill(**identity) == created

    assert client.update_skill(**identity, description="Updated").description == "Updated"
    # Omitting a field leaves it alone; an explicit None clears it.
    assert client.update_skill(**identity).description == "Updated"
    assert client.update_skill(**identity, description=None).description is None

    client.create_skill(name="docs", organization=org)
    query = {"filter_string": f"organization = '{org}'", "order_by": ["name ASC"]}
    first = client.search_skills(**query, max_results=1)
    assert [skill.name for skill in first] == ["docs"]
    second = client.search_skills(**query, max_results=1, page_token=first.token)
    assert second == [client.get_skill(**identity)]
    assert second.token is None
    assert [skill.name for skill in mlflow.genai.search_skills(**query)] == ["docs", "review"]

    assert client.delete_skill(**identity) is None
    with pytest.raises(MlflowException, match="not found") as exc_info:
        client.get_skill(**identity)
    assert exc_info.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_skill_versions_tags_and_aliases(client, org):
    identity = {"name": "review", "organization": org}
    source = GitSource("https://example.com/skills", ref="v1", subpath="skills/review")
    first = client.create_skill_version(**identity, source=source, digest="a" * 64)
    assert (first.version, first.status, first.source, first.digest) == (
        1,
        SkillStatus.ACTIVE,
        source,
        "a" * 64,
    )
    second = client.create_skill_version(
        **identity, source="oci://ghcr.io/acme/skills:v2", status="draft"
    )
    assert second.source == OCISource("ghcr.io/acme/skills:v2")
    assert client.get_skill_version(**identity, version=1) == first
    # The active version resolves as latest over the higher draft.
    assert client.get_latest_skill_version(**identity) == first
    versions = client.search_skill_versions(**identity, order_by=["version DESC"])
    assert [version.version for version in versions] == [2, 1]

    client.set_skill_tag(**identity, key="team", value="platform")
    client.set_skill_version_tag(**identity, version=1, key="scan", value="clean")
    assert client.get_skill(**identity).tags == {"team": "platform"}
    assert client.get_skill_version(**identity, version=1).tags == {"scan": "clean"}
    client.delete_skill_tag(**identity, key="team")
    client.delete_skill_version_tag(**identity, version=1, key="scan")
    assert client.get_skill(**identity).tags == {}
    assert client.get_skill_version(**identity, version=1).tags == {}

    client.set_skill_alias(**identity, alias="production", version=1)
    assert client.get_skill_version_by_alias(**identity, alias="production").version == 1
    assert client.get_skill(**identity).aliases == {"production": 1}
    client.delete_skill_alias(**identity, alias="production")
    with pytest.raises(MlflowException, match="not found") as exc_info:
        client.get_skill_version_by_alias(**identity, alias="production")
    assert exc_info.value.error_code == "RESOURCE_DOES_NOT_EXIST"

    client.update_skill_version(**identity, version=1, status="deprecated")
    client.delete_skill_version(**identity, version=1)
    assert client.get_latest_skill_version(**identity) == second
    client.delete_skill_version(**identity, version=2)
    with pytest.raises(MlflowException, match="No resolved latest") as exc_info:
        client.get_latest_skill_version(**identity)
    assert exc_info.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_bulk_register_skills(client, org):
    definitions = [
        {
            "name": name,
            "source_type": "git",
            "source": "https://example.com/skills.git",
            "ref": "main",
            "subpath": f"skills/{name}",
            "digest": digest * 64,
        }
        for name, digest in [("review", "a"), ("docs", "b")]
    ]
    versions = client.bulk_register_skills(skill_definitions=definitions, organization=org)

    assert [(v.name, v.version, v.status) for v in versions] == [
        ("review", 1, SkillStatus.ACTIVE),
        ("docs", 1, SkillStatus.ACTIVE),
    ]
    for version in versions:
        assert version.source == GitSource(
            "https://example.com/skills.git", ref="main", subpath=f"skills/{version.name}"
        )
        assert client.get_skill_version(name=version.name, version=1, organization=org) == version

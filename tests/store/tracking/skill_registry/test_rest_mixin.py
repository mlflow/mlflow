import io
import json
import os
import shutil
import subprocess
import tarfile
from copy import deepcopy
from pathlib import Path
from unittest import mock

import pytest
from fastapi import FastAPI
from requests import Response
from starlette.testclient import TestClient

from mlflow.entities.skill import Skill, SkillStatus
from mlflow.entities.skill_source import (
    GitSource,
    MlflowSource,
    OCISource,
    SkillSourceType,
    ZipSource,
)
from mlflow.entities.skill_version import SkillVersion
from mlflow.environment_variables import MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE
from mlflow.exceptions import MlflowException
from mlflow.genai import import_skills, register_skill, search_skills
from mlflow.genai.skill_content.archive import package_skill_tree
from mlflow.genai.skill_content.digest import compute_tree_digest
from mlflow.genai.skill_content.fetchers import FetchedContent
from mlflow.genai.skill_content.paths import tree_size
from mlflow.genai.skill_content.skill_md import inspect_skill_dir
from mlflow.genai.skill_content.sources import resolve_source_type
from mlflow.genai.skills import _filter_and_validate_skill_directories
from mlflow.protos.databricks_pb2 import PERMISSION_DENIED
from mlflow.server import ARTIFACTS_DESTINATION_ENV_VAR, SERVE_ARTIFACTS_ENV_VAR, handlers
from mlflow.server.fastapi_app import add_registry_exception_handlers
from mlflow.server.skill_registry_api import skill_registry_router
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import NOT_SET
from mlflow.store.tracking.rest_store import RestStore
from mlflow.store.tracking.skill_registry.abstract_mixin import SkillRegistryMixin
from mlflow.store.tracking.skill_registry.rest_mixin import RestSkillRegistryMixin, _skill_path
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking.client import MlflowClient
from mlflow.utils.rest_utils import MlflowHostCreds
from mlflow.utils.validation import (
    _MAX_BULK_REGISTER_SKILLS,
    MAX_MODEL_REGISTRY_TAG_KEY_LENGTH,
    MAX_MODEL_REGISTRY_TAG_VALUE_LENGTH,
    MAX_REGISTERED_MODEL_ALIAS_LENGTH,
    MAX_SKILL_VERSION,
)
from mlflow.utils.workspace_context import WorkspaceContext
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME


@pytest.fixture
def store():
    return RestStore(lambda: MlflowHostCreds("https://registry.example.com", token="test-token"))


@pytest.fixture
def mocked_skill_client():
    backend = mock.Mock(spec=SkillRegistryMixin)
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=backend):
        yield MlflowClient(tracking_uri="https://registry.example.com"), backend


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("create_skill", {}),
        ("create_skill_version", {}),
        ("update_skill", {}),
        ("update_skill_version", {"version": 1}),
        ("set_skill_tag", {"key": "team", "value": "platform"}),
        ("set_skill_version_tag", {"version": 1, "key": "team", "value": "platform"}),
        ("set_skill_alias", {"alias": "production", "version": 1}),
    ],
)
@pytest.mark.parametrize(
    "identity",
    [
        {"name": None},
        {"name": 123},
        {"name": ""},
        {"name": "invalid/name"},
        {"organization": None},
        {"organization": 123},
        {"organization": "invalid/org"},
    ],
)
def test_skill_client_rejects_invalid_identity(mocked_skill_client, method, kwargs, identity):
    client, backend = mocked_skill_client
    with pytest.raises(MlflowException, match="[Ss]kill name|[Oo]rganization") as exc:
        getattr(client, method)(**{"name": "review", **identity, **kwargs})

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    getattr(backend, method).assert_not_called()


@pytest.mark.parametrize("method", ["set_skill_tag", "set_skill_version_tag"])
@pytest.mark.parametrize(
    ("key", "value"),
    [
        (None, "platform"),
        (123, "platform"),
        ("team", None),
        ("team", 123),
        ("team/../owner", "platform"),
        ("a" * (MAX_MODEL_REGISTRY_TAG_KEY_LENGTH + 1), "platform"),
        ("team", "a" * (MAX_MODEL_REGISTRY_TAG_VALUE_LENGTH + 1)),
    ],
    ids=[
        "null-key",
        "numeric-key",
        "null-value",
        "numeric-value",
        "path",
        "long-key",
        "long-value",
    ],
)
def test_skill_client_rejects_invalid_tags(mocked_skill_client, method, key, value):
    client, backend = mocked_skill_client
    kwargs = {"version": 1} if method == "set_skill_version_tag" else {}
    with pytest.raises(MlflowException, match="key|value") as exc:
        getattr(client, method)(name="review", key=key, value=value, **kwargs)

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    getattr(backend, method).assert_not_called()


@pytest.mark.parametrize(
    "alias",
    [None, 123, "", "latest", "LATEST", "v1", "a/b", "a" * (MAX_REGISTERED_MODEL_ALIAS_LENGTH + 1)],
)
def test_skill_client_rejects_invalid_aliases(mocked_skill_client, alias):
    client, backend = mocked_skill_client
    with pytest.raises(MlflowException, match="alias") as exc:
        client.set_skill_alias(name="review", alias=alias, version=1)

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    backend.set_skill_alias.assert_not_called()


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("update_skill_version", {}),
        ("set_skill_version_tag", {"key": "team", "value": "platform"}),
        ("set_skill_alias", {"alias": "production"}),
    ],
)
@pytest.mark.parametrize("version", [None, True, "1", 1.0, 0, MAX_SKILL_VERSION + 1])
def test_skill_client_rejects_invalid_versions(mocked_skill_client, method, kwargs, version):
    client, backend = mocked_skill_client
    with pytest.raises(MlflowException, match="positive integer") as exc:
        getattr(client, method)(name="review", version=version, **kwargs)

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    getattr(backend, method).assert_not_called()


@pytest.mark.parametrize("organization", ["", "acme"])
@pytest.mark.parametrize(
    ("method", "kwargs", "defaults"),
    [
        ("create_skill", {}, {"description": None, "icons": None}),
        ("update_skill", {}, {"description": NOT_SET, "icons": NOT_SET}),
        ("update_skill", {"description": None, "icons": None}, {}),
        ("update_skill_version", {"version": 1}, {"status": NOT_SET}),
        ("update_skill_version", {"version": 1, "status": None}, {}),
        ("update_skill_version", {"version": 1, "status": "backend-status"}, {}),
        ("set_skill_tag", {"key": "team/owner", "value": ""}, {}),
        ("set_skill_version_tag", {"version": 1, "key": "team/owner", "value": ""}, {}),
        ("set_skill_alias", {"alias": "production", "version": MAX_SKILL_VERSION}, {}),
    ],
)
def test_skill_client_delegates_valid_inputs(
    mocked_skill_client, method, kwargs, defaults, organization
):
    client, backend = mocked_skill_client
    getattr(client, method)(name="review", organization=organization, **kwargs)

    getattr(backend, method).assert_called_once_with(
        name="review", organization=organization, **kwargs, **defaults
    )


@pytest.mark.parametrize(
    ("digest", "status"),
    [(None, "active"), ("a" * 64, "draft"), ("invalid", "deleted"), (123, None)],
)
def test_skill_client_delegates_digest_and_status(mocked_skill_client, digest, status):
    client, backend = mocked_skill_client
    source = GitSource("https://example.com/repo.git", ref="main", subpath="skills/review")
    result = client.create_skill_version(
        name="review", organization="acme", source=source, digest=digest, status=status
    )

    backend.create_skill_version.assert_called_once_with(
        name="review",
        organization="acme",
        source_type="git",
        source=source.url,
        ref="main",
        subpath="skills/review",
        digest=digest,
        status=status,
    )
    assert result is backend.create_skill_version.return_value


@pytest.mark.parametrize(
    "definitions",
    [None, {}, ["review"], [{}], [{"name": None}], [{"name": 123}], [{"name": "invalid/name"}]],
)
def test_skill_client_bulk_rejects_invalid_definitions(mocked_skill_client, definitions):
    client, backend = mocked_skill_client
    with pytest.raises(MlflowException, match="[Ss]kill") as exc:
        client.bulk_register_skills(skill_definitions=definitions)

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    backend.bulk_register_skills.assert_not_called()


def test_skill_client_bulk_validates_all_names_before_delegating(mocked_skill_client):
    client, backend = mocked_skill_client
    definitions = [{"name": "review"}, {"name": "invalid/name"}]
    with pytest.raises(MlflowException, match="Invalid skill name") as exc:
        client.bulk_register_skills(skill_definitions=definitions)

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    backend.bulk_register_skills.assert_not_called()


@pytest.mark.parametrize("organization", [None, 123, "invalid/org"])
def test_skill_client_bulk_rejects_invalid_organization(mocked_skill_client, organization):
    client, backend = mocked_skill_client
    with pytest.raises(MlflowException, match="Invalid organization") as exc:
        client.bulk_register_skills(
            skill_definitions=[{"name": "review"}], organization=organization
        )

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    backend.bulk_register_skills.assert_not_called()


@pytest.mark.parametrize("organization", ["", "acme"])
def test_skill_client_bulk_delegates_definitions_unchanged(mocked_skill_client, organization):
    client, backend = mocked_skill_client
    definitions = [
        {
            "name": name,
            "source_type": "git",
            "source": "https://example.com/repo.git",
            "ref": "main",
            "subpath": f"skills/{name}",
            "digest": "backend-digest",
            "status": "backend-status",
        }
        for name in ("review", "docs")
    ]
    original = deepcopy(definitions)
    result = client.bulk_register_skills(skill_definitions=definitions, organization=organization)

    backend.bulk_register_skills.assert_called_once_with(
        skill_definitions=original, organization=organization
    )
    assert definitions == original
    assert result is backend.bulk_register_skills.return_value


def test_rest_store_resolves_skill_methods_to_rest_mixin():
    for name, method in vars(RestSkillRegistryMixin).items():
        if callable(method) and not name.startswith("_"):
            assert getattr(RestStore, name) is method, name


@pytest.mark.parametrize(
    ("name", "organization", "expected"),
    [
        ("my-skill", "", "/my-skill"),
        ("my-skill", "team", "/@team/my-skill"),
        ("review-v1", "team.name", "/@team.name/review-v1"),
    ],
)
def test_skill_path(name, organization, expected):
    assert _skill_path(name, organization) == expected


def test_skill_path_defaults_to_unscoped():
    assert _skill_path("my-skill") == "/my-skill"


@pytest.mark.parametrize(
    ("name", "organization"),
    [
        ("a/b c?#%", ""),
        ("skill-\u2603", ""),
        ("@team", ""),
        ("review.v1", ""),
        ("review", "org/team @"),
        ("review", "team-\u2603"),
        ("review", "team..name"),
    ],
)
def test_skill_path_rejects_invalid_identity(name, organization):
    with pytest.raises(
        MlflowException, match="Invalid skill name|Invalid organization"
    ) as exc_info:
        _skill_path(name, organization)
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"


@pytest.mark.parametrize("workspace", [None, "team-a"])
def test_skill_request_preserves_transport_context(store, workspace):
    response = Response()
    response.status_code = 200
    response._content = b'{"skills": [], "next_page_token": "next"}'
    payload = {"description": "Updated"}
    params = {"page_token": "previous"}
    with (
        WorkspaceContext(workspace),
        mock.patch.object(store, "_probe_workspace_support", return_value=True) as probe,
        mock.patch(
            "mlflow.utils.rest_utils._get_http_response_with_retries", return_value=response
        ) as request,
    ):
        result = store._skill_request(
            "PATCH", _skill_path("my-skill", "team"), json=payload, params=params
        )

    assert result == {"skills": [], "next_page_token": "next"}
    if workspace:
        probe.assert_called_once_with()
    else:
        probe.assert_not_called()
    assert request.call_args.args[:2] == (
        "PATCH",
        "https://registry.example.com/api/3.0/mlflow/skills/@team/my-skill",
    )
    kwargs = request.call_args.kwargs
    assert kwargs["json"] == payload
    assert kwargs["params"] == params
    assert kwargs["headers"]["Authorization"] == "Bearer test-token"
    assert kwargs["headers"].get(WORKSPACE_HEADER_NAME) == workspace


def test_skill_request_rejects_unsupported_workspace_before_request(store):
    with (
        WorkspaceContext("team-a"),
        mock.patch.object(store, "_probe_workspace_support", return_value=False) as probe,
        mock.patch("mlflow.store.tracking.skill_registry.rest_mixin.http_request") as request,
        pytest.raises(MlflowException, match="does not support workspaces") as exc_info,
    ):
        store._skill_request("GET", _skill_path("my-skill"))

    assert exc_info.value.error_code == "FEATURE_DISABLED"
    probe.assert_called_once_with()
    request.assert_not_called()


@pytest.mark.parametrize(
    ("status_code", "error_code"),
    [(404, "RESOURCE_DOES_NOT_EXIST"), (403, "PERMISSION_DENIED")],
)
def test_skill_request_preserves_server_errors(store, status_code, error_code):
    response = Response()
    response.status_code = status_code
    response._content = json.dumps({"error_code": error_code, "message": "Skill error"}).encode()
    with (
        mock.patch(
            "mlflow.store.tracking.skill_registry.rest_mixin.http_request", return_value=response
        ) as request,
        pytest.raises(MlflowException, match="Skill error") as exc_info,
    ):
        store._skill_request("GET", _skill_path("my-skill"))

    assert exc_info.value.error_code == error_code
    request.assert_called_once_with(
        host_creds=store.get_host_creds(),
        endpoint="/api/3.0/mlflow/skills/my-skill",
        method="GET",
        json=None,
        params=None,
    )


@pytest.fixture
def registry_client(store, tmp_path, db_uri):
    db_store = SqlAlchemyStore(db_uri, (tmp_path / "artifacts").as_uri())
    app = FastAPI()
    add_registry_exception_handlers(app)
    app.include_router(skill_registry_router, prefix="/api/3.0/mlflow/skills")
    with (
        TestClient(app) as http_client,
        mock.patch("mlflow.server.handlers._get_tracking_store", return_value=db_store),
        mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=store),
        mock.patch(
            "mlflow.store.tracking.skill_registry.rest_mixin.http_request",
            side_effect=lambda host_creds, endpoint, method, **kwargs: http_client.request(
                method, endpoint, **kwargs
            ),
        ),
        mock.patch(
            "mlflow.utils.validation._resolve_hostname_with_timeout",
            return_value=[(None, None, None, None, ("8.8.8.8", 0))],
        ),
    ):
        yield MlflowClient(), db_store


@pytest.mark.parametrize("organization", ["", "acme"])
def test_parent_crud(registry_client, organization):
    client, db_store = registry_client
    icons = [{"src": "https://example.com/icon.png", "mimeType": "image/png", "sizes": ["48x48"]}]
    identity = {"name": "code-review", "organization": organization}
    created = client.create_skill(**identity, description="Reviews code", icons=icons)
    assert created == db_store.get_skill(**identity)
    assert created.description == "Reviews code"
    assert created.icons == icons
    assert client.get_skill(**identity) == created

    db_store.create_skill_version(
        **identity, source_type="git", source="https://example.com/repo.git"
    )
    db_store.set_skill_tag(**identity, key="team", value="platform")
    db_store.set_skill_alias(**identity, alias="production", version=1)
    fetched = client.get_skill(**identity)
    assert fetched == db_store.get_skill(**identity)
    assert fetched.status == SkillStatus.ACTIVE
    assert fetched.source_type == SkillSourceType.GIT
    assert fetched.latest_version == 1
    assert fetched.tags == {"team": "platform"}
    assert fetched.aliases == {"production": 1}

    updated = client.update_skill(**identity, description="Updated")
    assert updated.description == "Updated"
    assert updated.icons == icons
    unchanged = client.update_skill(**identity)
    assert unchanged.description == updated.description
    assert unchanged.icons == icons
    assert unchanged == db_store.get_skill(**identity)
    without_icons = client.update_skill(**identity, icons=[])
    assert without_icons.icons == []
    assert without_icons.description == "Updated"
    cleared = client.update_skill(**identity, description=None, icons=None)
    assert cleared.description is None
    assert cleared.icons is None
    assert cleared == client.get_skill(**identity)

    assert client.delete_skill(**identity) is None
    with pytest.raises(MlflowException, match="not found") as exc_info:
        client.get_skill(**identity)
    assert exc_info.value.error_code == "RESOURCE_DOES_NOT_EXIST"
    with pytest.raises(MlflowException, match="not found"):
        db_store.get_skill_version(**identity, version=1)


def test_parent_organizations_are_independent(registry_client):
    client, _ = registry_client
    client.create_skill(name="review", description="unscoped")
    client.create_skill(name="review", organization="acme", description="scoped")
    client.update_skill(name="review", organization="acme", description="updated")
    client.delete_skill(name="review", organization="acme")
    assert client.get_skill(name="review").description == "unscoped"


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("delete_skill", {"name": "@other/review"}),
        ("delete_skill", {"name": "review/versions/1"}),
        ("delete_skill", {"name": "1", "organization": "acme/review/versions"}),
        (
            "set_skill_tag",
            {"name": "review/versions/1", "key": "team", "value": "changed"},
        ),
        ("get_skill_version", {"name": "review", "version": "1/tags/team"}),
        ("update_skill_version", {"name": "review", "version": "1/tags/team"}),
        ("delete_skill_version", {"name": "review", "version": "1/tags/team"}),
        (
            "set_skill_version_tag",
            {"name": "review", "version": "1/tags/team", "key": "team", "value": "changed"},
        ),
        (
            "delete_skill_version_tag",
            {"name": "review", "version": "1/tags/team", "key": "team"},
        ),
    ],
)
def test_invalid_path_parameters_cannot_change_target(registry_client, method, kwargs):
    client, db_store = registry_client
    parents = []
    versions = []
    for organization in ("", "other", "acme"):
        identity = {"name": "review", "organization": organization}
        client.create_skill_version(
            **identity, source="https://example.com/repo.git", status="draft"
        )
        client.set_skill_tag(**identity, key="team", value="parent")
        client.set_skill_version_tag(**identity, version=1, key="team", value="version")
        parents.append(db_store.get_skill(**identity))
        versions.append(db_store.get_skill_version(**identity, version=1))

    with (
        mock.patch("mlflow.store.tracking.skill_registry.rest_mixin.http_request") as request,
        pytest.raises(
            MlflowException, match="Invalid skill name|Invalid organization|Skill version"
        ) as exc_info,
    ):
        getattr(client, method)(**kwargs)
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    request.assert_not_called()

    for parent, version in zip(parents, versions):
        identity = {"name": parent.name, "organization": parent.organization}
        assert db_store.get_skill(**identity) == parent
        assert db_store.get_skill_version(**identity, version=1) == version


@pytest.mark.parametrize("value", [".", "..", 123, None, True, 1.0, b"key", [], {}])
@pytest.mark.parametrize(
    ("method", "kwargs", "parameter"),
    [
        ("delete_skill_tag", {}, "key"),
        ("delete_skill_version_tag", {"version": 1}, "key"),
        ("delete_skill_alias", {}, "alias"),
        ("get_skill_version_by_alias", {}, "alias"),
    ],
)
def test_skill_rest_rejects_invalid_path_parameters(store, method, kwargs, parameter, value):
    message = (
        "Path parameters must not be"
        if isinstance(value, str)
        else "Path parameters must be strings"
    )
    with (
        mock.patch("mlflow.store.tracking.skill_registry.rest_mixin.http_request") as request,
        pytest.raises(MlflowException, match=message) as exc,
    ):
        getattr(store, method)(name="review", **kwargs, **{parameter: value})
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    request.assert_not_called()


@pytest.mark.parametrize("api", ["client", "genai"])
def test_search_skills_filters_ordering_and_pagination(registry_client, api):
    client, _ = registry_client
    search = client.search_skills if api == "client" else search_skills
    empty = search()
    assert isinstance(empty, PagedList)
    assert empty == []
    assert empty.token is None
    for name in ["alpha", "bravo", "charlie"]:
        client.create_skill(name=name, organization="acme")
    client.create_skill(name="other")
    query = {"filter_string": "organization = 'acme'", "order_by": ["name DESC"], "max_results": 2}
    first = search(**query)
    assert isinstance(first, PagedList)
    assert [skill.name for skill in first] == ["charlie", "bravo"]
    assert first.token is not None
    assert first[0] == client.get_skill(name="charlie", organization="acme")
    second = search(**query, page_token=first.token)
    assert isinstance(second, PagedList)
    assert [skill.name for skill in second] == ["alpha"]
    assert second.token is None


@pytest.mark.parametrize("api", ["client", "genai"])
@pytest.mark.parametrize(
    ("query", "message"),
    [
        ({"page_token": "invalid-token"}, "[Pp]age.token"),
        ({"filter_string": "unknown_field = 'value'"}, "unknown_field"),
        ({"order_by": ["unknown_field ASC"]}, "unknown_field"),
        ({"max_results": 0}, "max_results"),
    ],
)
def test_search_skills_propagates_server_errors(registry_client, api, query, message):
    client, _ = registry_client
    search = client.search_skills if api == "client" else search_skills
    with pytest.raises(MlflowException, match=message) as exc_info:
        search(**query)
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"


def test_parent_crud_propagates_server_errors(registry_client):
    client, _ = registry_client
    client.create_skill(name="review")
    with pytest.raises(MlflowException, match="already exists") as exc_info:
        client.create_skill(name="review")
    assert exc_info.value.error_code == "RESOURCE_ALREADY_EXISTS"
    for method in (client.get_skill, client.update_skill, client.delete_skill):
        with pytest.raises(MlflowException, match="not found") as exc_info:
            method(name="missing")
        assert exc_info.value.error_code == "RESOURCE_DOES_NOT_EXIST"


@pytest.mark.parametrize("method", ["create_skill", "get_skill", "update_skill", "search_skills"])
def test_parent_methods_preserve_complete_response(store, method):
    data = {
        "name": "review",
        "organization": "acme",
        "description": "Review",
        "icons": [{"src": "https://example.com/icon.png", "theme": "dark"}],
        "status": "deprecated",
        "latest_version": 7,
        "source_type": "git",
        "aliases": [{"alias": "production", "version": 7}],
        "tags": {"team": "platform"},
        "created_by": "creator",
        "last_updated_by": "editor",
        "creation_timestamp": 1000,
        "last_updated_timestamp": 2000,
    }
    expected = Skill(
        name="review",
        organization="acme",
        description="Review",
        icons=[{"src": "https://example.com/icon.png", "theme": "dark"}],
        status=SkillStatus.DEPRECATED,
        latest_version=7,
        source_type=SkillSourceType.GIT,
        aliases={"production": 7},
        tags={"team": "platform"},
        created_by="creator",
        last_updated_by="editor",
        creation_timestamp=1000,
        last_updated_timestamp=2000,
    )
    response = (
        {"skills": [data], "next_page_token": "opaque-token"} if method == "search_skills" else data
    )
    with mock.patch.object(store, "_skill_request", return_value=response) as request:
        if method == "search_skills":
            page = store.search_skills()
            assert page == [expected]
            assert page.token == "opaque-token"
        else:
            assert getattr(store, method)(name="review", organization="acme") == expected
    if method == "create_skill":
        request.assert_called_once_with("POST", "", json={"name": "review", "organization": "acme"})
    elif method == "get_skill":
        request.assert_called_once_with("GET", "/@acme/review")
    elif method == "update_skill":
        request.assert_called_once_with("PATCH", "/@acme/review", json={})
    else:
        request.assert_called_once_with("GET", "", params={"max_results": 1000})


def test_parent_requests_omit_audit_arguments(store):
    with mock.patch.object(store, "_skill_request", return_value={"name": "review"}) as request:
        store.create_skill(name="review", created_by="client-user")
        request.assert_called_once_with("POST", "", json={"name": "review", "organization": ""})
        request.reset_mock()
        store.update_skill(name="review", last_updated_by="client-user")
        request.assert_called_once_with("PATCH", "/review", json={})


@pytest.mark.parametrize("organization", ["", "acme"])
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            GitSource("https://example.com/repo", ref="v2", subpath="skills/review"),
            GitSource("https://example.com/repo", ref="v2", subpath="skills/review"),
        ),
        (
            OCISource("oci://ghcr.io/acme/skills:v1", subpath="review"),
            OCISource("ghcr.io/acme/skills:v1", subpath="review"),
        ),
        (
            ZipSource("https://example.com/download", subpath="review"),
            ZipSource("https://example.com/download", subpath="review"),
        ),
        ("https://example.com/repo.git", GitSource("https://example.com/repo.git")),
        ("oci://ghcr.io/acme/skills:v1", OCISource("ghcr.io/acme/skills:v1")),
        ("https://example.com/skills.zip", ZipSource("https://example.com/skills.zip")),
    ],
)
def test_create_and_get_skill_version(registry_client, organization, source, expected):
    client, db_store = registry_client
    identity = {"name": "review", "organization": organization}
    created = client.create_skill_version(
        **identity, source=source, digest="a" * 64, status="draft"
    )
    assert created.version == 1
    assert created.source == expected
    assert created.digest == "a" * 64
    assert created.status == SkillStatus.DRAFT
    assert created == db_store.get_skill_version(**identity, version=1)
    assert client.get_skill_version(**identity, version=1) == created
    assert client.get_skill(**identity).latest_version == 1


@pytest.mark.parametrize("supplied", [{}, {"source": None}], ids=["omitted", "explicit-none"])
def test_create_skill_version_delegates_optional_source_to_backend(
    registry_client, store, supplied
):
    client, db_store = registry_client
    with (
        mock.patch.object(
            store, "create_skill_version", wraps=store.create_skill_version
        ) as create,
        pytest.raises(MlflowException, match="requires a multipart/form-data body") as exc,
    ):
        client.create_skill_version(name="review", **supplied)
    create.assert_called_once_with(
        name="review",
        organization="",
        source_type=None,
        source=None,
        ref=None,
        subpath=None,
        digest=None,
        status="active",
    )
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    assert db_store.search_skills() == []


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("https://example.com/repo", "Cannot infer"),
        ("ssh://git@example.com/repo", "Cannot infer"),
        (
            MlflowSource("mlflow-artifacts:/skills/review/token"),
            "typed source or a non-empty string",
        ),
        ("mlflow-artifacts:/skills/review/token", "chosen by the server"),
    ],
)
def test_create_skill_version_rejects_invalid_source_before_request(
    registry_client, store, source, message
):
    client, _ = registry_client
    with (
        mock.patch.object(store, "_skill_request") as request,
        pytest.raises(MlflowException, match=message) as exc_info,
    ):
        client.create_skill_version(name="review", source=source)
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    request.assert_not_called()


@pytest.mark.parametrize("organization", ["", "acme"])
def test_skill_version_lifecycle_and_aliases(registry_client, organization):
    client, db_store = registry_client
    identity = {"name": "review", "organization": organization}
    client.create_skill(**identity)
    first = client.create_skill_version(**identity, source="https://example.com/repo.git")
    second = client.create_skill_version(
        **identity, source="https://example.com/repo.git", status="draft"
    )
    db_store.set_skill_version_tag(**identity, version=1, key="scan", value="clean")
    db_store.set_skill_alias(**identity, alias="production", version=1)
    fetched = client.get_skill_version(**identity, version=1)
    assert fetched.tags == {"scan": "clean"}
    assert fetched.aliases == ["production"]
    assert client.get_skill_version_by_alias(**identity, alias="production") == fetched
    assert client.get_latest_skill_version(**identity) == fetched
    assert client.update_skill_version(**identity, version=1) == fetched
    with pytest.raises(MlflowException, match="status cannot be null"):
        client.update_skill_version(**identity, version=1, status=None)
    with pytest.raises(MlflowException, match="Invalid status transition"):
        client.delete_skill_version(**identity, version=first.version)

    active = client.update_skill_version(**identity, version=second.version, status="active")
    assert active.status == SkillStatus.ACTIVE
    assert client.get_latest_skill_version(**identity) == active
    deprecated = client.update_skill_version(**identity, version=first.version, status="deprecated")
    assert deprecated.status == SkillStatus.DEPRECATED
    assert client.delete_skill_version(**identity, version=first.version) is None
    with pytest.raises(MlflowException, match="not found"):
        client.get_skill_version(**identity, version=first.version)
    with pytest.raises(MlflowException, match="not found"):
        client.get_skill_version_by_alias(**identity, alias="production")


@pytest.mark.parametrize("organization", ["", "acme"])
def test_search_skill_versions(registry_client, organization):
    client, _ = registry_client
    identity = {"name": "review", "organization": organization}
    client.create_skill(**identity)
    assert client.search_skill_versions(**identity) == []
    for status in ["active", "draft", "active", "active"]:
        client.create_skill_version(
            **identity, source="https://example.com/repo.git", status=status
        )
    query = {
        **identity,
        "filter_string": "status = 'active'",
        "order_by": ["version DESC"],
        "max_results": 2,
    }
    first = client.search_skill_versions(**query)
    assert isinstance(first, PagedList)
    assert [version.version for version in first] == [4, 3]
    assert first[0] == client.get_skill_version(**identity, version=4)
    assert first.token is not None
    second = client.search_skill_versions(**query, page_token=first.token)
    assert [version.version for version in second] == [1]
    assert second.token is None
    with pytest.raises(MlflowException, match="[Pp]age.token"):
        client.search_skill_versions(**identity, page_token="invalid-token")


def test_skill_version_errors(registry_client):
    client, _ = registry_client
    with pytest.raises(MlflowException, match="can be registered as"):
        client.create_skill_version(
            name="review", source="https://example.com/repo.git", status="deleted"
        )
    client.create_skill(name="review")
    with pytest.raises(MlflowException, match="requires a .*multipart"):
        client.create_skill_version(name="review")
    with pytest.raises(MlflowException, match="No resolved latest"):
        client.get_latest_skill_version(name="review")
    for method in (
        client.get_skill_version,
        client.update_skill_version,
        client.delete_skill_version,
    ):
        with pytest.raises(MlflowException, match="not found") as exc_info:
            method(name="review", version=999)
        assert exc_info.value.error_code == "RESOURCE_DOES_NOT_EXIST"


@pytest.mark.parametrize("deleted_versions", [0, 2], ids=["no-versions", "every-version-deleted"])
def test_get_latest_skill_version_without_a_live_version(registry_client, deleted_versions):
    client, _ = registry_client
    client.create_skill(name="review")
    for _ in range(deleted_versions):
        version = client.create_skill_version(
            name="review", source="https://example.com/repo.git", status="draft"
        )
        client.delete_skill_version(name="review", version=version.version)

    with pytest.raises(MlflowException, match="No resolved latest") as exc_info:
        client.get_latest_skill_version(name="review")
    assert exc_info.value.error_code == "RESOURCE_DOES_NOT_EXIST"
    # The parent stays readable; it just resolves to no version.
    skill = client.get_skill(name="review")
    assert skill.latest_version is None
    assert skill.status is None


@pytest.mark.parametrize(
    ("source_type", "source", "ref", "expected"),
    [
        (
            "git",
            "https://example.com/repo",
            "v2",
            GitSource("https://example.com/repo", ref="v2", subpath="review"),
        ),
        (
            "oci",
            "ghcr.io/acme/skills:v1",
            None,
            OCISource("ghcr.io/acme/skills:v1", subpath="review"),
        ),
        (
            "zip",
            "https://example.com/archive",
            None,
            ZipSource("https://example.com/archive", subpath="review"),
        ),
        (
            "mlflow",
            "mlflow-artifacts:/plugins/package/token",
            None,
            MlflowSource("mlflow-artifacts:/plugins/package/token", subpath="review"),
        ),
    ],
)
def test_get_skill_version_preserves_complete_response(store, source_type, source, ref, expected):
    data = {
        "name": "review",
        "organization": "acme",
        "version": 7,
        "source_type": source_type,
        "source": source,
        "ref": ref,
        "subpath": "review",
        "digest": "a" * 64,
        "status": "deprecated",
        "tags": {"scan": "clean"},
        "aliases": ["production"],
        "created_by": "creator",
        "last_updated_by": "editor",
        "creation_timestamp": 1000,
        "last_updated_timestamp": 2000,
    }
    with mock.patch.object(store, "_skill_request", return_value=data) as request:
        assert store.get_skill_version(
            name="review", version=7, organization="acme"
        ) == SkillVersion(
            name="review",
            organization="acme",
            version=7,
            source_type=SkillSourceType(source_type),
            source=expected,
            digest="a" * 64,
            status=SkillStatus.DEPRECATED,
            tags={"scan": "clean"},
            aliases=["production"],
            created_by="creator",
            last_updated_by="editor",
            creation_timestamp=1000,
            last_updated_timestamp=2000,
        )
    request.assert_called_once_with("GET", "/@acme/review/versions/7")


def test_skill_version_request_paths_and_audit_arguments(store):
    identity = {"name": "review", "organization": "acme"}
    with mock.patch.object(
        store, "_skill_request", return_value={"name": "review", "version": 1}
    ) as request:
        store.create_skill_version(
            **identity,
            source_type="git",
            source="https://example.com/repo",
            created_by="client-user",
        )
        assert "created_by" not in request.call_args.kwargs["json"]
        request.reset_mock()
        store.get_skill_version_by_alias(**identity, alias="a/b ?")
        request.assert_called_once_with("GET", "/@acme/review/aliases/a%2Fb%20%3F")
        request.reset_mock()
        store.update_skill_version(
            **identity, version=1, status=None, last_updated_by="client-user"
        )
        request.assert_called_once_with("PATCH", "/@acme/review/versions/1", json={"status": None})
        request.reset_mock()
        store.delete_skill_version(**identity, version=1, last_updated_by="client-user")
        request.assert_called_once_with("DELETE", "/@acme/review/versions/1")


@pytest.mark.parametrize("organization", ["", "acme"])
def test_skill_tags_and_version_tags(registry_client, organization):
    client, _ = registry_client
    identity = {"name": "review", "organization": organization}
    for _ in range(2):
        client.create_skill_version(**identity, source="https://example.com/repo.git")
    key = "team/review notes"
    assert client.set_skill_tag(**identity, key=key, value="parent") is None
    assert client.set_skill_version_tag(**identity, version=1, key=key, value="version") is None
    assert client.get_skill(**identity).tags == {key: "parent"}
    assert client.get_skill_version(**identity, version=1).tags == {key: "version"}
    assert client.get_skill_version(**identity, version=2).tags == {}

    client.set_skill_tag(**identity, key=key, value="")
    client.set_skill_version_tag(**identity, version=1, key=key, value="updated")
    assert client.get_skill(**identity).tags == {key: ""}
    assert client.get_skill_version(**identity, version=1).tags == {key: "updated"}
    assert client.delete_skill_tag(**identity, key=key) is None
    assert client.get_skill(**identity).tags == {}
    assert client.get_skill_version(**identity, version=1).tags == {key: "updated"}
    assert client.delete_skill_version_tag(**identity, version=1, key=key) is None
    assert client.get_skill_version(**identity, version=1).tags == {}


@pytest.mark.parametrize("organization", ["", "acme"])
def test_skill_alias_set_reassign_and_delete(registry_client, organization):
    client, _ = registry_client
    identity = {"name": "review", "organization": organization}
    for _ in range(2):
        client.create_skill_version(**identity, source="https://example.com/repo.git")
    assert client.set_skill_alias(**identity, alias="production", version=1) is None
    assert client.get_skill_version_by_alias(**identity, alias="production").version == 1
    assert client.get_skill(**identity).aliases == {"production": 1}
    client.set_skill_alias(**identity, alias="production", version=2)
    assert client.get_skill_version_by_alias(**identity, alias="production").version == 2
    assert client.get_skill_version(**identity, version=1).aliases == []
    assert client.get_skill_version(**identity, version=2).aliases == ["production"]
    assert client.delete_skill_alias(**identity, alias="production") is None
    assert client.get_skill(**identity).aliases == {}
    assert client.get_skill_version(**identity, version=2).aliases == []
    with pytest.raises(MlflowException, match="not found"):
        client.get_skill_version_by_alias(**identity, alias="production")


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("set_skill_tag", {"key": "team", "value": "platform"}),
        ("delete_skill_tag", {"key": "team"}),
        ("set_skill_version_tag", {"version": 1, "key": "team", "value": "platform"}),
        ("delete_skill_version_tag", {"version": 1, "key": "team"}),
        ("set_skill_alias", {"alias": "production", "version": 1}),
        ("delete_skill_alias", {"alias": "production"}),
    ],
)
def test_tag_and_alias_methods_propagate_missing_resource(registry_client, method, kwargs):
    client, _ = registry_client
    with pytest.raises(MlflowException, match="not found") as exc_info:
        getattr(client, method)(name="missing", organization="acme", **kwargs)
    assert exc_info.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_tag_and_alias_errors_leave_existing_metadata_intact(registry_client):
    client, _ = registry_client
    client.create_skill_version(name="review", source="https://example.com/repo.git")
    client.set_skill_alias(name="review", alias="production", version=1)
    with pytest.raises(MlflowException, match="not found") as exc_info:
        client.set_skill_alias(name="review", alias="production", version=999)
    assert exc_info.value.error_code == "RESOURCE_DOES_NOT_EXIST"
    with pytest.raises(MlflowException, match="reserved") as exc_info:
        client.set_skill_alias(name="review", alias="latest", version=1)
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    with pytest.raises(MlflowException, match="cannot be deleted"):
        client.delete_skill_alias(name="review", alias="latest")
    with pytest.raises(MlflowException, match="not found"):
        client.delete_skill_tag(name="review", key="missing")
    with pytest.raises(MlflowException, match="not found"):
        client.delete_skill_version_tag(name="review", version=1, key="missing")
    assert client.get_skill(name="review").aliases == {"production": 1}


@pytest.mark.parametrize(
    ("method", "kwargs", "suffix"),
    [
        ("delete_skill_tag", {"key": "a/b ?#%"}, "tags/a%2Fb%20%3F%23%25"),
        (
            "delete_skill_version_tag",
            {"version": 1, "key": "a/b ?#%"},
            "versions/1/tags/a%2Fb%20%3F%23%25",
        ),
        ("delete_skill_alias", {"alias": "a/b ?#%"}, "aliases/a%2Fb%20%3F%23%25"),
        ("delete_skill_tag", {"key": "."}, None),
        ("delete_skill_tag", {"key": ".."}, None),
        ("delete_skill_version_tag", {"version": 1, "key": "."}, None),
        ("delete_skill_version_tag", {"version": 1, "key": ".."}, None),
        ("delete_skill_alias", {"alias": "."}, None),
        ("delete_skill_alias", {"alias": ".."}, None),
    ],
)
def test_tag_and_alias_delete_paths_are_encoded(store, method, kwargs, suffix):
    with mock.patch.object(store, "_skill_request") as request:
        if suffix is None:
            with pytest.raises(MlflowException, match="Path parameters must not be") as exc_info:
                getattr(store, method)(name="my-skill", organization="my-org", **kwargs)
            assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
            request.assert_not_called()
            return
        getattr(store, method)(name="my-skill", organization="my-org", **kwargs)
    request.assert_called_once_with("DELETE", f"/@my-org/my-skill/{suffix}")


@pytest.fixture
def skill_definitions():
    return [
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


@pytest.mark.parametrize("organization", ["", "acme"])
@pytest.mark.parametrize("status", [None, "active", "draft"])
def test_bulk_register_skills(registry_client, store, skill_definitions, organization, status):
    client, db_store = registry_client
    if status is not None:
        for definition in skill_definitions:
            definition["status"] = status
    original = deepcopy(skill_definitions)
    with mock.patch.object(store, "_skill_request", wraps=store._skill_request) as request:
        result = client.bulk_register_skills(
            skill_definitions=skill_definitions, organization=organization
        )
    request.assert_called_once_with(
        "POST", "/bulk-register", json={"organization": organization, "skills": original}
    )
    assert [version.name for version in result] == ["review", "docs"]
    assert [version.version for version in result] == [1, 1]
    assert all(version.status == (status or "active") for version in result)
    assert result == [
        db_store.get_skill_version(name=definition["name"], version=1, organization=organization)
        for definition in skill_definitions
    ]
    assert result[0].source == GitSource(
        "https://example.com/skills.git", ref="main", subpath="skills/review"
    )
    assert skill_definitions == original
    assert (
        client.bulk_register_skills(skill_definitions=skill_definitions, organization=organization)
        == result
    )


@pytest.mark.parametrize("requested_status", ["active", "draft"])
@pytest.mark.parametrize("existing_status", ["active", "draft", "deprecated"])
def test_bulk_register_preserves_reused_version(
    registry_client, store, skill_definitions, requested_status, existing_status
):
    client, db_store = registry_client
    for _ in range(2):
        db_store.create_skill_version(
            **skill_definitions[0],
            organization="acme",
            created_by="author",
            status="draft" if existing_status == "active" else "active",
        )
    db_store.update_skill_version(
        name="review",
        version=2,
        organization="acme",
        status=existing_status,
        last_updated_by="editor",
    )
    db_store.set_skill_version_tag(
        name="review", version=2, organization="acme", key="scan", value="clean"
    )
    db_store.set_skill_alias(name="review", version=2, organization="acme", alias="production")
    expected = db_store.get_skill_version(name="review", version=2, organization="acme")
    for definition in skill_definitions:
        definition["status"] = requested_status
    with mock.patch.object(store, "_skill_request", wraps=store._skill_request) as request:
        result = client.bulk_register_skills(
            skill_definitions=skill_definitions, organization="acme"
        )
    assert request.call_count == 1
    assert result[0] == expected
    assert result[1].name == "docs"
    assert result[1].version == 1
    assert result[1].status == requested_status
    assert db_store.get_skill_version(name="review", version=2, organization="acme") == expected


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"status": "draft"}, "same status"),
        ({"digest": "invalid"}, "SHA-256 digest"),
        ({"name": "review"}, "Duplicate Skill name"),
    ],
)
def test_bulk_register_propagates_validation_errors_without_writes(
    registry_client, skill_definitions, changes, message
):
    client, db_store = registry_client
    skill_definitions[1].update(changes)
    with pytest.raises(MlflowException, match=message) as exc_info:
        client.bulk_register_skills(skill_definitions=skill_definitions)
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert db_store.search_skills() == []


def test_bulk_register_rejects_empty_batch(registry_client):
    client, _ = registry_client
    with pytest.raises(MlflowException, match="skills") as exc_info:
        client.bulk_register_skills(skill_definitions=[])
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"


def test_bulk_register_omits_audit_argument(store, skill_definitions):
    with mock.patch.object(store, "_skill_request", return_value={"skill_versions": []}) as request:
        assert store.bulk_register_skills(skill_definitions, created_by="client-user") == []
    request.assert_called_once_with(
        "POST", "/bulk-register", json={"organization": "", "skills": skill_definitions}
    )


@pytest.fixture
def skill_repository(tmp_path):
    repo = tmp_path / "repository"
    for path, name in [("skills/a-review", "review"), ("skills/nested/z-docs", "docs")]:
        root = repo / path
        root.mkdir(parents=True)
        (root / "SKILL.md").write_text(f"---\nname: {name}\n---\n# {name}\n")
        (root / "script.py").write_text(f"print('{name}')\n")
    (repo / "README.md").write_text("A repository of skills\n")
    return repo


@pytest.fixture
def remote_repository(skill_repository):
    roots = []

    def fetch(resolved, dest, scratch, limit):
        roots.append(dest)
        shutil.copytree(skill_repository, dest)
        return dest

    with mock.patch("mlflow.genai.skill_content.fetchers._fetch_remote", side_effect=fetch) as call:
        yield call, roots


@pytest.mark.parametrize(
    "source",
    [
        "https://example.com/skills.git",
        GitSource("https://example.com/skills", ref="v2", subpath=" skills// "),
    ],
)
@pytest.mark.parametrize("skill_names", [None, ["docs", "review"], ["docs"]])
@pytest.mark.parametrize("status", [None, "draft"])
def test_import_skills(
    registry_client, store, skill_repository, remote_repository, source, skill_names, status
):
    client, db_store = registry_client
    fetch, roots = remote_repository
    resolved = resolve_source_type(source)
    with (
        mock.patch("mlflow.genai.skills.MlflowClient", return_value=client) as client_factory,
        mock.patch.object(store, "_skill_request", wraps=store._skill_request) as request,
    ):
        result = import_skills(
            source=source,
            organization="acme",
            skill_names=skill_names,
            **({"status": status} if status else {}),
        )
    client_factory.assert_called_once_with()
    expected = [
        {
            "name": name,
            "source_type": "git",
            "source": resolved.source,
            "ref": resolved.ref,
            "subpath": path,
            "digest": compute_tree_digest(skill_repository / path),
            "status": status or "active",
        }
        for name, path in [("review", "skills/a-review"), ("docs", "skills/nested/z-docs")]
        if skill_names is None or name in skill_names
    ]
    request.assert_called_once_with(
        "POST", "/bulk-register", json={"organization": "acme", "skills": expected}
    )
    fetch.assert_called_once()
    assert fetch.call_args.args[0] == resolved
    assert [version.name for version in result] == [entry["name"] for entry in expected]
    assert [version.status for version in result] == [status or "active"] * len(expected)
    assert result == [
        db_store.get_skill_version(name=entry["name"], version=1, organization="acme")
        for entry in expected
    ]
    assert all(not root.exists() for root in roots)


@pytest.mark.parametrize("subpath", [None, "skills/a-review"])
def test_import_includes_skill_at_discovery_root(
    registry_client, skill_repository, remote_repository, subpath
):
    if subpath is None:
        shutil.rmtree(skill_repository / "skills")
    (skill_repository / "SKILL.md").write_text("---\nname: root-skill\n---\n")
    versions = import_skills(
        source=GitSource("https://example.com/skills", subpath=subpath),
        skill_names=["root-skill" if subpath is None else "review"],
    )
    assert len(versions) == 1
    assert versions[0].source.subpath == subpath
    assert versions[0].digest == compute_tree_digest(skill_repository / (subpath or ""))


def test_import_limits_discovery_to_subpath(registry_client, skill_repository, remote_repository):
    (skill_repository / "SKILL.md").write_text("---\nname: review\n---\n")
    versions = import_skills(source=GitSource("https://example.com/skills", subpath="skills"))
    assert [version.name for version in versions] == ["review", "docs"]


@pytest.mark.parametrize(
    "path", [" leading/skill-a", "\u00a0leading/skill-a", "skills/skill-a\u00a0"]
)
def test_import_rejects_subpath_changed_by_normalization(
    registry_client, store, skill_repository, remote_repository, path
):
    root = skill_repository / path
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text("---\nname: skill-a\n---\n")
    fetch, roots = remote_repository
    with (
        mock.patch("mlflow.genai.skills.compute_tree_digest") as digest,
        mock.patch.object(store, "_skill_request") as request,
        pytest.raises(MlflowException, match="would change during source normalization") as exc,
    ):
        import_skills(source="https://example.com/skills.git", skill_names=["skill-a"])
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    digest.assert_not_called()
    request.assert_not_called()
    assert registry_client[1].search_skills() == []
    fetch.assert_called_once()
    assert all(not root.exists() for root in roots)


@pytest.mark.parametrize("subpath", [None, "skills"])
def test_import_preserves_internal_subpath_whitespace(
    registry_client, skill_repository, remote_repository, subpath
):
    path = "skills/ leading/skill-a"
    root = skill_repository / path
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text("---\nname: skill-a\n---\n")
    versions = import_skills(
        source=GitSource("https://example.com/skills.git", subpath=subpath),
        skill_names=["skill-a"],
    )
    assert len(versions) == 1
    assert versions[0].source.subpath == path
    assert versions[0].digest == compute_tree_digest(root)


@pytest.mark.parametrize("directory", [False, True])
def test_import_ignores_case_insensitive_nested_manifest(
    registry_client, skill_repository, remote_repository, directory
):
    root = skill_repository / "skills/a-review/child"
    root.mkdir()
    manifest = root / "skill.md"
    if directory:
        manifest.mkdir()
    else:
        manifest.write_text("---\nname: child\n---\n")
    if not (root / "SKILL.md").exists():
        pytest.skip("Requires a case-insensitive filesystem")
    fetch, roots = remote_repository
    versions = import_skills(source="https://example.com/skills.git")
    assert [version.name for version in versions] == ["review", "docs"]
    assert versions[0].digest == compute_tree_digest(skill_repository / "skills/a-review")
    fetch.assert_called_once()
    assert all(not root.exists() for root in roots)


@pytest.mark.parametrize("nested_path", ["child", "AAA/deep-child"])
@pytest.mark.parametrize("skill_names", [None, ["review"], ["child"], ["docs"]])
@pytest.mark.parametrize(
    "content", ["---\nname: child\n---\n", "Invalid child manifest", "---\nname: docs\n---\n"]
)
def test_import_stops_discovery_at_skill_roots(
    registry_client, store, skill_repository, remote_repository, nested_path, skill_names, content
):
    root = skill_repository / "skills/a-review" / nested_path
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(content)
    fetch, roots = remote_repository
    with (
        mock.patch("mlflow.genai.skills.inspect_skill_dir", wraps=inspect_skill_dir) as inspect,
        mock.patch.object(store, "_skill_request", wraps=store._skill_request) as request,
    ):
        if skill_names == ["child"]:
            with pytest.raises(MlflowException, match="Requested skills were not found: child"):
                import_skills(source="https://example.com/skills.git", skill_names=skill_names)
            request.assert_not_called()
            assert registry_client[1].search_skills() == []
        else:
            versions = import_skills(
                source="https://example.com/skills.git", skill_names=skill_names
            )
            assert [version.name for version in versions] == (skill_names or ["review", "docs"])
            for version in versions:
                assert version.digest == compute_tree_digest(
                    skill_repository / version.source.subpath
                )
            request.assert_called_once()
    assert inspect.call_args_list == [
        mock.call(roots[0] / "skills/a-review"),
        mock.call(roots[0] / "skills/nested/z-docs"),
    ]
    fetch.assert_called_once()
    assert all(not root.exists() for root in roots)


def test_discovery_prunes_walk_beneath_unselected_skills(skill_repository):
    child = skill_repository / "skills/a-review/child"
    child.mkdir()
    (child / "SKILL.md").write_text("---\nname: child\n---\n")
    visited = []
    original_walk = os.walk

    def walk(*args, **kwargs):
        for entry in original_walk(*args, **kwargs):
            visited.append(Path(entry[0]))
            yield entry

    fetched = FetchedContent(
        skill_repository, resolve_source_type("https://example.com/skills.git")
    )
    with mock.patch("mlflow.genai.skills.os.walk", side_effect=walk) as walk_mock:
        manifests = _filter_and_validate_skill_directories(fetched, {"docs"})
    walk_mock.assert_called_once_with(
        skill_repository, topdown=True, onerror=mock.ANY, followlinks=False
    )
    assert [manifest.name for manifest in manifests] == ["docs"]
    assert child not in visited
    assert skill_repository / "skills/a-review" in visited
    assert skill_repository / "skills/nested/z-docs" in visited


def test_import_can_target_nested_skill_with_discovery_subpath(
    registry_client, skill_repository, remote_repository
):
    subpath = "skills/a-review/child"
    child = skill_repository / subpath
    child.mkdir()
    (child / "SKILL.md").write_text("---\nname: child\n---\n")
    versions = import_skills(
        source=GitSource("https://example.com/skills.git", subpath=subpath), skill_names=["child"]
    )
    assert [version.name for version in versions] == ["child"]
    assert versions[0].source.subpath == subpath
    assert versions[0].digest == compute_tree_digest(child)


def test_import_does_not_skip_invalid_parent_manifest(
    registry_client, store, skill_repository, remote_repository
):
    (skill_repository / "SKILL.md").write_text("---\nname: INVALID\n---\n")
    with (
        mock.patch.object(store, "_skill_request") as request,
        pytest.raises(MlflowException, match="Invalid skill name"),
    ):
        import_skills(source="https://example.com/skills.git", skill_names=["review"])
    request.assert_not_called()


def test_import_stops_at_manifest_in_discovery_root(
    registry_client, skill_repository, remote_repository
):
    (skill_repository / "SKILL.md").write_text("---\nname: root-skill\n---\n")
    fetch, roots = remote_repository
    with mock.patch("mlflow.genai.skills.inspect_skill_dir", wraps=inspect_skill_dir) as inspect:
        versions = import_skills(source="https://example.com/skills.git")
    assert [version.name for version in versions] == ["root-skill"]
    assert versions[0].source.subpath is None
    assert versions[0].digest == compute_tree_digest(skill_repository)
    inspect.assert_called_once_with(roots[0])
    fetch.assert_called_once()
    assert not roots[0].exists()


@pytest.mark.parametrize("path", ["SKILL.md", "skills/a-review/SKILL.md"])
@pytest.mark.parametrize("skill_names", [None, ["docs"]])
def test_import_rejects_manifest_directory(
    registry_client, store, skill_repository, remote_repository, path, skill_names
):
    directory = skill_repository / path
    if directory.is_file():
        directory.unlink()
    directory.mkdir(parents=True)
    (directory / "notes.txt").write_text("Supporting content\n")
    fetch, roots = remote_repository
    with (
        mock.patch("mlflow.genai.skills.compute_tree_digest") as digest,
        mock.patch.object(store, "_skill_request") as request,
        pytest.raises(MlflowException, match="must be a file|does not contain a SKILL.md") as exc,
    ):
        import_skills(source="https://example.com/skills.git", skill_names=skill_names)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    digest.assert_not_called()
    request.assert_not_called()
    assert registry_client[1].search_skills() == []
    fetch.assert_called_once()
    assert all(not root.exists() for root in roots)


@pytest.mark.parametrize(
    ("count", "skill_names"),
    [
        (_MAX_BULK_REGISTER_SKILLS, None),
        (_MAX_BULK_REGISTER_SKILLS + 1, None),
        (_MAX_BULK_REGISTER_SKILLS + 1, ["review"]),
    ],
)
def test_import_batch_size_limit(skill_repository, remote_repository, count, skill_names):
    for index in range(count - 2):
        root = skill_repository / "skills" / f"skill-{index}"
        root.mkdir()
        (root / "SKILL.md").write_text(f"---\nname: skill-{index}\n---\n")
    fetch, roots = remote_repository
    selected_count = count if skill_names is None else len(skill_names)
    with (
        mock.patch("mlflow.genai.skills.MlflowClient") as client,
        mock.patch("mlflow.genai.skills.compute_tree_digest", wraps=compute_tree_digest) as digest,
    ):
        client.return_value.bulk_register_skills.return_value = []
        if selected_count > _MAX_BULK_REGISTER_SKILLS:
            with pytest.raises(
                MlflowException, match=f"at most {_MAX_BULK_REGISTER_SKILLS} skills"
            ) as exc:
                import_skills(source="https://example.com/skills.git", skill_names=skill_names)
            assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
            client.assert_not_called()
            digest.assert_not_called()
        else:
            assert (
                import_skills(source="https://example.com/skills.git", skill_names=skill_names)
                == []
            )
            client.assert_called_once_with()
            register = client.return_value.bulk_register_skills
            register.assert_called_once()
            assert len(register.call_args.kwargs["skill_definitions"]) == selected_count
            assert digest.call_count == selected_count
    fetch.assert_called_once()
    assert all(not root.exists() for root in roots)


@pytest.mark.parametrize("skill_size", [600, 1000, 1001])
def test_import_enforces_individual_skill_size_limit(
    registry_client, store, skill_repository, remote_repository, monkeypatch, skill_size
):
    _, db_store = registry_client
    limit = 1000
    monkeypatch.setenv(MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE.name, str(limit))
    for path, size in [("skills/a-review", 600), ("skills/nested/z-docs", skill_size)]:
        root = skill_repository / path
        (root / "payload.txt").write_bytes(b"x" * (size - tree_size(root)))
    assert tree_size(skill_repository) > limit
    fetch, roots = remote_repository

    with mock.patch.object(store, "_skill_request", wraps=store._skill_request) as request:
        if skill_size > limit:
            with pytest.raises(MlflowException, match=f"size limit of {limit} bytes") as exc:
                import_skills(source="https://example.com/skills.git")
            assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
            assert f"Skill 'docs' content is {skill_size} bytes" in exc.value.message
            request.assert_not_called()
            assert db_store.search_skills() == []
        else:
            versions = import_skills(source="https://example.com/skills.git")
            request.assert_called_once()
            assert [version.name for version in versions] == ["review", "docs"]
            assert versions == [
                db_store.get_skill_version(name=version.name, version=version.version)
                for version in versions
            ]

    fetch.assert_called_once()
    assert all(not root.exists() for root in roots)


@pytest.mark.parametrize("oversized_path", ["skills/nested/z-docs/payload.txt", "unrelated.txt"])
def test_import_applies_skill_size_limit_only_to_selected_content(
    registry_client, store, skill_repository, remote_repository, monkeypatch, oversized_path
):
    _, db_store = registry_client
    limit = 1000
    monkeypatch.setenv(MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE.name, str(limit))
    (skill_repository / oversized_path).write_bytes(b"x" * (limit + 1))
    fetch, roots = remote_repository

    with (
        mock.patch.object(store, "_skill_request", wraps=store._skill_request) as request,
        mock.patch("mlflow.genai.skills.compute_tree_digest", wraps=compute_tree_digest) as digest,
    ):
        versions = import_skills(source="https://example.com/skills.git", skill_names=["review"])

    request.assert_called_once()
    digest.assert_called_once_with(roots[0] / "skills/a-review")
    assert versions == [db_store.get_skill_version(name="review", version=1)]
    assert [skill.name for skill in db_store.search_skills()] == ["review"]
    fetch.assert_called_once()
    assert all(not root.exists() for root in roots)


@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_import_enforces_discovery_size_limit(
    registry_client, store, skill_repository, remote_repository, monkeypatch, extra_bytes
):
    _, db_store = registry_client
    limit = 1000
    monkeypatch.setenv(MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE.name, str(limit))
    budget = limit * _MAX_BULK_REGISTER_SKILLS
    (skill_repository / "unrelated.txt").write_bytes(
        b"x" * (budget + extra_bytes - tree_size(skill_repository))
    )
    fetch, roots = remote_repository

    with (
        mock.patch.object(store, "_skill_request", wraps=store._skill_request) as request,
        mock.patch("mlflow.genai.skills.compute_tree_digest", wraps=compute_tree_digest) as digest,
    ):
        if extra_bytes:
            with pytest.raises(MlflowException, match=f"size limit of {budget} bytes") as exc:
                import_skills(source="https://example.com/skills.git", skill_names=["review"])
            assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
            request.assert_not_called()
            digest.assert_not_called()
            assert db_store.search_skills() == []
        else:
            versions = import_skills(
                source="https://example.com/skills.git", skill_names=["review"]
            )
            request.assert_called_once()
            digest.assert_called_once_with(roots[0] / "skills/a-review")
            assert versions == [db_store.get_skill_version(name="review", version=1)]

    fetch.assert_called_once()
    assert all(not root.exists() for root in roots)


def test_import_fetches_requested_git_ref(registry_client, skill_repository):
    for args in [
        ["init", "-q", "-b", "main"],
        ["add", "."],
        ["commit", "-q", "-m", "Initial skills"],
        ["tag", "v1"],
    ]:
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
            cwd=skill_repository,
            check=True,
            capture_output=True,
        )
    digest = compute_tree_digest(skill_repository / "skills/a-review")
    (skill_repository / "skills/a-review/SKILL.md").write_text("Uncommitted invalid manifest")
    source = GitSource(skill_repository.as_uri(), ref="v1", subpath="skills")
    versions = import_skills(source=source, skill_names=["review"])
    assert len(versions) == 1
    assert versions[0].digest == digest
    assert versions[0].source == GitSource(source.url, ref="v1", subpath="skills/a-review")


@pytest.mark.parametrize(
    "source",
    [
        None,
        "./skills",
        "https://example.com/skills",
        "https://example.com/skills.zip",
        OCISource("example.com/skills:v1"),
        "mlflow-artifacts:/skills",
    ],
)
def test_import_rejects_invalid_source_before_fetch(source):
    with (
        mock.patch("mlflow.genai.skills.fetch_source") as fetch,
        pytest.raises(MlflowException, match="source|Cannot infer") as exc,
    ):
        import_skills(source=source)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    fetch.assert_not_called()


@pytest.mark.parametrize(
    "metadata",
    [
        {"organization": "bad/org"},
        {"status": "deprecated"},
        {"skill_names": []},
        {"skill_names": "review"},
        {"skill_names": [123]},
        {"skill_names": ["BAD NAME"]},
    ],
)
def test_import_rejects_invalid_metadata_before_fetch(metadata):
    with (
        mock.patch("mlflow.genai.skills.fetch_source") as fetch,
        pytest.raises(MlflowException, match="name|imported") as exc,
    ):
        import_skills(source="https://example.com/skills.git", **metadata)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    fetch.assert_not_called()


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("duplicate", "Duplicate discovered"),
        ("missing", "not found"),
        ("directory-name", "not found"),
        ("empty", "No skills"),
        ("invalid", "name"),
        ("digest", "Unreadable content"),
    ],
)
def test_import_prepares_entire_batch_before_request(
    registry_client, store, skill_repository, remote_repository, failure, message
):
    _, db_store = registry_client
    fetch, roots = remote_repository
    names = None
    if failure == "duplicate":
        (skill_repository / "skills/nested/z-docs/SKILL.md").write_text("---\nname: review\n---\n")
        names = ["review"]
    elif failure == "missing":
        names = ["review", "missing"]
    elif failure == "directory-name":
        names = ["a-review"]
    elif failure == "empty":
        shutil.rmtree(skill_repository / "skills")
    elif failure == "invalid":
        (skill_repository / "skills/nested/z-docs/SKILL.md").write_text("No declared name")

    def digest(root):
        if failure == "digest" and root.name == "z-docs":
            raise MlflowException.invalid_parameter_value("Unreadable content")
        return compute_tree_digest(root)

    with (
        mock.patch("mlflow.genai.skills.compute_tree_digest", side_effect=digest) as compute_digest,
        mock.patch.object(store, "_skill_request") as request,
        pytest.raises(MlflowException, match=message) as exc,
    ):
        import_skills(source="https://example.com/skills.git", skill_names=names)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    fetch.assert_called_once()
    request.assert_not_called()
    if failure != "digest":
        compute_digest.assert_not_called()
    else:
        assert compute_digest.call_args_list == [
            mock.call(roots[0] / "skills/a-review"),
            mock.call(roots[0] / "skills/nested/z-docs"),
        ]
    assert db_store.search_skills() == []
    assert all(not root.exists() for root in roots)


def test_import_preserves_reused_version_status(registry_client, remote_repository):
    _, db_store = registry_client
    source = "https://example.com/skills.git"
    first = import_skills(source=source, skill_names=["review"], status="draft")[0]
    db_store.set_skill_version_tag(name="review", version=first.version, key="scan", value="clean")
    expected = db_store.get_skill_version(name="review", version=first.version)
    versions = import_skills(source=source)
    assert versions[0] == expected
    assert versions[0].status == "draft"
    assert versions[1].name == "docs"
    assert versions[1].status == "active"
    assert len(db_store.search_skill_versions(name="review")) == 1


def test_import_cleans_up_and_preserves_server_error(registry_client, store, remote_repository):
    fetch, roots = remote_repository
    with (
        mock.patch.object(
            store,
            "_skill_request",
            side_effect=MlflowException("Not allowed", error_code=PERMISSION_DENIED),
        ) as request,
        pytest.raises(MlflowException, match="Not allowed") as exc,
    ):
        import_skills(source="https://example.com/skills.git")
    assert exc.value.error_code == "PERMISSION_DENIED"
    request.assert_called_once()
    fetch.assert_called_once()
    assert all(not root.exists() for root in roots)


@pytest.fixture
def skill_tree(tmp_path):
    root = tmp_path / "skill"
    root.mkdir()
    (root / "SKILL.md").write_text("---\nname: review\ndescription: Reviews code\n---\n# Review\n")
    (root / "scripts").mkdir()
    (root / "scripts" / "run.py").write_text("print('review')\n")
    return root


@pytest.fixture
def skill_archive_path(skill_tree, tmp_path):
    return package_skill_tree(skill_tree, tmp_path / "content.tar.gz")


@pytest.fixture
def remote_content(skill_tree):
    roots = []

    def fetch(resolved, dest, scratch, limit):
        roots.append(dest)
        shutil.copytree(skill_tree, dest / (resolved.subpath or ""))
        return dest

    with mock.patch("mlflow.genai.skill_content.fetchers._fetch_remote", side_effect=fetch) as call:
        yield call, roots


@pytest.fixture
def skill_artifacts(tmp_path, monkeypatch):
    root = tmp_path / "skill-artifacts"
    root.mkdir()
    monkeypatch.setenv(SERVE_ARTIFACTS_ENV_VAR, "true")
    monkeypatch.setenv(ARTIFACTS_DESTINATION_ENV_VAR, str(root))
    monkeypatch.setattr(handlers, "_artifact_repo", None)
    return root


@pytest.mark.parametrize("name", [None, "custom-review"])
@pytest.mark.parametrize(
    "source",
    [
        GitSource("https://example.com/repo", ref="v2", subpath="skills/review"),
        OCISource("oci://ghcr.io/acme/skills:v1", subpath="skills/review"),
        ZipSource("https://example.com/archive", subpath="skills/review"),
        "https://example.com/repo.git",
        "oci://ghcr.io/acme/skills:v1",
        "https://example.com/skills.zip",
    ],
)
def test_register_remote_skill(registry_client, store, skill_tree, remote_content, source, name):
    client, db_store = registry_client
    fetch, roots = remote_content
    resolved = resolve_source_type(source)
    with (
        mock.patch("mlflow.genai.skills.MlflowClient", return_value=client) as client_factory,
        mock.patch.object(
            client, "create_skill_version", wraps=client.create_skill_version
        ) as create,
        mock.patch.object(store, "_skill_request", wraps=store._skill_request) as request,
    ):
        version = register_skill(source=source, name=name, organization="acme", status="draft")
    client_factory.assert_called_once_with()
    create.assert_called_once_with(
        name=name or "review",
        organization="acme",
        source=source,
        digest=compute_tree_digest(skill_tree),
        status="draft",
    )
    fetch.assert_called_once()
    assert fetch.call_args.args[0] == resolved
    request.assert_called_once_with(
        "POST",
        f"/@acme/{name or 'review'}/versions",
        json={
            "digest": compute_tree_digest(skill_tree),
            "status": "draft",
            "source_type": resolved.source_type.value,
            "source": resolved.source,
            "ref": resolved.ref,
            "subpath": resolved.subpath,
        },
    )
    assert version == db_store.get_skill_version(name or "review", 1, organization="acme")
    assert version.status == SkillStatus.DRAFT
    parent = client.get_skill(name=version.name, organization="acme")
    assert parent.description is None
    assert parent.icons is None
    assert all(not root.exists() for root in roots)


@pytest.mark.parametrize(
    "source",
    [
        GitSource("https://example.com/repo", ref="v2", subpath="skills/review"),
        OCISource("oci://ghcr.io/acme/skills:v1", subpath="skills/review"),
        ZipSource("https://example.com/archive", subpath="skills/review"),
    ],
    ids=["git", "oci", "zip"],
)
@pytest.mark.parametrize(
    "high_level", [True, False], ids=["register_skill", "create_skill_version"]
)
def test_remote_registration_defaults_to_active(
    registry_client, remote_content, source, high_level
):
    client, db_store = registry_client
    fetch, _ = remote_content
    if high_level:
        version = register_skill(source=source)
        fetch.assert_called_once()
    else:
        version = client.create_skill_version(name="review", source=source)
        fetch.assert_not_called()

    assert version.status == SkillStatus.ACTIVE
    assert db_store.get_skill_version("review", version.version).status == SkillStatus.ACTIVE
    assert client.get_latest_skill_version(name="review") == version


@pytest.mark.parametrize("name", [None, "custom-review"])
@pytest.mark.parametrize("organization", ["", "acme"])
@pytest.mark.parametrize("high_level", [True, False])
def test_register_local_skill(
    registry_client,
    store,
    skill_tree,
    skill_archive_path,
    skill_artifacts,
    name,
    organization,
    high_level,
):
    client, db_store = registry_client
    with (
        mock.patch.object(store, "_skill_request", wraps=store._skill_request) as request,
        mock.patch("mlflow.genai.skills.inspect_skill_dir", wraps=inspect_skill_dir) as inspect,
        mock.patch(
            "mlflow.genai.skills.compute_tree_digest", wraps=compute_tree_digest
        ) as sdk_digest,
        mock.patch(
            "mlflow.genai.skill_content.digest.compute_tree_digest", wraps=compute_tree_digest
        ) as digest,
    ):
        if high_level:
            version = register_skill(source=str(skill_tree), name=name, organization=organization)
        else:
            version = client.create_skill_version(
                source=str(skill_archive_path),
                name=name or "review",
                organization=organization,
                digest=compute_tree_digest(skill_tree),
            )
    if high_level:
        sdk_digest.assert_called_once()
        snapshot = sdk_digest.call_args.args[0]
        assert snapshot != skill_tree
        assert not snapshot.parent.exists()
        inspect.assert_called_once_with(snapshot)
    else:
        inspect.assert_not_called()
        sdk_digest.assert_not_called()
    digest.assert_not_called()
    request.assert_called_once()
    assert request.call_args.args == ("POST", "/register")
    files = request.call_args.kwargs["files"]
    assert json.loads(files["metadata"][1]) == {
        "name": name or "review",
        "organization": organization,
        "digest": compute_tree_digest(skill_tree),
        "status": "active",
    }
    content = files["content"][1]
    assert content.closed
    if high_level:
        assert not Path(content.name).parent.exists()
    else:
        assert Path(content.name) == skill_archive_path
        assert skill_archive_path.exists()
    assert version == db_store.get_skill_version(name or "review", 1, organization=organization)
    assert version.source_type == SkillSourceType.MLFLOW
    assert version.status == SkillStatus.ACTIVE
    stored = skill_artifacts / version.source.artifact_path.removeprefix("mlflow-artifacts:/")
    assert compute_tree_digest(stored) == version.digest
    assert (stored / "SKILL.md").read_bytes() == (skill_tree / "SKILL.md").read_bytes()
    assert version.source.subpath is None
    assert skill_tree.exists()
    parent = client.get_skill(name=version.name, organization=organization)
    assert parent.description is None
    assert parent.icons is None


def test_registration_preserves_parent_metadata(registry_client, skill_tree, skill_artifacts):
    client, _ = registry_client
    icons = [{"src": "https://example.com/icon.png"}]
    client.create_skill(name="review", description="Curated description", icons=icons)
    client.set_skill_tag(name="review", key="team", value="platform")
    first = register_skill(source=str(skill_tree), status="draft")
    second = register_skill(source=str(skill_tree))
    assert (first.version, second.version) == (1, 2)
    assert first.source != second.source
    parent = client.get_skill(name="review")
    assert parent.description == "Curated description"
    assert parent.icons == icons
    assert parent.tags == {"team": "platform"}


@pytest.mark.parametrize(
    "source",
    [
        None,
        "",
        "https://example.com/ambiguous",
        MlflowSource("mlflow-artifacts:/skills/review"),
        "mlflow-artifacts:/skills/review",
    ],
)
def test_register_rejects_invalid_source_before_fetch(source):
    with (
        mock.patch("mlflow.genai.skills.fetch_source") as fetch,
        pytest.raises(MlflowException, match="source|Source|Cannot infer") as exc,
    ):
        register_skill(source=source)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    fetch.assert_not_called()


def test_register_skill_source_is_required():
    with pytest.raises(TypeError, match="source"):
        register_skill()


@pytest.mark.parametrize(
    "metadata", [{"name": ""}, {"organization": "bad/org"}, {"status": "deleted"}]
)
def test_register_rejects_invalid_metadata_before_fetch(metadata):
    with (
        mock.patch("mlflow.genai.skills.fetch_source") as fetch,
        pytest.raises(MlflowException, match="name|status|registered") as exc,
    ):
        register_skill(source="https://example.com/repo.git", **metadata)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    fetch.assert_not_called()


@pytest.mark.parametrize(
    "manifest", ["# No name", "---\nname: BAD NAME\n---\n", "---\nname: [review]\n---\n"]
)
def test_register_explicit_name_still_validates_content(
    registry_client, store, skill_tree, remote_content, manifest
):
    fetch, roots = remote_content
    (skill_tree / "SKILL.md").write_text(manifest)
    with (
        mock.patch.object(store, "_skill_request") as request,
        pytest.raises(MlflowException, match="name"),
    ):
        register_skill(source="https://example.com/repo.git", name="custom-review")
    fetch.assert_called_once()
    request.assert_not_called()
    assert all(not root.exists() for root in roots)


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("name", [None, "custom-review"])
def test_register_rejects_root_manifest_directory_before_digest(
    registry_client, store, skill_tree, remote_content, local, name
):
    invalid = skill_tree / "SKILL.md"
    invalid.unlink()
    invalid.mkdir()
    (invalid / "notes.txt").write_text("Supporting content\n")
    fetch, roots = remote_content
    with (
        mock.patch("mlflow.genai.skills.compute_tree_digest") as digest,
        mock.patch.object(store, "_skill_request") as request,
        pytest.raises(MlflowException, match="does not contain a SKILL.md") as exc,
    ):
        register_skill(
            source=str(skill_tree) if local else "https://example.com/repo.git", name=name
        )
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    digest.assert_not_called()
    request.assert_not_called()
    assert registry_client[1].search_skills() == []
    assert fetch.call_count == (0 if local else 1)
    assert all(not root.exists() for root in roots)
    assert invalid.exists()


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("name", [None, "custom-review"])
@pytest.mark.parametrize("kind", ["manifest", "invalid-manifest", "directory"])
def test_register_includes_nested_manifest_content(
    registry_client, skill_tree, remote_content, skill_artifacts, local, name, kind
):
    nested = skill_tree / "examples/SKILL.md"
    nested.parent.mkdir()
    if kind == "directory":
        nested.mkdir()
        nested = nested / "notes.txt"
    content = "---\nname: child\n---\n" if kind == "manifest" else "Supporting content\n"
    nested.write_text(content)
    version = register_skill(
        source=str(skill_tree) if local else "https://example.com/repo.git", name=name
    )
    assert version.name == (name or "review")
    assert version.digest == compute_tree_digest(skill_tree)
    if local:
        stored = skill_artifacts / version.source.artifact_path.removeprefix("mlflow-artifacts:/")
        assert (stored / nested.relative_to(skill_tree)).read_text() == content
        assert compute_tree_digest(stored) == version.digest
    assert registry_client[1].get_skill_version(version.name, version.version) == version


@pytest.mark.parametrize("local", [False, True])
def test_register_cleans_up_after_request_failure(
    registry_client, store, skill_tree, remote_content, local
):
    fetch, roots = remote_content
    with (
        mock.patch.object(
            store, "_skill_request", side_effect=MlflowException("conflict")
        ) as request,
        pytest.raises(MlflowException, match="conflict"),
    ):
        register_skill(source=str(skill_tree) if local else "https://example.com/repo.git")
    request.assert_called_once()
    if local:
        fetch.assert_not_called()
        content = request.call_args.kwargs["files"]["content"][1]
        assert content.closed
        assert not Path(content.name).parent.exists()
    else:
        fetch.assert_called_once()
        assert all(not root.exists() for root in roots)
    assert skill_tree.exists()


def test_register_local_propagates_artifact_capability_error(
    registry_client, skill_tree, monkeypatch
):
    monkeypatch.setenv(SERVE_ARTIFACTS_ENV_VAR, "false")
    with pytest.raises(MlflowException, match="does not serve artifacts") as exc:
        register_skill(source=str(skill_tree))
    assert exc.value.error_code == "NOT_IMPLEMENTED"
    assert registry_client[1].search_skills() == []


def test_register_with_direct_sql_store(registry_client, skill_tree, remote_content):
    client, db_store = registry_client
    fetch, roots = remote_content
    with mock.patch(
        "mlflow.tracking._tracking_service.utils._get_store", return_value=db_store
    ) as get_store:
        version = register_skill(source="https://example.com/repo.git")
        assert version == db_store.get_skill_version("review", 1)
        with pytest.raises(MlflowException, match="HTTP tracking server") as exc:
            register_skill(source=str(skill_tree))
    assert exc.value.error_code == "NOT_IMPLEMENTED"
    get_store.assert_called_with(client.tracking_uri)
    fetch.assert_called_once()
    assert all(not root.exists() for root in roots)


@pytest.mark.parametrize("workspace", [None, "team-a"])
def test_register_multipart_preserves_transport_context(store, workspace):
    response = Response()
    response.status_code = 200
    response._content = b'{"name": "review", "version": 1, "status": "active"}'
    with (
        WorkspaceContext(workspace),
        mock.patch.object(store, "_probe_workspace_support", return_value=True) as probe,
        mock.patch(
            "mlflow.utils.rest_utils._get_http_response_with_retries", return_value=response
        ) as request,
        io.BytesIO(b"archive") as content,
    ):
        store._register_skill(name="review", content=content)
        request.assert_called_once()
        assert request.call_args.args[:2] == (
            "POST",
            "https://registry.example.com/api/3.0/mlflow/skills/register",
        )
        kwargs = request.call_args.kwargs
        assert kwargs["headers"]["Authorization"] == "Bearer test-token"
        assert kwargs["headers"].get(WORKSPACE_HEADER_NAME) == workspace
        assert kwargs["json"] is None
        assert kwargs["files"]["metadata"][0] is None
        assert kwargs["files"]["metadata"][2] == "application/json"
        assert json.loads(kwargs["files"]["metadata"][1]) == {
            "name": "review",
            "organization": "",
            "digest": None,
            "status": "active",
        }
        assert kwargs["files"]["content"] == ("content.tar.gz", content, "application/gzip")
    if workspace:
        probe.assert_called_once_with()
    else:
        probe.assert_not_called()


@pytest.mark.parametrize("edit_at", ["before-packaging", "after-hashing"])
def test_register_local_skill_digest_matches_uploaded_snapshot(
    registry_client, skill_tree, skill_artifacts, edit_at
):
    manifest_path = skill_tree / "SKILL.md"
    original = manifest_path.read_bytes()
    edited = original.replace(b"name: review", b"name: edited")
    assert len(original) == len(edited)

    def package(root, output):
        if edit_at == "before-packaging":
            manifest_path.write_bytes(edited)
        return package_skill_tree(root, output)

    def hash_snapshot(root):
        result = compute_tree_digest(root)
        if edit_at == "after-hashing":
            manifest_path.write_bytes(edited)
        return result

    with (
        mock.patch("mlflow.genai.skills.package_skill_tree", side_effect=package) as package_mock,
        mock.patch("mlflow.genai.skills.compute_tree_digest", side_effect=hash_snapshot) as digest,
    ):
        version = register_skill(source=str(skill_tree))

    package_mock.assert_called_once()
    assert package_mock.call_args.args[0] == skill_tree
    digest.assert_called_once()
    snapshot = digest.call_args.args[0]
    assert snapshot != skill_tree
    assert not snapshot.parent.exists()
    assert not Path(package_mock.call_args.args[1]).exists()
    stored = skill_artifacts / version.source.artifact_path.removeprefix("mlflow-artifacts:/")
    assert compute_tree_digest(stored) == version.digest
    assert version.name == ("edited" if edit_at == "before-packaging" else "review")
    assert (stored / "SKILL.md").read_bytes() == (
        edited if edit_at == "before-packaging" else original
    )
    assert manifest_path.read_bytes() == edited
    if edit_at == "after-hashing":
        assert compute_tree_digest(skill_tree) != version.digest
    assert registry_client[1].get_skill_version(version.name, version.version) == version


@pytest.mark.parametrize(
    ("change", "message"),
    [("oversized", "exceeds the skill content size limit"), ("manifest", "Invalid skill name")],
)
def test_register_revalidates_local_snapshot(
    registry_client, store, skill_tree, monkeypatch, change, message
):
    monkeypatch.setenv(MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE.name, "1024")

    def package(root, output):
        if change == "oversized":
            (skill_tree / "payload.txt").write_bytes(b"x" * 1025)
        else:
            (skill_tree / "SKILL.md").write_text("---\nname: INVALID\n---\n")
        return package_skill_tree(root, output)

    with (
        mock.patch("mlflow.genai.skills.package_skill_tree", side_effect=package) as package_mock,
        mock.patch("mlflow.genai.skills.compute_tree_digest") as digest,
        mock.patch.object(store, "_skill_request") as request,
        pytest.raises(MlflowException, match=message) as exc,
    ):
        register_skill(source=str(skill_tree))

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    package_mock.assert_called_once()
    assert not Path(package_mock.call_args.args[1]).parent.exists()
    digest.assert_not_called()
    request.assert_not_called()
    assert skill_tree.exists()
    assert registry_client[1].search_skills() == []


def test_register_cleans_up_after_packaging_failure(registry_client, store, skill_tree):
    with (
        mock.patch(
            "mlflow.genai.skills.package_skill_tree",
            side_effect=OSError("disk full"),
        ) as package,
        mock.patch.object(store, "_skill_request") as request,
        pytest.raises(OSError, match="disk full"),
    ):
        register_skill(source=str(skill_tree))
    package.assert_called_once()
    assert not Path(package.call_args.args[1]).parent.exists()
    request.assert_not_called()
    assert skill_tree.exists()


@pytest.mark.parametrize("supplied", [{}, {"digest": None}, {"digest": "a" * 64}])
def test_create_local_skill_version_preserves_optional_digest(
    registry_client, skill_archive_path, skill_artifacts, supplied
):
    client, db_store = registry_client
    with mock.patch("mlflow.genai.skill_content.digest.compute_tree_digest") as digest:
        version = client.create_skill_version(
            name="custom-review", source=str(skill_archive_path), status="draft", **supplied
        )
    digest.assert_not_called()
    assert version.digest == supplied.get("digest")
    assert version.status == SkillStatus.DRAFT
    assert db_store.get_skill_version("custom-review", version.version).digest == supplied.get(
        "digest"
    )


@pytest.mark.parametrize("directory", [True, False])
def test_create_local_skill_version_requires_archive_file(
    registry_client, store, tmp_path, directory
):
    client, db_store = registry_client
    with (
        mock.patch.object(store, "_skill_request") as request,
        pytest.raises(MlflowException, match="gzip-compressed tar archive file") as exc,
    ):
        client.create_skill_version(
            name="review", source=str(tmp_path if directory else tmp_path / "missing")
        )
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    request.assert_not_called()
    assert db_store.search_skills() == []


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("corrupt", "not a readable tar archive"),
        ("uncompressed", "not a readable tar archive"),
        ("traversal", "unsafe path"),
        ("symlink", "not a regular file or directory"),
    ],
)
def test_create_local_skill_version_rejects_invalid_archive(
    registry_client, store, tmp_path, kind, message
):
    client, db_store = registry_client
    archive = tmp_path / "content.tar.gz"
    if kind == "corrupt":
        archive.write_bytes(b"not an archive")
    else:
        with tarfile.open(archive, "w" if kind == "uncompressed" else "w:gz") as tar:
            member = tarfile.TarInfo("../escape" if kind == "traversal" else "SKILL.md")
            if kind == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = "target"
            tar.addfile(member)
    with (
        mock.patch.object(store, "_skill_request") as request,
        pytest.raises(MlflowException, match=message) as exc,
    ):
        client.create_skill_version(name="review", source=str(archive))
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    request.assert_not_called()
    assert db_store.search_skills() == []
    assert archive.exists()


def test_create_local_skill_version_enforces_archive_size_limit(
    registry_client, store, skill_archive_path, monkeypatch
):
    client, db_store = registry_client
    monkeypatch.setenv(MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE.name, "1")
    with (
        mock.patch.object(store, "_skill_request") as request,
        pytest.raises(MlflowException, match="skill content size limit") as exc,
    ):
        client.create_skill_version(name="review", source=str(skill_archive_path))
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    request.assert_not_called()
    assert db_store.search_skills() == []
    assert skill_archive_path.exists()


@pytest.mark.parametrize("manifest", ["# No name", "---\nname: BAD NAME\n---\n"])
def test_local_registration_validates_manifest_with_explicit_name(
    registry_client, store, skill_tree, manifest
):
    _, db_store = registry_client
    (skill_tree / "SKILL.md").write_text(manifest)
    with (
        mock.patch.object(store, "_skill_request") as request,
        pytest.raises(MlflowException, match="name"),
    ):
        register_skill(source=str(skill_tree), name="custom-review")
    request.assert_not_called()
    assert db_store.search_skills() == []

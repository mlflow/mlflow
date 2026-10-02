import asyncio
import io
import json
from pathlib import Path
from unittest import mock

import pytest
from fastapi import FastAPI, HTTPException
from starlette.requests import Request
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
from mlflow.exceptions import MlflowException
from mlflow.server import skill_registry_api
from mlflow.server.fastapi_app import add_registry_exception_handlers
from mlflow.server.skill_registry_api import (
    _MAX_BULK_REGISTER_SKILLS,
    _SKILL_NAME_PATH_PATTERN,
    get_skill_registry_api_route_prefixes,
    is_skill_registry_api_path,
    skill_registry_router,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking.skill_registry.abstract_mixin import NOT_SET
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.validation import _MAX_REGISTRY_ICONS_PER_LIST

PREFIX = "/ajax-api/3.0/mlflow/skills"


@pytest.fixture
def mock_icon_hostname_resolution():
    with mock.patch(
        "mlflow.utils.validation._resolve_hostname_with_timeout",
        return_value=[(None, None, None, None, ("8.8.8.8", 0))],
    ):
        yield


EXPECTED_ROUTE_METHODS = {
    "": {"get", "post"},
    "/register": {"post"},
    "/bulk-register": {"post"},
    "/{name}": {"get", "patch", "delete"},
    "/@{organization}/{name}": {"get", "patch", "delete"},
    "/{name}/versions": {"get", "post"},
    "/@{organization}/{name}/versions": {"get", "post"},
    "/{name}/versions/{version}": {"get", "patch", "delete"},
    "/@{organization}/{name}/versions/{version}": {"get", "patch", "delete"},
    "/{name}/tags": {"post"},
    "/@{organization}/{name}/tags": {"post"},
    "/{name}/tags/{key}": {"delete"},
    "/@{organization}/{name}/tags/{key}": {"delete"},
    "/{name}/versions/{version}/tags": {"post"},
    "/@{organization}/{name}/versions/{version}/tags": {"post"},
    "/{name}/versions/{version}/tags/{key}": {"delete"},
    "/@{organization}/{name}/versions/{version}/tags/{key}": {"delete"},
    "/{name}/aliases": {"post"},
    "/@{organization}/{name}/aliases": {"post"},
    "/{name}/aliases/{alias}": {"get", "delete"},
    "/@{organization}/{name}/aliases/{alias}": {"get", "delete"},
}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            GitSource(url="https://github.com/acme/skills.git", ref="main", subpath="reviewer"),
            ("https://github.com/acme/skills.git", "main", "reviewer"),
        ),
        (
            OCISource(image="ghcr.io/acme/skills:latest", subpath="reviewer"),
            ("ghcr.io/acme/skills:latest", None, "reviewer"),
        ),
        (
            ZipSource(url="https://example.com/skills.zip", subpath="reviewer"),
            ("https://example.com/skills.zip", None, "reviewer"),
        ),
        (
            MlflowSource(artifact_path="mlflow-artifacts:/skills/reviewer"),
            ("mlflow-artifacts:/skills/reviewer", None, None),
        ),
        ("assembled://reviewer", ("assembled://reviewer", None, None)),
        (None, (None, None, None)),
    ],
)
def test_skill_source_response_fields(source, expected):
    assert skill_registry_api._skill_source_response_fields(source) == expected


def _create_registry_fastapi_app() -> FastAPI:
    fastapi_app = FastAPI()
    add_registry_exception_handlers(fastapi_app)
    for route_prefix in get_skill_registry_api_route_prefixes():
        fastapi_app.include_router(skill_registry_router, prefix=route_prefix)
    return fastapi_app


def _create_client(tmp_path: Path, db_uri: str) -> tuple[TestClient, SqlAlchemyStore]:
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir()
    store = SqlAlchemyStore(db_uri, artifact_path.as_uri())
    return TestClient(_create_registry_fastapi_app()), store


def test_skill_registry_route_prefixes_and_path_detection():
    prefixes = get_skill_registry_api_route_prefixes()

    assert prefixes == (
        "/ajax-api/3.0/mlflow/skills",
        "/api/3.0/mlflow/skills",
    )
    assert is_skill_registry_api_path("/ajax-api/3.0/mlflow/skills")
    assert is_skill_registry_api_path("/api/3.0/mlflow/skills/code-review")
    assert not is_skill_registry_api_path("/ajax-api/3.0/mlflow/assistant/skills/install")


def test_skill_name_path_parameters_reserve_leading_at_sign(tmp_path: Path, db_uri: str):
    client, _ = _create_client(tmp_path, db_uri)
    schema = client.get("/openapi.json").json()

    name_parameters = []
    for path_item in schema["paths"].values():
        for operation in path_item.values():
            if isinstance(operation, dict):
                name_parameters.extend(
                    parameter
                    for parameter in operation.get("parameters", [])
                    if parameter["name"] == "name"
                )

    assert name_parameters
    assert {parameter["schema"]["pattern"] for parameter in name_parameters} == {
        _SKILL_NAME_PATH_PATTERN
    }


def test_owned_routes_are_registered_under_each_prefix(tmp_path: Path, db_uri: str):
    client, _ = _create_client(tmp_path, db_uri)
    schema = client.get("/openapi.json").json()

    for prefix in get_skill_registry_api_route_prefixes():
        for suffix, methods in EXPECTED_ROUTE_METHODS.items():
            route = f"{prefix}{suffix}"
            assert route in schema["paths"], route
            assert methods <= set(schema["paths"][route]), route


def test_version_creation_documents_json_and_multipart_bodies(tmp_path: Path, db_uri: str):
    client, _ = _create_client(tmp_path, db_uri)
    schema = client.get("/openapi.json").json()

    for suffix in (
        "/{name}/versions",
        "/@{organization}/{name}/versions",
        "/register",
    ):
        for prefix in get_skill_registry_api_route_prefixes():
            request_body = schema["paths"][f"{prefix}{suffix}"]["post"]["requestBody"]
            assert request_body["required"] is True
            assert set(request_body["content"]) == {
                "application/json",
                "multipart/form-data",
            }
            json_schema = request_body["content"]["application/json"]["schema"]
            if suffix == "/register":
                assert json_schema["required"] == ["name"]
                assert json_schema["properties"]["name"] == {
                    "type": "string",
                    "title": "Name",
                }
            multipart_schema = request_body["content"]["multipart/form-data"]["schema"]
            assert set(multipart_schema["required"]) == {"metadata", "content"}


@pytest.mark.parametrize("prefix", get_skill_registry_api_route_prefixes())
def test_skill_routes_are_available_under_each_prefix(prefix: str, tmp_path: Path, db_uri: str):
    client, store = _create_client(tmp_path, db_uri)
    store.create_skill("versions")
    store.create_skill("versions", organization="acme")

    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.get(f"{prefix}/versions")
        assert response.status_code == 200, response.text
        assert response.json()["name"] == "versions"
        assert response.json()["organization"] == ""

        response = client.get(f"{prefix}/@acme/versions")
        assert response.status_code == 200, response.text
        assert response.json()["name"] == "versions"
        assert response.json()["organization"] == "acme"


def test_invalid_skill_name_is_rejected_before_store_call(tmp_path: Path, db_uri: str):
    client, store = _create_client(tmp_path, db_uri)
    with mock.patch(
        "mlflow.server.handlers._get_tracking_store", return_value=store
    ) as get_tracking_store:
        response = client.post(PREFIX, json={"name": "invalid_name"})

    assert response.status_code == 400, response.text
    assert "Invalid skill name" in response.json()["message"]
    get_tracking_store.assert_not_called()


def test_leading_at_sign_is_rejected_for_organization_skill_name(tmp_path: Path, db_uri: str):
    client, _ = _create_client(tmp_path, db_uri)
    response = client.get(f"{PREFIX}/@acme/@code-review")

    assert response.status_code == 400, response.text
    assert "should match pattern" in response.json()["message"]


def test_malformed_organization_version_route_is_rejected_without_store_call(
    tmp_path: Path, db_uri: str
):
    client, _ = _create_client(tmp_path, db_uri)
    with mock.patch("mlflow.server.handlers._get_tracking_store") as get_tracking_store:
        response = client.post(f"{PREFIX}/@acme/versions", json={})

    assert response.status_code == 400, response.text
    assert "should match pattern" in response.json()["message"]
    get_tracking_store.assert_not_called()


def test_invalid_skill_version_is_rejected_before_store_call(tmp_path: Path, db_uri: str):
    client, store = _create_client(tmp_path, db_uri)
    with mock.patch(
        "mlflow.server.handlers._get_tracking_store", return_value=store
    ) as get_tracking_store:
        response = client.get(f"{PREFIX}/code-review/versions/0")

    assert response.status_code == 400, response.text
    assert "Skill version must be a positive integer" in response.json()["message"]
    get_tracking_store.assert_not_called()


def test_register_requires_an_explicit_skill_name(tmp_path: Path, db_uri: str):
    client, _ = _create_client(tmp_path, db_uri)
    with mock.patch("mlflow.server.skill_registry_api.register_skill_version") as register:
        response = client.post(
            f"{PREFIX}/register",
            json={
                "source_type": "git",
                "source": "https://github.com/acme/skills.git",
            },
        )

    assert response.status_code == 400, response.text
    assert "'name' must be provided explicitly" in response.json()["message"]
    register.assert_not_called()


def test_register_rejects_unknown_request_fields(tmp_path: Path, db_uri: str):
    client, _ = _create_client(tmp_path, db_uri)
    with mock.patch("mlflow.server.skill_registry_api.register_skill_version") as register:
        response = client.post(
            f"{PREFIX}/register",
            json={
                "name": "code-review",
                "source_type": "git",
                "source": "https://github.com/acme/skills.git",
                "sub_path": "skills/code-review",
            },
        )

    assert response.status_code == 400, response.text
    assert "sub_path" in response.json()["message"]
    assert "extra" in response.json()["message"]
    register.assert_not_called()


def test_create_and_get_skill_without_organization(
    tmp_path: Path, db_uri: str, mock_icon_hostname_resolution
):
    client, store = _create_client(tmp_path, db_uri)
    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.post(
            PREFIX,
            json={
                "name": "code-review",
                "description": "Reviews pull requests",
                "icons": [
                    {
                        "src": "https://example.com/icon.svg",
                        "mimeType": " IMAGE/SVG+XML ",
                        "theme": "light",
                        "futureField": "preserved",
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["name"] == "code-review"
        assert response.json()["organization"] == ""
        assert response.json()["description"] == "Reviews pull requests"
        assert response.json()["icons"] == [
            {
                "src": "https://example.com/icon.svg",
                "mimeType": "image/svg+xml",
                "theme": "light",
                "futureField": "preserved",
            }
        ]
        assert response.json()["source_type"] is None

        response = client.get(f"{PREFIX}/code-review")
        assert response.status_code == 200, response.text
        assert response.json()["name"] == "code-review"
        assert response.json()["source_type"] is None

        store.create_skill_version(
            "code-review",
            source_type=SkillSourceType.GIT,
            source="https://example.com/skills.git",
        )
        response = client.get(f"{PREFIX}/code-review")
        assert response.status_code == 200, response.text
        assert response.json()["latest_version"] == 1
        assert response.json()["source_type"] == "git"


def test_create_skill_rejects_invalid_icon(tmp_path: Path, db_uri: str):
    client, store = _create_client(tmp_path, db_uri)
    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.post(
            PREFIX,
            json={
                "name": "code-review",
                "icons": [{"src": "javascript:alert(1)"}],
            },
        )

    assert response.status_code == 400, response.text
    assert "Invalid Icon URL scheme" in response.json()["message"]
    with pytest.raises(MlflowException, match="not found"):
        store.get_skill("code-review")


def test_create_skill_rejects_invalid_icon_mime_type(
    tmp_path: Path, db_uri: str, mock_icon_hostname_resolution
):
    client, store = _create_client(tmp_path, db_uri)
    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.post(
            PREFIX,
            json={
                "name": "code-review",
                "icons": [{"src": "https://example.com/icon.svg", "mimeType": "text/plain"}],
            },
        )

    assert response.status_code == 400, response.text
    assert "Invalid icon mimeType" in response.json()["message"]
    with pytest.raises(MlflowException, match="not found"):
        store.get_skill("code-review")


def test_create_skill_rejects_too_many_icons(
    tmp_path: Path, db_uri: str, mock_icon_hostname_resolution
):
    client, store = _create_client(tmp_path, db_uri)
    icons = [
        {"src": f"https://example.com/icon-{index}.svg"}
        for index in range(_MAX_REGISTRY_ICONS_PER_LIST + 1)
    ]

    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.post(PREFIX, json={"name": "code-review", "icons": icons})

    assert response.status_code == 400, response.text
    assert f"at most {_MAX_REGISTRY_ICONS_PER_LIST} items" in response.json()["message"]
    with pytest.raises(MlflowException, match="not found"):
        store.get_skill("code-review")


def test_skill_response_icon_serialization_does_not_validate_url():
    skill = Skill(
        name="code-review",
        icons=[{"src": "https://example.com/icon.svg", "mimeType": "image/svg+xml"}],
    )

    with mock.patch(
        "mlflow.utils.validation._resolve_hostname_with_timeout",
        side_effect=AssertionError("response serialization must not resolve icon hosts"),
    ):
        response = skill_registry_api.SkillResponse.from_entity(skill)

    assert response.model_dump()["icons"] == [
        {"src": "https://example.com/icon.svg", "mimeType": "image/svg+xml"}
    ]


def test_create_and_get_organization_skill(tmp_path: Path, db_uri: str):
    client, store = _create_client(tmp_path, db_uri)
    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.post(
            PREFIX,
            json={"name": "code-review", "organization": "acme"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["organization"] == "acme"

        response = client.get(f"{PREFIX}/@acme/code-review")
        assert response.status_code == 200, response.text
        assert response.json()["name"] == "code-review"
        assert response.json()["organization"] == "acme"


def test_update_skill_distinguishes_omitted_and_explicit_null_fields(
    tmp_path: Path, db_uri: str, mock_icon_hostname_resolution
):
    client, store = _create_client(tmp_path, db_uri)
    store.create_skill(
        "code-review",
        organization="acme",
        description="Original description",
        icons=[{"src": "https://example.com/original.svg"}],
    )

    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.patch(
            f"{PREFIX}/@acme/code-review",
            json={"description": "Updated description"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["description"] == "Updated description"
        assert response.json()["icons"] == [{"src": "https://example.com/original.svg"}]

        response = client.patch(
            f"{PREFIX}/@acme/code-review",
            json={"description": None, "icons": None},
        )
        assert response.status_code == 200, response.text
        assert response.json()["description"] is None
        assert response.json()["icons"] is None


def test_search_skills_forwards_query_parameters(tmp_path: Path, db_uri: str):
    client, store = _create_client(tmp_path, db_uri)
    results = PagedList(
        [Skill(name="code-review", organization="acme", source_type=SkillSourceType.GIT)],
        token="next-token",
    )

    with (
        mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store),
        mock.patch.object(store, "search_skills", return_value=results) as search_skills,
    ):
        response = client.get(
            PREFIX,
            params=[
                ("filter_string", "source_type = 'git'"),
                ("max_results", "20"),
                ("order_by", "name ASC"),
                ("order_by", "organization ASC"),
                ("page_token", "token-1"),
            ],
        )

    assert response.status_code == 200, response.text
    assert response.json()["skills"][0]["name"] == "code-review"
    assert response.json()["skills"][0]["organization"] == "acme"
    assert response.json()["skills"][0]["source_type"] == "git"
    assert response.json()["next_page_token"] == "next-token"
    search_skills.assert_called_once_with(
        filter_string="source_type = 'git'",
        max_results=20,
        order_by=["name ASC", "organization ASC"],
        page_token="token-1",
    )


@pytest.mark.parametrize(
    ("path", "organization"),
    [
        ("code-review/versions", ""),
        ("@acme/code-review/versions", "acme"),
    ],
)
def test_search_skill_versions_forwards_query_parameters(
    path: str, organization: str, tmp_path: Path, db_uri: str
):
    client, _ = _create_client(tmp_path, db_uri)
    store = mock.Mock()
    results = PagedList(
        [
            SkillVersion(
                name="code-review",
                version=2,
                organization=organization,
                source_type="git",
                source="https://github.com/acme/skills.git",
            )
        ],
        token="next-token",
    )
    store.search_skill_versions.return_value = results

    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.get(
            f"{PREFIX}/{path}",
            params=[
                ("filter_string", "status = 'active'"),
                ("max_results", "20"),
                ("order_by", "version DESC"),
                ("page_token", "token-1"),
            ],
        )

    assert response.status_code == 200, response.text
    assert response.json()["skill_versions"][0]["version"] == 2
    assert response.json()["skill_versions"][0]["organization"] == organization
    assert response.json()["next_page_token"] == "next-token"
    store.search_skill_versions.assert_called_once_with(
        name="code-review",
        organization=organization,
        filter_string="status = 'active'",
        max_results=20,
        order_by=["version DESC"],
        page_token="token-1",
    )


@pytest.mark.parametrize(
    ("path", "method_name"),
    [
        ("", "search_skills"),
        ("/code-review/versions", "search_skill_versions"),
    ],
)
def test_search_routes_surface_store_page_token_errors(
    path: str, method_name: str, tmp_path: Path, db_uri: str
):
    client, _ = _create_client(tmp_path, db_uri)
    store = mock.Mock()
    getattr(store, method_name).side_effect = MlflowException.invalid_parameter_value(
        "Page token does not match the search query."
    )

    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.get(f"{PREFIX}{path}", params={"page_token": "foreign-token"})

    assert response.status_code == 400, response.text
    assert "Page token does not match" in response.json()["message"]


def test_skill_tag_routes_forward_identity(tmp_path: Path, db_uri: str):
    client, _ = _create_client(tmp_path, db_uri)
    store = mock.Mock()

    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.post(
            f"{PREFIX}/@acme/code-review/tags",
            json={"key": "team", "value": "platform"},
        )
        assert response.status_code == 200, response.text
        assert response.json() == {}
        store.set_skill_tag.assert_called_once_with(
            name="code-review",
            key="team",
            value="platform",
            organization="acme",
        )

        response = client.delete(f"{PREFIX}/code-review/tags/team")
        assert response.status_code == 200, response.text
        assert response.json() == {}
        store.delete_skill_tag.assert_called_once_with(
            name="code-review",
            key="team",
            organization="",
        )

        response = client.post(
            f"{PREFIX}/@acme/code-review/versions/2/tags",
            json={"key": "approved", "value": "true"},
        )
        assert response.status_code == 200, response.text
        store.set_skill_version_tag.assert_called_once_with(
            name="code-review",
            version=2,
            key="approved",
            value="true",
            organization="acme",
        )

        response = client.delete(f"{PREFIX}/code-review/versions/2/tags/approved")
        assert response.status_code == 200, response.text
        store.delete_skill_version_tag.assert_called_once_with(
            name="code-review",
            version=2,
            key="approved",
            organization="",
        )

        response = client.post(
            f"{PREFIX}/@acme/code-review/tags",
            json={"key": "team/owner", "value": "platform"},
        )
        assert response.status_code == 200, response.text

        response = client.delete(f"{PREFIX}/@acme/code-review/tags/team/owner")
        assert response.status_code == 200, response.text
        assert response.json() == {}
        store.delete_skill_tag.assert_called_with(
            name="code-review",
            key="team/owner",
            organization="acme",
        )

        response = client.post(
            f"{PREFIX}/@acme/code-review/versions/2/tags",
            json={"key": "release/channel", "value": "stable"},
        )
        assert response.status_code == 200, response.text

        response = client.delete(f"{PREFIX}/@acme/code-review/versions/2/tags/release/channel")
        assert response.status_code == 200, response.text
        assert response.json() == {}
        store.delete_skill_version_tag.assert_called_with(
            name="code-review",
            version=2,
            key="release/channel",
            organization="acme",
        )


def test_skill_tag_routes_preserve_key_case_and_value(tmp_path: Path, db_uri: str):
    client, _ = _create_client(tmp_path, db_uri)
    store = mock.Mock()
    value = "x" * 9000

    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.post(
            f"{PREFIX}/@acme/code-review/tags",
            json={"key": "Team", "value": value},
        )

    assert response.status_code == 200, response.text
    store.set_skill_tag.assert_called_once_with(
        name="code-review",
        key="Team",
        value=value,
        organization="acme",
    )


def test_skill_tag_routes_surface_store_validation_errors(tmp_path: Path, db_uri: str):
    client, _ = _create_client(tmp_path, db_uri)
    store = mock.Mock()
    store.set_skill_tag.side_effect = MlflowException.invalid_parameter_value(
        "Invalid Skill tag value."
    )

    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.post(
            f"{PREFIX}/code-review/tags",
            json={"key": "team", "value": "invalid"},
        )

    assert response.status_code == 400, response.text
    assert "Invalid Skill tag value" in response.json()["message"]


def test_alias_mutation_routes_forward_parent_identity(tmp_path: Path, db_uri: str):
    client, store = _create_client(tmp_path, db_uri)

    with (
        mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store),
        mock.patch.object(store, "set_skill_alias") as set_skill_alias,
        mock.patch.object(store, "delete_skill_alias") as delete_skill_alias,
    ):
        response = client.post(
            f"{PREFIX}/@acme/code-review/aliases",
            json={"alias": "stable", "version": 2},
        )
        assert response.status_code == 200, response.text
        assert response.json() == {}
        set_skill_alias.assert_called_once_with(
            name="code-review",
            alias="stable",
            version=2,
            organization="acme",
        )

        response = client.delete(f"{PREFIX}/code-review/aliases/stable")
        assert response.status_code == 200, response.text
        assert response.json() == {}
        delete_skill_alias.assert_called_once_with(
            name="code-review",
            alias="stable",
            organization="",
        )


def test_organization_parent_update_and_delete_forward_identity(tmp_path: Path, db_uri: str):
    client, store = _create_client(tmp_path, db_uri)

    with (
        mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store),
        mock.patch.object(
            store,
            "update_skill",
            return_value=Skill(
                name="code-review",
                organization="acme",
                description="Updated description",
            ),
        ) as update_skill,
        mock.patch("mlflow.server.skill_registry.deletion.delete_skill") as delete_skill,
    ):
        response = client.patch(
            f"{PREFIX}/@acme/code-review",
            json={"description": "Updated description"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["organization"] == "acme"
        update_skill.assert_called_once_with(
            name="code-review",
            organization="acme",
            description="Updated description",
            icons=NOT_SET,
            last_updated_by=None,
        )

        response = client.delete(f"{PREFIX}/@acme/code-review")
        assert response.status_code == 200, response.text
        delete_skill.assert_called_once_with(name="code-review", organization="acme")


def test_deletion_routes_forward_parent_identity(tmp_path: Path, db_uri: str):
    client, store = _create_client(tmp_path, db_uri)

    with (
        mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store),
        mock.patch.object(store, "delete_skill_version") as delete_skill_version,
        mock.patch("mlflow.server.skill_registry.deletion.delete_skill") as delete_skill,
    ):
        response = client.delete(f"{PREFIX}/@acme/code-review/versions/2")
        assert response.status_code == 200, response.text
        assert response.json() == {}
        delete_skill_version.assert_called_once_with(
            name="code-review",
            version=2,
            organization="acme",
            last_updated_by=None,
        )

        response = client.delete(f"{PREFIX}/code-review")
        assert response.status_code == 200, response.text
        assert response.json() == {}
        delete_skill.assert_called_once_with(name="code-review", organization="")


def test_create_get_and_update_skill_version(tmp_path: Path, db_uri: str):
    client, store = _create_client(tmp_path, db_uri)
    store.create_skill("code-review")
    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.post(
            f"{PREFIX}/code-review/versions",
            json={
                "source_type": "git",
                "source": "https://github.com/acme/skills.git",
                "ref": "v1.0.0",
                "subpath": "skills/code-review",
                "digest": "a" * 64,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["version"] == 1
        assert response.json()["source_type"] == "git"
        assert response.json()["source"] == "https://github.com/acme/skills.git"
        assert response.json()["ref"] == "v1.0.0"
        assert response.json()["subpath"] == "skills/code-review"

        response = client.get(f"{PREFIX}/code-review/versions/1")
        assert response.status_code == 200, response.text
        assert response.json()["digest"] == "a" * 64

        response = client.patch(
            f"{PREFIX}/code-review/versions/1",
            json={"status": "deprecated"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "deprecated"


def test_create_and_get_organization_skill_version(tmp_path: Path, db_uri: str):
    client, store = _create_client(tmp_path, db_uri)
    store.create_skill("code-review", organization="acme")
    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.post(
            f"{PREFIX}/@acme/code-review/versions",
            json={
                "source_type": "zip",
                "source": "https://example.com/skills.zip",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["organization"] == "acme"

        response = client.get(f"{PREFIX}/@acme/code-review/versions/1")
        assert response.status_code == 200, response.text
        assert response.json()["name"] == "code-review"


@pytest.mark.parametrize(
    ("path", "organization"),
    [
        ("code-review/versions", ""),
        ("@acme/code-review/versions", "acme"),
    ],
)
def test_create_skill_version_parses_multipart_request(
    path: str, organization: str, tmp_path: Path, db_uri: str
):
    client, _ = _create_client(tmp_path, db_uri)
    version = SkillVersion(
        name="code-review",
        version=1,
        organization=organization,
        source_type="mlflow",
        source=MlflowSource(artifact_path="mlflow-artifacts:/skills/code-review/token"),
        status=SkillStatus.ACTIVE,
    )
    captured_content = []

    def register_with_content_capture(*args, **kwargs):
        captured_content.append(kwargs["content"].read())
        return version

    with mock.patch(
        "mlflow.server.skill_registry_api.register_skill_version",
        side_effect=register_with_content_capture,
    ) as register:
        response = client.post(
            f"{PREFIX}/{path}",
            files={
                "metadata": (
                    "metadata.json",
                    json.dumps({
                        "name": "metadata-name",
                        "organization": "metadata-org",
                        "digest": "a" * 64,
                    }),
                    "application/json",
                ),
                "content": ("content.tar.gz", io.BytesIO(b"archive"), "application/gzip"),
            },
        )

    assert response.status_code == 200, response.text
    registration = register.call_args.args[0]
    assert registration.name == "code-review"
    assert registration.organization == organization
    assert registration.digest == "a" * 64
    assert register.call_args.kwargs["multipart"] is True
    assert captured_content == [b"archive"]
    assert register.call_args.kwargs["content"].closed


def test_multipart_registration_rejects_oversized_content_length_before_parsing(
    tmp_path: Path, db_uri: str, monkeypatch: pytest.MonkeyPatch
):
    client, _ = _create_client(tmp_path, db_uri)
    monkeypatch.setattr(skill_registry_api, "_get_multipart_request_size_limit", lambda: 1)

    with (
        mock.patch.object(Request, "form") as form,
        mock.patch("mlflow.server.skill_registry_api.register_skill_version") as register,
    ):
        response = client.post(
            f"{PREFIX}/register",
            files={
                "metadata": ("metadata.json", '{"name": "code-review"}', "application/json"),
                "content": ("content.tar.gz", io.BytesIO(b"archive"), "application/gzip"),
            },
        )

    assert response.status_code == 413, response.text
    assert "maximum allowed size" in response.json()["message"]
    form.assert_not_called()
    register.assert_not_called()


def test_multipart_registration_rejects_oversized_chunked_body_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(skill_registry_api, "_get_multipart_request_size_limit", lambda: 1)
    receive_calls = 0

    async def receive():
        nonlocal receive_calls
        receive_calls += 1
        return {"type": "http.request", "body": b"too large", "more_body": True}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": f"{PREFIX}/register",
            "raw_path": f"{PREFIX}/register".encode(),
            "query_string": b"",
            "headers": [(b"content-type", b"multipart/form-data; boundary=boundary")],
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "http_version": "1.1",
            "state": {},
        },
        receive,
    )

    async def parse_request():
        async with skill_registry_api._parse_registration_request(request):
            pass

    with pytest.raises(HTTPException, match="maximum allowed size") as exc_info:
        asyncio.run(parse_request())

    assert exc_info.value.status_code == 413
    assert receive_calls == 1


def test_multipart_registration_rejects_oversized_metadata(
    tmp_path: Path, db_uri: str, monkeypatch: pytest.MonkeyPatch
):
    client, _ = _create_client(tmp_path, db_uri)
    monkeypatch.setattr(skill_registry_api, "_MAX_REGISTRATION_METADATA_SIZE", 16)

    with mock.patch("mlflow.server.skill_registry_api.register_skill_version") as register:
        response = client.post(
            f"{PREFIX}/register",
            files={
                "metadata": ("metadata.json", b"x" * 17, "application/json"),
                "content": ("content.tar.gz", io.BytesIO(b"archive"), "application/gzip"),
            },
        )

    assert response.status_code == 413, response.text
    assert "registration metadata" in response.json()["message"]
    register.assert_not_called()


def test_multipart_registration_rejects_extra_file_parts(tmp_path: Path, db_uri: str):
    client, _ = _create_client(tmp_path, db_uri)

    with mock.patch("mlflow.server.skill_registry_api.register_skill_version") as register:
        response = client.post(
            f"{PREFIX}/register",
            files={
                "metadata": ("metadata.json", '{"name": "code-review"}', "application/json"),
                "content": ("content.tar.gz", io.BytesIO(b"archive"), "application/gzip"),
                "extra": ("extra.bin", io.BytesIO(b"unexpected"), "application/octet-stream"),
            },
        )

    assert response.status_code == 400, response.text
    assert "maximum number of files" in response.json()["message"].lower()
    register.assert_not_called()


def test_multipart_registration_rejects_extra_form_fields(tmp_path: Path, db_uri: str):
    client, _ = _create_client(tmp_path, db_uri)

    with mock.patch("mlflow.server.skill_registry_api.register_skill_version") as register:
        response = client.post(
            f"{PREFIX}/register",
            data={"metadata": '{"name": "code-review"}', "extra": "unexpected"},
            files={"content": ("content.tar.gz", io.BytesIO(b"archive"), "application/gzip")},
        )

    assert response.status_code == 400, response.text
    assert "maximum number of fields" in response.json()["message"].lower()
    register.assert_not_called()


def test_multipart_registration_rejects_unknown_part_with_expected_files(
    tmp_path: Path, db_uri: str
):
    client, _ = _create_client(tmp_path, db_uri)

    with mock.patch("mlflow.server.skill_registry_api.register_skill_version") as register:
        response = client.post(
            f"{PREFIX}/register",
            data={"extra": "unexpected"},
            files={
                "metadata": ("metadata.json", '{"name": "code-review"}', "application/json"),
                "content": ("content.tar.gz", io.BytesIO(b"archive"), "application/gzip"),
            },
        )

    assert response.status_code == 400, response.text
    assert "exactly one 'metadata' part" in response.json()["message"]
    register.assert_not_called()


def test_create_skill_version_rejects_remote_source_in_multipart(tmp_path: Path, db_uri: str):
    client, _ = _create_client(tmp_path, db_uri)
    response = client.post(
        f"{PREFIX}/code-review/versions",
        files={
            "metadata": (
                "metadata.json",
                json.dumps({
                    "name": "code-review",
                    "source_type": "git",
                    "source": "https://github.com/acme/skills.git",
                }),
                "application/json",
            ),
            "content": ("content.tar.gz", io.BytesIO(b"archive"), "application/gzip"),
        },
    )

    assert response.status_code == 400, response.text
    assert "must use an application/json body" in response.json()["message"]


def test_register_remote_skill_version_creates_parent(tmp_path: Path, db_uri: str):
    client, store = _create_client(tmp_path, db_uri)
    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.post(
            f"{PREFIX}/register",
            json={
                "name": "code-review",
                "source_type": "git",
                "source": "https://github.com/acme/skills.git",
                "ref": "v1.0.0",
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["name"] == "code-review"
    assert response.json()["version"] == 1
    assert store.get_skill("code-review").created_by is None


def test_register_local_skill_version_parses_multipart_request(tmp_path: Path, db_uri: str):
    client, _ = _create_client(tmp_path, db_uri)
    version = SkillVersion(
        name="code-review",
        version=1,
        source_type="mlflow",
        source=MlflowSource(artifact_path="mlflow-artifacts:/skills/code-review/token"),
        status=SkillStatus.ACTIVE,
    )
    captured_content = []

    def register_with_content_capture(*args, **kwargs):
        captured_content.append(kwargs["content"].read())
        return version

    with mock.patch(
        "mlflow.server.skill_registry_api.register_skill_version",
        side_effect=register_with_content_capture,
    ) as register:
        response = client.post(
            f"{PREFIX}/register",
            files={
                "metadata": (
                    "metadata.json",
                    json.dumps({"name": "code-review", "digest": "a" * 64}),
                    "application/json",
                ),
                "content": ("content.tar.gz", io.BytesIO(b"archive"), "application/gzip"),
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["source_type"] == "mlflow"
    registration = register.call_args.args[0]
    assert registration.name == "code-review"
    assert registration.digest == "a" * 64
    assert register.call_args.kwargs["multipart"] is True
    assert captured_content == [b"archive"]
    assert register.call_args.kwargs["content"].closed


def test_register_local_skill_version_closes_multipart_file_on_failure(tmp_path: Path, db_uri: str):
    client, _ = _create_client(tmp_path, db_uri)
    uploaded_file = None

    def fail_registration(*args, **kwargs):
        nonlocal uploaded_file
        uploaded_file = kwargs["content"]
        raise MlflowException.invalid_parameter_value("registration failed")

    with mock.patch(
        "mlflow.server.skill_registry_api.register_skill_version",
        side_effect=fail_registration,
    ):
        response = client.post(
            f"{PREFIX}/register",
            files={
                "metadata": (
                    "metadata.json",
                    json.dumps({"name": "code-review", "digest": "a" * 64}),
                    "application/json",
                ),
                "content": ("content.tar.gz", io.BytesIO(b"archive"), "application/gzip"),
            },
        )

    assert response.status_code == 400, response.text
    assert uploaded_file is not None
    assert uploaded_file.closed


def test_register_local_skill_version_rejects_invalid_utf8_metadata(tmp_path: Path, db_uri: str):
    client, _ = _create_client(tmp_path, db_uri)
    with mock.patch("mlflow.server.skill_registry_api.register_skill_version") as register:
        response = client.post(
            f"{PREFIX}/register",
            files={
                "metadata": ("metadata.json", b'{"name": "code-review"}\xff', "application/json"),
                "content": ("content.tar.gz", io.BytesIO(b"archive"), "application/gzip"),
            },
        )

    assert response.status_code == 400, response.text
    assert "metadata" in response.json()["message"]
    assert "valid JSON" in response.json()["message"]
    register.assert_not_called()


@pytest.mark.parametrize("prefix", get_skill_registry_api_route_prefixes())
def test_bulk_register_skill_versions_forwards_client_prepared_batch(
    prefix: str, tmp_path: Path, db_uri: str
):
    client, _ = _create_client(tmp_path, db_uri)
    versions = [
        SkillVersion(
            name="code-review",
            version=1,
            organization="acme",
            source_type="git",
            source="https://github.com/acme/skills.git",
            status=SkillStatus.ACTIVE,
        ),
        SkillVersion(
            name="release-notes",
            version=1,
            organization="acme",
            source_type="git",
            source="https://github.com/acme/skills.git",
            status=SkillStatus.ACTIVE,
        ),
    ]
    with mock.patch(
        "mlflow.server.skill_registry_api.bulk_register_skill_versions",
        return_value=versions,
    ) as bulk_register:
        response = client.post(
            f"{prefix}/bulk-register",
            json={
                "organization": "acme",
                "skills": [
                    {
                        "name": "code-review",
                        "source_type": "git",
                        "source": "https://github.com/acme/skills.git",
                        "ref": "main",
                        "subpath": "skills/code-review",
                        "digest": "a" * 64,
                    },
                    {
                        "name": "release-notes",
                        "source": "https://github.com/acme/skills.git",
                        "ref": "main",
                        "subpath": "skills/release-notes",
                        "digest": "b" * 64,
                    },
                ],
            },
        )

    assert response.status_code == 200, response.text
    assert [version["name"] for version in response.json()["skill_versions"]] == [
        "code-review",
        "release-notes",
    ]
    registrations = bulk_register.call_args.args[0]
    assert [registration.name for registration in registrations] == [
        "code-review",
        "release-notes",
    ]
    assert all(registration.created_by is None for registration in registrations)
    assert all(registration.organization == "acme" for registration in registrations)


def test_bulk_register_skill_versions_rejects_oversized_batch(tmp_path: Path, db_uri: str):
    client, _ = _create_client(tmp_path, db_uri)
    skills = [
        {
            "name": f"skill-{index}",
            "source": "https://github.com/acme/skills.git",
            "ref": "main",
            "subpath": f"skills/skill-{index}",
            "digest": "a" * 64,
        }
        for index in range(_MAX_BULK_REGISTER_SKILLS + 1)
    ]

    with mock.patch(
        "mlflow.server.skill_registry_api.bulk_register_skill_versions"
    ) as bulk_register:
        response = client.post(f"{PREFIX}/bulk-register", json={"skills": skills})

    assert response.status_code == 400, response.text
    assert f"at most {_MAX_BULK_REGISTER_SKILLS} items" in response.json()["message"]
    bulk_register.assert_not_called()


def test_bulk_register_skill_versions_uses_transactional_store_and_is_idempotent(
    tmp_path: Path, db_uri: str
):
    client, store = _create_client(tmp_path, db_uri)
    request_body = {
        "organization": "acme",
        "skills": [
            {
                "name": "code-review",
                "source": "https://github.com/acme/skills.git",
                "ref": "main",
                "subpath": "skills/code-review",
                "digest": "a" * 64,
            },
            {
                "name": "release-notes",
                "source": "https://github.com/acme/skills.git",
                "ref": "main",
                "subpath": "skills/release-notes",
                "digest": "b" * 64,
            },
        ],
    }

    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.post(f"{PREFIX}/bulk-register", json=request_body)
        assert response.status_code == 200, response.text
        first_versions = response.json()["skill_versions"]

        response = client.post(f"{PREFIX}/bulk-register", json=request_body)
        assert response.status_code == 200, response.text
        second_versions = response.json()["skill_versions"]

    assert [version["version"] for version in first_versions] == [1, 1]
    assert [version["version"] for version in second_versions] == [1, 1]
    assert store.get_skill_version("code-review", 1, organization="acme").digest == "a" * 64
    assert store.get_skill_version("release-notes", 1, organization="acme").digest == "b" * 64


def test_bulk_register_skill_versions_rejects_invalid_organization_before_persistence(
    tmp_path: Path, db_uri: str
):
    client, store = _create_client(tmp_path, db_uri)
    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.post(
            f"{PREFIX}/bulk-register",
            json={
                "organization": "invalid organization",
                "skills": [
                    {
                        "name": "code-review",
                        "source": "https://github.com/acme/skills.git",
                        "ref": "main",
                        "subpath": "skills/code-review",
                        "digest": "a" * 64,
                    },
                    {
                        "name": "release-notes",
                        "source": "https://github.com/acme/skills.git",
                        "ref": "main",
                        "subpath": "skills/release-notes",
                        "digest": "b" * 64,
                    },
                ],
            },
        )

    assert response.status_code == 400, response.text
    assert "Invalid organization name" in response.json()["message"]
    assert store.search_skills().token is None
    assert list(store.search_skills()) == []


def test_bulk_register_skill_versions_rejects_per_skill_organization(tmp_path: Path, db_uri: str):
    client, store = _create_client(tmp_path, db_uri)
    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.post(
            f"{PREFIX}/bulk-register",
            json={
                "organization": "acme",
                "skills": [
                    {
                        "name": "code-review",
                        "organization": "other",
                        "source": "https://github.com/acme/skills.git",
                        "ref": "main",
                        "subpath": "skills/code-review",
                        "digest": "a" * 64,
                    }
                ],
            },
        )

    assert response.status_code == 400, response.text
    assert "organization" in response.json()["message"]
    assert store.search_skills().token is None
    assert list(store.search_skills()) == []


def test_bulk_register_skill_versions_supports_omitted_organization(tmp_path: Path, db_uri: str):
    client, store = _create_client(tmp_path, db_uri)
    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.post(
            f"{PREFIX}/bulk-register",
            json={
                "skills": [
                    {
                        "name": "code-review",
                        "source": "https://github.com/acme/skills.git",
                        "ref": "main",
                        "subpath": "skills/code-review",
                        "digest": "a" * 64,
                    }
                ]
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["skill_versions"][0]["organization"] == ""
    assert store.get_skill_version("code-review", 1).organization == ""


def test_get_skill_version_by_alias_and_latest(tmp_path: Path, db_uri: str):
    client, store = _create_client(tmp_path, db_uri)
    store.create_skill_version(
        "code-review",
        source_type="git",
        source="https://github.com/acme/skills.git",
    )
    store.create_skill_version(
        "code-review",
        source_type="git",
        source="https://github.com/acme/skills.git",
        ref="v2.0.0",
    )
    store.set_skill_alias("code-review", "stable", 1)
    store.create_skill_version(
        "code-review",
        organization="acme",
        source_type="git",
        source="https://github.com/acme/skills.git",
    )
    store.set_skill_alias("code-review", "stable", 1, organization="acme")
    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store):
        response = client.get(f"{PREFIX}/code-review/aliases/stable")
        assert response.status_code == 200, response.text
        assert response.json()["version"] == 1

        response = client.get(f"{PREFIX}/code-review/aliases/latest")
        assert response.status_code == 200, response.text
        assert response.json()["version"] == 2

        response = client.get(f"{PREFIX}/@acme/code-review/aliases/stable")
        assert response.status_code == 200, response.text
        assert response.json()["organization"] == "acme"
        assert response.json()["version"] == 1

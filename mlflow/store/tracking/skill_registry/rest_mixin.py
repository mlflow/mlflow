"""REST implementation of SkillRegistryMixin."""

import json
from typing import Any, BinaryIO
from urllib.parse import quote

from mlflow.entities.skill import RegistryIcon, Skill, SkillStatus
from mlflow.entities.skill_version import SkillVersion
from mlflow.exceptions import MlflowException, MlflowNotImplementedException
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import NOT_SET, SEARCH_MAX_RESULTS_DEFAULT
from mlflow.utils.rest_utils import http_request, verify_rest_response
from mlflow.utils.validation import (
    _validate_organization_name,
    _validate_skill_name,
    _validate_skill_version,
)

_SKILL_API_PREFIX = "/api/3.0/mlflow/skills"


def _encode_path_param(value: str) -> str:
    if not isinstance(value, str):
        raise MlflowException.invalid_parameter_value(
            f"Path parameters must be strings, got {type(value).__name__}."
        )
    if value in (".", ".."):
        raise MlflowException.invalid_parameter_value("Path parameters must not be '.' or '..'.")
    return quote(value, safe="")


def _skill_path(name: str, organization: str = "") -> str:
    _validate_skill_name(name)
    _validate_organization_name(organization)
    if organization:
        return f"/@{_encode_path_param(organization)}/{_encode_path_param(name)}"
    return f"/{_encode_path_param(name)}"


class RestSkillRegistryMixin:
    """REST implementation of SkillRegistryMixin.

    Expects the implementing class to provide ``get_host_creds()``.
    Uses ``http_request`` directly (no protobuf) since the Skill
    registry endpoints use Pydantic/JSON.
    """

    def _skill_request(self, method: str, path: str, json=None, params=None, **kwargs):
        self._validate_workspace_support_if_specified()
        endpoint = f"{_SKILL_API_PREFIX}{path}"
        response = http_request(
            host_creds=self.get_host_creds(),
            endpoint=endpoint,
            method=method,
            json=json,
            params=params,
            **kwargs,
        )
        verify_rest_response(response, endpoint)
        if not response.text:
            return None
        return response.json()

    def _register_skill(
        self,
        *,
        name: str,
        content: BinaryIO,
        digest: str | None = None,
        organization: str = "",
        status: str = "active",
    ) -> SkillVersion:
        """Upload a prepared gzip-compressed tar archive through atomic registration."""
        metadata = {
            "name": name,
            "organization": organization,
            "digest": digest,
            "status": status,
        }

        data = self._skill_request(
            "POST",
            f"{_skill_path(name, organization)}/versions",
            files={
                "metadata": (None, json.dumps(metadata), "application/json"),
                "content": ("content.tar.gz", content, "application/gzip"),
            },
        )
        return SkillVersion.from_dict(data)

    def create_skill(
        self,
        name: str,
        organization: str = "",
        description: str | None = None,
        icons: list[RegistryIcon] | None = None,
        created_by: str | None = None,
    ) -> Skill:
        body = {"name": name, "organization": organization}
        if description is not None:
            body["description"] = description
        if icons is not None:
            body["icons"] = icons
        return Skill.from_dict(self._skill_request("POST", "", json=body))

    def get_skill(self, name: str, organization: str = "") -> Skill:
        return Skill.from_dict(self._skill_request("GET", _skill_path(name, organization)))

    def search_skills(
        self,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
        include_skill_identities: list[tuple[str, str]] | None = None,
        exclude_skill_identities: list[tuple[str, str]] | None = None,
    ) -> PagedList[Skill]:
        # The REST API cannot express exclusions from an unrestricted result set.
        if exclude_skill_identities and include_skill_identities is None:
            raise MlflowNotImplementedException(
                "REST-backed Skill search cannot enforce exclusions without an include selector. "
                "Use a SQL tracking backend for server-side Skill authorization."
            )
        if include_skill_identities is not None and exclude_skill_identities:
            excluded = set(exclude_skill_identities)
            include_skill_identities = [
                identity for identity in include_skill_identities if identity not in excluded
            ]
        params: dict[str, Any] = {"max_results": max_results}
        if filter_string is not None:
            params["filter_string"] = filter_string
        if order_by is not None:
            params["order_by"] = order_by
        if page_token is not None:
            params["page_token"] = page_token
        if include_skill_identities is not None:
            params["include_skill_identities"] = [
                f"@{organization}/{name}" if organization else name
                for organization, name in include_skill_identities
            ] or [""]
        data = self._skill_request("GET", "", params=params)
        return PagedList(
            [Skill.from_dict(skill) for skill in data["skills"]], data.get("next_page_token")
        )

    def update_skill(
        self,
        name: str,
        organization: str = "",
        description: str | None = NOT_SET,
        icons: list[RegistryIcon] | None = NOT_SET,
        last_updated_by: str | None = None,
    ) -> Skill:
        body: dict[str, Any] = {}
        if description is not NOT_SET:
            body["description"] = description
        if icons is not NOT_SET:
            body["icons"] = icons
        return Skill.from_dict(
            self._skill_request("PATCH", _skill_path(name, organization), json=body)
        )

    def delete_skill(self, name: str, organization: str = "") -> None:
        self._skill_request("DELETE", _skill_path(name, organization))

    def create_skill_version(
        self,
        name: str,
        organization: str = "",
        source_type: str | None = None,
        source: str | None = None,
        ref: str | None = None,
        subpath: str | None = None,
        digest: str | None = None,
        status: str = "active",
        created_by: str | None = None,
        expected_parent_exists: bool | None = None,
    ) -> SkillVersion:
        if expected_parent_exists is not None:
            raise MlflowNotImplementedException(
                "REST-backed Skill registration cannot atomically enforce a parent-existence "
                "precondition. Use a SQL tracking backend for server-side Skill authorization."
            )
        body = {
            "source_type": source_type,
            "source": source,
            "ref": ref,
            "subpath": subpath,
            "digest": digest,
            "status": status,
        }
        return SkillVersion.from_dict(
            self._skill_request("POST", f"{_skill_path(name, organization)}/versions", json=body)
        )

    def bulk_register_skills(
        self,
        skill_definitions: list[dict[str, Any]],
        organization: str = "",
        created_by: str | None = None,
        expected_parent_exists: dict[str, bool] | None = None,
    ) -> list[SkillVersion]:
        if expected_parent_exists:
            raise MlflowNotImplementedException(
                "REST-backed bulk Skill registration cannot atomically enforce parent-existence "
                "preconditions. Use a SQL tracking backend for server-side Skill authorization."
            )
        data = self._skill_request(
            "POST",
            "/bulk-register",
            json={"organization": organization, "skills": skill_definitions},
        )
        return [SkillVersion.from_dict(version) for version in data["skill_versions"]]

    def get_skill_version(
        self,
        name: str,
        version: int,
        organization: str = "",
    ) -> SkillVersion:
        _validate_skill_version(version)
        return SkillVersion.from_dict(
            self._skill_request("GET", f"{_skill_path(name, organization)}/versions/{version}")
        )

    def get_skill_version_by_alias(
        self,
        name: str,
        alias: str,
        organization: str = "",
    ) -> SkillVersion:
        return SkillVersion.from_dict(
            self._skill_request(
                "GET", f"{_skill_path(name, organization)}/aliases/{_encode_path_param(alias)}"
            )
        )

    def get_latest_skill_version(
        self,
        name: str,
        organization: str = "",
    ) -> SkillVersion:
        return self.get_skill_version_by_alias(name=name, alias="latest", organization=organization)

    def search_skill_versions(
        self,
        name: str,
        organization: str = "",
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[SkillVersion]:
        params: dict[str, Any] = {"max_results": max_results}
        if filter_string is not None:
            params["filter_string"] = filter_string
        if order_by is not None:
            params["order_by"] = order_by
        if page_token is not None:
            params["page_token"] = page_token
        data = self._skill_request(
            "GET", f"{_skill_path(name, organization)}/versions", params=params
        )
        return PagedList(
            [SkillVersion.from_dict(version) for version in data["skill_versions"]],
            data.get("next_page_token"),
        )

    def update_skill_version(
        self,
        name: str,
        version: int,
        organization: str = "",
        status: SkillStatus | None = NOT_SET,
        last_updated_by: str | None = None,
    ) -> SkillVersion:
        _validate_skill_version(version)
        body = {}
        if status is not NOT_SET:
            body["status"] = str(status) if status is not None else None
        return SkillVersion.from_dict(
            self._skill_request(
                "PATCH",
                f"{_skill_path(name, organization)}/versions/{version}",
                json=body,
            )
        )

    def delete_skill_version(
        self,
        name: str,
        version: int,
        organization: str = "",
        last_updated_by: str | None = None,
    ) -> None:
        _validate_skill_version(version)
        self._skill_request("DELETE", f"{_skill_path(name, organization)}/versions/{version}")

    def set_skill_tag(
        self,
        name: str,
        key: str,
        value: str,
        organization: str = "",
    ) -> None:
        self._skill_request(
            "POST", f"{_skill_path(name, organization)}/tags", json={"key": key, "value": value}
        )

    def delete_skill_tag(
        self,
        name: str,
        key: str,
        organization: str = "",
    ) -> None:
        self._skill_request(
            "DELETE", f"{_skill_path(name, organization)}/tags/{_encode_path_param(key)}"
        )

    def set_skill_version_tag(
        self,
        name: str,
        version: int,
        key: str,
        value: str,
        organization: str = "",
    ) -> None:
        _validate_skill_version(version)
        self._skill_request(
            "POST",
            f"{_skill_path(name, organization)}/versions/{version}/tags",
            json={"key": key, "value": value},
        )

    def delete_skill_version_tag(
        self,
        name: str,
        version: int,
        key: str,
        organization: str = "",
    ) -> None:
        _validate_skill_version(version)
        self._skill_request(
            "DELETE",
            f"{_skill_path(name, organization)}/versions/{version}/tags/{_encode_path_param(key)}",
        )

    def set_skill_alias(
        self,
        name: str,
        alias: str,
        version: int,
        organization: str = "",
    ) -> None:
        self._skill_request(
            "POST",
            f"{_skill_path(name, organization)}/aliases",
            json={"alias": alias, "version": version},
        )

    def delete_skill_alias(
        self,
        name: str,
        alias: str,
        organization: str = "",
    ) -> None:
        self._skill_request(
            "DELETE", f"{_skill_path(name, organization)}/aliases/{_encode_path_param(alias)}"
        )

"""REST implementation of MCPServerRegistryMixin."""

from typing import Any
from urllib.parse import quote

from mlflow.entities.skill import RegistryIcon, Skill, SkillStatus
from mlflow.entities.skill_version import SkillVersion
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import NOT_SET, SEARCH_MAX_RESULTS_DEFAULT
from mlflow.utils.rest_utils import http_request, verify_rest_response

_SKILL_API_PREFIX = "/api/3.0/mlflow/skills"


def _encode_path_param(value: str) -> str:
    return quote(str(value), safe="")


def _server_path(name: str) -> str:
    return f"/{_encode_path_param(name)}"


class RestSkillRegistryMixin:
    """REST implementation of SkillRegistryMixin.

    Expects the implementing class to provide ``get_host_creds()``.
    Uses ``http_request`` directly (no protobuf) since the Skill
    registry endpoints use Pydantic/JSON.
    """

    def _skill_request(self, method: str, path: str, json=None, params=None):
        self._validate_workspace_support_if_specified()
        endpoint = f"{_SKILL_API_PREFIX}{path}"
        response = http_request(
            host_creds=self.get_host_creds(),
            endpoint=endpoint,
            method=method,
            json=json,
            params=params,
        )
        verify_rest_response(response, endpoint)
        if not response.text:
            return None
        return response.json()

    def create_skill(
        self,
        name: str,
        organization: str = "",
        description: str | None = None,
        icons: list[RegistryIcon] | None = None,
        created_by: str | None = None,
    ) -> Skill: ...

    def get_skill(self, name: str, organization: str = "") -> Skill: ...

    def search_skills(
        self,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[Skill]: ...

    def update_skill(
        self,
        name: str,
        organization: str = "",
        description: str | None = NOT_SET,
        icons: list[RegistryIcon] | None = NOT_SET,
        last_updated_by: str | None = None,
    ) -> Skill: ...

    def delete_skill(self, name: str, organization: str = "") -> None: ...

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
    ) -> SkillVersion: ...

    def bulk_register_skills(
        self,
        skill_definitions: list[dict[str, Any]],
        organization: str = "",
        created_by: str | None = None,
    ) -> list[SkillVersion]: ...

    def get_skill_version(
        self,
        name: str,
        version: int,
        organization: str = "",
    ) -> SkillVersion: ...

    def get_skill_version_by_alias(
        self,
        name: str,
        alias: str,
        organization: str = "",
    ) -> SkillVersion: ...

    def get_latest_skill_version(
        self,
        name: str,
        organization: str = "",
    ) -> SkillVersion: ...

    def search_skill_versions(
        self,
        name: str,
        organization: str = "",
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[SkillVersion]: ...

    def update_skill_version(
        self,
        name: str,
        version: int,
        organization: str = "",
        status: SkillStatus | None = NOT_SET,
        last_updated_by: str | None = None,
    ) -> SkillVersion: ...

    def delete_skill_version(
        self,
        name: str,
        version: int,
        organization: str = "",
        last_updated_by: str | None = None,
    ) -> None: ...

    def set_skill_tag(
        self,
        name: str,
        key: str,
        value: str,
        organization: str = "",
    ) -> None: ...

    def delete_skill_tag(
        self,
        name: str,
        key: str,
        organization: str = "",
    ) -> None: ...

    def set_skill_version_tag(
        self,
        name: str,
        version: int,
        key: str,
        value: str,
        organization: str = "",
    ) -> None: ...

    def delete_skill_version_tag(
        self,
        name: str,
        version: int,
        key: str,
        organization: str = "",
    ) -> None: ...

    def set_skill_alias(
        self,
        name: str,
        alias: str,
        version: int,
        organization: str = "",
    ) -> None: ...

    def delete_skill_alias(
        self,
        name: str,
        alias: str,
        organization: str = "",
    ) -> None: ...

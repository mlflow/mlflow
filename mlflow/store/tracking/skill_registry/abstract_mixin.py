from __future__ import annotations

from typing import Any

from mlflow.entities.skill import RegistryIcon, Skill, SkillStatus
from mlflow.entities.skill_version import SkillVersion
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import NOT_SET, SEARCH_MAX_RESULTS_DEFAULT


class SkillRegistryMixin:
    """Mixin providing the Skill Registry interface for tracking stores."""

    def create_skill(
        self,
        name: str,
        organization: str = "",
        description: str | None = None,
        icons: list[RegistryIcon] | None = None,
        created_by: str | None = None,
    ) -> Skill:
        raise NotImplementedError(self.__class__.__name__)

    def get_skill(self, name: str, organization: str = "") -> Skill:
        raise NotImplementedError(self.__class__.__name__)

    def update_skill(
        self,
        name: str,
        organization: str = "",
        description: str | None = NOT_SET,
        icons: list[RegistryIcon] | None = NOT_SET,
        last_updated_by: str | None = None,
    ) -> Skill:
        raise NotImplementedError(self.__class__.__name__)

    def delete_skill(self, name: str, organization: str = "") -> None:
        """
        Hard-delete a skill with its versions, tags, and aliases.

        The delete fails, removing nothing, while any of the skill's versions is a member of a
        live (non-``deleted``) agent plugin version.
        """
        raise NotImplementedError(self.__class__.__name__)

    def delete_skill_and_collect_artifacts(self, name: str, organization: str = "") -> list[str]:
        """
        Hard-delete a skill like ``delete_skill`` and return the artifact paths its versions owned.

        Artifact storage is not transactional with the registry database, so the paths are
        captured and the row deletion committed in one transaction, and the caller reclaims the
        returned paths afterwards, best-effort. Only paths written by the standalone upload flow
        are returned; a version that references a package tree owns nothing.
        """
        raise NotImplementedError(self.__class__.__name__)

    def search_skills(
        self,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
        include_skill_identities: list[tuple[str, str]] | None = None,
        exclude_skill_identities: list[tuple[str, str]] | None = None,
    ) -> PagedList[Skill]:
        """Search with optional exact ``(organization, name)`` filters before pagination."""
        raise NotImplementedError(self.__class__.__name__)

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
        raise NotImplementedError(self.__class__.__name__)

    def bulk_register_skills(
        self,
        skill_definitions: list[dict[str, Any]],
        organization: str = "",
        created_by: str | None = None,
        expected_parent_exists: dict[str, bool] | None = None,
    ) -> list[SkillVersion]:
        """Atomically register standalone skills from one Git repository and ref.

        Reuse the highest non-deleted version with matching source, ref, subpath, and digest,
        preserving its status even when it differs from the requested status. Otherwise create
        a version with the requested status using ordinary registration's allocation.
        Results follow input order; any failure rolls back the entire batch.

        Args:
            skill_definitions: Nonempty list of normalized definitions with unique names,
                Git sources, and digests from the same repository and ref. Each definition's
                ``status`` must be ``active`` (default) or ``draft`` and must be the same
                across the batch. Status is validated per definition and applies only to
                newly created versions.
            organization: Organization shared by all definitions.
            created_by: Authenticated creator for new records.

        Returns:
            Reused or created versions in input order, potentially with different statuses.

        Raises:
            MlflowException: If metadata, requested status, or batch constraints are invalid,
                or a skill name conflicts with a packaged member.
        """
        raise NotImplementedError(self.__class__.__name__)

    def get_skill_version(
        self,
        name: str,
        version: int,
        organization: str = "",
    ) -> SkillVersion:
        raise NotImplementedError(self.__class__.__name__)

    def search_skill_versions(
        self,
        name: str,
        organization: str = "",
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[SkillVersion]:
        raise NotImplementedError(self.__class__.__name__)

    def set_skill_tag(
        self,
        name: str,
        key: str,
        value: str,
        organization: str = "",
    ) -> None:
        raise NotImplementedError(self.__class__.__name__)

    def delete_skill_tag(
        self,
        name: str,
        key: str,
        organization: str = "",
    ) -> None:
        raise NotImplementedError(self.__class__.__name__)

    def set_skill_version_tag(
        self,
        name: str,
        version: int,
        key: str,
        value: str,
        organization: str = "",
    ) -> None:
        raise NotImplementedError(self.__class__.__name__)

    def delete_skill_version_tag(
        self,
        name: str,
        version: int,
        key: str,
        organization: str = "",
    ) -> None:
        raise NotImplementedError(self.__class__.__name__)

    def get_skill_version_by_alias(
        self,
        name: str,
        alias: str,
        organization: str = "",
    ) -> SkillVersion:
        raise NotImplementedError(self.__class__.__name__)

    def get_latest_skill_version(
        self,
        name: str,
        organization: str = "",
    ) -> SkillVersion:
        """Retrieve the version resolved as latest for a skill."""
        raise NotImplementedError(self.__class__.__name__)

    def update_skill_version(
        self,
        name: str,
        version: int,
        organization: str = "",
        status: SkillStatus | None = NOT_SET,
        last_updated_by: str | None = None,
    ) -> SkillVersion:
        raise NotImplementedError(self.__class__.__name__)

    def delete_skill_version(
        self,
        name: str,
        version: int,
        organization: str = "",
        last_updated_by: str | None = None,
    ) -> None:
        raise NotImplementedError(self.__class__.__name__)

    def set_skill_alias(
        self,
        name: str,
        alias: str,
        version: int,
        organization: str = "",
    ) -> None:
        """Set an alias on a non-deleted skill version."""
        raise NotImplementedError(self.__class__.__name__)

    def delete_skill_alias(
        self,
        name: str,
        alias: str,
        organization: str = "",
    ) -> None:
        raise NotImplementedError(self.__class__.__name__)

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from mlflow.entities.skill import RegistryIcon, Skill, SkillStatus
from mlflow.entities.skill_version import SkillVersion
from mlflow.exceptions import MlflowNotImplementedException
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import NOT_SET, SEARCH_MAX_RESULTS_DEFAULT

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


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

    def delete_skill_and_collect_artifacts(
        self,
        name: str,
        organization: str = "",
        before_commit: Callable[[Session], None] | None = None,
    ) -> list[str]:
        """
        Hard-delete a skill like ``delete_skill`` and return the artifact paths its versions owned.

        Artifact storage is not transactional with the registry database, so the paths are
        captured and the row deletion committed in one transaction, and the caller reclaims the
        returned paths afterwards, best-effort. Only paths written by the standalone upload flow
        are returned; a version that references a package tree owns nothing.

        ``before_commit`` runs with the SQL transaction after integrity checks and deletion
        have flushed, while the parent lock still prevents identity reuse. Any exception
        rolls back the deletion. Backends unable to enforce this must reject the callback.
        """
        if before_commit is not None:
            raise MlflowNotImplementedException(
                "This tracking backend cannot enforce Skill deletion cleanup before commit. "
                "Use a SQL tracking backend for server-side Skill authorization."
            )
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
        """Search with exact ``(organization, name)`` filters before pagination.

        ``include_skill_identities`` is the effective selector: ``None`` selects all Skills,
        and an empty list selects none. The API handler intersects the caller's selector
        with the auth app's scope before calling this method. ``exclude_skill_identities``
        removes exact identities. Both filters apply before pagination and may change
        between pages; tokens bind to the workspace, query filter, and ordering only.
        The REST store can subtract exclusions from a finite include selector; it rejects
        exclusions from an unrestricted result set before fetching results.
        """
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
        """Create a version, creating its parent Skill when necessary.

        ``expected_parent_exists`` guards the parent state observed by the caller:
        ``True`` requires an existing parent, ``False`` requires a missing parent,
        and ``None`` leaves either state valid. A mismatch raises ``RESOURCE_CONFLICT``
        before creating a parent or version. This is a storage consistency check;
        the caller must authorize the expected operation separately.
        Stores that cannot enforce the precondition atomically, including the REST store,
        must reject a non-``None`` expectation before writing.
        """
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
            expected_parent_exists: Optional map of Skill names to required parent states.
                ``True`` requires an existing parent and ``False`` requires a missing one;
                omitted names are unconstrained. A mismatch raises ``RESOURCE_CONFLICT``
                and rolls back the entire batch before any result is committed.
                Stores that cannot enforce these preconditions atomically, including the
                REST store, must reject a nonempty map before writing.

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

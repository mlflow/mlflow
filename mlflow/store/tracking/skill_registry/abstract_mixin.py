from __future__ import annotations

from mlflow.entities.skill import RegistryIcon, Skill
from mlflow.entities.skill_version import SkillVersion
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT

NOT_SET = object()


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
    ) -> PagedList[Skill]:
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
    ) -> SkillVersion:
        raise NotImplementedError(self.__class__.__name__)

    def get_skill_version(
        self,
        name: str,
        version: int,
        organization: str = "",
    ) -> SkillVersion:
        raise NotImplementedError(self.__class__.__name__)

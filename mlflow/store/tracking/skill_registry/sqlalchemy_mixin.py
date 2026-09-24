from __future__ import annotations

import logging

import sqlalchemy as sa
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import subqueryload

from mlflow.entities.skill import VALID_SKILL_STATUS_TRANSITIONS, RegistryIcon, Skill, SkillStatus
from mlflow.entities.skill_source import SkillSourceType
from mlflow.entities.skill_version import SkillVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_CONFLICT,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.tracking.dbmodels.models import (
    SqlAgentPluginVersion,
    SqlAgentPluginVersionMember,
    SqlSkill,
    SqlSkillAlias,
    SqlSkillVersion,
)
from mlflow.store.tracking.skill_registry.abstract_mixin import NOT_SET
from mlflow.store.tracking.skill_registry.artifact_paths import owned_skill_upload_path
from mlflow.store.tracking.skill_registry.constants import (
    SKILL_VERSION_DIGEST_LENGTH,
    SKILL_VERSION_SOURCE_MAX_LENGTH,
)
from mlflow.store.tracking.skill_registry_pagination import validate_max_results
from mlflow.utils.search_utils import SearchUtils
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
    _validate_alias_name,
    _validate_alias_name_reserved,
    _validate_organization_name,
    _validate_skill_name,
    _validate_skill_version,
)

_logger = logging.getLogger(__name__)


class SqlAlchemySkillRegistryMixin:
    """SQLAlchemy implementation of the Skill Registry store interface."""

    CREATE_SKILL_VERSION_RETRIES = 3
    MAX_REPORTED_BLOCKING_REFERENCES = 10

    def _skill_query(self, session):
        return SqlSkill.with_resolved_latest(
            self._get_query(session, SqlSkill).options(
                subqueryload(SqlSkill.tags),
                subqueryload(SqlSkill.skill_aliases),
            )
        )

    @staticmethod
    def _validate_skill_identity(name: str, organization: str) -> None:
        _validate_skill_name(name)
        _validate_organization_name(organization)

    @staticmethod
    def _validate_skill_version_source(
        source_type: str | None,
        source: str | None,
        ref: str | None,
        subpath: str | None,
        digest: str | None,
    ) -> None:
        try:
            parsed_source_type = SkillSourceType(source_type) if source_type is not None else None
        except (TypeError, ValueError) as e:
            raise MlflowException.invalid_parameter_value(
                f"Invalid Skill source type: {source_type!r}"
            ) from e

        for field_name, value in (
            ("source", source),
            ("ref", ref),
            ("subpath", subpath),
        ):
            if value is not None and (
                not isinstance(value, str) or len(value) > SKILL_VERSION_SOURCE_MAX_LENGTH
            ):
                raise MlflowException.invalid_parameter_value(
                    f"Skill version {field_name} must be a string of at most "
                    f"{SKILL_VERSION_SOURCE_MAX_LENGTH} characters."
                )

        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != SKILL_VERSION_DIGEST_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise MlflowException.invalid_parameter_value(
                "Skill version digest must be a "
                f"{SKILL_VERSION_DIGEST_LENGTH}-character lowercase hexadecimal string."
            )

        if ref is not None and parsed_source_type not in (None, SkillSourceType.GIT):
            raise MlflowException.invalid_parameter_value(
                "Skill version ref is only supported for Git sources."
            )

    @staticmethod
    def _validate_skill_version_status(status: str) -> str:
        try:
            parsed_status = SkillStatus(status)
        except (TypeError, ValueError) as e:
            raise MlflowException.invalid_parameter_value(
                f"Invalid SkillVersion status: {status!r}"
            ) from e

        if parsed_status not in (SkillStatus.ACTIVE, SkillStatus.DRAFT):
            raise MlflowException.invalid_parameter_value(
                "A newly created SkillVersion must have status 'active' or 'draft'."
            )
        return parsed_status.value

    def _assert_name_not_a_packaged_member(self, session, name: str, organization: str) -> None:
        """
        Refuse standalone creation for a name held by a member of a packaged agent plugin.

        A skill name is unique within its workspace and organization whether the skill was
        registered standalone or created by importing a package. A packaged plugin's members
        take their source from the package, so a standalone version registered onto one would
        break that binding; the RFC has standalone creation fail instead. Members of assembled
        plugins are ordinary standalone skills pinned by reference and stay unaffected.
        """
        workspace = self._with_workspace_field(SqlSkill(name=name, organization=organization))
        member = SqlAgentPluginVersionMember
        plugin_version = SqlAgentPluginVersion
        held_by = (
            session
            .query(plugin_version.organization, plugin_version.name, plugin_version.version)
            .join(
                member,
                (plugin_version.workspace == member.plugin_workspace)
                & (plugin_version.organization == member.plugin_organization)
                & (plugin_version.name == member.plugin_name)
                & (plugin_version.version == member.plugin_version),
            )
            .filter(
                member.plugin_workspace == workspace.workspace,
                member.member_organization == organization,
                member.member_name == name,
                plugin_version.source_type.isnot(None),
                plugin_version.source_type != SkillSourceType.ASSEMBLED.value,
            )
            .order_by(plugin_version.organization, plugin_version.name, plugin_version.version)
            .first()
        )
        if held_by is not None:
            plugin_organization, plugin_name, version = held_by
            plugin = (f"@{plugin_organization}/" if plugin_organization else "") + plugin_name
            raise MlflowException(
                f"Skill '{name}' in organization '{organization}' is a member of the packaged "
                f"agent plugin '{plugin}' (version {version}) and cannot receive standalone "
                "versions; re-import the plugin to update it, or register under a different "
                "name or organization.",
                error_code=RESOURCE_ALREADY_EXISTS,
            )

    def create_skill(
        self,
        name: str,
        organization: str = "",
        description: str | None = None,
        icons: list[RegistryIcon] | None = None,
        created_by: str | None = None,
    ) -> Skill:
        self._validate_skill_identity(name, organization)
        now = get_current_time_millis()
        with self.ManagedSessionMaker(read_only=False) as session:
            self._assert_name_not_a_packaged_member(session, name, organization)
            skill = self._with_workspace_field(
                SqlSkill(
                    name=name,
                    organization=organization,
                    description=description,
                    icons=icons,
                    created_by=created_by,
                    last_updated_by=created_by,
                    created_at=now,
                    last_updated_at=now,
                )
            )
            session.add(skill)
            try:
                session.flush()
            except IntegrityError as e:
                raise MlflowException(
                    f"Skill '{name}' already exists in organization '{organization}'",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from e
            return skill.to_mlflow_entity()

    def get_skill(self, name: str, organization: str = "") -> Skill:
        self._validate_skill_identity(name, organization)
        with self.ManagedSessionMaker() as session:
            skill = (
                self
                ._skill_query(session)
                .filter(SqlSkill.name == name, SqlSkill.organization == organization)
                .one_or_none()
            )
            if skill is None:
                raise MlflowException(
                    f"Skill '{name}' not found in organization '{organization}'",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            return skill.to_mlflow_entity()

    def update_skill(
        self,
        name: str,
        organization: str = "",
        description: str | None = NOT_SET,
        icons: list[RegistryIcon] | None = NOT_SET,
        last_updated_by: str | None = None,
    ) -> Skill:
        self._validate_skill_identity(name, organization)
        with self.ManagedSessionMaker(read_only=False) as session:
            skill = (
                self
                ._skill_query(session)
                .filter(SqlSkill.name == name, SqlSkill.organization == organization)
                .one_or_none()
            )
            if skill is None:
                raise MlflowException(
                    f"Skill '{name}' not found in organization '{organization}'",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            if description is not NOT_SET:
                skill.description = description
            if icons is not NOT_SET:
                skill.icons = icons
            skill.last_updated_by = last_updated_by
            skill.last_updated_at = get_current_time_millis()
            session.flush()
            skill = (
                self
                ._skill_query(session)
                .filter(SqlSkill.name == name, SqlSkill.organization == organization)
                .one()
            )
            return skill.to_mlflow_entity()

    def delete_skill(self, name: str, organization: str = "") -> None:
        self.delete_skill_and_collect_artifacts(name, organization)

    def delete_skill_and_collect_artifacts(self, name: str, organization: str = "") -> list[str]:
        self._validate_skill_identity(name, organization)
        with self.ManagedSessionMaker(read_only=False) as session:
            # Lock the parent row before reading anything, so the versions captured below are
            # exactly the ones the cascade will remove: a concurrent registration's version
            # insert takes a key-share lock on this row and waits until the delete commits,
            # then either fails or recreates the skill from version 1. ``FOR UPDATE`` is the
            # lock strength that conflicts with that foreign-key lock on PostgreSQL (a plain
            # non-key update only takes ``FOR NO KEY UPDATE``, which does not) and on MySQL.
            skill = (
                self
                ._get_query(session, SqlSkill)
                .filter(SqlSkill.name == name, SqlSkill.organization == organization)
                .with_for_update()
                .one_or_none()
            )
            if skill is None:
                raise MlflowException(
                    f"Skill '{name}' not found in organization '{organization}'",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            # SQLite ignores ``FOR UPDATE``; a no-op write starts its write transaction, which
            # serializes writers. On SQL Server the update's exclusive lock is what blocks the
            # insert's shared lock, since ``UPDLOCK`` alone would not.
            self._get_query(session, SqlSkill).filter(
                SqlSkill.name == name, SqlSkill.organization == organization
            ).update(
                {SqlSkill.last_updated_at: SqlSkill.last_updated_at}, synchronize_session=False
            )
            self._purge_stale_skill_memberships(session, skill)
            owned_paths = [
                path
                for version in skill.skill_versions
                if (
                    path := owned_skill_upload_path(
                        name=version.name,
                        organization=version.organization,
                        source_type=version.source_type,
                        source=version.source,
                        subpath=version.subpath,
                    )
                )
            ]
            session.delete(skill)
            try:
                session.flush()
            except IntegrityError as e:
                # A plugin version that started referencing the skill after the check above
                # is caught by the membership foreign key instead.
                raise MlflowException(
                    f"Skill '{name}' became referenced by an agent plugin version while it was "
                    "being deleted; nothing was removed. Retry the delete.",
                    error_code=RESOURCE_CONFLICT,
                ) from e
            # The session commits when this block exits; the paths are only handed back
            # once the rows are gone, so a rolled-back delete never reclaims anything.
        return owned_paths

    def _purge_stale_skill_memberships(self, session, skill: SqlSkill) -> None:
        """
        Fail if a live agent plugin version still contains one of the skill's versions, and
        otherwise remove the membership rows held only by soft-deleted plugin versions.

        Both steps run before anything else is removed. The stale rows have to go first
        because the membership foreign key blocks deletion of a referenced skill version.
        """
        member = SqlAgentPluginVersionMember
        plugin_version = SqlAgentPluginVersion
        memberships = (
            session
            .query(member, plugin_version.status)
            .join(
                plugin_version,
                (plugin_version.workspace == member.plugin_workspace)
                & (plugin_version.organization == member.plugin_organization)
                & (plugin_version.name == member.plugin_name)
                & (plugin_version.version == member.plugin_version),
            )
            .filter(
                member.plugin_workspace == skill.workspace,
                member.member_organization == skill.organization,
                member.member_name == skill.name,
            )
            .order_by(
                member.plugin_organization,
                member.plugin_name,
                member.plugin_version,
                member.member_version,
            )
            .all()
        )
        if live := [row for row, status in memberships if status != SkillStatus.DELETED.value]:
            shown = ", ".join(
                (f"@{row.plugin_organization}/" if row.plugin_organization else "")
                + f"{row.plugin_name}/{row.plugin_version} (skill version {row.member_version})"
                for row in live[: self.MAX_REPORTED_BLOCKING_REFERENCES]
            )
            remaining = len(live) - self.MAX_REPORTED_BLOCKING_REFERENCES
            more = f", and {remaining} more" if remaining > 0 else ""
            raise MlflowException(
                f"Skill '{skill.name}' cannot be deleted while live agent plugin versions "
                f"contain it: {shown}{more}. Delete or re-version those plugins first.",
                error_code=RESOURCE_CONFLICT,
            )
        for row, _ in memberships:
            session.delete(row)
        session.flush()

    def search_skills(
        self,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[Skill]:
        if filter_string is not None or order_by is not None:
            raise MlflowException(
                "Skill search filters and custom ordering are not supported yet",
                error_code=INVALID_PARAMETER_VALUE,
            )
        validate_max_results(max_results)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        with self.ManagedSessionMaker() as session:
            query = self._skill_query(session).order_by(
                SqlSkill.organization.asc(), SqlSkill.name.asc()
            )
            rows = query.offset(offset).limit(max_results + 1).all()
            skills = [skill.to_mlflow_entity() for skill in rows]
            next_token = None
            if len(skills) > max_results:
                next_token = SearchUtils.create_page_token(offset + max_results)
            return PagedList(skills[:max_results], token=next_token)

    def _get_or_create_skill_for_version(
        self,
        session,
        name: str,
        organization: str,
        created_by: str | None = None,
    ) -> SqlSkill:
        skill = (
            self
            ._get_query(session, SqlSkill)
            .filter(SqlSkill.name == name, SqlSkill.organization == organization)
            .one_or_none()
        )
        if skill is not None:
            return skill

        skill = self._with_workspace_field(
            SqlSkill(
                name=name,
                organization=organization,
                created_by=created_by,
                last_updated_by=created_by,
            )
        )
        try:
            session.add(skill)
            session.flush()
        except IntegrityError as e:
            raise MlflowException(
                f"Skill '{name}' already exists in organization '{organization}'",
                error_code=RESOURCE_ALREADY_EXISTS,
            ) from e
        return skill

    def _persist_skill_version(
        self,
        session,
        name: str,
        organization: str,
        version: int,
        source_type: str | None = None,
        source: str | None = None,
        ref: str | None = None,
        subpath: str | None = None,
        digest: str | None = None,
        status: str = SkillStatus.ACTIVE.value,
        created_by: str | None = None,
    ) -> SkillVersion:
        self._validate_skill_identity(name, organization)
        self._validate_skill_version_source(source_type, source, ref, subpath, digest)
        _validate_skill_version(version)
        status = self._validate_skill_version_status(status)

        skill = self._get_or_create_skill_for_version(
            session, name, organization, created_by=created_by
        )
        now = get_current_time_millis()
        skill_version = SqlSkillVersion(
            workspace=skill.workspace,
            organization=organization,
            name=name,
            version=version,
            source_type=source_type,
            source=source,
            ref=ref,
            subpath=subpath,
            digest=digest,
            status=status,
            created_by=created_by,
            last_updated_by=created_by,
            created_at=now,
            last_updated_at=now,
        )
        session.add(skill_version)
        try:
            session.flush()
        except IntegrityError as e:
            raise MlflowException(
                f"Skill version '{name}' version '{version}' already exists",
                error_code=RESOURCE_ALREADY_EXISTS,
            ) from e
        return skill_version.to_mlflow_entity()

    def create_skill_version(
        self,
        name: str,
        organization: str = "",
        source_type: str | None = None,
        source: str | None = None,
        ref: str | None = None,
        subpath: str | None = None,
        digest: str | None = None,
        status: str = SkillStatus.ACTIVE.value,
        created_by: str | None = None,
    ) -> SkillVersion:
        self._validate_skill_identity(name, organization)
        self._validate_skill_version_source(source_type, source, ref, subpath, digest)
        self._validate_skill_version_status(status)
        # Checked once, ahead of the retry loop: the loop treats RESOURCE_ALREADY_EXISTS as a
        # version-number collision, and a bound name is not something a retry can resolve.
        with self.ManagedSessionMaker() as session:
            self._assert_name_not_a_packaged_member(session, name, organization)
        for attempt in range(self.CREATE_SKILL_VERSION_RETRIES):
            try:
                with self.ManagedSessionMaker(read_only=False) as session:
                    max_version = (
                        self
                        ._get_query(session, SqlSkillVersion)
                        .with_entities(func.max(SqlSkillVersion.version))
                        .filter(
                            SqlSkillVersion.name == name,
                            SqlSkillVersion.organization == organization,
                        )
                        .scalar()
                    )
                    version = (max_version or 0) + 1
                    return self._persist_skill_version(
                        session=session,
                        name=name,
                        organization=organization,
                        version=version,
                        source_type=source_type,
                        source=source,
                        ref=ref,
                        subpath=subpath,
                        digest=digest,
                        status=status,
                        created_by=created_by,
                    )
            except MlflowException as e:
                if e.error_code != ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
                    raise
                more_retries = self.CREATE_SKILL_VERSION_RETRIES - attempt - 1
                _logger.info(
                    "Skill version creation conflict (name=%s, organization=%s); "
                    "retrying %s more time%s.",
                    name,
                    organization,
                    more_retries,
                    "s" if more_retries != 1 else "",
                )

        raise MlflowException(
            f"Skill version creation error (name={name}, organization={organization}). "
            f"Giving up after {self.CREATE_SKILL_VERSION_RETRIES} attempts.",
            error_code=RESOURCE_ALREADY_EXISTS,
        )

    def get_skill_version(
        self,
        name: str,
        version: int,
        organization: str = "",
    ) -> SkillVersion:
        self._validate_skill_identity(name, organization)
        _validate_skill_version(version)
        with self.ManagedSessionMaker() as session:
            skill_version = (
                self
                ._get_query(session, SqlSkillVersion)
                .filter(
                    SqlSkillVersion.name == name,
                    SqlSkillVersion.organization == organization,
                    SqlSkillVersion.version == version,
                    SqlSkillVersion.status != SkillStatus.DELETED.value,
                )
                .one_or_none()
            )
            if skill_version is None:
                raise MlflowException(
                    f"Skill version '{name}' version '{version}' not found",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            return skill_version.to_mlflow_entity()

    # --- Skill version lifecycle operations ---

    SET_SKILL_ALIAS_RETRIES = 3

    def _skill_version_query(self, session, *, include_deleted=False, load_version_tags=True):
        query = self._get_query(session, SqlSkillVersion)
        if not include_deleted:
            query = query.filter(SqlSkillVersion.status != SkillStatus.DELETED.value)
        if load_version_tags:
            query = query.options(subqueryload(SqlSkillVersion.version_tags))
        return query

    def _get_skill_version_or_raise(
        self,
        session,
        name: str,
        version: int,
        organization: str,
        *,
        include_deleted: bool = False,
        load_version_tags: bool = True,
        columns_only: bool = False,
    ) -> SqlSkillVersion:
        query = self._skill_version_query(
            session,
            include_deleted=include_deleted,
            load_version_tags=load_version_tags and not columns_only,
        ).filter(
            SqlSkillVersion.name == name,
            SqlSkillVersion.organization == organization,
            SqlSkillVersion.version == version,
        )
        if columns_only:
            query = query.with_entities(
                SqlSkillVersion.workspace,
                SqlSkillVersion.organization,
                SqlSkillVersion.name,
                SqlSkillVersion.version,
                SqlSkillVersion.status,
            )
        skill_version = query.one_or_none()
        if skill_version is None:
            raise MlflowException(
                f"Skill version '{name}' version '{version}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return skill_version

    @staticmethod
    def _validate_skill_status_transition(current: SkillStatus, new: SkillStatus) -> None:
        allowed = VALID_SKILL_STATUS_TRANSITIONS.get(current, set())
        if new not in allowed:
            raise MlflowException(
                f"Invalid status transition from '{current}' to '{new}'. "
                f"Allowed transitions: {sorted(str(status) for status in allowed)}",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def update_skill_version(
        self,
        name: str,
        version: int,
        organization: str = "",
        status: SkillStatus | str | None = NOT_SET,
        last_updated_by: str | None = None,
    ):
        self._validate_skill_identity(name, organization)
        _validate_skill_version(version)
        if status is None:
            raise MlflowException.invalid_parameter_value(
                "status cannot be null; omit the field to leave it unchanged"
            )

        with self.ManagedSessionMaker(read_only=False) as session:
            skill_version = self._get_skill_version_or_raise(session, name, version, organization)
            current_status = SkillStatus(skill_version.status)
            if status is NOT_SET:
                return skill_version.to_mlflow_entity()

            try:
                new_status = SkillStatus(status)
            except (TypeError, ValueError) as e:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid SkillVersion status: {status!r}"
                ) from e
            self._validate_skill_status_transition(current_status, new_status)

            now = get_current_time_millis()
            updated = (
                self
                ._get_query(session, SqlSkillVersion)
                .filter(
                    SqlSkillVersion.name == name,
                    SqlSkillVersion.organization == organization,
                    SqlSkillVersion.version == version,
                    SqlSkillVersion.status == current_status.value,
                )
                .update(
                    {
                        SqlSkillVersion.status: new_status.value,
                        SqlSkillVersion.last_updated_by: last_updated_by,
                        SqlSkillVersion.last_updated_at: now,
                    },
                    synchronize_session=False,
                )
            )
            if updated != 1:
                raise MlflowException(
                    "Skill version changed while being updated; retry the operation",
                    error_code=RESOURCE_CONFLICT,
                )

            session.expire(
                skill_version,
                attribute_names=["status", "last_updated_by", "last_updated_at"],
            )
            if new_status is SkillStatus.DELETED:
                self._delete_skill_aliases_for_version(session, skill_version)
                return skill_version.to_mlflow_entity(alias_names=[])
            return skill_version.to_mlflow_entity()

    def delete_skill_version(
        self,
        name: str,
        version: int,
        organization: str = "",
        last_updated_by: str | None = None,
    ) -> None:
        self._validate_skill_identity(name, organization)
        _validate_skill_version(version)
        with self.ManagedSessionMaker(read_only=False) as session:
            skill_version = self._get_skill_version_or_raise(
                session,
                name,
                version,
                organization,
                include_deleted=True,
                columns_only=True,
            )
            current_status = SkillStatus(skill_version.status)
            if current_status is SkillStatus.DELETED:
                raise MlflowException(
                    f"Skill version '{name}' version '{version}' not found",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )

            now = get_current_time_millis()
            updated = (
                self
                ._get_query(session, SqlSkillVersion)
                .filter(
                    SqlSkillVersion.name == name,
                    SqlSkillVersion.organization == organization,
                    SqlSkillVersion.version == version,
                    SqlSkillVersion.status == current_status.value,
                )
                .update(
                    {
                        SqlSkillVersion.status: SkillStatus.DELETED.value,
                        SqlSkillVersion.last_updated_by: last_updated_by,
                        SqlSkillVersion.last_updated_at: now,
                    },
                    synchronize_session=False,
                )
            )
            if updated != 1:
                raise MlflowException(
                    "Skill version changed while being deleted; retry the operation",
                    error_code=RESOURCE_CONFLICT,
                )

            self._delete_skill_aliases_for_version(session, skill_version)

    def _resolve_latest_skill_version(self, session, name: str, organization: str):
        self._get_entity_or_raise(
            session,
            SqlSkill,
            {"name": name, "organization": organization},
            "Skill",
        )
        skill_version = self._latest_resolved_skill_version_query(
            session, name, organization
        ).first()
        if skill_version is None:
            raise MlflowException(
                f"No resolved latest version found for skill '{name}'",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return skill_version

    def _latest_resolved_skill_version_query(self, session, name: str, organization: str):
        status_priority = sa.case(
            (SqlSkillVersion.status == SkillStatus.ACTIVE.value, 0),
            else_=1,
        )
        return (
            self
            ._skill_version_query(session)
            .filter(
                SqlSkillVersion.name == name,
                SqlSkillVersion.organization == organization,
            )
            .order_by(status_priority.asc(), SqlSkillVersion.version.desc())
        )

    def get_latest_skill_version(self, name: str, organization: str = ""):
        self._validate_skill_identity(name, organization)
        with self.ManagedSessionMaker() as session:
            return self._resolve_latest_skill_version(
                session, name, organization
            ).to_mlflow_entity()

    # --- Skill alias operations ---

    def get_skill_version_by_alias(self, name: str, alias: str, organization: str = ""):
        self._validate_skill_identity(name, organization)
        if isinstance(alias, str) and alias.lower() == "latest":
            return self.get_latest_skill_version(name, organization)
        _validate_alias_name(alias, resource_type="Skill")
        _validate_alias_name_reserved(alias)

        with self.ManagedSessionMaker() as session:
            alias_row = (
                self
                ._get_query(session, SqlSkillAlias)
                .filter(
                    SqlSkillAlias.name == name,
                    SqlSkillAlias.organization == organization,
                    SqlSkillAlias.alias == alias,
                )
                .one_or_none()
            )
            if alias_row is None:
                raise MlflowException(
                    f"Alias '{alias}' not found for skill '{name}'",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            skill_version = (
                self
                ._skill_version_query(session)
                .filter(
                    SqlSkillVersion.name == name,
                    SqlSkillVersion.organization == organization,
                    SqlSkillVersion.version == alias_row.version,
                )
                .one_or_none()
            )
            if skill_version is None:
                raise MlflowException(
                    f"Alias '{alias}' for skill '{name}' points to a missing or deleted version",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            return skill_version.to_mlflow_entity()

    def set_skill_alias(self, name: str, alias: str, version: int, organization: str = "") -> None:
        self._validate_skill_identity(name, organization)
        _validate_skill_version(version)
        _validate_alias_name(alias, resource_type="Skill")
        _validate_alias_name_reserved(alias)
        for attempt in range(self.SET_SKILL_ALIAS_RETRIES):
            try:
                with self.ManagedSessionMaker(read_only=False) as session:
                    skill_version = (
                        self
                        ._get_query(session, SqlSkillVersion)
                        .filter(
                            SqlSkillVersion.name == name,
                            SqlSkillVersion.organization == organization,
                            SqlSkillVersion.version == version,
                        )
                        .one_or_none()
                    )
                    if skill_version is None or skill_version.status == SkillStatus.DELETED.value:
                        raise MlflowException(
                            f"Skill version '{name}' version '{version}' not found or is deleted",
                            error_code=RESOURCE_DOES_NOT_EXIST,
                        )

                    # Acquire the version row's write lock before checking aliases. This
                    # serializes alias creation with a concurrent soft delete on databases
                    # that support row locks and starts a write transaction on SQLite.
                    locked = (
                        self
                        ._get_query(session, SqlSkillVersion)
                        .filter(
                            SqlSkillVersion.name == name,
                            SqlSkillVersion.organization == organization,
                            SqlSkillVersion.version == version,
                            SqlSkillVersion.status != SkillStatus.DELETED.value,
                        )
                        .update(
                            {SqlSkillVersion.status: SqlSkillVersion.status},
                            synchronize_session=False,
                        )
                    )
                    if locked != 1:
                        raise MlflowException(
                            f"Skill version '{name}' version '{version}' not found or is deleted",
                            error_code=RESOURCE_DOES_NOT_EXIST,
                        )

                    alias_row = (
                        self
                        ._get_query(session, SqlSkillAlias)
                        .filter(
                            SqlSkillAlias.name == name,
                            SqlSkillAlias.organization == organization,
                            SqlSkillAlias.alias == alias,
                        )
                        .one_or_none()
                    )
                    if alias_row is None:
                        alias_row = self._with_workspace_field(
                            SqlSkillAlias(
                                name=name,
                                organization=organization,
                                alias=alias,
                                version=version,
                            )
                        )
                        session.add(alias_row)
                    else:
                        alias_row.version = version
                    session.flush()
                    return
            except MlflowException as e:
                if not isinstance(e.__cause__, IntegrityError):
                    raise
                if attempt == self.SET_SKILL_ALIAS_RETRIES - 1:
                    raise

    def delete_skill_alias(self, name: str, alias: str, organization: str = "") -> None:
        self._validate_skill_identity(name, organization)
        if isinstance(alias, str) and alias.lower() == "latest":
            raise MlflowException(
                "The 'latest' alias is resolved automatically and cannot be deleted",
                error_code=INVALID_PARAMETER_VALUE,
            )
        _validate_alias_name(alias, resource_type="Skill")
        _validate_alias_name_reserved(alias)
        with self.ManagedSessionMaker(read_only=False) as session:
            alias_row = (
                self
                ._get_query(session, SqlSkillAlias)
                .filter(
                    SqlSkillAlias.name == name,
                    SqlSkillAlias.organization == organization,
                    SqlSkillAlias.alias == alias,
                )
                .one_or_none()
            )
            if alias_row is None:
                raise MlflowException(
                    f"Alias '{alias}' not found on skill '{name}'",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            session.delete(alias_row)

    @staticmethod
    def _delete_skill_aliases_for_version(session, skill_version: SqlSkillVersion) -> None:
        session.query(SqlSkillAlias).filter(
            SqlSkillAlias.workspace == skill_version.workspace,
            SqlSkillAlias.organization == skill_version.organization,
            SqlSkillAlias.name == skill_version.name,
            SqlSkillAlias.version == skill_version.version,
        ).delete(synchronize_session=False)

from __future__ import annotations

import logging

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from mlflow.entities.skill import RegistryIcon, Skill, SkillStatus
from mlflow.entities.skill_source import SkillSourceType
from mlflow.entities.skill_version import SkillVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.tracking.dbmodels.models import SqlSkill, SqlSkillVersion
from mlflow.store.tracking.skill_registry.abstract_mixin import NOT_SET
from mlflow.utils.search_utils import SearchUtils
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
    _validate_organization_name,
    _validate_skill_name,
    _validate_skill_version,
)

_logger = logging.getLogger(__name__)


class SqlAlchemySkillRegistryMixin:
    """SQLAlchemy implementation of the Skill Registry store interface."""

    CREATE_SKILL_VERSION_RETRIES = 3

    def _skill_query(self, session):
        return SqlSkill.with_resolved_latest(
            self._get_query(session, SqlSkill).options(
                selectinload(SqlSkill.tags),
                selectinload(SqlSkill.skill_aliases),
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
            if value is not None and (not isinstance(value, str) or len(value) > 2048):
                raise MlflowException.invalid_parameter_value(
                    f"Skill version {field_name} must be a string of at most 2048 characters."
                )

        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise MlflowException.invalid_parameter_value(
                "Skill version digest must be a 64-character lowercase hexadecimal string."
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
            return skill.to_mlflow_entity()

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
        self._validate_max_results_param(max_results)
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

    def _get_or_create_skill_for_version(self, session, name: str, organization: str) -> SqlSkill:
        skill = (
            self
            ._get_query(session, SqlSkill)
            .filter(SqlSkill.name == name, SqlSkill.organization == organization)
            .one_or_none()
        )
        if skill is not None:
            return skill

        skill = self._with_workspace_field(SqlSkill(name=name, organization=organization))
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
    ) -> SkillVersion:
        self._validate_skill_identity(name, organization)
        self._validate_skill_version_source(source_type, source, ref, subpath, digest)
        _validate_skill_version(version)
        status = self._validate_skill_version_status(status)

        skill = self._get_or_create_skill_for_version(session, name, organization)
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
    ) -> SkillVersion:
        self._validate_skill_identity(name, organization)
        self._validate_skill_version_source(source_type, source, ref, subpath, digest)
        self._validate_skill_version_status(status)
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

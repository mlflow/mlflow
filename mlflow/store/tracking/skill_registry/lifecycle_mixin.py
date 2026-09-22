from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import subqueryload

from mlflow.entities.skill import (
    VALID_SKILL_STATUS_TRANSITIONS,
    SkillStatus,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_CONFLICT,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.tracking.dbmodels.models import SqlSkill, SqlSkillAlias, SqlSkillVersion
from mlflow.store.tracking.skill_registry.abstract_mixin import NOT_SET
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import _validate_skill_alias, _validate_skill_version


class SqlAlchemySkillRegistryLifecycleMixin:
    """SQLAlchemy lifecycle and alias operations for the Skill Registry."""

    SET_SKILL_ALIAS_RETRIES = 3

    @staticmethod
    def _validate_skill_alias_name(alias: str) -> None:
        try:
            _validate_skill_alias(alias)
        except MlflowException as e:
            message = str(e).replace("Registered model alias name", "Skill alias name")
            raise MlflowException.invalid_parameter_value(message) from e

    def _skill_version_query(self, session):
        return self._get_query(session, SqlSkillVersion).options(
            subqueryload(SqlSkillVersion.version_tags),
            subqueryload(SqlSkillVersion.skill).subqueryload(SqlSkill.skill_aliases),
        )

    def _get_skill_version_or_raise(
        self,
        session,
        name: str,
        version: int,
        organization: str,
    ) -> SqlSkillVersion:
        skill_version = (
            self
            ._skill_version_query(session)
            .filter(
                SqlSkillVersion.name == name,
                SqlSkillVersion.organization == organization,
                SqlSkillVersion.version == version,
            )
            .one_or_none()
        )
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
            if status is NOT_SET:
                return skill_version.to_mlflow_entity()

            try:
                new_status = SkillStatus(status)
            except (TypeError, ValueError) as e:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid SkillVersion status: {status!r}"
                ) from e
            current_status = SkillStatus(skill_version.status)
            if new_status is current_status:
                return skill_version.to_mlflow_entity()
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

            skill_version.status = new_status.value
            skill_version.last_updated_by = last_updated_by
            skill_version.last_updated_at = now
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
            skill_version = self._get_skill_version_or_raise(session, name, version, organization)
            current_status = SkillStatus(skill_version.status)
            if current_status is SkillStatus.DELETED:
                return
            self._validate_skill_status_transition(current_status, SkillStatus.DELETED)

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

            skill_version.status = SkillStatus.DELETED.value
            skill_version.last_updated_by = last_updated_by
            skill_version.last_updated_at = now
            self._delete_skill_aliases_for_version(session, skill_version)

    def _resolve_latest_skill_version(self, session, name: str, organization: str):
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

        if skill.resolved_latest_version is None:
            raise MlflowException(
                f"No resolved latest version found for skill '{name}'",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        skill_version = (
            self
            ._skill_version_query(session)
            .filter(
                SqlSkillVersion.name == name,
                SqlSkillVersion.organization == organization,
                SqlSkillVersion.version == skill.resolved_latest_version,
            )
            .one_or_none()
        )
        if skill_version is None:
            raise MlflowException(
                f"No resolved latest version found for skill '{name}'",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return skill_version

    def get_latest_skill_version(self, name: str, organization: str = ""):
        self._validate_skill_identity(name, organization)
        with self.ManagedSessionMaker() as session:
            return self._resolve_latest_skill_version(
                session, name, organization
            ).to_mlflow_entity()

    def get_skill_version_by_alias(self, name: str, alias: str, organization: str = ""):
        self._validate_skill_identity(name, organization)
        if isinstance(alias, str) and alias.lower() == "latest":
            return self.get_latest_skill_version(name, organization)
        self._validate_skill_alias_name(alias)

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
                    SqlSkillVersion.status != SkillStatus.DELETED.value,
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
        self._validate_skill_alias_name(alias)
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
                f"Alias '{alias}' not found on skill '{name}'",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        self._validate_skill_alias_name(alias)
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

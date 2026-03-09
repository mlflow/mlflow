import json
import logging
from datetime import datetime, timezone

_logger = logging.getLogger(__name__)

OAUTH_MANAGED_PASSWORD = "__OAUTH_MANAGED__"


class UserProvisioner:
    def __init__(self, store, oauth_config):
        self._store = store
        self._config = oauth_config

    def provision_user(
        self,
        username: str,
        provider_config,
        groups: list[str] | None = None,
    ) -> tuple[int, bool]:
        if not self._config.auto_provision_users:
            if not self._store.has_user(username):
                raise PermissionError(
                    f"User {username} not found and auto-provisioning is disabled"
                )

        is_admin = self._check_admin(groups or [], provider_config)
        role_mappings = self._config.parse_role_mappings(provider_config.role_mappings)
        default_permission = self._resolve_permission(groups or [], role_mappings)

        if self._store.has_user(username):
            user = self._store.get_user(username)
            if is_admin and not user.is_admin:
                self._store.update_user(username, is_admin=True)
            elif not is_admin and user.is_admin:
                # Only demote if admin was IdP-granted (not the config admin)
                if username != self._config.auth_config.admin_username:
                    self._store.update_user(username, is_admin=False)
            user_id = user.id
            _logger.info("Linked existing user %s via OAuth", username)
        else:
            user = self._store.create_user(
                username=username,
                password=OAUTH_MANAGED_PASSWORD,
                is_admin=is_admin,
            )
            user_id = user.id
            _logger.info("Auto-provisioned new user %s via OAuth", username)

        self._update_role_override(user_id, default_permission, groups or [])
        return user_id, is_admin

    def _check_admin(self, groups: list[str], provider_config) -> bool:
        admin_groups = self._config.parse_admin_groups(provider_config.admin_groups)
        if not admin_groups:
            return False
        return any(g in admin_groups for g in groups)

    def _resolve_permission(self, groups: list[str], role_mappings: dict[str, str]) -> str:
        permission_order = {"NO_PERMISSIONS": 0, "READ": 1, "USE": 2, "EDIT": 3, "MANAGE": 4}
        best_permission = self._config.auth_config.default_permission
        best_rank = permission_order.get(best_permission, 0)

        for group in groups:
            if group in role_mappings:
                perm = role_mappings[group]
                rank = permission_order.get(perm, 0)
                if rank > best_rank:
                    best_permission = perm
                    best_rank = rank

        return best_permission

    def _update_role_override(self, user_id: int, default_permission: str, groups: list[str]):
        # Access the session maker through the store's engine
        from sqlalchemy.orm import Session

        from mlflow.server.auth.oauth.db.models import SqlUserRoleOverride

        engine = self._store.engine
        with Session(engine) as db:
            override = (
                db.query(SqlUserRoleOverride).filter(SqlUserRoleOverride.user_id == user_id).first()
            )

            now = datetime.now(timezone.utc)
            if override:
                override.default_permission = default_permission
                override.idp_groups = json.dumps(groups)
                override.last_synced_at = now
            else:
                override = SqlUserRoleOverride(
                    user_id=user_id,
                    default_permission=default_permission,
                    idp_groups=json.dumps(groups),
                    last_synced_at=now,
                )
                db.add(override)
            db.commit()

    def get_user_default_permission(self, user_id: int) -> str | None:
        from sqlalchemy.orm import Session

        from mlflow.server.auth.oauth.db.models import SqlUserRoleOverride

        engine = self._store.engine
        with Session(engine) as db:
            override = (
                db.query(SqlUserRoleOverride).filter(SqlUserRoleOverride.user_id == user_id).first()
            )
            return override.default_permission if override else None

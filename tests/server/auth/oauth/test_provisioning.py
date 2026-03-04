import json
from dataclasses import dataclass

import pytest
import sqlalchemy
from sqlalchemy.orm import Session as DbSession

from mlflow.server.auth.oauth.db.models import Base as OAuthBase
from mlflow.server.auth.oauth.db.models import SqlUserRoleOverride
from mlflow.server.auth.oauth.provisioning import UserProvisioner


@dataclass
class FakeProviderConfig:
    name: str = "primary"
    role_mappings: str = "readers:READ, editors:EDIT, managers:MANAGE"
    admin_groups: str = "admins"


class FakeAuthConfig:
    default_permission: str = "READ"
    admin_username: str = "admin"


class FakeOAuthConfig:
    auto_provision_users: bool = True
    auth_config: FakeAuthConfig | None = None

    def __init__(self):
        self.auth_config = FakeAuthConfig()

    def parse_role_mappings(self, mappings_str: str) -> dict[str, str]:
        if not mappings_str:
            return {}
        result = {}
        for pair in mappings_str.split(","):
            pair = pair.strip()
            if ":" in pair:
                group, perm = pair.rsplit(":", 1)
                result[group.strip()] = perm.strip()
        return result

    def parse_admin_groups(self, admin_groups_str: str) -> list[str]:
        if not admin_groups_str:
            return []
        return [g.strip() for g in admin_groups_str.split(",") if g.strip()]


@dataclass
class FakeUser:
    id: int = 1
    username: str = "jane"
    is_admin: bool = False


class FakeStore:
    def __init__(self, engine):
        self.engine = engine
        self._users = {}
        self._next_id = 1

    def has_user(self, username):
        return username in self._users

    def get_user(self, username):
        return self._users[username]

    def create_user(self, username, password, is_admin=False):
        user = FakeUser(id=self._next_id, username=username, is_admin=is_admin)
        self._next_id += 1
        self._users[username] = user
        return user

    def update_user(self, username, is_admin):
        self._users[username].is_admin = is_admin


@pytest.fixture
def db_engine():
    engine = sqlalchemy.create_engine("sqlite:///:memory:")
    OAuthBase.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def store(db_engine):
    return FakeStore(db_engine)


@pytest.fixture
def provisioner(store):
    return UserProvisioner(store, FakeOAuthConfig())


def test_user_provisioner_provision_new_user(provisioner, store):
    provider = FakeProviderConfig()
    user_id, is_admin = provisioner.provision_user(
        username="jane",
        provider_config=provider,
        groups=["readers"],
    )
    assert store.has_user("jane")
    assert is_admin is False
    assert user_id == 1


def test_user_provisioner_provision_existing_user(provisioner, store):
    store.create_user("jane", "password")
    provider = FakeProviderConfig()
    user_id, is_admin = provisioner.provision_user(
        username="jane",
        provider_config=provider,
        groups=["readers"],
    )
    assert user_id == 1


def test_user_provisioner_admin_group_sets_admin(provisioner, store):
    provider = FakeProviderConfig()
    user_id, is_admin = provisioner.provision_user(
        username="jane",
        provider_config=provider,
        groups=["admins"],
    )
    assert is_admin is True
    assert store.get_user("jane").is_admin is True


def test_user_provisioner_highest_permission_wins(provisioner, store, db_engine):
    provider = FakeProviderConfig()
    provisioner.provision_user(
        username="jane",
        provider_config=provider,
        groups=["readers", "managers"],
    )

    with DbSession(db_engine) as db:
        override = db.query(SqlUserRoleOverride).first()
        assert override.default_permission == "MANAGE"


def test_user_provisioner_role_override_updated_on_each_login(provisioner, store, db_engine):
    provider = FakeProviderConfig()

    provisioner.provision_user(
        username="jane",
        provider_config=provider,
        groups=["readers"],
    )

    with DbSession(db_engine) as db:
        override = db.query(SqlUserRoleOverride).first()
        assert override.default_permission == "READ"

    provisioner.provision_user(
        username="jane",
        provider_config=provider,
        groups=["editors"],
    )

    with DbSession(db_engine) as db:
        override = db.query(SqlUserRoleOverride).first()
        assert override.default_permission == "EDIT"


def test_user_provisioner_no_matching_groups(provisioner, store, db_engine):
    provider = FakeProviderConfig()
    provisioner.provision_user(
        username="jane",
        provider_config=provider,
        groups=["unrelated-group"],
    )

    with DbSession(db_engine) as db:
        override = db.query(SqlUserRoleOverride).first()
        # Falls back to default_permission from auth_config
        assert override.default_permission == "READ"


def test_user_provisioner_empty_groups(provisioner, store):
    provider = FakeProviderConfig()
    user_id, is_admin = provisioner.provision_user(
        username="jane",
        provider_config=provider,
        groups=[],
    )
    assert is_admin is False


def test_user_provisioner_idp_groups_stored_as_json(provisioner, store, db_engine):
    provider = FakeProviderConfig()
    provisioner.provision_user(
        username="jane",
        provider_config=provider,
        groups=["readers", "editors"],
    )

    with DbSession(db_engine) as db:
        override = db.query(SqlUserRoleOverride).first()
        groups = json.loads(override.idp_groups)
        assert "readers" in groups
        assert "editors" in groups


def test_user_provisioner_get_user_default_permission(provisioner, store, db_engine):
    provider = FakeProviderConfig()
    provisioner.provision_user(
        username="jane",
        provider_config=provider,
        groups=["editors"],
    )

    user = store.get_user("jane")
    perm = provisioner.get_user_default_permission(user.id)
    assert perm == "EDIT"


def test_user_provisioner_auto_provision_disabled(store):
    config = FakeOAuthConfig()
    config.auto_provision_users = False
    provisioner = UserProvisioner(store, config)
    provider = FakeProviderConfig()

    with pytest.raises(PermissionError, match="not found and auto-provisioning is disabled"):
        provisioner.provision_user(
            username="unknown",
            provider_config=provider,
            groups=[],
        )

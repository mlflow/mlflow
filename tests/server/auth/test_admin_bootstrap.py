import logging
from pathlib import Path
from unittest import mock

import pytest

from mlflow.environment_variables import (
    _MLFLOW_AUTH_ADMIN_BOOTSTRAPPED,
    MLFLOW_AUTH_ADMIN_PASSWORD,
    MLFLOW_AUTH_ADMIN_USERNAME,
)
from mlflow.exceptions import MlflowException
from mlflow.server import auth as auth_module
from mlflow.server.auth.config import read_auth_config

_PACKAGED_BASIC_AUTH_INI = Path(auth_module.__file__).parent / "basic_auth.ini"
_LEGACY_DEFAULT_PASSWORD = "password1234"


@pytest.fixture(autouse=True)
def _clear_admin_env(monkeypatch):
    monkeypatch.delenv(MLFLOW_AUTH_ADMIN_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_AUTH_ADMIN_PASSWORD.name, raising=False)
    monkeypatch.delenv(_MLFLOW_AUTH_ADMIN_BOOTSTRAPPED.name, raising=False)


def _write_ini(tmp_path: Path, *, admin_password: str | None = None) -> Path:
    lines = [
        "[mlflow]",
        "default_permission = READ",
        f"database_uri = sqlite:///{tmp_path / 'auth.db'}",
        "admin_username = admin",
    ]
    if admin_password is not None:
        lines.append(f"admin_password = {admin_password}")
    path = tmp_path / "auth.ini"
    path.write_text("\n".join(lines) + "\n")
    return path


def _read_config(path: Path):
    with mock.patch("mlflow.server.auth.config._get_auth_config_path", return_value=str(path)):
        return read_auth_config()


def test_packaged_config_ships_no_admin_password():
    config = _read_config(_PACKAGED_BASIC_AUTH_INI)
    assert config.admin_username == "admin"
    assert config.admin_password is None
    assert _LEGACY_DEFAULT_PASSWORD not in _PACKAGED_BASIC_AUTH_INI.read_text()


@pytest.mark.parametrize(
    ("ini_password", "env_password", "expected"),
    [
        (None, None, None),
        ("", None, None),
        ("ini-admin-password", None, "ini-admin-password"),
        (None, "env-admin-password", "env-admin-password"),
        ("ini-admin-password", "env-admin-password", "env-admin-password"),
        # A set-but-empty variable wins over the file so the misconfiguration is not masked.
        ("ini-admin-password", "", None),
    ],
)
def test_admin_password_resolution(tmp_path, monkeypatch, ini_password, env_password, expected):
    if env_password is not None:
        monkeypatch.setenv(MLFLOW_AUTH_ADMIN_PASSWORD.name, env_password)
    config = _read_config(_write_ini(tmp_path, admin_password=ini_password))
    assert config.admin_password == expected


@pytest.mark.parametrize(
    ("env_username", "expected"),
    [
        ("root", "root"),
        # A set-but-empty variable wins over the file and normalizes to None, like the password.
        ("", None),
    ],
)
def test_admin_username_env_overrides_ini(tmp_path, monkeypatch, env_username, expected):
    monkeypatch.setenv(MLFLOW_AUTH_ADMIN_USERNAME.name, env_username)
    assert _read_config(_write_ini(tmp_path)).admin_username == expected


@pytest.fixture
def store():
    with mock.patch.object(auth_module, "store") as store:
        yield store


@pytest.mark.parametrize("username", [None, ""])
def test_create_admin_user_requires_username(store, username):
    with pytest.raises(MlflowException, match="needs a username for the admin user") as exc_info:
        auth_module.create_admin_user(username, "a-strong-admin-password")
    assert MLFLOW_AUTH_ADMIN_USERNAME.name in str(exc_info.value)
    store.has_user.assert_not_called()
    store.create_user.assert_not_called()


@pytest.mark.parametrize(
    ("password", "match"),
    [
        (None, "needs a password to create the admin user"),
        ("", "needs a password to create the admin user"),
        ("short", "at least 12 characters"),
    ],
)
def test_create_admin_user_rejects_missing_or_short_password(store, password, match):
    store.has_user.return_value = False
    with pytest.raises(MlflowException, match=match) as exc_info:
        auth_module.create_admin_user("admin", password)
    # Every misconfiguration names the setting to fix.
    assert MLFLOW_AUTH_ADMIN_PASSWORD.name in str(exc_info.value)
    store.create_user.assert_not_called()


def test_create_admin_user_rejects_legacy_default_password(store):
    store.has_user.return_value = False
    with pytest.raises(MlflowException, match="insecure default password"):
        auth_module.create_admin_user("admin", _LEGACY_DEFAULT_PASSWORD)
    store.create_user.assert_not_called()


def test_create_admin_user_creates_admin_with_configured_password(store):
    store.has_user.return_value = False
    store.authenticate_user.return_value = False
    auth_module.create_admin_user("admin", "a-strong-admin-password")
    store.has_user.assert_called_once_with("admin", use_primary=True)
    store.create_user.assert_called_once_with("admin", "a-strong-admin-password", is_admin=True)


def test_create_admin_user_skips_bootstrap_when_admin_exists(store, caplog):
    store.has_user.return_value = True
    store.authenticate_user.return_value = False
    with caplog.at_level(logging.WARNING, logger=auth_module.__name__):
        # Missing / legacy configured passwords are irrelevant once the admin exists.
        auth_module.create_admin_user("admin", None)
        auth_module.create_admin_user("admin", _LEGACY_DEFAULT_PASSWORD)
    store.create_user.assert_not_called()
    assert not caplog.records


def test_create_admin_user_warns_when_existing_admin_uses_legacy_default(store, caplog):
    store.has_user.return_value = True
    store.authenticate_user.return_value = True
    with caplog.at_level(logging.WARNING, logger=auth_module.__name__):
        auth_module.create_admin_user("admin", None)
    # Reads go to the primary so replica lag cannot hide an existing admin during a restart.
    store.authenticate_user.assert_called_once_with(
        "admin", _LEGACY_DEFAULT_PASSWORD, use_primary=True
    )
    store.create_user.assert_not_called()
    store.update_user.assert_not_called()
    assert any(
        "user 'admin' still uses the insecure default password" in r.message for r in caplog.records
    )


def test_create_admin_user_rotates_legacy_default_password_when_password_configured(store, caplog):
    store.has_user.return_value = True
    # The stored credential is the legacy default until update_user replaces it.
    store.authenticate_user.side_effect = lambda *_, **__: not store.update_user.called
    with caplog.at_level(logging.WARNING, logger=auth_module.__name__):
        auth_module.create_admin_user("admin", "a-strong-admin-password")
    store.update_user.assert_called_once_with("admin", password="a-strong-admin-password")
    store.create_user.assert_not_called()
    messages = [r.message for r in caplog.records]
    assert len(messages) == 1
    assert "Replaced the insecure default password of admin user 'admin'" in messages[0]


def test_create_admin_user_refuses_to_rotate_to_a_short_password(store):
    store.has_user.return_value = True
    store.authenticate_user.return_value = True
    with pytest.raises(MlflowException, match="at least 12 characters"):
        auth_module.create_admin_user("admin", "short")
    store.update_user.assert_not_called()


def test_create_admin_user_keeps_starting_with_a_stale_legacy_ini_password(store, caplog):
    # An upgraded deployment that copied the old ini still has `admin_password = password1234`
    # next to an admin row on that password. That is not a rotation request and must not fail
    # startup; the login block and the warning cover it.
    store.has_user.return_value = True
    store.authenticate_user.return_value = True
    with caplog.at_level(logging.WARNING, logger=auth_module.__name__):
        auth_module.create_admin_user("admin", _LEGACY_DEFAULT_PASSWORD)
    store.update_user.assert_not_called()
    store.create_user.assert_not_called()
    messages = [r.message for r in caplog.records]
    assert len(messages) == 1
    assert "user 'admin' still uses the insecure default password" in messages[0]


def test_create_admin_user_leaves_a_rotated_password_alone(store):
    store.has_user.return_value = True
    store.authenticate_user.return_value = False
    auth_module.create_admin_user("admin", "a-strong-admin-password")
    store.update_user.assert_not_called()
    store.create_user.assert_not_called()


@pytest.mark.parametrize("root_exists", [True, False])
def test_create_admin_user_warns_about_legacy_admin_when_username_overridden(
    store, caplog, root_exists
):
    # Overriding the bootstrap username on an upgraded deployment must not hide the historical
    # `admin` account that still carries the publicly known password.
    store.has_user.return_value = root_exists
    store.authenticate_user.side_effect = lambda username, password, **_: username == "admin"
    with caplog.at_level(logging.WARNING, logger=auth_module.__name__):
        auth_module.create_admin_user("root", "a-strong-admin-password")
    assert store.create_user.call_count == (0 if root_exists else 1)
    # The configured password only rotates the configured bootstrap admin, never another user.
    store.update_user.assert_not_called()
    messages = [r.message for r in caplog.records]
    assert len(messages) == 1
    assert "user 'admin' still uses the insecure default password" in messages[0]


def test_bootstrap_admin_user_initializes_store_then_creates_admin(store, monkeypatch):
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(
            database_uri="sqlite:///auth.db",
            read_database_uri=None,
            admin_username="admin",
            admin_password=None,
        ),
    )
    store.has_user.return_value = False
    with pytest.raises(MlflowException, match="needs a password"):
        auth_module.bootstrap_admin_user()
    store.init_db.assert_called_once_with("sqlite:///auth.db", read_db_uri=None)
    store.create_user.assert_not_called()


@pytest.mark.parametrize("already_bootstrapped", [True, False])
def test_init_store_for_app_skips_bootstrap_when_cli_already_did_it(
    store, monkeypatch, already_bootstrapped
):
    if already_bootstrapped:
        monkeypatch.setenv(_MLFLOW_AUTH_ADMIN_BOOTSTRAPPED.name, "true")
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(
            database_uri="sqlite:///auth.db",
            read_database_uri=None,
            admin_username="admin",
            admin_password="a-strong-admin-password",
        ),
    )
    store.has_user.return_value = False
    store.authenticate_user.return_value = False
    auth_module._init_store_for_app()
    store.init_db.assert_called_once_with("sqlite:///auth.db", read_db_uri=None)
    if already_bootstrapped:
        store.has_user.assert_not_called()
        store.authenticate_user.assert_not_called()
        store.create_user.assert_not_called()
    else:
        store.create_user.assert_called_once_with("admin", "a-strong-admin-password", is_admin=True)

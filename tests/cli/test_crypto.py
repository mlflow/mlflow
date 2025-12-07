import logging
from contextlib import contextmanager
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow.cli.crypto import commands
from mlflow.exceptions import MlflowException


@pytest.fixture(autouse=True)
def suppress_logging():
    original_root = logging.root.level
    original_mlflow = logging.getLogger("mlflow").level
    original_alembic = logging.getLogger("alembic").level

    logging.root.setLevel(logging.CRITICAL)
    logging.getLogger("mlflow").setLevel(logging.CRITICAL)
    logging.getLogger("alembic").setLevel(logging.CRITICAL)

    yield

    logging.root.setLevel(original_root)
    logging.getLogger("mlflow").setLevel(original_mlflow)
    logging.getLogger("alembic").setLevel(original_alembic)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def old_passphrase_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", "old-passphrase")


@pytest.fixture
def mock_session():
    session = mock.MagicMock()
    session.__enter__ = mock.Mock(return_value=session)
    session.__exit__ = mock.Mock(return_value=False)
    return session


@pytest.fixture
def mock_store(mock_session):
    store = mock.Mock()
    store.ManagedSessionMaker.return_value = mock_session
    return store


@pytest.fixture
def mock_secret():
    secret = mock.Mock()
    secret.secret_id = "test-secret-id-123"
    secret.encrypted_value = b"encrypted-data"
    secret.wrapped_dek = b"wrapped-dek-data"
    secret.kek_version = 1
    return secret


@pytest.fixture
def empty_db(mock_session):
    mock_session.query.return_value.filter.return_value.all.return_value = []


@pytest.fixture
def db_with_secret(mock_session, mock_secret):
    mock_session.query.return_value.filter.return_value.all.return_value = [mock_secret]


@contextmanager
def patch_backend(mock_store):
    mock_sql_secret = mock.Mock()
    with (
        mock.patch("mlflow.cli.crypto._get_store", return_value=mock_store),
        mock.patch.dict(
            "sys.modules",
            {"mlflow.store.tracking.dbmodels.models": mock.Mock(SqlGatewaySecret=mock_sql_secret)},
        ),
    ):
        yield mock_sql_secret


@contextmanager
def patch_rotation(return_value=None):
    result = mock.Mock()
    result.wrapped_dek = b"new-wrapped-dek"
    with mock.patch(
        "mlflow.cli.crypto.rotate_secret_encryption",
        return_value=return_value or result,
    ):
        yield


def test_crypto_group_exists():
    assert commands.name == "crypto"
    assert commands.help is not None
    assert "cryptographic" in commands.help.lower()


def test_rotate_kek_command_exists():
    rotate_cmd = next((cmd for cmd in commands.commands.values() if cmd.name == "rotate-kek"), None)
    assert rotate_cmd is not None
    assert "rotate" in rotate_cmd.help.lower()


def test_rotate_kek_has_required_parameters():
    rotate_cmd = next((cmd for cmd in commands.commands.values() if cmd.name == "rotate-kek"), None)
    param_names = [p.name for p in rotate_cmd.params]
    assert "new_passphrase" in param_names
    assert "backend_store_uri" in param_names
    assert "yes" in param_names


def test_new_passphrase_is_required():
    rotate_cmd = next((cmd for cmd in commands.commands.values() if cmd.name == "rotate-kek"), None)
    new_pass_param = next((p for p in rotate_cmd.params if p.name == "new_passphrase"), None)
    assert new_pass_param.required
    assert new_pass_param.prompt
    assert new_pass_param.hide_input
    assert new_pass_param.confirmation_prompt


def test_yes_flag_is_optional():
    rotate_cmd = next((cmd for cmd in commands.commands.values() if cmd.name == "rotate-kek"), None)
    yes_param = next((p for p in rotate_cmd.params if p.name == "yes"), None)
    assert yes_param.is_flag
    assert yes_param.default is False


def test_missing_old_passphrase_raises_error(runner, monkeypatch):
    monkeypatch.delenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", raising=False)
    result = runner.invoke(commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"])
    assert result.exit_code != 0
    assert result.exception is not None
    assert "MLFLOW_CRYPTO_KEK_PASSPHRASE" in str(result.exception)


def test_old_passphrase_from_env(runner, old_passphrase_env, mock_store, empty_db):
    with patch_backend(mock_store):
        result = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"]
        )
        assert result.exit_code == 0
        assert "No secrets found" in result.output


def test_kek_version_defaults_to_1(runner, old_passphrase_env, mock_store, empty_db, monkeypatch):
    monkeypatch.delenv("MLFLOW_CRYPTO_KEK_VERSION", raising=False)
    with patch_backend(mock_store), mock.patch("mlflow.cli.crypto.KEKManager") as mock_kek:
        runner.invoke(commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"])
        assert mock_kek.call_args_list[0][1]["kek_version"] == 1


def test_kek_version_from_env(runner, mock_store, empty_db):
    with patch_backend(mock_store), mock.patch("mlflow.cli.crypto.os.getenv") as mock_getenv:

        def getenv_side_effect(key, default=None):
            if key == "MLFLOW_CRYPTO_KEK_PASSPHRASE":
                return "old-passphrase"
            elif key == "MLFLOW_CRYPTO_KEK_VERSION":
                return "5"
            return default

        mock_getenv.side_effect = getenv_side_effect
        result = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"]
        )
        assert result.exit_code == 0


def test_version_increments_correctly(runner, mock_store, empty_db):
    with patch_backend(mock_store), mock.patch("mlflow.cli.crypto.os.getenv") as mock_getenv:

        def getenv_side_effect(key, default=None):
            if key == "MLFLOW_CRYPTO_KEK_PASSPHRASE":
                return "old-passphrase"
            elif key == "MLFLOW_CRYPTO_KEK_VERSION":
                return "3"
            return default

        mock_getenv.side_effect = getenv_side_effect
        result = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"]
        )
        assert result.exit_code == 0


def test_interactive_prompt_shows_warning(runner, old_passphrase_env, mock_store, empty_db):
    with patch_backend(mock_store):
        result = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase"], input="n\n"
        )
        assert "⚠️  WARNING: KEK Rotation Operation" in result.output
        assert "Re-wrap all encryption DEKs" in result.output
        assert "MLFLOW_CRYPTO_KEK_PASSPHRASE" in result.output
        assert "MLFLOW_CRYPTO_KEK_VERSION" in result.output
        assert "Continue with KEK rotation?" in result.output


def test_yes_flag_skips_confirmation(runner, old_passphrase_env, mock_store, empty_db):
    with patch_backend(mock_store):
        result = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"]
        )
        assert "⚠️  WARNING" not in result.output
        assert "Continue with KEK rotation?" not in result.output


def test_cancellation_exits_gracefully(runner, old_passphrase_env, mock_store, empty_db):
    with patch_backend(mock_store):
        result = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase"], input="n\n"
        )
        assert result.exit_code == 0
        assert "KEK rotation cancelled" in result.output


def test_connects_to_backend_store(runner, old_passphrase_env, mock_store, empty_db):
    with patch_backend(mock_store):
        result = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"]
        )
        assert result.exit_code == 0


def test_uses_custom_backend_store_uri(runner, old_passphrase_env, mock_store, empty_db):
    with patch_backend(mock_store):
        result = runner.invoke(
            commands,
            [
                "rotate-kek",
                "--new-passphrase",
                "new-passphrase",
                "--backend-store-uri",
                "sqlite:///test.db",
                "--yes",
            ],
        )
        assert result.exit_code == 0


def test_filters_secrets_by_kek_version(
    runner, old_passphrase_env, mock_store, mock_secret, monkeypatch
):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_VERSION", "2")
    mock_session = mock_store.ManagedSessionMaker.return_value
    mock_session.query.return_value.filter.return_value.all.return_value = [mock_secret]
    with patch_backend(mock_store), patch_rotation():
        runner.invoke(commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"])
        assert mock_session.query.return_value.filter.call_args is not None


def test_commits_transaction_on_success(runner, old_passphrase_env, mock_store, db_with_secret):
    with patch_backend(mock_store), patch_rotation():
        runner.invoke(commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"])
        mock_store.ManagedSessionMaker.return_value.commit.assert_called_once()


def test_no_secrets_returns_success(runner, old_passphrase_env, mock_store, empty_db):
    with patch_backend(mock_store):
        result = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"]
        )
        assert result.exit_code == 0
        assert "No secrets found" in result.output
        assert "Nothing to rotate" in result.output


def test_wrong_old_passphrase_fails(runner, old_passphrase_env, mock_store, db_with_secret):
    with (
        patch_backend(mock_store),
        mock.patch(
            "mlflow.cli.crypto.rotate_secret_encryption",
            side_effect=MlflowException("Failed to rotate secret encryption"),
        ),
    ):
        result = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"]
        )
        assert result.exit_code != 0
        assert "Failed to rotate encryption key" in result.output


def test_rotation_failure_rolls_back(runner, old_passphrase_env, mock_store, db_with_secret):
    with (
        patch_backend(mock_store),
        mock.patch(
            "mlflow.cli.crypto.rotate_secret_encryption",
            side_effect=Exception("Rotation failed"),
        ),
    ):
        result = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"]
        )
        assert result.exit_code != 0
        mock_store.ManagedSessionMaker.return_value.rollback.assert_called_once()
        assert "No changes were made" in str(result.exception)


def test_database_connection_error(runner, old_passphrase_env):
    mock_sql_secret = mock.Mock()
    with (
        mock.patch("mlflow.cli.crypto._get_store", side_effect=Exception("Connection failed")),
        mock.patch.dict(
            "sys.modules",
            {"mlflow.store.tracking.dbmodels.models": mock.Mock(SqlGatewaySecret=mock_sql_secret)},
        ),
    ):
        result = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"]
        )
        assert result.exit_code != 0
        assert "Failed to connect to backend store" in str(result.exception)


def test_kek_manager_creation_error(runner, old_passphrase_env, mock_store):
    with (
        patch_backend(mock_store),
        mock.patch("mlflow.cli.crypto.KEKManager", side_effect=Exception("KEK creation failed")),
    ):
        result = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"]
        )
        assert result.exit_code != 0
        assert "Failed to create KEK managers" in str(result.exception)


def test_shows_progress_for_multiple_secrets(runner, old_passphrase_env, mock_store, mock_session):
    secrets = [
        mock.Mock(
            secret_id=f"secret-{i}", encrypted_value=b"enc", wrapped_dek=b"wrap", kek_version=1
        )
        for i in range(5)
    ]
    mock_session.query.return_value.filter.return_value.all.return_value = secrets
    with patch_backend(mock_store), patch_rotation():
        result = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"]
        )
        assert result.exit_code == 0
        assert "Found 5 secrets to rotate" in result.output


def test_success_message_includes_version_info(runner, mock_store, mock_session):
    secret = mock.Mock(secret_id="test", encrypted_value=b"enc", wrapped_dek=b"wrap", kek_version=3)
    mock_session.query.return_value.filter.return_value.all.return_value = [secret]
    with (
        patch_backend(mock_store),
        patch_rotation(),
        mock.patch("mlflow.cli.crypto.os.getenv") as mock_getenv,
    ):

        def getenv_side_effect(key, default=None):
            if key == "MLFLOW_CRYPTO_KEK_PASSPHRASE":
                return "old-passphrase"
            elif key == "MLFLOW_CRYPTO_KEK_VERSION":
                return "3"
            return default

        mock_getenv.side_effect = getenv_side_effect
        result = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"]
        )
        assert result.exit_code == 0
        assert "Successfully rotated 1 encryption key" in result.output


def test_shows_environment_variable_warning(runner, old_passphrase_env, mock_store, db_with_secret):
    with patch_backend(mock_store), patch_rotation():
        result = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"]
        )
        assert result.exit_code == 0
        assert "CRITICAL: Update BOTH environment variables" in result.output
        assert "MLFLOW_CRYPTO_KEK_PASSPHRASE='<new-passphrase>'" in result.output
        assert "MLFLOW_CRYPTO_KEK_VERSION='2'" in result.output
        assert "Failure to update BOTH variables will cause decryption failures" in result.output


def test_large_number_of_secrets(runner, old_passphrase_env, mock_store, mock_session):
    secrets = [
        mock.Mock(
            secret_id=f"secret-{i}", encrypted_value=b"enc", wrapped_dek=b"wrap", kek_version=1
        )
        for i in range(1000)
    ]
    mock_session.query.return_value.filter.return_value.all.return_value = secrets
    with patch_backend(mock_store), patch_rotation():
        result = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"]
        )
        assert result.exit_code == 0
        assert "Found 1000 secrets to rotate" in result.output
        assert "Successfully rotated 1000 encryption keys" in result.output


def test_mixed_kek_versions_only_rotates_old_version(
    runner, old_passphrase_env, mock_store, mock_session, monkeypatch
):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_VERSION", "1")
    secret = mock.Mock(
        secret_id="secret-v1", encrypted_value=b"enc", wrapped_dek=b"wrap", kek_version=1
    )
    mock_session.query.return_value.filter.return_value.all.return_value = [secret]
    with patch_backend(mock_store), patch_rotation():
        result = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"]
        )
        assert result.exit_code == 0
        assert "Found 1 secrets to rotate" in result.output


def test_rotation_with_special_characters_in_passphrase(runner, mock_store, empty_db, monkeypatch):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", "old-p@$$phrase!#$%")
    with patch_backend(mock_store):
        result = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-p@$$phrase!#$%", "--yes"]
        )
        assert result.exit_code == 0


def test_idempotency_running_twice_with_same_version(
    runner, old_passphrase_env, mock_store, empty_db
):
    with patch_backend(mock_store):
        result1 = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase", "--yes"]
        )
        result2 = runner.invoke(
            commands, ["rotate-kek", "--new-passphrase", "new-passphrase-2", "--yes"]
        )
        assert result1.exit_code == 0
        assert result2.exit_code == 0
        assert "No secrets found" in result1.output
        assert "No secrets found" in result2.output

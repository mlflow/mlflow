from unittest.mock import Mock

from cryptography.fernet import Fernet

from mlflow.secrets.validation import SecretKeyValidator, ValidationResult, log_validation_results


def test_validation_result_empty():
    result = ValidationResult()
    assert not result.has_errors()
    assert not result.has_warnings()
    assert len(result.errors) == 0
    assert len(result.warnings) == 0
    assert len(result.info) == 0


def test_validation_result_with_errors():
    result = ValidationResult()
    result.add_error("Test error")
    assert result.has_errors()
    assert len(result.errors) == 1
    assert result.errors[0] == "Test error"


def test_validation_result_with_warnings():
    result = ValidationResult()
    result.add_warning("Test warning")
    assert result.has_warnings()
    assert len(result.warnings) == 1
    assert result.warnings[0] == "Test warning"


def test_detect_key_source_temporary(monkeypatch):
    monkeypatch.delenv("MLFLOW_SECRET_MASTER_KEY", raising=False)
    monkeypatch.delenv("MLFLOW_SECRET_MASTER_KEY_FILE", raising=False)

    validator = SecretKeyValidator()
    assert validator._detect_key_source() == "temporary"


def test_detect_key_source_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_SECRET_MASTER_KEY", "test_key")
    monkeypatch.delenv("MLFLOW_SECRET_MASTER_KEY_FILE", raising=False)

    validator = SecretKeyValidator()
    assert validator._detect_key_source() == "env"


def test_detect_key_source_file(tmp_path, monkeypatch):
    key_file = tmp_path / "master.key"
    key_file.write_text("test_key")

    monkeypatch.delenv("MLFLOW_SECRET_MASTER_KEY", raising=False)
    monkeypatch.setenv("MLFLOW_SECRET_MASTER_KEY_FILE", str(key_file))

    validator = SecretKeyValidator()
    assert validator._detect_key_source() == "file"


def test_detect_key_source_both(tmp_path, monkeypatch):
    key_file = tmp_path / "master.key"
    key_file.write_text("test_key")

    monkeypatch.setenv("MLFLOW_SECRET_MASTER_KEY", "test_key")
    monkeypatch.setenv("MLFLOW_SECRET_MASTER_KEY_FILE", str(key_file))

    validator = SecretKeyValidator()
    assert validator._detect_key_source() == "both"


def test_validate_at_startup_temporary_key(monkeypatch):
    monkeypatch.delenv("MLFLOW_SECRET_MASTER_KEY", raising=False)
    monkeypatch.delenv("MLFLOW_SECRET_MASTER_KEY_FILE", raising=False)

    validator = SecretKeyValidator()
    result = validator.validate_at_startup()

    assert not result.has_errors()
    assert result.has_warnings()
    assert any("temporary" in w.lower() for w in result.warnings)


def test_validate_at_startup_both_keys(tmp_path, monkeypatch):
    key_file = tmp_path / "master.key"
    key_file.write_text(Fernet.generate_key().decode())

    monkeypatch.setenv("MLFLOW_SECRET_MASTER_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("MLFLOW_SECRET_MASTER_KEY_FILE", str(key_file))

    validator = SecretKeyValidator()
    result = validator.validate_at_startup()

    assert result.has_errors()
    assert any("both" in e.lower() for e in result.errors)


def test_validate_at_startup_env_key(monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("MLFLOW_SECRET_MASTER_KEY", key)
    monkeypatch.delenv("MLFLOW_SECRET_MASTER_KEY_FILE", raising=False)

    validator = SecretKeyValidator()
    result = validator.validate_at_startup()

    assert not result.has_errors()
    assert any("environment variable" in i.lower() for i in result.info)


def test_validate_file_permissions_secure(tmp_path, monkeypatch):
    key_file = tmp_path / "master.key"
    key = Fernet.generate_key().decode()
    key_file.write_text(key)
    key_file.chmod(0o600)

    monkeypatch.delenv("MLFLOW_SECRET_MASTER_KEY", raising=False)
    monkeypatch.setenv("MLFLOW_SECRET_MASTER_KEY_FILE", str(key_file))

    validator = SecretKeyValidator()
    result = validator.validate_at_startup()

    assert not result.has_errors()
    assert not result.has_warnings()
    assert any("600" in i and "secure" in i.lower() for i in result.info)


def test_validate_file_permissions_insecure(tmp_path, monkeypatch):
    key_file = tmp_path / "master.key"
    key = Fernet.generate_key().decode()
    key_file.write_text(key)
    key_file.chmod(0o644)

    monkeypatch.delenv("MLFLOW_SECRET_MASTER_KEY", raising=False)
    monkeypatch.setenv("MLFLOW_SECRET_MASTER_KEY_FILE", str(key_file))

    validator = SecretKeyValidator()
    result = validator.validate_at_startup()

    assert result.has_errors()
    assert any("insecure" in e.lower() for e in result.errors)


def test_validate_file_not_found(tmp_path, monkeypatch):
    key_file = tmp_path / "nonexistent.key"

    monkeypatch.delenv("MLFLOW_SECRET_MASTER_KEY", raising=False)
    monkeypatch.setenv("MLFLOW_SECRET_MASTER_KEY_FILE", str(key_file))

    validator = SecretKeyValidator()
    result = validator.validate_at_startup()

    assert result.has_errors()
    assert any("does not exist" in e.lower() for e in result.errors)


def test_encryption_roundtrip_test(tmp_path, monkeypatch):
    key_file = tmp_path / "master.key"
    key = Fernet.generate_key().decode()
    key_file.write_text(key)
    key_file.chmod(0o600)

    monkeypatch.delenv("MLFLOW_SECRET_MASTER_KEY", raising=False)
    monkeypatch.setenv("MLFLOW_SECRET_MASTER_KEY_FILE", str(key_file))

    validator = SecretKeyValidator()
    result = validator.validate_at_startup()

    assert not result.has_errors()
    assert any("roundtrip" in i.lower() and "passed" in i.lower() for i in result.info)


def test_log_validation_results_errors():
    result = ValidationResult()
    result.add_error("Test error")

    log_validation_results(result, mode="warn")

    assert result.has_errors()
    assert "Test error" in result.errors


def test_log_validation_results_warnings():
    result = ValidationResult()
    result.add_warning("Test warning")

    log_validation_results(result, mode="warn")

    assert result.has_warnings()
    assert "Test warning" in result.warnings


def test_log_validation_results_info():
    result = ValidationResult()
    result.add_info("Test info")

    log_validation_results(result, mode="warn")

    assert "Test info" in result.info


def test_validate_with_store_no_secrets(tmp_path, monkeypatch):
    key_file = tmp_path / "master.key"
    key = Fernet.generate_key().decode()
    key_file.write_text(key)
    key_file.chmod(0o600)

    monkeypatch.delenv("MLFLOW_SECRET_MASTER_KEY", raising=False)
    monkeypatch.setenv("MLFLOW_SECRET_MASTER_KEY_FILE", str(key_file))

    validator = SecretKeyValidator()

    mock_store = Mock()
    mock_store.list_secret_names = Mock(return_value=[])

    result = validator.validate_with_store(mock_store)

    assert not result.has_errors()
    mock_store.list_secret_names.assert_called_once()


def test_validate_with_store_with_secrets(tmp_path, monkeypatch):
    key_file = tmp_path / "master.key"
    key = Fernet.generate_key().decode()
    key_file.write_text(key)
    key_file.chmod(0o600)

    monkeypatch.delenv("MLFLOW_SECRET_MASTER_KEY", raising=False)
    monkeypatch.setenv("MLFLOW_SECRET_MASTER_KEY_FILE", str(key_file))

    validator = SecretKeyValidator()

    mock_store = Mock()
    mock_store.list_secret_names = Mock(return_value=["secret1", "secret2"])
    mock_secret = Mock()
    mock_store.get_secret = Mock(return_value=mock_secret)

    result = validator.validate_with_store(mock_store)

    assert not result.has_errors()
    assert any("validated" in i.lower() for i in result.info)
    mock_store.get_secret.assert_called_once()

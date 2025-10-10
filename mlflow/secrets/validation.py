import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

_logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def add_error(self, message: str):
        self.errors.append(message)

    def add_warning(self, message: str):
        self.warnings.append(message)

    def add_info(self, message: str):
        self.info.append(message)

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class SecretKeyValidator:
    def validate_at_startup(self):
        result = ValidationResult()

        key_source = self._detect_key_source()

        if key_source == "both":
            result.add_error(
                "Both MLFLOW_SECRET_MASTER_KEY and MLFLOW_SECRET_MASTER_KEY_FILE are set. "
                "Please set only one."
            )
            return result

        if key_source == "temporary":
            result.add_warning(
                "Using temporary auto-generated master key. "
                "Secrets will be lost on restart. "
                "Set MLFLOW_SECRET_MASTER_KEY_FILE for production."
            )
        elif key_source == "file":
            self._validate_file_permissions(result)
        elif key_source == "env":
            result.add_info("Using master key from MLFLOW_SECRET_MASTER_KEY environment variable")

        self._test_encryption_roundtrip(result)

        return result

    def _detect_key_source(self) -> str:
        env_key = os.environ.get("MLFLOW_SECRET_MASTER_KEY")
        key_file = os.environ.get("MLFLOW_SECRET_MASTER_KEY_FILE")

        if env_key and key_file:
            return "both"
        elif env_key:
            return "env"
        elif key_file:
            return "file"
        else:
            return "temporary"

    def _validate_file_permissions(self, result: ValidationResult):
        key_file = os.environ.get("MLFLOW_SECRET_MASTER_KEY_FILE")
        if not key_file:
            return

        key_file_path = Path(key_file).expanduser().resolve()

        if not key_file_path.exists():
            result.add_error(f"Key file does not exist: {key_file_path}")
            return

        try:
            stat_info = key_file_path.stat()
            mode = oct(stat_info.st_mode)[-3:]

            if mode in ("600", "400"):
                result.add_info(f"Key file permissions: {mode} (secure)")
            else:
                if stat_info.st_mode & 0o044:
                    result.add_error(
                        f"Key file has insecure permissions: {mode} (group/world readable). "
                        f"Fix: chmod 600 {key_file_path}"
                    )
                else:
                    result.add_warning(
                        f"Key file permissions: {mode}. Recommended: 600 (owner read/write) "
                        "or 400 (owner read-only)"
                    )

            fingerprint = self._get_key_fingerprint()
            if fingerprint:
                result.add_info(f"Key fingerprint: sha256:{fingerprint}")
        except Exception as e:
            result.add_error(f"Cannot read key file: {e}")

    def _get_key_fingerprint(self) -> str | None:
        try:
            env_key = os.environ.get("MLFLOW_SECRET_MASTER_KEY")
            key_file = os.environ.get("MLFLOW_SECRET_MASTER_KEY_FILE")

            if env_key:
                return hashlib.sha256(env_key.encode()).hexdigest()[:8]
            elif key_file:
                key_file_path = Path(key_file).expanduser().resolve()
                key_content = key_file_path.read_text().strip()
                return hashlib.sha256(key_content.encode()).hexdigest()[:8]
        except Exception:
            return None

    def _test_encryption_roundtrip(self, result: ValidationResult):
        try:
            from mlflow.secrets.crypto import SecretManager

            secret_manager = SecretManager()
            test_value = "validation_test_12345"
            encrypted = secret_manager.encrypt(test_value)
            decrypted = secret_manager.decrypt(encrypted)

            if decrypted == test_value:
                result.add_info("Encryption roundtrip test passed")
            else:
                result.add_error("Encryption roundtrip test failed: decrypted value doesn't match")
        except Exception as e:
            result.add_error(f"Encryption test failed: {e}")

    def validate_with_store(self, store):
        result = self.validate_at_startup()

        if hasattr(store, "list_secret_names") and hasattr(store, "get_secret"):
            try:
                from mlflow.secrets.scope import SecretScope

                global_secrets = store.list_secret_names(SecretScope.GLOBAL, None)

                if global_secrets and len(global_secrets) > 0:
                    try:
                        store.get_secret(global_secrets[0], SecretScope.GLOBAL, None)
                        result.add_info(
                            f"Successfully validated {len(global_secrets)} existing secrets"
                        )
                    except Exception as e:
                        result.add_error(
                            f"Cannot decrypt existing secrets. Wrong master key? Error: {e}"
                        )
            except Exception as e:
                result.add_warning(f"Could not validate existing secrets: {e}")

        return result


def log_validation_results(result: ValidationResult, mode: str = "warn"):
    if result.has_errors():
        _logger.error("=" * 80)
        _logger.error("SECRET KEY VALIDATION ERRORS:")
        for error in result.errors:
            _logger.error(f"  ✗ {error}")
        if mode == "strict":
            _logger.error("Server startup aborted due to validation errors")
        else:
            _logger.error("Secrets functionality may not work correctly!")
        _logger.error("=" * 80)

    if result.has_warnings():
        for warning in result.warnings:
            _logger.warning(f"  ⚠️  {warning}")

    if result.info:
        for info in result.info:
            _logger.info(f"  ✓ {info}")

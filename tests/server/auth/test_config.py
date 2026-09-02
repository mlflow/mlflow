import os
import tempfile
from pathlib import Path

import pytest

from mlflow.server.auth.config import AuthConfig, EnvInterpolation, read_auth_config


def test_env_interpolation():
    """Test the EnvInterpolation class for environment variable substitution."""
    # Create a temporary config file with environment variables
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
        f.write("""[mlflow]
default_permission = READ
database_uri = sqlite:///${DB_NAME}.db
admin_username = ${ADMIN_USER}
admin_password = ${ADMIN_PASS}
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
""")
        config_path = f.name

    try:
        # Set environment variables
        os.environ["DB_NAME"] = "test_db"
        os.environ["ADMIN_USER"] = "test_admin"
        os.environ["ADMIN_PASS"] = "test_password"

        # Read config and verify interpolation
        original_config_path = os.environ.get("MLFLOW_AUTH_CONFIG_PATH")
        os.environ["MLFLOW_AUTH_CONFIG_PATH"] = config_path

        try:
            config = read_auth_config()
            assert config.database_uri == "sqlite:///test_db.db"
            assert config.admin_username == "test_admin"
            assert config.admin_password == "test_password"
            assert config.default_permission == "READ"
            assert config.authorization_function == "mlflow.server.auth:authenticate_request_basic_auth"
        finally:
            # Restore original config path
            if original_config_path is not None:
                os.environ["MLFLOW_AUTH_CONFIG_PATH"] = original_config_path
            else:
                os.environ.pop("MLFLOW_AUTH_CONFIG_PATH", None)
    finally:
        # Clean up environment variables and temp file
        os.unlink(config_path)
        for var in ["DB_NAME", "ADMIN_USER", "ADMIN_PASS"]:
            os.environ.pop(var, None)


def test_env_interpolation_with_defaults():
    """Test EnvInterpolation with default values when environment variables are not set."""
    # Create a temporary config file with environment variables
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
        f.write("""[mlflow]
default_permission = READ
database_uri = sqlite:///basic_auth.db
admin_username = admin
admin_password = password
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
""")
        config_path = f.name

    try:
        # Don't set environment variables to test defaults
        original_config_path = os.environ.get("MLFLOW_AUTH_CONFIG_PATH")
        os.environ["MLFLOW_AUTH_CONFIG_PATH"] = config_path

        try:
            config = read_auth_config()
            assert config.default_permission == "READ"
            assert config.admin_username == "admin"
            assert config.admin_password == "password"
            assert config.database_uri == "sqlite:///basic_auth.db"
            assert config.authorization_function == "mlflow.server.auth:authenticate_request_basic_auth"
        finally:
            # Restore original config path
            if original_config_path is not None:
                os.environ["MLFLOW_AUTH_CONFIG_PATH"] = original_config_path
            else:
                os.environ.pop("MLFLOW_AUTH_CONFIG_PATH", None)
    finally:
        os.unlink(config_path)


def test_auth_config_named_tuple():
    """Test that AuthConfig is properly created as a NamedTuple."""
    config = AuthConfig(
        default_permission="READ",
        database_uri="sqlite:///test.db",
        admin_username="admin",
        admin_password="password",
        authorization_function="mlflow.server.auth:authenticate_request_basic_auth"
    )

    # Verify it's a NamedTuple
    assert isinstance(config, tuple)
    assert hasattr(config, "_fields")
    assert "default_permission" in config._fields
    assert "database_uri" in config._fields
    assert "admin_username" in config._fields
    assert "admin_password" in config._fields
    assert "authorization_function" in config._fields

    # Verify values
    assert config.default_permission == "READ"
    assert config.database_uri == "sqlite:///test.db"
    assert config.admin_username == "admin"
    assert config.admin_password == "password"
    assert config.authorization_function == "mlflow.server.auth:authenticate_request_basic_auth"

    # Verify we can access fields by name (NamedTuple feature)
    assert config.default_permission == config[0]
    assert config.database_uri == config[1]
    assert config.admin_username == config[2]
    assert config.admin_password == config[3]
    assert config.authorization_function == config[4]


def test_read_auth_config_default():
    """Test reading the default auth configuration."""
    # This test will use the default basic_auth.ini file
    config = read_auth_config()

    # Verify default values from basic_auth.ini
    assert config.default_permission == "READ"
    assert config.database_uri == "sqlite:///basic_auth.db"
    assert config.admin_username == "admin"
    assert config.admin_password == "password1234"
    assert config.authorization_function == "mlflow.server.auth:authenticate_request_basic_auth"


def test_env_interpolation_edge_cases():
    """Test edge cases for environment variable interpolation."""
    # Create a temporary config file with edge cases
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
        f.write("""[mlflow]
default_permission = ${EMPTY_VAR}
database_uri = sqlite:///${SPECIAL_CHARS}.db
admin_username = user${NUMERIC}name
admin_password = ${MIXED_CONTENT}
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
""")
        config_path = f.name

    try:
        # Set environment variables with edge cases
        os.environ["EMPTY_VAR"] = ""
        os.environ["SPECIAL_CHARS"] = "test-with.special_chars"
        os.environ["NUMERIC"] = "123"
        os.environ["MIXED_CONTENT"] = "prefix_suffix"

        original_config_path = os.environ.get("MLFLOW_AUTH_CONFIG_PATH")
        os.environ["MLFLOW_AUTH_CONFIG_PATH"] = config_path

        try:
            config = read_auth_config()
            assert config.default_permission == ""  # Empty value
            assert config.database_uri == "sqlite:///test-with.special_chars.db"  # Special chars
            assert config.admin_username == "user123name"  # Numeric interpolation
            assert config.admin_password == "prefix_suffix"  # Simple variable
        finally:
            # Restore original config path
            if original_config_path is not None:
                os.environ["MLFLOW_AUTH_CONFIG_PATH"] = original_config_path
            else:
                os.environ.pop("MLFLOW_AUTH_CONFIG_PATH", None)
    finally:
        # Clean up
        os.unlink(config_path)
        for var in ["EMPTY_VAR", "SPECIAL_CHARS", "NUMERIC", "MIXED_CONTENT"]:
            os.environ.pop(var, None)


if __name__ == "__main__":
    pytest.main([__file__])
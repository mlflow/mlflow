"""REST API handlers for secrets management with integrity validation."""

import hashlib

from flask import request

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.secrets.crypto import SecretManager
from mlflow.secrets.scope import SecretScope
from mlflow.server import handlers


def create_secret():
    """
    Create or update a secret with client-side encryption and integrity validation.

    Request body:
        encrypted_name (str): Secret name encrypted with DEK
        encrypted_value (str): Secret value encrypted with DEK
        encrypted_dek (str): DEK encrypted with master key
        scope (int): Secret scope (0=GLOBAL, 1=SCORER)
        scope_id (int, optional): Scope ID for SCORER scope
        integrity_hash (str): SHA256 hash of "name:value" for validation

    Flow:
        1. Decrypt DEK with master key
        2. Decrypt name and value with DEK
        3. Recompute integrity hash and validate
        4. Re-encrypt with server's envelope encryption
        5. Store in database

    Returns:
        {"success": true}

    Raises:
        INVALID_PARAMETER_VALUE: If integrity validation fails
        INTERNAL_ERROR: If encryption/decryption fails
    """
    request_data = request.get_json()

    encrypted_name = request_data.get("encrypted_name")
    encrypted_value = request_data.get("encrypted_value")
    encrypted_dek = request_data.get("encrypted_dek")
    scope = request_data.get("scope", SecretScope.GLOBAL.value)
    scope_id = request_data.get("scope_id")
    client_integrity_hash = request_data.get("integrity_hash")

    if not all([encrypted_name, encrypted_value, encrypted_dek, client_integrity_hash]):
        raise MlflowException(
            "Missing required fields: encrypted_name, encrypted_value, "
            "encrypted_dek, integrity_hash",
            INVALID_PARAMETER_VALUE,
        )

    secret_manager = SecretManager()

    try:
        dek = secret_manager.decrypt_dek(encrypted_dek)
        plaintext_name = secret_manager.decrypt_with_dek(encrypted_name, dek)
        plaintext_value = secret_manager.decrypt_with_dek(encrypted_value, dek)
    except Exception as e:
        raise MlflowException(
            f"Failed to decrypt secret data. Ensure master key is correctly configured: {e}",
            INTERNAL_ERROR,
        )

    server_integrity_hash = _compute_integrity_hash(plaintext_name, plaintext_value)

    if server_integrity_hash != client_integrity_hash:
        raise MlflowException(
            "Integrity validation failed. The decrypted secret does not match the original. "
            "This indicates a potential encryption/decryption error or data corruption.",
            INVALID_PARAMETER_VALUE,
        )

    store = handlers._get_tracking_store()

    try:
        store.set_secret(
            name=plaintext_name,
            value=plaintext_value,
            scope=scope,
            scope_id=scope_id,
        )
    except Exception as e:
        raise MlflowException(f"Failed to store secret: {e}", INTERNAL_ERROR)

    return {"success": True}


def list_secrets():
    """
    List secret names for a scope.

    Query parameters:
        scope (int): Secret scope (0=GLOBAL, 1=SCORER)
        scope_id (int, optional): Scope ID for SCORER scope

    Returns:
        {"secret_names": ["name1", "name2", ...]}

    Note:
        This endpoint returns ONLY names, never values. This is intentional
        to prevent credential harvesting.
    """
    scope = request.args.get("scope", SecretScope.GLOBAL.value, type=int)
    scope_id = request.args.get("scope_id", type=int)

    store = handlers._get_tracking_store()

    try:
        names = store.list_secret_names(scope=scope, scope_id=scope_id)
    except Exception as e:
        raise MlflowException(f"Failed to list secrets: {e}", INTERNAL_ERROR)

    return {"secret_names": names}


def delete_secret():
    """
    Delete a secret.

    Request body:
        name (str): Secret name
        scope (int): Secret scope (0=GLOBAL, 1=SCORER)
        scope_id (int, optional): Scope ID for SCORER scope

    Returns:
        {"success": true}

    Raises:
        RESOURCE_DOES_NOT_EXIST: If secret does not exist
    """
    request_data = request.get_json()

    name = request_data.get("name")
    scope = request_data.get("scope", SecretScope.GLOBAL.value)
    scope_id = request_data.get("scope_id")

    if not name:
        raise MlflowException("Missing required field: name", INVALID_PARAMETER_VALUE)

    store = handlers._get_tracking_store()

    try:
        store.delete_secret(name=name, scope=scope, scope_id=scope_id)
    except Exception as e:
        raise MlflowException(f"Failed to delete secret: {e}", INTERNAL_ERROR)

    return {"success": True}


def _compute_integrity_hash(name: str, value: str) -> str:
    """
    Compute SHA256 hash for integrity validation.

    This must match the client-side computation exactly:
    SHA256(name:value)

    Args:
        name: Plaintext secret name.
        value: Plaintext secret value.

    Returns:
        Hex-encoded SHA256 hash.
    """
    combined = f"{name}:{value}"
    return hashlib.sha256(combined.encode()).hexdigest()

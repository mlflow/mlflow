"""Client-side API for secure secrets management with end-to-end encryption."""

import hashlib

from mlflow.secrets.crypto import SecretManager
from mlflow.secrets.scope import SecretScope


class SecretsClient:
    """
    Client for managing secrets with client-side encryption.

    This client encrypts secrets on the client side before transmission to ensure
    end-to-end security. Even if HTTPS is compromised, secrets remain encrypted.

    Security Flow:
    1. Generate random DEK (Data Encryption Key)
    2. Encrypt secret name and value with DEK
    3. Encrypt DEK with master key
    4. Compute integrity hash for validation
    5. Send encrypted payload to server
    6. Server validates integrity and re-encrypts with envelope encryption
    """

    def __init__(self, tracking_uri=None):
        """
        Initialize secrets client.

        Args:
            tracking_uri: MLflow tracking URI. If None, uses current tracking URI.
        """
        self.tracking_uri = tracking_uri
        self._secret_manager = SecretManager()

    def set_secret(
        self,
        name: str,
        value: str,
        scope: SecretScope = SecretScope.GLOBAL,
        scope_id: int | None = None,
    ):
        """
        Set a secret with client-side encryption.

        The secret is encrypted on the client before transmission. The server
        will validate the integrity and re-encrypt with server-side envelope encryption.

        Args:
            name: Secret name (will be encrypted).
            value: Secret value (will be encrypted).
            scope: Secret scope (GLOBAL or SCORER).
            scope_id: Scope ID (required for SCORER, None for GLOBAL).

        Raises:
            MlflowException: If validation fails or encryption errors occur.

        Example:
            >>> client = SecretsClient()
            >>> client.set_secret("api_key", "sk-1234", SecretScope.GLOBAL)
        """
        dek = self._secret_manager.generate_dek()

        encrypted_name = self._secret_manager.encrypt_with_dek(name, dek)
        encrypted_value = self._secret_manager.encrypt_with_dek(value, dek)
        encrypted_dek = self._secret_manager.encrypt_dek(dek)

        integrity_hash = self._compute_integrity_hash(name, value)

        from mlflow.utils.rest_utils import http_request

        endpoint = f"{self._get_base_uri()}/api/3.0/mlflow/secrets/create"
        request_data = {
            "encrypted_name": encrypted_name,
            "encrypted_value": encrypted_value,
            "encrypted_dek": encrypted_dek,
            "scope": scope.value,
            "integrity_hash": integrity_hash,
        }
        if scope_id is not None:
            request_data["scope_id"] = scope_id

        return http_request(
            host_creds=self._get_host_creds(),
            endpoint=endpoint,
            method="POST",
            json=request_data,
        )

    def list_secret_names(
        self, scope: SecretScope = SecretScope.GLOBAL, scope_id: int | None = None
    ) -> list[str]:
        """
        List secret names for a scope.

        This returns ONLY the names of secrets, never the values. This is
        intentional to prevent credential harvesting.

        Args:
            scope: Secret scope (GLOBAL or SCORER).
            scope_id: Scope ID (required for SCORER, None for GLOBAL).

        Returns:
            List of secret names (decrypted).

        Example:
            >>> client = SecretsClient()
            >>> names = client.list_secret_names(SecretScope.GLOBAL)
            >>> print(names)
            ['api_key', 'db_password']
        """
        from mlflow.utils.rest_utils import http_request

        endpoint = f"{self._get_base_uri()}/api/3.0/mlflow/secrets/list"
        params = {"scope": scope.value}
        if scope_id is not None:
            params["scope_id"] = scope_id

        response = http_request(
            host_creds=self._get_host_creds(), endpoint=endpoint, method="GET", params=params
        )

        return response.get("secret_names", [])

    def delete_secret(
        self, name: str, scope: SecretScope = SecretScope.GLOBAL, scope_id: int | None = None
    ):
        """
        Delete a secret.

        Args:
            name: Secret name to delete.
            scope: Secret scope (GLOBAL or SCORER).
            scope_id: Scope ID (required for SCORER, None for GLOBAL).

        Raises:
            MlflowException: If secret does not exist.

        Example:
            >>> client = SecretsClient()
            >>> client.delete_secret("old_api_key", SecretScope.GLOBAL)
        """
        from mlflow.utils.rest_utils import http_request

        endpoint = f"{self._get_base_uri()}/api/3.0/mlflow/secrets/delete"
        request_data = {"name": name, "scope": scope.value}
        if scope_id is not None:
            request_data["scope_id"] = scope_id

        return http_request(
            host_creds=self._get_host_creds(),
            endpoint=endpoint,
            method="DELETE",
            json=request_data,
        )

    def _compute_integrity_hash(self, name: str, value: str) -> str:
        """
        Compute SHA256 hash for integrity validation.

        The server will decrypt the name and value, recompute this hash,
        and verify it matches. This ensures the encryption/decryption
        roundtrip preserves the original data.

        Args:
            name: Plaintext secret name.
            value: Plaintext secret value.

        Returns:
            Hex-encoded SHA256 hash.
        """
        combined = f"{name}:{value}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _get_base_uri(self):
        """Get the base tracking URI."""
        if self.tracking_uri:
            return self.tracking_uri

        from mlflow.tracking import _get_store

        store = _get_store()
        if hasattr(store, "get_host_creds"):
            return store.get_host_creds().host
        return "http://localhost:5000"

    def _get_host_creds(self):
        """Get host credentials for REST API calls."""
        from mlflow.tracking import _get_store

        store = _get_store()
        if hasattr(store, "get_host_creds"):
            return store.get_host_creds()

        from mlflow.utils.rest_utils import MlflowHostCreds

        return MlflowHostCreds(self._get_base_uri())

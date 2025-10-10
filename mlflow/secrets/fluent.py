"""Fluent API for secrets management."""

from mlflow.secrets.scope import SecretScope


def set_secret(
    name: str,
    value: str,
    scope: SecretScope = SecretScope.GLOBAL,
    scope_id: int | None = None,
):
    """
    Set a secret with client-side encryption.

    The secret is encrypted on the client before transmission to ensure
    end-to-end security. Even if HTTPS is compromised, secrets remain encrypted.

    .. note::
        This function encrypts secrets before transmission. There is NO ``get_secret()``
        function to prevent credential harvesting. Secrets can only be retrieved
        server-side for authorized operations like scorer execution.

    Args:
        name: Secret name (will be encrypted).
        value: Secret value (will be encrypted).
        scope: Secret scope (GLOBAL or SCORER). Defaults to GLOBAL.
        scope_id: Scope ID (required for SCORER scope, None for GLOBAL).

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.secrets.scope import SecretScope

            # Set a global secret
            mlflow.set_secret("api_key", "sk-1234567890")

            # Set a scorer-scoped secret
            mlflow.set_secret("scorer_key", "secret", SecretScope.SCORER, scope_id=123)
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    return client.set_secret(name, value, scope, scope_id)


def list_secret_names(
    scope: SecretScope = SecretScope.GLOBAL,
    scope_id: int | None = None,
) -> list[str]:
    """
    List secret names for a scope.

    This returns ONLY the names of secrets, never their values. This is
    intentional to prevent credential harvesting.

    Args:
        scope: Secret scope (GLOBAL or SCORER). Defaults to GLOBAL.
        scope_id: Scope ID (required for SCORER scope, None for GLOBAL).

    Returns:
        List of secret names (decrypted).

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.secrets.scope import SecretScope

            # List global secrets
            names = mlflow.list_secret_names()
            print(names)  # ['api_key', 'db_password']

            # List scorer-scoped secrets
            names = mlflow.list_secret_names(SecretScope.SCORER, scope_id=123)
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    return client.list_secret_names(scope, scope_id)


def delete_secret(
    name: str,
    scope: SecretScope = SecretScope.GLOBAL,
    scope_id: int | None = None,
):
    """
    Delete a secret.

    Args:
        name: Secret name to delete.
        scope: Secret scope (GLOBAL or SCORER). Defaults to GLOBAL.
        scope_id: Scope ID (required for SCORER scope, None for GLOBAL).

    Example:
        .. code-block:: python

            import mlflow

            mlflow.delete_secret("old_api_key")
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    return client.delete_secret(name, scope, scope_id)

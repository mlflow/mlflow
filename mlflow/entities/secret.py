"""Entity class for MLflow secrets."""

from mlflow.entities._mlflow_object import _MlflowObject


class Secret(_MlflowObject):
    """
    A secret entity in MLflow.

    Secrets store encrypted key-value pairs scoped to experiments or scorers.

    Args:
        secret_id (str): Unique identifier for the secret.
        scope (int): Scope type (0=GLOBAL, 1=SCORER).
        scope_id (int | None): Database ID of the scope entity (None for GLOBAL).
        name_hash (str): Hash of the secret name for lookup.
        secret_value (str): Encrypted secret value.
        created_at (int): Unix timestamp (milliseconds) when created.
        updated_at (int): Unix timestamp (milliseconds) when last updated.
    """

    def __init__(
        self,
        secret_id: str,
        scope: int,
        scope_id: int | None,
        name_hash: str,
        secret_value: str,
        created_at: int,
        updated_at: int,
    ):
        self._secret_id = secret_id
        self._scope = scope
        self._scope_id = scope_id
        self._name_hash = name_hash
        self._secret_value = secret_value
        self._created_at = created_at
        self._updated_at = updated_at

    @property
    def secret_id(self):
        """The unique identifier for the secret."""
        return self._secret_id

    @property
    def scope(self):
        """The scope type (0=GLOBAL, 1=SCORER)."""
        return self._scope

    @property
    def scope_id(self):
        """The database ID of the scope entity (None for GLOBAL)."""
        return self._scope_id

    @property
    def name_hash(self):
        """The hash of the secret name."""
        return self._name_hash

    @property
    def secret_value(self):
        """The encrypted secret value."""
        return self._secret_value

    @property
    def created_at(self):
        """Unix timestamp (milliseconds) when the secret was created."""
        return self._created_at

    @property
    def updated_at(self):
        """Unix timestamp (milliseconds) when the secret was last updated."""
        return self._updated_at

    def __repr__(self):
        return f"<Secret(secret_id={self.secret_id}, scope={self.scope}, scope_id={self.scope_id})>"

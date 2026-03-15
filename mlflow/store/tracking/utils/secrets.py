from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import SqlGatewaySecret
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.utils.crypto import KEKManager, _decrypt_secret


def get_decrypted_secret(
    secret_id: str,
    store: SqlAlchemyStore | None = None,
) -> dict[str, Any]:
    """
    Get decrypted secret value by ID (server-side only).

    This is a privileged operation that decrypts a secret stored in the database.
    It should only be called server-side and never exposed to clients.

    Args:
        secret_id: ID of the secret to decrypt.
        store: Optional SqlAlchemyStore instance. If not provided, the current
            tracking store is used.

    Returns:
        Decrypted secret value as a dict (for compound credentials like
        {"api_key": "sk-xxx", ...}) or string (for simple secrets).

    Raises:
        MlflowException: If the tracking store is not a SqlAlchemyStore,
            or if the secret is not found, or if decryption fails.
    """
    if store is None:
        store = _get_store()
    if not isinstance(store, SqlAlchemyStore):
        raise MlflowException(
            "Secret retrieval is only supported with SqlAlchemyStore backends. "
            f"Current store type: {type(store).__name__}"
        )

    with store.ManagedSessionMaker() as session:
        sql_secret = store._get_entity_or_raise(
            session,
            SqlGatewaySecret,
            {"secret_id": secret_id},
            "GatewaySecret",
        )

        kek_manager = KEKManager()
        return _decrypt_secret(
            encrypted_value=sql_secret.encrypted_value,
            wrapped_dek=sql_secret.wrapped_dek,
            kek_manager=kek_manager,
            secret_id=sql_secret.secret_id,
            secret_name=sql_secret.secret_name,
        )

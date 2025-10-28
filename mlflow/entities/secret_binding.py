from dataclasses import dataclass

from mlflow.entities._mlflow_object import _MlflowObject


@dataclass
class SecretBinding(_MlflowObject):
    """
    MLflow entity representing a Secret Binding.

    Secret bindings map secrets to resources and define how the secret should be injected
    as environment variables. This allows secrets to be reused across multiple resources
    when marked as shared.

    Args:
        binding_id: String containing binding ID (UUID).
        secret_id: String containing the secret ID this binding references.
        resource_type: String containing the type of resource this secret is bound to.
        resource_id: String containing the ID of the resource using this secret.
        field_name: String containing the environment variable name (e.g., "OPENAI_API_KEY").
        created_at: Creation timestamp in milliseconds since the UNIX epoch.
        last_updated_at: Last update timestamp in milliseconds since the UNIX epoch.
        created_by: String containing the user ID who created the binding, or None.
        last_updated_by: String containing the user ID who last updated the binding, or None.
    """

    binding_id: str
    secret_id: str
    resource_type: str
    resource_id: str
    field_name: str
    created_at: int = 0
    last_updated_at: int = 0
    created_by: str | None = None
    last_updated_by: str | None = None

    def to_proto(self):
        from mlflow.protos.service_pb2 import SecretBinding as ProtoSecretBinding

        return ProtoSecretBinding(
            binding_id=self.binding_id,
            route_id=self.route_id,
            resource_type=self.resource_type,
            resource_id=self.resource_id,
            field_name=self.field_name,
            created_at=self.created_at,
            last_updated_at=self.last_updated_at,
            created_by=self.created_by or "",
            last_updated_by=self.last_updated_by or "",
        )

    @classmethod
    def from_proto(cls, proto):
        return cls(
            binding_id=proto.binding_id,
            route_id=proto.route_id,
            secret_id=proto.secret_id if proto.HasField("secret_id") else "",
            resource_type=proto.resource_type,
            resource_id=proto.resource_id,
            field_name=proto.field_name,
            created_at=proto.created_at,
            last_updated_at=proto.last_updated_at,
            created_by=proto.created_by if proto.HasField("created_by") else None,
            last_updated_by=proto.last_updated_by if proto.HasField("last_updated_by") else None,
        )

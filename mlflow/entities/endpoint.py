from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import Endpoint as ProtoEndpoint

if TYPE_CHECKING:
    from mlflow.entities.endpoint_model import EndpointModel
    from mlflow.entities.endpoint_tag import EndpointTag


@dataclass
class Endpoint(_MlflowObject):
    """
    MLflow entity representing an Endpoint.

    Endpoints are routing configurations that can direct traffic to multiple models.
    Each model in the endpoint has its own secret (API key) for authentication.

    This entity contains only metadata about the endpoint - cryptographic fields
    (encrypted_model_config, wrapped_model_config_dek) are never exposed outside the store layer.

    Args:
        endpoint_id: String containing endpoint ID (UUID).
        created_at: Creation timestamp in milliseconds since the UNIX epoch.
        last_updated_at: Last update timestamp in milliseconds since the UNIX epoch.
        name: String containing optional display name, or None.
        description: String containing optional user-provided description, or None.
        created_by: String containing the user ID who created the endpoint, or None.
        last_updated_by: String containing the user ID who last updated the endpoint, or None.
        models: List of EndpointModel objects associated with this endpoint.
        tags: List of EndpointTag objects associated with this endpoint.
    """

    endpoint_id: str
    created_at: int
    last_updated_at: int
    name: str | None = None
    description: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    models: list[EndpointModel] = field(default_factory=list)
    tags: list[EndpointTag] = field(default_factory=list)

    def to_proto(self):
        from mlflow.entities.endpoint_model import EndpointModel as EndpointModelEntity
        from mlflow.protos.service_pb2 import EndpointModel as ProtoEndpointModel

        proto = ProtoEndpoint()
        proto.endpoint_id = self.endpoint_id
        proto.created_at = self.created_at
        proto.last_updated_at = self.last_updated_at
        proto.name = self.name or ""
        proto.description = self.description or ""
        proto.created_by = self.created_by or ""
        proto.last_updated_by = self.last_updated_by or ""
        proto.tags.extend([tag.to_proto() for tag in self.tags])

        for model in self.models:
            model_proto = ProtoEndpointModel()
            model_proto.model_id = model.model_id
            model_proto.endpoint_id = model.endpoint_id
            model_proto.model_name = model.model_name
            model_proto.secret_id = model.secret_id
            model_proto.weight = model.weight
            model_proto.priority = model.priority
            model_proto.created_at = model.created_at
            model_proto.last_updated_at = model.last_updated_at
            model_proto.created_by = model.created_by or ""
            model_proto.last_updated_by = model.last_updated_by or ""
            model_proto.secret_name = model.secret_name or ""
            model_proto.provider = model.provider or ""
            proto.models.append(model_proto)

        return proto

    @classmethod
    def from_proto(cls, proto):
        from mlflow.entities.endpoint_model import EndpointModel as EndpointModelEntity
        from mlflow.entities.endpoint_tag import EndpointTag as EndpointTagEntity

        models = []
        for model_proto in proto.models:
            models.append(
                EndpointModelEntity(
                    model_id=model_proto.model_id,
                    endpoint_id=model_proto.endpoint_id,
                    model_name=model_proto.model_name,
                    secret_id=model_proto.secret_id,
                    weight=model_proto.weight,
                    priority=model_proto.priority,
                    created_at=model_proto.created_at,
                    last_updated_at=model_proto.last_updated_at,
                    created_by=model_proto.created_by if model_proto.HasField("created_by") else None,
                    last_updated_by=(
                        model_proto.last_updated_by
                        if model_proto.HasField("last_updated_by")
                        else None
                    ),
                    secret_name=(
                        model_proto.secret_name if model_proto.HasField("secret_name") else ""
                    ),
                    provider=model_proto.provider if model_proto.HasField("provider") else "",
                )
            )

        return cls(
            endpoint_id=proto.endpoint_id,
            created_at=proto.created_at,
            last_updated_at=proto.last_updated_at,
            name=proto.name if proto.HasField("name") else None,
            description=proto.description if proto.HasField("description") else None,
            created_by=proto.created_by if proto.HasField("created_by") else None,
            last_updated_by=proto.last_updated_by if proto.HasField("last_updated_by") else None,
            tags=[EndpointTagEntity.from_proto(tag) for tag in proto.tags],
            models=models,
        )

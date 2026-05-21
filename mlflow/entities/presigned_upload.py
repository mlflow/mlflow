from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CreatePresignedUploadResponse:
    """Response from creating a presigned upload URL."""

    presigned_url: str
    headers: dict[str, str] = field(default_factory=dict)

    def to_proto(self):
        from mlflow.protos.service_pb2 import (
            CreatePresignedUploadUrl as ProtoCreatePresignedUploadUrl,
        )

        response = ProtoCreatePresignedUploadUrl.Response()
        response.presigned_url = self.presigned_url
        response.headers.update(self.headers)
        return response

    @classmethod
    def from_proto(cls, proto) -> CreatePresignedUploadResponse:
        return cls(
            presigned_url=proto.presigned_url,
            headers=dict(proto.headers),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CreatePresignedUploadResponse:
        return cls(
            presigned_url=d["presigned_url"],
            headers=d.get("headers", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "presigned_url": self.presigned_url,
            "headers": self.headers,
        }

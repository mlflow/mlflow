from dataclasses import dataclass
from typing import Any, Optional

from mlflow.protos.mlflow_artifacts_pb2 import (
    CreateMultipartUpload as ProtoCreateMultipartUpload,
)
from mlflow.protos.mlflow_artifacts_pb2 import (
    MultipartUploadCredential as ProtoMultipartUploadCredential,
)


@dataclass
class MultipartUploadPart:
    part_number: int
    etag: str
    url: Optional[str] = None

    @classmethod
    def from_proto(cls, proto):
        return cls(
            proto.part_number,
            proto.etag or None,
            proto.url or None,
        )

    def to_dict(self):
        return {
            "part_number": self.part_number,
            "etag": self.etag,
            "url": self.url,
        }


@dataclass
class MultipartUploadCredential:
    url: str
    part_number: int
    headers: dict[str, Any]

    def to_proto(self):
        credential = ProtoMultipartUploadCredential()
        credential.url = self.url
        credential.part_number = self.part_number
        credential.headers.update(self.headers)
        return credential

    @classmethod
    def from_dict(cls, dict_):
        return cls(
            url=dict_["url"],
            part_number=dict_["part_number"],
            headers=dict_.get("headers", {}),
        )


@dataclass
class CreateMultipartUploadResponse:
    upload_id: Optional[str]
    credentials: list[MultipartUploadCredential]

    def to_proto(self):
        response = ProtoCreateMultipartUpload.Response()
        if self.upload_id:
            response.upload_id = self.upload_id
        response.credentials.extend([credential.to_proto() for credential in self.credentials])
        return response

    @classmethod
    def from_dict(cls, dict_):
        credentials = [MultipartUploadCredential.from_dict(cred) for cred in dict_["credentials"]]
        return cls(
            upload_id=dict_.get("upload_id"),
            credentials=credentials,
        )

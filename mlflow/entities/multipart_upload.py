from mlflow.protos.mlflow_artifacts_pb2 import (
    CreateMultipartUpload as ProtoCreateMultipartUpload,
    MultipartUploadCredential as ProtoMultipartUploadCredential,
)

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class MultipartUploadPart:
    part_number: int
    etag: str

    @classmethod
    def from_proto(cls, proto):
        return cls(
            proto.part_number,
            proto.etag or None,
        )


@dataclass
class MultipartUploadCredential:
    url: str
    part_number: int
    headers: Dict[str, Any]

    def to_proto(self):
        credential = ProtoMultipartUploadCredential()
        credential.url = self.url
        credential.part_number = self.part_number
        credential.headers.update(self.headers)
        return credential


@dataclass
class CreateMultipartUploadResponse:
    upload_id: Optional[str]
    credentials: List[MultipartUploadCredential]

    def to_proto(self):
        response = ProtoCreateMultipartUpload.Response()
        response.upload_id = self.upload_id
        response.credentials.extend([credential.to_proto() for credential in self.credentials])
        return response

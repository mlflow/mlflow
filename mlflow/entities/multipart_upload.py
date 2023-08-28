from mlflow.protos.mlflow_artifacts_pb2 import (
    MultipartUploadCredential as ProtoMultipartUploadCredential,
)

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class MultipartUploadPart:
    part_number: int
    etag: str


@dataclass
class MultipartUploadCredential:
    url: str
    part_number: int
    headers: Dict[str, Any]

    def to_proto(self):
        credential = ProtoMultipartUploadCredential()
        credential.url = self.url
        credential.part_number = self.part_number
        credential.headers = self.headers
        return credential


@dataclass
class CreateMultipartUploadResponse:
    credentials: List[MultipartUploadCredential]
    upload_id: Optional[str]

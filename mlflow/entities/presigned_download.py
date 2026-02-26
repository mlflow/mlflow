from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PresignedDownloadUrlResponse:
    """
    Response containing a presigned URL for downloading an artifact directly
    from cloud storage.
    """

    url: str
    headers: dict[str, str]
    file_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "url": self.url,
            "headers": self.headers,
        }
        if self.file_size is not None:
            result["file_size"] = self.file_size
        return result

    @classmethod
    def from_dict(cls, dict_: dict[str, Any]) -> PresignedDownloadUrlResponse:
        return cls(
            url=dict_["url"],
            headers=dict_.get("headers", {}),
            file_size=dict_.get("file_size"),
        )

import mimetypes
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class Attachment:
    """
    Represents a binary attachment (image, audio, PDF, etc.) that can be logged
    as part of a trace span's inputs or outputs.

    When an Attachment is set as a span input/output value, it is automatically:
    1. Replaced with a reference URI in the span data
    2. Uploaded as a separate artifact file alongside the trace data
    """

    def __init__(self, *, content_type: str, content_bytes: bytes):
        self._id = str(uuid.uuid4())
        self._content_type = content_type
        self._content_bytes = content_bytes

    @classmethod
    def from_file(cls, path: str | Path, content_type: str | None = None) -> "Attachment":
        path = Path(path)
        if content_type is None:
            guessed, _ = mimetypes.guess_type(str(path))
            content_type = guessed or "application/octet-stream"
        return cls(content_type=content_type, content_bytes=path.read_bytes())

    @property
    def id(self) -> str:
        return self._id

    @property
    def content_type(self) -> str:
        return self._content_type

    @property
    def content_bytes(self) -> bytes:
        return self._content_bytes

    def ref(self, trace_id: str) -> str:
        return (
            f"mlflow-attachment://{self._id}?content_type={self._content_type}&trace_id={trace_id}"
        )

    @staticmethod
    def parse_ref(ref_uri: str) -> dict[str, str] | None:
        parsed = urlparse(ref_uri)
        if parsed.scheme != "mlflow-attachment":
            return None
        params = parse_qs(parsed.query)
        attachment_id = parsed.hostname
        content_type = params.get("content_type", [None])[0]
        trace_id = params.get("trace_id", [None])[0]
        if not attachment_id or not content_type or not trace_id:
            return None
        return {
            "attachment_id": attachment_id,
            "content_type": content_type,
            "trace_id": trace_id,
        }

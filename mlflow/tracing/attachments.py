import uuid
from pathlib import Path


class Attachment:
    """
    Represents an attachment file. When this file is logged as a part of trace

    1. This attachment file is uploaded as an artifact.
    2. A reference to this attachment is created and can be used to retrieve the file later.
    """

    def __init__(self, *, content_type: str, content_bytes: bytes):
        self.id = str(uuid.uuid4())
        self.content_bytes = content_bytes
        self.content_type = content_type

    @classmethod
    def from_file(cls, file: str | Path, content_type: str | None = None) -> "Attachment":
        return cls(
            content_type=content_type or cls._infer_content_type(file),
            content_bytes=Path(file).read_bytes(),
        )

    @staticmethod
    def _infer_content_type(file: str | Path) -> str:
        if isinstance(file, Path):
            file = str(file)
        if file.endswith(".png"):
            return "image/png"
        elif file.endswith(".jpg") or file.endswith(".jpeg"):
            return "image/jpeg"
        elif file.endswith(".pdf"):
            return "application/pdf"
        elif file.endswith(".mp3"):
            return "audio/mpeg"
        elif file.endswith(".wav"):
            return "audio/wav"
        elif file.endswith(".ogg"):
            return "audio/ogg"
        elif file.endswith(".m4a"):
            return "audio/mp4"
        elif file.endswith(".aac"):
            return "audio/aac"
        elif file.endswith(".flac"):
            return "audio/flac"
        elif file.endswith(".webm"):
            return "audio/webm"
        return "application/octet-stream"

    def ref(self, trace_id: str) -> str:
        """
        A string representation of the attachment reference.
        """
        return (
            f"mlflow-attachment://{self.id}?content_type={self.content_type}&trace_id=tr-{trace_id}"
        )

    @classmethod
    def from_ref(cls, ref: str) -> "Attachment":
        raise NotImplementedError("TODO")

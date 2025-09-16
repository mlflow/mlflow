import mimetypes
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlparse


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
        """Create an Attachment from a file path.

        Args:
            file: Path to the file to attach
            content_type: Optional content type override. If not provided, will be inferred from
                file extension.

        Returns:
            Attachment object
        """
        return cls(
            content_type=content_type or cls._infer_content_type(file),
            content_bytes=Path(file).read_bytes(),
        )

    @staticmethod
    def _infer_content_type(file: str | Path) -> str:
        """Infer content type from file extension."""
        if isinstance(file, Path):
            file = str(file)

        # Use mimetypes module for more comprehensive type detection
        content_type, _ = mimetypes.guess_type(file)
        if content_type:
            return content_type

        # Fallback to manual mapping for common types
        file_lower = file.lower()
        if file_lower.endswith((".png",)):
            return "image/png"
        elif file_lower.endswith((".jpg", ".jpeg")):
            return "image/jpeg"
        elif file_lower.endswith((".gif",)):
            return "image/gif"
        elif file_lower.endswith((".webp",)):
            return "image/webp"
        elif file_lower.endswith((".pdf",)):
            return "application/pdf"
        elif file_lower.endswith((".mp3",)):
            return "audio/mpeg"
        elif file_lower.endswith((".wav",)):
            return "audio/wav"
        elif file_lower.endswith((".ogg",)):
            return "audio/ogg"
        elif file_lower.endswith((".m4a",)):
            return "audio/mp4"
        elif file_lower.endswith((".aac",)):
            return "audio/aac"
        elif file_lower.endswith((".flac",)):
            return "audio/flac"
        elif file_lower.endswith((".webm",)):
            return "audio/webm"
        elif file_lower.endswith((".mp4",)):
            return "video/mp4"
        elif file_lower.endswith((".mov",)):
            return "video/quicktime"
        elif file_lower.endswith((".avi",)):
            return "video/x-msvideo"
        return "application/octet-stream"

    def ref(self, trace_id: str) -> str:
        """
        A string representation of the attachment reference.

        Args:
            trace_id: The trace ID this attachment belongs to

        Returns:
            Reference string in the format: mlflow-attachments://{id}?content_type={type}&trace_id={trace_id}
        """
        return (
            f"mlflow-attachments://{self.id}?content_type={self.content_type}&trace_id={trace_id}"
        )

    @classmethod
    def from_ref(cls, ref: str) -> "AttachmentRef":
        """Create an AttachmentRef from a reference string.

        Args:
            ref: Reference string in the format: mlflow-attachments://{id}?content_type={type}&trace_id={trace_id}

        Returns:
            AttachmentRef object that can be used to retrieve the attachment
        """
        parsed = urlparse(ref)
        if parsed.scheme != "mlflow-attachments":
            raise ValueError(f"Invalid attachment reference scheme: {parsed.scheme}")

        attachment_id = parsed.netloc
        if not attachment_id:
            raise ValueError("Attachment ID not found in reference")

        params = parse_qs(parsed.query)
        content_type = params.get("content_type", [None])[0]
        trace_id = params.get("trace_id", [None])[0]

        if not content_type:
            raise ValueError("Content type not found in attachment reference")
        if not trace_id:
            raise ValueError("Trace ID not found in attachment reference")

        return AttachmentRef(id=attachment_id, content_type=content_type, trace_id=trace_id)


class AttachmentRef:
    """
    Reference to an attachment that can be used to retrieve attachment data.
    """

    def __init__(self, id: str, content_type: str, trace_id: str):
        self.id = id
        self.content_type = content_type
        self.trace_id = trace_id

    def download(self) -> bytes:
        """Download the attachment content from the MLflow server.

        Returns:
            Attachment content as bytes
        """
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        # Use the internal client method to download trace attachment
        trace_info = client._tracking_client.get_trace_info(self.trace_id)
        artifact_repo = client._tracking_client._get_artifact_repo_for_trace(trace_info)
        return artifact_repo.download_trace_attachment(self.id)

    def to_attachment(self) -> Attachment:
        """Convert this reference to a full Attachment object by downloading the content.

        Returns:
            Attachment object with downloaded content
        """
        content_bytes = self.download()
        return Attachment(content_type=self.content_type, content_bytes=content_bytes)

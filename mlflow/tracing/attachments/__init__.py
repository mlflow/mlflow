import base64
import json
import mimetypes
import uuid
from pathlib import Path


class Attachment:
    """
    Represents an attachment file. When this file is logged as a part of trace

    1. This attachment file is uploaded as an artifact.
    2. A reference to this attachment is created and can be used to retrieve the file later.
    """

    def __init__(self, *, content_type: str, content_bytes: bytes, filename: str | None = None):
        self.id = str(uuid.uuid4())
        self.content_bytes = content_bytes
        self.content_type = content_type
        self.filename = filename

    def __repr__(self) -> str:
        return (
            f"Attachment(id={self.id}, content_type={self.content_type}, "
            f"size={len(self.content_bytes)} bytes)"
        )

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
        path = Path(file)
        return cls(
            content_type=content_type or cls._infer_content_type(file),
            content_bytes=path.read_bytes(),
            filename=path.name,
        )

    @staticmethod
    def _infer_content_type(file: str | Path) -> str:
        """Infer content type from file extension."""
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

    def ref(self, trace_id: str, span_id: str) -> str:
        """
        A string representation of the attachment reference.

        Args:
            trace_id: The trace ID this attachment belongs to
            span_id: The span ID this attachment belongs to

        Returns:
            JSON-based reference string encoded in base64 for URL safety
        """
        metadata = {
            "attachment_id": self.id,
            "trace_id": trace_id,
            "span_id": span_id,
            "content_type": self.content_type,
            "size": len(self.content_bytes),
        }
        # Add optional fields if present
        if self.filename:
            metadata["filename"] = self.filename
        # Use compact JSON encoding (no spaces) for efficiency
        json_str = json.dumps(metadata, separators=(",", ":"))
        # Base64 encode for URL safety
        encoded = base64.urlsafe_b64encode(json_str.encode()).decode()
        return f"mlflow-attachment:{encoded}"

    @classmethod
    def from_ref(cls, ref: str) -> "Attachment":
        """Create an Attachment from a reference string.

        Args:
            ref: JSON-based reference string

        Returns:
            Attachment object with downloaded content
        """
        if not ref.startswith("mlflow-attachment:"):
            raise ValueError(f"Invalid attachment reference format: {ref}")

        encoded_json = ref[len("mlflow-attachment:") :]
        try:
            json_str = base64.urlsafe_b64decode(encoded_json).decode()
            metadata = json.loads(json_str)

            attachment_id = metadata["attachment_id"]  # noqa: F841
            trace_id = metadata["trace_id"]  # noqa: F841
            span_id = metadata["span_id"]  # noqa: F841
            content_type = metadata["content_type"]
            size = metadata.get("size")  # noqa: F841
            filename = metadata.get("filename")

        except (json.JSONDecodeError, KeyError, Exception) as e:
            raise ValueError(f"Invalid attachment reference: {e}")

        # TODO: Implement actual download logic
        # The variables above will be used to:
        # 1. Download attachment using trace_id
        # 2. Validate size matches expected
        content_bytes = b""

        return cls(content_type=content_type, content_bytes=content_bytes, filename=filename)

    @staticmethod
    def parse_ref(ref: str) -> dict[str, str | int | None]:
        """Parse an attachment reference string to extract metadata.

        Args:
            ref: JSON-based reference string

        Returns:
            Dictionary containing: trace_id, span_id, attachment_id, content_type, size
        """
        if not ref.startswith("mlflow-attachment:"):
            raise ValueError(f"Invalid attachment reference format: {ref}")

        encoded_json = ref[len("mlflow-attachment:") :]
        try:
            json_str = base64.urlsafe_b64decode(encoded_json).decode()
            return json.loads(json_str)
        except (json.JSONDecodeError, Exception) as e:
            raise ValueError(f"Invalid attachment reference: {e}")

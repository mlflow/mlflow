import json
import os
import signal
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlflow.assistant.types import Message

SESSION_DIR = Path(tempfile.gettempdir()) / "mlflow-assistant-sessions"


@dataclass
class Session:
    """Session state for assistant conversations."""

    context: dict[str, Any] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)
    pending_message: Message | None = None
    provider_session_id: str | None = None
    working_dir: Path | None = None  # Working directory for the session (e.g. project path)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the session history.

        Args:
            role: Role of the message sender (user, assistant, system)
            content: Text content of the message
        """
        self.messages.append(Message(role=role, content=content))

    def set_pending_message(self, role: str, content: str) -> None:
        """Set the pending message to be processed.

        Args:
            role: Role of the message sender
            content: Text content of the message
        """
        self.pending_message = Message(role=role, content=content)

    def clear_pending_message(self) -> Message | None:
        """Clear and return the pending message.

        Returns:
            The pending message, or None if no message was pending
        """
        msg = self.pending_message
        self.pending_message = None
        return msg

    def update_context(self, context: dict[str, Any]) -> None:
        """Update session context.

        Args:
            context: Context data to merge into session context
        """
        self.context.update(context)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of session
        """
        return {
            "context": self.context,
            "messages": [msg.model_dump() for msg in self.messages],
            "pending_message": self.pending_message.model_dump() if self.pending_message else None,
            "provider_session_id": self.provider_session_id,
            "working_dir": self.working_dir.as_posix() if self.working_dir else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Load from dictionary.

        Args:
            data: Dictionary representation of session

        Returns:
            Session instance
        """
        messages = [Message.model_validate(m) for m in data.get("messages", [])]
        pending = data.get("pending_message")
        pending_msg = Message.model_validate(pending) if pending else None

        return cls(
            context=data.get("context", {}),
            messages=messages,
            pending_message=pending_msg,
            provider_session_id=data.get("provider_session_id"),
            working_dir=Path(data.get("working_dir")) if data.get("working_dir") else None,
        )


class SessionManager:
    """Manages session storage and retrieval.

    Provides static methods for session operations, keeping
    Session as a simple data container.
    """

    @staticmethod
    def validate_session_id(session_id: str) -> None:
        """Validate that session_id is a valid UUID to prevent path traversal.

        Args:
            session_id: Session ID to validate

        Raises:
            ValueError: If session ID is not a valid UUID
        """
        try:
            uuid.UUID(session_id)
        except (ValueError, TypeError) as e:
            raise ValueError("Invalid session ID format") from e

    @staticmethod
    def get_session_file(session_id: str) -> Path:
        """Get the file path for a session.

        Args:
            session_id: Session ID

        Returns:
            Path to session file

        Raises:
            ValueError: If session ID is invalid
        """
        SessionManager.validate_session_id(session_id)
        return SESSION_DIR / f"{session_id}.json"

    @staticmethod
    def save(session_id: str, session: Session) -> None:
        """Save session to disk atomically.

        Args:
            session_id: Session ID
            session: Session to save

        Raises:
            ValueError: If session ID is invalid
        """
        SessionManager.validate_session_id(session_id)
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        session_file = SessionManager.get_session_file(session_id)

        # Write to temp file, then rename (atomic on POSIX)
        fd, temp_path = tempfile.mkstemp(dir=SESSION_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(session.to_dict(), f)
            os.replace(temp_path, session_file)
        except Exception:
            os.unlink(temp_path)
            raise

    @staticmethod
    def load(session_id: str) -> Session | None:
        """Load session from disk. Returns a Session instance, or None if not found"""
        try:
            session_file = SessionManager.get_session_file(session_id)
        except ValueError:
            return None
        if not session_file.exists():
            return None
        data = json.loads(session_file.read_text())
        return Session.from_dict(data)

    @staticmethod
    def create(context: dict[str, Any] | None = None, working_dir: Path | None = None) -> Session:
        """Create a new session.

        Args:
            context: Initial context data, or None
            working_dir: Working directory for the session

        Returns:
            New Session instance
        """
        return Session(context=context or {}, working_dir=working_dir)


def get_process_file(session_id: str) -> Path:
    """Get the file path for storing process PID."""
    SessionManager.validate_session_id(session_id)
    return SESSION_DIR / f"{session_id}.process.json"


def save_process_pid(session_id: str, pid: int) -> None:
    """Save process PID to file for cancellation support."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    process_file = get_process_file(session_id)
    process_file.write_text(json.dumps({"pid": pid}))


def get_process_pid(session_id: str) -> int | None:
    try:
        process_file = get_process_file(session_id)
    except ValueError:
        return None
    if not process_file.exists():
        return None
    data = json.loads(process_file.read_text())
    return data.get("pid")


def clear_process_pid(session_id: str) -> None:
    try:
        process_file = get_process_file(session_id)
    except ValueError:
        return
    if process_file.exists():
        process_file.unlink()


def terminate_session_process(session_id: str) -> bool:
    """Terminate the process associated with a session.

    Args:
        session_id: Session ID

    Returns:
        True if process was terminated, False otherwise
    """
    if pid := get_process_pid(session_id):
        try:
            os.kill(pid, signal.SIGTERM)
            clear_process_pid(session_id)
            return True
        except ProcessLookupError:
            clear_process_pid(session_id)
    return False

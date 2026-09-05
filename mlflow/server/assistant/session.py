import json
import logging
import os
import shutil
import signal
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mlflow.assistant.types import Message

_logger = logging.getLogger(__name__)

SESSION_DIR = Path(tempfile.gettempdir()) / "mlflow-assistant-sessions"

# Per-session sandbox $HOME directories with no activity within this window are reaped.
_SANDBOX_HOME_MAX_AGE_SECONDS = 24 * 60 * 60


@dataclass
class Session:
    """Session state for assistant conversations."""

    context: dict[str, Any] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)
    pending_message: Message | None = None
    provider_session_id: str | None = None
    working_dir: Path | None = None  # Working directory for the session (e.g. project path)
    # tool_call_id -> "allow" | "deny": a decision awaiting the next stream so a
    # turn paused at a permission prompt can resume. Set by the resume endpoint,
    # consumed (and cleared) by the stream.
    pending_tool_decisions: dict[str, Literal["allow", "deny"]] = field(default_factory=dict)
    # tool_call_id -> {"content": str, "is_error": bool}: the result of a CLIENT
    # -executed tool call (e.g. render_custom_view), awaiting the next stream so a
    # turn paused at a client_tool_call can resume. Set by the tool-result
    # endpoint, consumed (and cleared) by the stream.
    pending_client_tool_results: dict[str, dict[str, Any]] = field(default_factory=dict)

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
            "pending_tool_decisions": self.pending_tool_decisions,
            "pending_client_tool_results": self.pending_client_tool_results,
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
            pending_tool_decisions=data.get("pending_tool_decisions") or {},
            pending_client_tool_results=data.get("pending_client_tool_results") or {},
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
    try:
        data = json.loads(process_file.read_text())
    except (OSError, ValueError):
        # A truncated/corrupt file (writes are not atomic) or one that vanished after the
        # exists() check must not turn a cancel lookup into an error; treat it as absent.
        return None
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
        except (ProcessLookupError, PermissionError):
            clear_process_pid(session_id)
    return False


def get_container_file(session_id: str) -> Path:
    """Get the file path for storing a session's sandbox container id."""
    SessionManager.validate_session_id(session_id)
    return SESSION_DIR / f"{session_id}.container.json"


def save_container_id(session_id: str, container_id: str) -> None:
    """Record the sandbox container running a session's turn, for cancellation support."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    get_container_file(session_id).write_text(json.dumps({"container_id": container_id}))


def get_container_id(session_id: str) -> str | None:
    try:
        container_file = get_container_file(session_id)
    except ValueError:
        return None
    if not container_file.exists():
        return None
    try:
        return json.loads(container_file.read_text()).get("container_id")
    except (OSError, ValueError):
        # A truncated/corrupt file (writes are not atomic) or one that vanished after the
        # exists() check must not turn a cancel lookup into an error; treat it as absent.
        return None


def clear_container_id(session_id: str) -> None:
    try:
        container_file = get_container_file(session_id)
    except ValueError:
        return
    # missing_ok: the stream's finally and a concurrent cancel can both clear the same id; tolerate
    # the file already being gone rather than raising a spurious error on the losing path.
    container_file.unlink(missing_ok=True)


def get_session_sandbox_home(session_id: str) -> Path:
    """Server-owned host directory bind-mounted as the sandbox container's ``$HOME``.

    Persists the CLI's ``--resume`` state and caches across turns of a session. It is created
    here (owned by the server user) so the container, which runs as that same uid:gid, can
    write to it. These directories accumulate per session; reaping stale ones is handled by
    the session-lifecycle work.
    """
    SessionManager.validate_session_id(session_id)
    home = SESSION_DIR / "sandbox-home" / session_id
    home.mkdir(parents=True, exist_ok=True)
    # This HOME holds the CLI's login credentials and session state, so keep it private to the
    # server user (0700). mkdir() above is subject to the process umask, and a pre-existing dir
    # may have looser modes, so enforce it explicitly.
    home.chmod(0o700)
    # Bump the top-level mtime so reap_stale_sandbox_homes sees each turn as recent activity:
    # the CLI writes --resume state into nested subdirs (e.g. .claude/), which does not advance
    # the directory's own mtime.
    os.utime(home, None)
    return home


def reap_stale_sandbox_homes(max_age_seconds: float = _SANDBOX_HOME_MAX_AGE_SECONDS) -> int:
    """Remove per-session sandbox ``$HOME`` directories untouched within the window.

    These accumulate one per session (see ``get_session_sandbox_home``, which bumps a
    directory's mtime on every turn); a directory whose mtime is older than ``max_age_seconds``
    is treated as belonging to a finished session. Best-effort: returns the number removed.
    """
    base = SESSION_DIR / "sandbox-home"
    if not base.exists():
        return 0
    cutoff = time.time() - max_age_seconds
    removed = 0
    try:
        entries = list(base.iterdir())
    except OSError:
        return 0
    for entry in entries:
        try:
            # Skip symlinks (do not follow them out of the base dir) and non-directories.
            if entry.is_symlink() or not entry.is_dir() or entry.stat().st_mtime >= cutoff:
                continue
        except OSError:
            continue
        shutil.rmtree(entry, ignore_errors=True)
        if entry.exists():
            # rmtree with ignore_errors swallows failures; note it rather than reap silently.
            _logger.debug("Could not remove stale sandbox home directory %s", entry)
            continue
        removed += 1
        # The reaped HOME held the CLI's --resume state; drop the stored provider session id so
        # the next turn starts a fresh CLI session instead of resuming deleted state. The
        # directory name is the session id (see get_session_sandbox_home). A corrupt/unreadable
        # session file must not abort the whole sweep, so this is best-effort.
        try:
            session = SessionManager.load(entry.name)
            if session and session.provider_session_id is not None:
                session.provider_session_id = None
                SessionManager.save(entry.name, session)
        except Exception:
            _logger.debug("Could not clear provider session id for reaped session %s", entry.name)
    if removed:
        _logger.info("Reaped %d stale sandbox home directories.", removed)
    return removed


def terminate_session_container(session_id: str) -> bool:
    """Kill the sandbox container running a session's turn, if any.

    Returns True only if a container was found and killed. A container that is already gone
    clears the stored id and returns False. A transient failure (e.g. a daemon hiccup) against
    a possibly-live container is left as-is — the id is kept so a retry can still reach it.
    """
    container_id = get_container_id(session_id)
    if not container_id:
        return False
    try:
        import docker
        import docker.errors

        client = docker.from_env()
        try:
            container = client.containers.get(container_id)
        except docker.errors.NotFound:
            clear_container_id(session_id)
            return False
        container.kill()
        clear_container_id(session_id)
        return True
    except Exception:
        # Keep the id (a retry can still reach a possibly-live container) but log it — otherwise a
        # failed kill is invisible while the cancel endpoint reports the session as cancelled.
        _logger.warning(
            "Failed to terminate sandbox container for session %s; leaving its id for retry.",
            session_id,
            exc_info=True,
        )
        return False

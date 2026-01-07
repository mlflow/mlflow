"""
Assistant API endpoints for MLflow Server.

This module provides endpoints for integrating AI assistants with MLflow UI,
enabling AI-powered helper through a chat interface.
"""

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mlflow.server.assistant.providers.claude_code import ClaudeCodeProvider

# Hardcoded provider for now
_provider = ClaudeCodeProvider()

# TODO: Consider using a database for session storage instead of file-based storage.
#       Also implement periodic cleanup of old sessions (e.g., older than 1 week)
#       to prevent disk space accumulation.
SESSION_DIR = Path(tempfile.gettempdir()) / "mlflow-assistant-sessions"


@dataclass
class Session:
    """Session state for assistant conversations."""

    context: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, str]] = field(default_factory=list)
    pending_message: str | None = None
    # We need a separate provider_session_id because the API must return a session_id
    # immediately in the POST /message response, before the provider runs.
    # The provider's session_id is only available after the first streaming response completes.
    provider_session_id: str | None = None


def _validate_session_id(session_id: str) -> None:
    """Validate that session_id is a valid UUID to prevent path traversal."""
    try:
        uuid.UUID(session_id)
    except (ValueError, TypeError) as e:
        raise ValueError("Invalid session ID format") from e


def _get_session_file(session_id: str) -> Path:
    """Get the file path for a session."""
    _validate_session_id(session_id)
    return SESSION_DIR / f"{session_id}.json"


def _save_session(session_id: str, session: Session) -> None:
    """Save session to disk atomically."""
    _validate_session_id(session_id)
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    session_file = _get_session_file(session_id)
    # Write to temp file, then rename (atomic on POSIX)
    fd, temp_path = tempfile.mkstemp(dir=SESSION_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(asdict(session), f)
        os.replace(temp_path, session_file)
    except Exception:
        os.unlink(temp_path)
        raise


def _load_session(session_id: str) -> Session | None:
    """Load session from disk."""
    try:
        session_file = _get_session_file(session_id)
    except ValueError:
        return None
    if not session_file.exists():
        return None
    data = json.loads(session_file.read_text())
    return Session(**data)


async def _require_localhost(request: Request) -> None:
    """
    Dependency that restricts access to localhost only.

    Raises:
        HTTPException: If request is not from localhost
    """
    client_host = request.client.host if request.client else None
    localhost_addresses = {"127.0.0.1", "::1", "localhost"}

    if client_host not in localhost_addresses:
        raise HTTPException(
            status_code=403,
            detail="Assistant API is only accessible from localhost",
        )


assistant_router = APIRouter(
    prefix="/ajax-api/3.0/mlflow/assistant",
    tags=["assistant"],
    dependencies=[Depends(_require_localhost)],
)


class MessageRequest(BaseModel):
    """Request body for sending messages to the assistant."""

    message: str
    session_id: str | None = None  # empty for the first message
    context: dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    """Response for message endpoint."""

    session_id: str
    stream_url: str


def _format_sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Format an event as SSE."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@assistant_router.post("/message")
async def send_message(request: MessageRequest) -> MessageResponse:
    """
    Send a message to the assistant and get a session for streaming the response.

    Args:
        request: MessageRequest with message, context, and optional session_id

    Returns:
        MessageResponse with session_id and stream_url
    """
    # Generate or use existing session ID
    session_id = request.session_id or str(uuid.uuid4())

    # Create or update session
    session = _load_session(session_id)
    if session is None:
        session = Session(context=request.context)
    elif request.context:
        session.context.update(request.context)

    # Store the pending message
    session.pending_message = request.message
    session.messages.append({"role": "user", "content": request.message})
    _save_session(session_id, session)

    return MessageResponse(
        session_id=session_id,
        stream_url=f"/ajax-api/3.0/mlflow/assistant/stream/{session_id}",
    )


@assistant_router.get("/stream/{session_id}")
async def stream_response(session_id: str) -> StreamingResponse:
    """
    Stream the assistant's response via Server-Sent Events.

    Args:
        session_id: The session ID returned from /message

    Returns:
        StreamingResponse with SSE events
    """
    session = _load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get and clear the pending message
    pending_message = session.pending_message
    if not pending_message:
        raise HTTPException(status_code=400, detail="No pending message to process")
    session.pending_message = None
    _save_session(session_id, session)

    async def event_generator() -> AsyncGenerator[str, None]:
        nonlocal session
        async for event in _provider.run(
            prompt=pending_message,
            session_id=session.provider_session_id,
        ):
            event_type = event["type"]
            event_data = event["data"]

            # Store provider session ID if returned (for conversation continuity)
            if event_type == "done" and event_data.get("session_id"):
                session.provider_session_id = event_data["session_id"]
                _save_session(session_id, session)

            yield _format_sse_event(event_type, event_data)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@assistant_router.get("/status")
async def get_status() -> dict[str, Any]:
    """
    Get the status of the assistant provider.

    Returns:
        Provider status including availability
    """
    return {
        "provider": _provider.name,
        "available": _provider.is_available(),
    }

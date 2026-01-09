"""
Assistant API endpoints for MLflow Server.

This module provides endpoints for integrating AI assistants with MLflow UI,
enabling AI-powered helper through a chat interface.
"""

import ipaddress
import json
import uuid
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mlflow.assistant import get_project_path
from mlflow.assistant.providers.claude_code import ClaudeCodeProvider
from mlflow.assistant.types import EventType
from mlflow.server.assistant.session import SessionManager

# TODO: Hardcoded provider until supporting multiple providers
_provider = ClaudeCodeProvider()

_BLOCK_REMOTE_ACCESS_ERROR_MSG = "Assistant API is only accessible from the same host"


async def _require_localhost(request: Request) -> None:
    """
    Dependency that restricts access to localhost only.

    Uses ipaddress library for robust loopback detection.

    Raises:
        HTTPException: If request is not from localhost
    """
    client_host = request.client.host if request.client else None

    if not client_host:
        raise HTTPException(status_code=403, detail=_BLOCK_REMOTE_ACCESS_ERROR_MSG)

    try:
        ip = ipaddress.ip_address(client_host)
    except ValueError:
        raise HTTPException(status_code=403, detail=_BLOCK_REMOTE_ACCESS_ERROR_MSG)

    if not ip.is_loopback:
        raise HTTPException(status_code=403, detail=_BLOCK_REMOTE_ACCESS_ERROR_MSG)


assistant_router = APIRouter(
    prefix="/ajax-api/3.0/mlflow/assistant",
    tags=["assistant"],
    dependencies=[Depends(_require_localhost)],
)


class MessageRequest(BaseModel):
    message: str
    session_id: str | None = None  # empty for the first message
    experiment_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    session_id: str
    stream_url: str


class GetStatusResponse(BaseModel):
    provider: str
    available: bool


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

    project_path = get_project_path(request.experiment_id) if request.experiment_id else None

    # Create or update session
    session = SessionManager.load(session_id)
    if session is None:
        session = SessionManager.create(context=request.context, working_dir=project_path)
    elif request.context:
        session.update_context(request.context)

    # Store the pending message with role
    session.set_pending_message(role="user", content=request.message)
    session.add_message(role="user", content=request.message)
    SessionManager.save(session_id, session)

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
    session = SessionManager.load(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get and clear the pending message
    pending_message = session.clear_pending_message()
    if not pending_message:
        raise HTTPException(status_code=400, detail="No pending message to process")
    SessionManager.save(session_id, session)

    async def event_generator() -> AsyncGenerator[str, None]:
        nonlocal session
        async for event in _provider.astream(
            prompt=pending_message.content,
            session_id=session.provider_session_id,
            cwd=session.working_dir,
        ):
            # Store provider session ID if returned (for conversation continuity)
            if event.type == EventType.DONE:
                session.provider_session_id = event.data.get("session_id")
                SessionManager.save(session_id, session)

            yield event.to_sse_event()

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
async def get_status() -> GetStatusResponse:
    """
    Get the status of the assistant provider.

    Returns:
        Provider status including availability
    """
    return GetStatusResponse(
        provider=_provider.name,
        available=_provider.is_available(),
    )

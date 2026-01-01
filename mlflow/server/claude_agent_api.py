"""
Claude Agent API endpoints for MLflow Server.

This module provides endpoints for integrating Claude Code Agent SDK with MLflow UI,
enabling AI-powered trace analysis through a chat interface.
"""

import asyncio
import json
import logging
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

_logger = logging.getLogger(__name__)

# Config file location
CLAUDE_CONFIG_FILE = Path.home() / ".mlflow" / "claude-config.json"

# Session storage (in-memory with TTL)
# In production, consider using Redis or similar for persistence
_sessions: dict[str, dict[str, Any]] = {}

# Create FastAPI router
claude_agent_router = APIRouter(prefix="/api/claude-agent", tags=["claude-agent"])


class AnalyzeRequest(BaseModel):
    """Request body for trace analysis."""

    trace_context: str  # Serialized trace data
    prompt: str | None = None  # Optional user prompt
    session_id: str | None = None  # For follow-up messages


class MessageRequest(BaseModel):
    """Request body for follow-up messages."""

    session_id: str
    message: str


class AnalyzeResponse(BaseModel):
    """Response for analyze endpoint."""

    session_id: str
    stream_url: str


def _load_config() -> dict[str, Any]:
    """Load Claude config from file."""
    if CLAUDE_CONFIG_FILE.exists():
        try:
            with open(CLAUDE_CONFIG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _get_claude_path() -> str | None:
    """Get path to Claude CLI executable."""
    return shutil.which("claude")


async def _run_claude_agent(
    prompt: str,
    cwd: str | None = None,
    model: str | None = None,
    session_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Run Claude Agent and stream responses.

    Uses the Claude Agent SDK via subprocess to invoke Claude Code CLI.

    Args:
        prompt: The prompt to send to Claude
        cwd: Working directory for Claude (to read source files)
        model: Model to use (or None for default)
        session_id: Session ID for resume (or None for new session)

    Yields:
        SSE-formatted event strings
    """
    claude_path = _get_claude_path()
    if not claude_path:
        yield f"event: error\ndata: Claude CLI not found. Run 'mlflow claude init' first.\n\n"
        return

    # Build command
    cmd = [claude_path, "-p", prompt, "--output-format", "stream-json"]

    if model and model != "default":
        cmd.extend(["--model", model])

    if session_id and session_id in _sessions:
        stored_session = _sessions[session_id].get("claude_session_id")
        if stored_session:
            cmd.extend(["--resume", stored_session])

    try:
        # Start the Claude process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        # Stream stdout
        claude_session_id = None
        async for line in process.stdout:
            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue

            try:
                data = json.loads(line_str)
                msg_type = data.get("type", "")

                # Extract session ID if present
                if "session_id" in data:
                    claude_session_id = data["session_id"]

                # Handle different message types
                if msg_type == "assistant":
                    content = data.get("message", {}).get("content", [])
                    text_parts = [
                        block.get("text", "")
                        for block in content
                        if block.get("type") == "text"
                    ]
                    if text_parts:
                        text = " ".join(text_parts)
                        yield f"event: message\ndata: {json.dumps({'text': text})}\n\n"

                elif msg_type == "result":
                    # Final result
                    if claude_session_id and session_id:
                        _sessions[session_id]["claude_session_id"] = claude_session_id
                    yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"

                elif msg_type == "error":
                    error_msg = data.get("error", {}).get("message", "Unknown error")
                    yield f"event: error\ndata: {json.dumps({'error': error_msg})}\n\n"

            except json.JSONDecodeError:
                # Non-JSON output, treat as plain text
                yield f"event: message\ndata: {json.dumps({'text': line_str})}\n\n"

        # Wait for process to complete
        await process.wait()

        if process.returncode != 0:
            stderr = await process.stderr.read()
            error_msg = stderr.decode("utf-8").strip() or f"Process exited with code {process.returncode}"
            yield f"event: error\ndata: {json.dumps({'error': error_msg})}\n\n"

    except Exception as e:
        _logger.exception("Error running Claude agent")
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


@claude_agent_router.post("/analyze")
async def analyze_trace(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Start a trace analysis session with Claude.

    This endpoint creates a new session and returns a stream URL
    for receiving Claude's analysis via Server-Sent Events.

    Args:
        request: AnalyzeRequest containing trace context and optional prompt

    Returns:
        AnalyzeResponse with session_id and stream_url
    """
    # Generate or use existing session ID
    session_id = request.session_id or str(uuid.uuid4())

    # Store session data
    _sessions[session_id] = {
        "trace_context": request.trace_context,
        "prompt": request.prompt,
        "messages": [],
    }

    return AnalyzeResponse(
        session_id=session_id,
        stream_url=f"/api/claude-agent/stream/{session_id}",
    )


@claude_agent_router.get("/stream/{session_id}")
async def stream_response(session_id: str) -> StreamingResponse:
    """
    Stream Claude's response via Server-Sent Events.

    Args:
        session_id: The session ID returned from /analyze

    Returns:
        StreamingResponse with SSE events
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    config = _load_config()

    # Build the prompt
    trace_context = session["trace_context"]
    user_prompt = session.get("prompt", "")

    full_prompt = f"""Analyze this MLflow trace and help identify any issues:

{trace_context}

{user_prompt if user_prompt else "Please analyze this trace and explain what happened."}"""

    # Get config values
    cwd = config.get("projectPath")
    model = config.get("model")

    async def event_generator() -> AsyncGenerator[str, None]:
        async for event in _run_claude_agent(
            prompt=full_prompt,
            cwd=cwd,
            model=model,
            session_id=session_id,
        ):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@claude_agent_router.post("/message")
async def send_message(request: MessageRequest) -> StreamingResponse:
    """
    Send a follow-up message in an existing session.

    Args:
        request: MessageRequest with session_id and message

    Returns:
        StreamingResponse with SSE events
    """
    if request.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[request.session_id]
    config = _load_config()

    # Add message to session history
    session["messages"].append({"role": "user", "content": request.message})

    # Get config values
    cwd = config.get("projectPath")
    model = config.get("model")

    async def event_generator() -> AsyncGenerator[str, None]:
        async for event in _run_claude_agent(
            prompt=request.message,
            cwd=cwd,
            model=model,
            session_id=request.session_id,
        ):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@claude_agent_router.get("/health")
async def health_check() -> dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Health status and whether Claude CLI is available
    """
    claude_available = _get_claude_path() is not None
    config_exists = CLAUDE_CONFIG_FILE.exists()

    return {
        "status": "ok",
        "claude_available": str(claude_available),
        "config_exists": str(config_exists),
    }

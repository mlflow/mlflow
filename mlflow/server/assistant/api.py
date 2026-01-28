"""
Assistant API endpoints for MLflow Server.

This module provides endpoints for integrating AI assistants with MLflow UI,
enabling AI-powered helper through a chat interface.
"""

import ipaddress
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mlflow.assistant import clear_project_path_cache, get_project_path
from mlflow.assistant.config import AssistantConfig, PermissionsConfig, ProjectConfig
from mlflow.assistant.providers.base import (
    CLINotInstalledError,
    NotAuthenticatedError,
    clear_config_cache,
)
from mlflow.assistant.providers.claude_code import ClaudeCodeProvider
from mlflow.assistant.skill_installer import install_skills, list_installed_skills
from mlflow.assistant.types import EventType
from mlflow.server.assistant.session import SessionManager, terminate_session_process

# TODO: Hardcoded provider until supporting multiple providers
_provider = ClaudeCodeProvider()


# Update the message when we support proxy access
_BLOCK_REMOTE_ACCESS_ERROR_MSG = (
    "Assistant API is only accessible from the same host where the mLflow server is running."
)


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


# Config-related models
class ConfigResponse(BaseModel):
    providers: dict[str, Any] = Field(default_factory=dict)
    projects: dict[str, Any] = Field(default_factory=dict)


class ConfigUpdateRequest(BaseModel):
    providers: dict[str, Any] | None = None
    projects: dict[str, Any] | None = None


class SessionPatchRequest(BaseModel):
    status: Literal["cancelled"]


class SessionPatchResponse(BaseModel):
    message: str


# Skills-related models
class SkillsInstallRequest(BaseModel):
    type: Literal["global", "project", "custom"] = "global"
    custom_path: str | None = None  # Required if type="custom"
    experiment_id: str | None = None  # Used to get project_path for type="project"


class SkillsInstallResponse(BaseModel):
    installed_skills: list[str]
    skills_directory: str


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
        session = SessionManager.create(
            context=request.context, working_dir=Path(project_path) if project_path else None
        )
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


@assistant_router.get("/sessions/{session_id}/stream")
async def stream_response(request: Request, session_id: str) -> StreamingResponse:
    """
    Stream the assistant's response via Server-Sent Events.

    Args:
        request: The FastAPI request object
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

    # Extract the MLflow server URL from the request for the assistant to use.
    # This assumes the assistant is accessing the same MLflow server that serves this API,
    # which works because the assistant endpoint is localhost-only.
    # TODO: Extend this to support remote/proxy scenarios where the tracking URI may differ.
    tracking_uri = str(request.base_url).rstrip("/")

    async def event_generator() -> AsyncGenerator[str, None]:
        nonlocal session
        async for event in _provider.astream(
            prompt=pending_message.content,
            tracking_uri=tracking_uri,
            session_id=session.provider_session_id,
            mlflow_session_id=session_id,
            cwd=session.working_dir,
            context=session.context,
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


@assistant_router.patch("/sessions/{session_id}")
async def patch_session(session_id: str, request: SessionPatchRequest) -> SessionPatchResponse:
    """
    Update session status.

    Currently supports cancelling an active session, which terminates
    the running assistant process.

    Args:
        session_id: The session ID
        request: SessionPatchRequest with status to set

    Returns:
        SessionPatchResponse indicating success
    """
    session = SessionManager.load(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.status == "cancelled":
        terminated = terminate_session_process(session_id)
        msg = "Session cancelled and process terminated" if terminated else "Session cancelled"
        return SessionPatchResponse(message=msg)

    # This branch is unreachable due to Literal type, but satisfies type checker
    raise HTTPException(status_code=400, detail=f"Unknown status: {request.status}")


@assistant_router.get("/providers/{provider}/health")
async def provider_health_check(provider: str) -> dict[str, str]:
    """
    Check if a specific provider is ready (CLI installed and authenticated).

    Args:
        provider: The provider name (e.g., "claude_code").

    Returns:
        200 with { status: "ok" } if ready.

    Raises:
        HTTPException 404: If provider is not found.
        HTTPException 412: If preconditions not met (CLI not installed or not authenticated).
    """
    # TODO: Support multiple providers via registry
    if provider != _provider.name:
        raise HTTPException(status_code=404, detail=f"Provider '{provider}' not found")

    try:
        _provider.check_connection()
    except CLINotInstalledError as e:
        raise HTTPException(status_code=412, detail=str(e))
    except NotAuthenticatedError as e:
        raise HTTPException(status_code=401, detail=str(e))

    return {"status": "ok"}


@assistant_router.get("/config")
async def get_config() -> ConfigResponse:
    """
    Get the current assistant configuration.

    Returns:
        Current configuration including providers and projects.
    """
    config = AssistantConfig.load()
    return ConfigResponse(
        providers={name: p.model_dump() for name, p in config.providers.items()},
        projects={exp_id: p.model_dump() for exp_id, p in config.projects.items()},
    )


@assistant_router.put("/config")
async def update_config(request: ConfigUpdateRequest) -> ConfigResponse:
    """
    Update the assistant configuration.

    Args:
        request: Partial configuration update.

    Returns:
        Updated configuration.
    """
    config = AssistantConfig.load()

    # Update providers
    if request.providers:
        for name, provider_data in request.providers.items():
            model = provider_data.get("model", "default")
            permissions = None
            if "permissions" in provider_data:
                perm_data = provider_data["permissions"]
                permissions = PermissionsConfig(
                    allow_edit_files=perm_data.get("allow_edit_files", True),
                    allow_read_docs=perm_data.get("allow_read_docs", True),
                    full_access=perm_data.get("full_access", False),
                )
            config.set_provider(name, model, permissions)

    # Update projects
    if request.projects:
        for exp_id, project_data in request.projects.items():
            if project_data is None:
                # Remove project mapping
                config.projects.pop(exp_id, None)
            else:
                location = project_data.get("location", "")
                project_path = Path(location).expanduser()
                if not project_path or not project_path.exists():
                    raise HTTPException(
                        status_code=400,
                        detail=f"Project path does not exist: {location}",
                    )
                config.projects[exp_id] = ProjectConfig(
                    type=project_data.get("type", "local"),
                    location=str(project_path),
                )

    config.save()

    # Clear caches so provider and project path lookups pick up new settings
    clear_config_cache()
    clear_project_path_cache()

    return ConfigResponse(
        providers={name: p.model_dump() for name, p in config.providers.items()},
        projects={exp_id: p.model_dump() for exp_id, p in config.projects.items()},
    )


@assistant_router.post("/skills/install")
async def install_skills_endpoint(request: SkillsInstallRequest) -> SkillsInstallResponse:
    """
    Install skills bundled with MLflow.
    This endpoint only handles installation. Config updates should be done via PUT /config.

    Args:
        request: SkillsInstallRequest with type, custom_path, and experiment_id.

    Returns:
        SkillsInstallResponse with installed skill names and directory.

    Raises:
        HTTPException 400: If custom type without custom_path or project type without experiment_id.
    """
    config = AssistantConfig.load()

    # Resolve project_path for "project" type
    project_path: Path | None = None
    if request.type == "project":
        if not request.experiment_id:
            raise HTTPException(status_code=400, detail="experiment_id required for 'project' type")
        project_location = config.get_project_path(request.experiment_id)
        if not project_location:
            raise HTTPException(
                status_code=400,
                detail=f"No project path configured for experiment {request.experiment_id}",
            )
        project_path = Path(project_location)

    # Get the destination path to install skills to
    match request.type:
        case "global":
            destination = _provider.resolve_skills_path(Path.home())
        case "project":
            destination = _provider.resolve_skills_path(project_path)
        case "custom":
            destination = Path(request.custom_path).expanduser()

    # Check if skills already exist - skip re-installation
    if destination.exists():
        if current_skills := list_installed_skills(destination):
            return SkillsInstallResponse(
                installed_skills=current_skills, skills_directory=str(destination)
            )

    installed = install_skills(destination)

    return SkillsInstallResponse(installed_skills=installed, skills_directory=str(destination))

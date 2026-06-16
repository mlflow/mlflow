import ipaddress
import logging
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Literal

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mlflow.assistant import clear_project_path_cache, get_project_path
from mlflow.assistant.config import AssistantConfig, PermissionsConfig, ProjectConfig
from mlflow.assistant.providers import list_providers
from mlflow.assistant.providers.base import (
    AssistantProvider,
    CLINotInstalledError,
    NotAuthenticatedError,
    ProviderNotConfiguredError,
    clear_config_cache,
)
from mlflow.assistant.skill_installer import install_skills, list_installed_skills
from mlflow.assistant.types import EventType
from mlflow.environment_variables import MLFLOW_ALLOW_REMOTE_ASSISTANT
from mlflow.server.assistant.session import SessionManager, terminate_session_process

_logger = logging.getLogger(__name__)


def _get_provider(name: str):
    for p in list_providers():
        if p.name == name:
            return p
    return None


def _get_selected_provider():
    config = AssistantConfig.load()
    for provider_name, provider_config in config.providers.items():
        if provider_config.selected:
            return _get_provider(provider_name)
    return None


_BLOCK_REMOTE_ACCESS_ERROR_MSG = (
    "Assistant API is only accessible from the same host where the MLflow server is running."
)
_REMOTE_ACCESS_MODES = ("off", "api-only", "all")


def _is_localhost(request: Request) -> bool:
    client_host = request.client.host if request.client else None
    if not client_host:
        return False
    try:
        ip = ipaddress.ip_address(client_host)
    except ValueError:
        return False
    return ip.is_loopback


def _get_remote_access_mode() -> str:
    mode = MLFLOW_ALLOW_REMOTE_ASSISTANT.get()
    if mode not in _REMOTE_ACCESS_MODES:
        _logger.warning(
            "Invalid value %r for %s, falling back to 'off'. Valid values are: %s",
            mode,
            MLFLOW_ALLOW_REMOTE_ASSISTANT.name,
            _REMOTE_ACCESS_MODES,
        )
        return "off"
    return mode


def _provider_allows_remote_access(provider: AssistantProvider | None) -> bool:
    mode = _get_remote_access_mode()
    if mode == "all":
        return True
    if mode == "api-only":
        return provider is not None and not provider.requires_local_execution
    return False


def _enforce_remote_access(request: Request, provider: AssistantProvider | None) -> None:
    if _is_localhost(request):
        return
    if not _provider_allows_remote_access(provider):
        raise HTTPException(status_code=403, detail=_BLOCK_REMOTE_ACCESS_ERROR_MSG)


assistant_router = APIRouter(
    prefix="/ajax-api/3.0/mlflow/assistant",
    tags=["assistant"],
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
    remote_chat_allowed: bool = False


class ConfigUpdateRequest(BaseModel):
    providers: dict[str, Any] | None = None
    projects: dict[str, Any] | None = None


class SessionPatchRequest(BaseModel):
    status: Literal["cancelled"]


class SessionPatchResponse(BaseModel):
    message: str


class PermissionDecision(BaseModel):
    request_id: str  # the paused tool_call's id
    decision: Literal["allow", "deny"]


# Skills-related models
class SkillsInstallRequest(BaseModel):
    type: Literal["global", "project", "custom"] = "global"
    custom_path: str | None = None  # Required if type="custom"
    experiment_id: str | None = None  # Used to get project_path for type="project"


class SkillsInstallResponse(BaseModel):
    installed_skills: list[str]
    skills_directory: str


@assistant_router.post("/message")
async def send_message(http_request: Request, request: MessageRequest) -> MessageResponse:
    """
    Send a message to the assistant and get a session for streaming the response.

    Args:
        http_request: The FastAPI request object, used for remote-access gating
        request: MessageRequest with message, context, and optional session_id

    Returns:
        MessageResponse with session_id and stream_url
    """
    _enforce_remote_access(http_request, _get_selected_provider())

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
    _enforce_remote_access(request, _get_selected_provider())

    session = SessionManager.load(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # A turn is driven by either a pending user message (a new turn) or pending
    # tool-call decisions (resuming a turn paused at a permission prompt). Both
    # are consumed here so the stream is replay-safe.
    pending_message = session.clear_pending_message()
    tool_decisions = session.pending_tool_decisions
    session.pending_tool_decisions = {}
    if not pending_message and not tool_decisions:
        raise HTTPException(status_code=400, detail="No pending message to process")
    SessionManager.save(session_id, session)

    prompt = pending_message.content if pending_message else ""
    # On resume the decision rides in the context; the provider detects the
    # pending tool_calls in history and applies it instead of starting a turn.
    # A new message supersedes a pending decision: if both are present (e.g. a
    # resume stream never consumed the decision and the user typed again),
    # forwarding the stale tool_decisions would make the provider resume the
    # abandoned turn and silently drop the new message. Prefer the message.
    context = dict(session.context)
    if tool_decisions and not pending_message:
        context["tool_decisions"] = tool_decisions

    # Extract the MLflow server URL from the request for the assistant to use.
    # This assumes the assistant is accessing the same MLflow server that serves this API.
    # TODO: Extend this to support remote/proxy scenarios where the tracking URI may differ.
    tracking_uri = str(request.base_url).rstrip("/")

    async def event_generator() -> AsyncGenerator[str, None]:
        nonlocal session
        provider = _get_selected_provider()
        if provider is None:
            from mlflow.assistant.types import Event

            yield Event.from_error(
                "No assistant provider is configured or available."
            ).to_sse_event()
            return
        async for event in provider.astream(
            prompt=prompt,
            tracking_uri=tracking_uri,
            session_id=session.provider_session_id,
            mlflow_session_id=session_id,
            cwd=session.working_dir,
            context=context,
        ):
            # Store provider session ID if returned (for conversation continuity).
            # On a paused turn this persists the history with the unanswered
            # tool_call so a later resume can continue from it.
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
async def patch_session(
    http_request: Request, session_id: str, request: SessionPatchRequest
) -> SessionPatchResponse:
    """
    Update session status.

    Currently supports cancelling an active session, which terminates
    the running assistant process.

    Args:
        http_request: The FastAPI request object, used for remote-access gating
        session_id: The session ID
        request: SessionPatchRequest with status to set

    Returns:
        SessionPatchResponse indicating success
    """
    _enforce_remote_access(http_request, _get_selected_provider())

    session = SessionManager.load(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.status == "cancelled":
        # Terminate any associated subprocess. The OpenAI-compatible provider
        # holds no in-process state to release (the turn ends at each prompt).
        # Drop any tool permissions so later stream doesn't see stale decisions.
        session.pending_tool_decisions = {}
        SessionManager.save(session_id, session)
        terminated = terminate_session_process(session_id)
        msg = "Session cancelled and process terminated" if terminated else "Session cancelled"
        return SessionPatchResponse(message=msg)

    # This branch is unreachable due to Literal type, but satisfies type checker
    raise HTTPException(status_code=400, detail=f"Unknown status: {request.status}")


@assistant_router.post("/sessions/{session_id}/permission")
async def resolve_permission(session_id: str, request: PermissionDecision) -> MessageResponse:
    """Deliver a tool-call permission decision and resume the paused turn on a new stream.

    The decision is stored on the session and consumed by the next stream, which
    re-enters the provider with the choice in context. Stateless across requests:
    any worker can serve the decision because the pending state lives in the
    session, not process memory.
    """
    try:
        SessionManager.validate_session_id(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session = SessionManager.load(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    session.pending_tool_decisions = {request.request_id: request.decision}
    SessionManager.save(session_id, session)

    return MessageResponse(
        session_id=session_id,
        stream_url=f"/ajax-api/3.0/mlflow/assistant/sessions/{session_id}/stream",
    )


@assistant_router.get("/providers/{provider}/health")
async def provider_health_check(http_request: Request, provider: str) -> dict[str, str]:
    p = _get_provider(provider)
    if p is None:
        raise HTTPException(status_code=404, detail=f"Provider '{provider}' not found")

    _enforce_remote_access(http_request, p)

    try:
        p.check_connection()
    except NotImplementedError as e:
        # Presets that delegate verification to the frontend (e.g. the
        # in-server MLflow AI Gateway). Returning a clear 501 prevents the
        # wizard from claiming a successful probe that never ran.
        raise HTTPException(status_code=501, detail=str(e))
    except CLINotInstalledError as e:
        raise HTTPException(status_code=412, detail=str(e))
    except NotAuthenticatedError as e:
        raise HTTPException(status_code=401, detail=str(e))

    return {"status": "ok"}


@assistant_router.get("/config")
async def get_config(request: Request) -> ConfigResponse:
    """
    Get the current assistant configuration.

    Returns:
        Current configuration including providers and projects.
    """
    config = AssistantConfig.load()
    providers = {name: p.model_dump() for name, p in config.providers.items()}
    if not _is_localhost(request):
        for provider_data in providers.values():
            provider_data.pop("api_key", None)

    return ConfigResponse(
        providers=providers,
        projects={exp_id: p.model_dump() for exp_id, p in config.projects.items()},
        remote_chat_allowed=_provider_allows_remote_access(_get_selected_provider()),
    )


@assistant_router.put("/config")
async def update_config(http_request: Request, request: ConfigUpdateRequest) -> ConfigResponse:
    """
    Update the assistant configuration.

    Args:
        http_request: The FastAPI request object, used for remote-access gating
        request: Partial configuration update.

    Returns:
        Updated configuration.
    """
    _enforce_remote_access(http_request, None)

    config = AssistantConfig.load()

    # Update providers
    if request.providers:
        for name, provider_data in request.providers.items():
            existing = config.providers.get(name)
            model = provider_data.get("model") or (existing.model if existing else "default")
            base_url = provider_data.get("base_url")
            api_key = provider_data.get("api_key")
            permissions = None
            if "permissions" in provider_data:
                perm_data = provider_data["permissions"]
                permissions = PermissionsConfig(
                    allow_edit_files=perm_data.get("allow_edit_files", True),
                    allow_read_docs=perm_data.get("allow_read_docs", True),
                    full_access=perm_data.get("full_access", False),
                )
            selected = provider_data.get("selected", False)
            if selected:
                config.set_provider(name, model, permissions, base_url=base_url, api_key=api_key)
            else:
                config.update_provider(
                    name,
                    model=model,
                    permissions=permissions,
                    base_url=base_url,
                    api_key=api_key,
                )

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
        remote_chat_allowed=_provider_allows_remote_access(_get_selected_provider()),
    )


@assistant_router.post("/skills/install")
async def install_skills_endpoint(
    http_request: Request, request: SkillsInstallRequest
) -> SkillsInstallResponse:
    """
    Install skills bundled with MLflow.
    This endpoint only handles installation. Config updates should be done via PUT /config.

    Args:
        http_request: The FastAPI request object, used for remote-access gating
        request: SkillsInstallRequest with type, custom_path, and experiment_id.

    Returns:
        SkillsInstallResponse with installed skill names and directory.

    Raises:
        HTTPException 400: If custom type without custom_path or project type without experiment_id.
    """
    _enforce_remote_access(http_request, _get_selected_provider())

    config = AssistantConfig.load()

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

    provider = _get_selected_provider()
    if provider is None:
        raise HTTPException(
            status_code=412,
            detail="No assistant provider is configured or available.",
        )

    match request.type:
        case "global":
            destination = provider.resolve_skills_path(Path.home())
        case "project":
            destination = provider.resolve_skills_path(project_path)
        case "custom":
            if not request.custom_path:
                raise HTTPException(
                    status_code=400,
                    detail="custom_path is required when type='custom'.",
                )
            destination = Path(request.custom_path).expanduser()

    # Check if skills already exist - skip re-installation
    if destination.exists():
        if current_skills := list_installed_skills(destination):
            return SkillsInstallResponse(
                installed_skills=current_skills, skills_directory=str(destination)
            )

    installed = install_skills(destination)

    return SkillsInstallResponse(installed_skills=installed, skills_directory=str(destination))


@assistant_router.get("/providers/{provider}/models")
async def list_provider_models(
    http_request: Request,
    provider: str,
    base_url: str | None = None,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    # api_key is read from the X-API-Key header (not a query param) so the
    # bearer token doesn't land in access logs, browser history, or referer
    # headers. Remote-access gating mitigates remote exposure but not
    # local logging.
    api_key = x_api_key
    p = _get_provider(provider)
    if p is None:
        raise HTTPException(
            status_code=404,
            detail=f"Provider '{provider}' not found",
        )

    _enforce_remote_access(http_request, p)

    try:
        models = p.list_models(base_url, api_key)
        return {"models": models}
    except NotImplementedError:
        raise HTTPException(
            status_code=404,
            detail=f"Model listing is not supported for provider '{provider}'",
        )
    except CLINotInstalledError as e:
        raise HTTPException(status_code=412, detail=str(e))
    except ProviderNotConfiguredError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )

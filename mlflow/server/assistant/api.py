import asyncio
import enum
import functools
import ipaddress
import logging
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, AsyncGenerator, Literal

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field
from starlette.responses import Response

from mlflow.assistant import clear_project_path_cache, get_project_path
from mlflow.assistant.config import (
    AssistantConfig,
    PermissionsConfig,
    ProjectConfig,
    set_config_user,
)
from mlflow.assistant.config import ProviderConfig as AssistantProviderConfig
from mlflow.assistant.gateway_connection import (
    _GATEWAY_VENDOR_MODELS,
    GatewayUnsupportedError,
    ensure_gateway_connection,
)
from mlflow.assistant.providers import (
    MlflowGatewayProvider,
    list_providers,
    resolve_default_provider,
)
from mlflow.assistant.providers.base import (
    AssistantProvider,
    ClientToolDelivery,
    CLINotInstalledError,
    NotAuthenticatedError,
    ProviderNotConfiguredError,
    assistant_sandbox_enabled,
    clear_config_cache,
)
from mlflow.assistant.providers.tool_executor import set_remote_caller
from mlflow.assistant.skill_installer import install_skills, list_installed_skills
from mlflow.assistant.types import TURN_CONTROL_CONTEXT_KEYS, Event, EventType
from mlflow.environment_variables import MLFLOW_ENABLE_REMOTE_ASSISTANT
from mlflow.server.asgi_utils import get_server_base_url
from mlflow.server.assistant.gateway_permissions import ensure_assistant_gateway_use_permission
from mlflow.server.assistant.identity import (
    BASIC_AUTH_CHALLENGE_HEADERS,
    AssistantAuthError,
    auth_plugin_active,
    resolve_authenticated_username,
)
from mlflow.server.assistant.session import (
    Session,
    SessionManager,
    terminate_session_container,
    terminate_session_process,
)
from mlflow.server.handlers import _add_static_prefix

_logger = logging.getLogger(__name__)


def _get_provider(name: str):
    for p in list_providers():
        if p.name == name:
            return p
    return None


def _get_selected_provider(config: AssistantConfig | None = None):
    """Return only the provider explicitly selected in Assistant config."""
    if config is None:
        config = AssistantConfig.load()
    for provider_name, provider_config in config.providers.items():
        if provider_config.selected:
            return _get_provider(provider_name)
    return None


def _resolve_provider(
    config: AssistantConfig | None = None, remote: bool = False
) -> AssistantProvider | None:
    """Return the explicit provider, or a runtime default for chat routes."""
    selected = _get_selected_provider(config)
    if selected is not None:
        if remote and not selected.allows_remote_access:
            return None
        return selected
    return resolve_default_provider(remote=remote)


_BLOCK_REMOTE_ACCESS_ERROR_MSG = (
    "Assistant API is only accessible from the same host where the MLflow server is running."
)


def _is_localhost(request: Request) -> bool:
    # This app is only ever served via uvicorn (see mlflow/server/__init__.py), which by
    # default trusts X-Forwarded-For from 127.0.0.1 and rewrites request.client.host
    # accordingly. So a same-host reverse proxy on 127.0.0.1 does not defeat this check.
    client_host = request.client.host if request.client else None
    if not client_host:
        return False
    try:
        ip = ipaddress.ip_address(client_host)
    except ValueError:
        return False
    return ip.is_loopback


def _provider_allows_remote_access(provider: AssistantProvider | None) -> bool:
    if provider is None:
        return False
    # Remote access requires the sandbox: the Assistant's server-side tools (Bash and the file
    # tools, including python) run on the host without it, so a remote caller could execute
    # arbitrary code there. With the sandbox on, tool execution runs isolated in a container. The
    # CLI providers already gate their own allows_remote_access on the sandbox; requiring it here
    # makes the gateway provider require it too, so remote tool execution is always sandboxed.
    return (
        MLFLOW_ENABLE_REMOTE_ASSISTANT.get()
        and assistant_sandbox_enabled()
        and provider.allows_remote_access
    )


def _enforce_remote_access(request: Request, provider: AssistantProvider | None) -> None:
    if _is_localhost(request):
        return
    if not _provider_allows_remote_access(provider):
        raise HTTPException(status_code=403, detail=_BLOCK_REMOTE_ACCESS_ERROR_MSG)


# Per-route remote-access policy:
#   ONLY_SAFE_PROVIDER — gate on the provider identified by a {provider} path parameter,
#                        falling back to whichever provider the user has currently selected
#   AUTHENTICATED      — allow a remote caller only on an authenticated server (so the request has
#                        an identity to attribute it to); used for per-user config writes, which
#                        are not tool execution and so are not gated on a safe provider/the sandbox
#   DENY               — always block remote access (stays localhost-only regardless of mode)
#   NONE               — no gating (e.g. GET /config, which redacts secrets instead)
class _RemoteAccessPolicy(str, enum.Enum):
    ONLY_SAFE_PROVIDER = "only_safe_provider"
    AUTHENTICATED = "authenticated"
    DENY = "deny"
    NONE = "none"


_REMOTE_ACCESS_POLICY_ATTR = "_assistant_remote_access_policy"


def _remote_access_policy(policy: _RemoteAccessPolicy):
    def decorator(func):
        setattr(func, _REMOTE_ACCESS_POLICY_ATTR, policy)
        return func

    return decorator


def _get_route_provider(request: Request) -> AssistantProvider | None:
    if provider_name := request.path_params.get("provider"):
        return _get_provider(provider_name)
    return _resolve_provider(remote=not _is_localhost(request))


def _current_username(request: Request) -> str | None:
    # Set by _AssistantAPIRoute.route_handler before any endpoint runs; None on a no-auth server.
    return request.state.assistant_username


def _session_owned_by(session: Session, username: str | None) -> bool:
    """Whether ``session`` belongs to ``username``.

    On a no-auth server both sides are None, so this is a no-op match; on an authenticated server
    ``username`` is always a real user (never None), so a session owned by a different user (or an
    unowned legacy session, ``owner is None``) does not match.
    """
    return session.owner == username


def _load_owned_session(session_id: str, username: str | None) -> Session | None:
    """Load a session only if it belongs to ``username``.

    Returns None when the session does not exist OR is owned by a different user, so callers treat
    "not yours" the same as "not found" (a 404) and one user cannot read or drive another user's
    session by its id.
    """
    session = SessionManager.load(session_id)
    if session is None:
        return None
    if not _session_owned_by(session, username):
        _logger.debug(
            "Assistant session %s requested by a user that does not own it; denying", session_id
        )
        return None
    return session


class _AssistantAPIRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        original_route_handler = super().get_route_handler()
        policy: _RemoteAccessPolicy | None = getattr(
            self.endpoint, _REMOTE_ACCESS_POLICY_ATTR, None
        )
        if policy is None:
            raise RuntimeError(
                f"Assistant route {self.path!r} ({self.endpoint.__name__}) is missing a "
                f"remote-access policy. Add @_remote_access_policy(...) to the endpoint."
            )

        async def route_handler(request: Request) -> Response:
            if policy != _RemoteAccessPolicy.NONE and not _is_localhost(request):
                if policy == _RemoteAccessPolicy.DENY or not MLFLOW_ENABLE_REMOTE_ASSISTANT.get():
                    raise HTTPException(status_code=403, detail=_BLOCK_REMOTE_ACCESS_ERROR_MSG)
                if policy == _RemoteAccessPolicy.AUTHENTICATED:
                    # Per-user config writes: allowed remotely only on an authenticated server, so
                    # the write can be attributed to a user (a no-auth server has no identity and
                    # stays localhost-only). No provider/sandbox gate -- this is not tool execution.
                    # The identity resolution below rejects an unauthenticated remote caller (401).
                    if not auth_plugin_active():
                        raise HTTPException(status_code=403, detail=_BLOCK_REMOTE_ACCESS_ERROR_MSG)
                else:
                    provider = _get_route_provider(request)
                    # A {provider} path param that doesn't resolve to a known provider is a
                    # 404, not a remote-access decision; let the endpoint handle it.
                    if not ("provider" in request.path_params and provider is None):
                        _enforce_remote_access(request, provider)
            # Establish the caller's authenticated identity (None on a no-auth server) so per-user
            # features can key on it. On an authenticated server this also stops the Assistant from
            # being driven anonymously, since the FastAPI routes are not covered by the auth
            # plugin's Flask before-request handlers.
            try:
                request.state.assistant_username = resolve_authenticated_username(request)
            except AssistantAuthError as e:
                raise HTTPException(
                    status_code=401, detail=str(e), headers=BASIC_AUTH_CHALLENGE_HEADERS
                ) from e
            # Bind the user for per-user config resolution (providers). Set on the request's own
            # asyncio context, so it also applies while the streaming response body runs; each
            # request runs in its own context, so this does not leak across requests.
            set_config_user(request.state.assistant_username)
            # Cap a remote (non-localhost) caller at the restricted tool-permission profile, so
            # server-side tool execution cannot be driven with full_access over the network.
            set_remote_caller(not _is_localhost(request))
            return await original_route_handler(request)

        return route_handler


assistant_router = APIRouter(
    prefix="/ajax-api/3.0/mlflow/assistant",
    tags=["assistant"],
    route_class=_AssistantAPIRoute,
)

_TURN_SCOPED_CONTEXT_KEYS = {"customTraceView"}


class MessageRequest(BaseModel):
    message: str
    session_id: str | None = None  # empty for the first message
    experiment_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    session_id: str
    stream_url: str


class ChatRequest(BaseModel):
    message: str
    experiment_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    # Full conversation history as a JSON blob carried by the client. It is passed
    # to the provider as the provider session ID and is not persisted server-side.
    conversation_history: str | None = None
    # tool_call_id -> "allow" | "deny". Carried by the client when resuming a turn paused at a
    # permission prompt; the provider applies it to the matching pending tool_call already in the
    # carried history. Keeps permission state off the server on the stateless path.
    tool_decisions: dict[str, Literal["allow", "deny"]] | None = None
    # Results for browser-executed tools, keyed by the pending tool-call ID. Like permission
    # decisions, these are turn controls rather than model-visible page context.
    client_tool_results: dict[str, "ClientToolResultPayload"] | None = None


class ClientToolResultPayload(BaseModel):
    content: str
    is_error: bool = False


# Config-related models
class ConfigResponse(BaseModel):
    providers: dict[str, Any] = Field(default_factory=dict)
    projects: dict[str, Any] = Field(default_factory=dict)
    remote_access_allowed: bool = False


class ConfigUpdateRequest(BaseModel):
    providers: dict[str, Any] | None = None
    projects: dict[str, Any] | None = None


class ProviderInfo(BaseModel):
    name: str
    display_name: str
    description: str
    available: bool
    selected: bool
    requires_api_key: bool
    has_api_key: bool
    allows_remote_access: bool
    client_carries_history: bool
    # How client-executed actions are delivered: as native tool calls, terminal
    # structured output, or not supported by this provider.
    client_tool_delivery: ClientToolDelivery = "unsupported"
    model_options: list[str] = Field(default_factory=list)


class ResolvedProviderInfo(BaseModel):
    name: str
    model: str | None = None
    auto_selected: bool
    requires_api_key: bool
    has_api_key: bool
    client_carries_history: bool
    client_tool_delivery: ClientToolDelivery = "unsupported"
    model_provider: str | None = None
    model_options: list[str] = Field(default_factory=list)
    provider_model: str | None = None


class ProvidersResponse(BaseModel):
    providers: list[ProviderInfo]
    resolved: ResolvedProviderInfo | None
    gateway_vendor_options: dict[str, list[str]] = Field(default_factory=dict)


class SessionPatchRequest(BaseModel):
    status: Literal["cancelled"]


class SessionPatchResponse(BaseModel):
    message: str


class PermissionDecision(BaseModel):
    request_id: str  # the paused tool_call's id
    decision: Literal["allow", "deny"]


class ClientToolResult(BaseModel):
    request_id: str  # the paused tool_call's id
    content: str
    is_error: bool = False


def _store_gateway_api_key(name: str, provider_data: dict[str, Any]) -> str | None:
    api_key = provider_data.get("api_key")
    gateway_vendor = provider_data.get("gateway_vendor")
    if not api_key and gateway_vendor is None:
        return None
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Gateway vendor connections require an API key.",
        )
    if name != MlflowGatewayProvider.GATEWAY_PROVIDER_NAME:
        raise HTTPException(
            status_code=400,
            detail="API keys must be stored in LLM Connections through the "
            "'mlflow_gateway' provider.",
        )
    if gateway_vendor is None:
        raise HTTPException(
            status_code=400,
            detail="Gateway API keys require a gateway_vendor.",
        )
    try:
        return ensure_gateway_connection(gateway_vendor, api_key)
    except (GatewayUnsupportedError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _gateway_vendor_options() -> dict[str, list[str]]:
    return {vendor: [model] for vendor, model in _GATEWAY_VENDOR_MODELS.items()}


def _gateway_vendor_from_managed_endpoint(model: str | None) -> str | None:
    if not model:
        return None
    prefix = "mlflow-assistant-"
    vendor = model.removeprefix(prefix)
    if vendor == model:
        return None
    return vendor if vendor in _GATEWAY_VENDOR_MODELS else None


def _resolved_provider_info(
    provider: AssistantProvider,
    provider_config: Any | None,
    *,
    auto_selected: bool,
) -> ResolvedProviderInfo:
    model = (
        provider_config.model if provider_config and provider_config.model != "default" else None
    )
    resolved = ResolvedProviderInfo(
        name=provider.name,
        model=model,
        auto_selected=auto_selected,
        requires_api_key=False,
        has_api_key=False,
        client_carries_history=provider.client_carries_history,
        client_tool_delivery=provider.client_tool_delivery,
    )
    if provider.name == MlflowGatewayProvider.GATEWAY_PROVIDER_NAME:
        if vendor := _gateway_vendor_from_managed_endpoint(model):
            provider_model = _GATEWAY_VENDOR_MODELS[vendor]
            resolved.model_provider = vendor
            resolved.model_options = [provider_model]
            resolved.provider_model = provider_model
            resolved.has_api_key = True
    return resolved


def _resolve_assistant_provider(
    config: AssistantConfig,
    providers: list[AssistantProvider],
) -> ResolvedProviderInfo | None:
    for provider in providers:
        provider_config = config.providers.get(provider.name)
        if provider_config and provider_config.selected:
            return _resolved_provider_info(provider, provider_config, auto_selected=False)

    for provider in providers:
        if provider.name == MlflowGatewayProvider.GATEWAY_PROVIDER_NAME:
            continue
        if provider.is_available():
            return _resolved_provider_info(provider, None, auto_selected=True)
    return None


# Skills-related models
class SkillsInstallRequest(BaseModel):
    type: Literal["global", "project", "custom"] = "global"
    custom_path: str | None = None  # Required if type="custom"
    experiment_id: str | None = None  # Used to get project_path for type="project"


class SkillsInstallResponse(BaseModel):
    installed_skills: list[str]
    skills_directory: str


@assistant_router.post("/message")
@_remote_access_policy(_RemoteAccessPolicy.ONLY_SAFE_PROVIDER)
async def send_message(request: MessageRequest, http_request: Request) -> MessageResponse:
    """
    Send a message to the assistant and get a session for streaming the response.

    Args:
        request: MessageRequest with message, context, and optional session_id
        http_request: The FastAPI request object, carrying the authenticated user

    Returns:
        MessageResponse with session_id and stream_url
    """
    username = _current_username(http_request)
    # Generate or use existing session ID
    session_id = request.session_id or str(uuid.uuid4())

    project_path = get_project_path(request.experiment_id) if request.experiment_id else None

    # Create or update session
    session = SessionManager.load(session_id)
    if session is not None and not _session_owned_by(session, username):
        # The id belongs to another user; treat as not found rather than reading or overwriting it.
        _logger.debug(
            "Assistant session %s requested by a user that does not own it; denying", session_id
        )
        raise HTTPException(status_code=404, detail="Session not found")
    if session is None:
        session = SessionManager.create(
            context=request.context,
            working_dir=Path(project_path) if project_path else None,
            owner=username,
        )
    else:
        # Page context is merged for conversation continuity, but feature modes
        # are turn-scoped. Remove omitted transient keys so leaving a feature
        # cannot keep later turns in its provider/output mode.
        for key in _TURN_SCOPED_CONTEXT_KEYS - request.context.keys():
            session.context.pop(key, None)
        session.update_context(request.context)
        # A session created without a project directory (e.g. the first message
        # had no experiment_id) never got a working_dir. If a later message
        # resolves one, fill it in instead of leaving the session permanently
        # without file-tool access. Only fills a missing working_dir; an
        # already-configured one is never replaced.
        if session.working_dir is None and project_path:
            session.working_dir = Path(project_path)

    # Store the pending message with role
    session.set_pending_message(role="user", content=request.message)
    session.add_message(role="user", content=request.message)
    SessionManager.save(session_id, session)

    return MessageResponse(
        session_id=session_id,
        stream_url=_add_static_prefix(
            f"/ajax-api/3.0/mlflow/assistant/sessions/{session_id}/stream"
        ),
    )


async def stream_provider_events(
    start_stream: Callable[[], AsyncGenerator[Event, None]] | None,
) -> AsyncGenerator[Event, None]:
    """Relay a provider's event stream, or a single error event if none is configured.

    ``start_stream`` is a thunk that opens the provider's ``astream``/``astream_stateless``
    generator (the caller binds the right one for its path), or ``None`` when no provider is
    available. The thunk is invoked *inside* the try block so a provider that raises on entry
    (e.g. one missing the method for this path) still terminates the turn with a clean error
    event instead of dropping the connection. Yields ``Event`` objects so callers can both
    serialize them to SSE and react to specific events (e.g. the stateful path persisting the
    provider session id on DONE).
    """
    if start_stream is None:
        yield Event.from_error("No assistant provider is configured or available.")
        return
    try:
        async for event in start_stream():
            yield event
    except Exception:
        # A provider blowing up mid-stream would otherwise drop the connection with no terminal
        # event, leaving the client spinning forever. Emit a clean error event instead so every
        # turn ends with either a done or an error frame. This path is reachable by a remote
        # client (the stateless backend is meant for remotely hosted MLflow), so don't leak the
        # raw exception — log the full detail server-side and return a generic message.
        _logger.exception("Assistant provider stream failed")
        yield Event.from_error("The assistant encountered an unexpected error. Please try again.")


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@assistant_router.get("/sessions/{session_id}/stream")
@_remote_access_policy(_RemoteAccessPolicy.ONLY_SAFE_PROVIDER)
async def stream_response(request: Request, session_id: str) -> StreamingResponse:
    """
    Stream the assistant's response via Server-Sent Events.

    Args:
        request: The FastAPI request object
        session_id: The session ID returned from /message

    Returns:
        StreamingResponse with SSE events
    """
    username = _current_username(request)
    session = _load_owned_session(session_id, username)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # A turn is driven by a pending user message (a new turn) or pending tool-call
    # decisions/results (resuming a turn paused at a permission prompt or a
    # client-executed tool call). All are consumed here so the stream is replay-safe.
    pending_message = session.clear_pending_message()
    tool_decisions = session.pending_tool_decisions
    session.pending_tool_decisions = {}
    client_tool_results = session.pending_client_tool_results
    session.pending_client_tool_results = {}
    if not pending_message and not tool_decisions and not client_tool_results:
        raise HTTPException(status_code=400, detail="No pending message to process")
    SessionManager.save(session_id, session)

    prompt = pending_message.content if pending_message else ""
    # On resume the decision/result rides in the context; the provider detects the
    # pending tool_calls in history and applies it instead of starting a turn.
    # A new message supersedes pending decisions/results: if both are present (e.g. a
    # resume stream never consumed them and the user typed again), forwarding the
    # stale state would make the provider resume the abandoned turn and silently
    # drop the new message. Prefer the message.
    context = dict(session.context)
    if not pending_message:
        if tool_decisions:
            context["tool_decisions"] = tool_decisions
        if client_tool_results:
            context["client_tool_results"] = client_tool_results

    # Extract the MLflow server URL from the request for the assistant to use.
    # This assumes the assistant is accessing the same MLflow server that serves this API.
    # TODO: Extend this to support remote/proxy scenarios where the tracking URI may differ.
    tracking_uri = get_server_base_url(request)
    is_remote = not _is_localhost(request)
    provider = await asyncio.to_thread(_resolve_provider, remote=is_remote)
    if provider is not None and provider.client_carries_history:
        raise HTTPException(
            status_code=400,
            detail="This provider requires the stateless /chat endpoint.",
        )
    if provider is not None and provider.name == MlflowGatewayProvider.GATEWAY_PROVIDER_NAME:
        # The in-server gateway enforces a per-endpoint USE permission. The Assistant's
        # managed endpoints are created outside the HTTP route that would grant it, so
        # authorize this caller for them before the turn calls the gateway.
        await asyncio.to_thread(ensure_assistant_gateway_use_permission, username)

    async def event_generator() -> AsyncGenerator[str, None]:
        nonlocal session
        start_stream = (
            functools.partial(
                provider.astream,
                prompt=prompt,
                tracking_uri=tracking_uri,
                session_id=session.provider_session_id,
                mlflow_session_id=session_id,
                cwd=session.working_dir,
                context=context,
            )
            if provider is not None
            else None
        )
        async for event in stream_provider_events(start_stream):
            # Store provider session ID if returned (for conversation continuity).
            # On a paused or failed turn this lets a later request resume the same
            # provider conversation instead of losing its history.
            provider_session_id = event.data.get("session_id")
            if event.type in {EventType.DONE, EventType.ERROR} and provider_session_id:
                session.provider_session_id = provider_session_id
                SessionManager.save(session_id, session)

            yield event.to_sse_event()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@assistant_router.post("/chat")
@_remote_access_policy(_RemoteAccessPolicy.ONLY_SAFE_PROVIDER)
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    """Stateless streaming chat for client-carried-history providers."""
    is_remote = not _is_localhost(request)
    provider = await asyncio.to_thread(_resolve_provider, remote=is_remote)
    if provider is not None and not provider.client_carries_history:
        raise HTTPException(
            status_code=400,
            detail="This provider does not support the stateless /chat endpoint.",
        )
    username = _current_username(request)
    if provider is not None and provider.name == MlflowGatewayProvider.GATEWAY_PROVIDER_NAME:
        await asyncio.to_thread(ensure_assistant_gateway_use_permission, username)
    project_path = get_project_path(body.experiment_id) if body.experiment_id else None
    cwd = Path(project_path) if project_path else None
    tracking_uri = get_server_base_url(request)

    # On resume the decision rides in the context; the provider detects the pending tool_calls in
    # the carried history and applies it instead of starting a new turn.
    # Turn controls have typed top-level fields. Never accept lookalikes from the arbitrary page
    # context map: doing so would bypass validation and let context metadata drive tool execution.
    context = {
        key: value for key, value in body.context.items() if key not in TURN_CONTROL_CONTEXT_KEYS
    }
    if body.tool_decisions:
        context["tool_decisions"] = body.tool_decisions
    if body.client_tool_results:
        context["client_tool_results"] = {
            request_id: result.model_dump()
            for request_id, result in body.client_tool_results.items()
        }

    async def event_generator() -> AsyncGenerator[str, None]:
        start_stream = (
            functools.partial(
                provider.astream_stateless,
                prompt=body.message,
                tracking_uri=tracking_uri,
                conversation_history=body.conversation_history,
                cwd=cwd,
                context=context,
            )
            if provider is not None
            else None
        )
        async for event in stream_provider_events(start_stream):
            yield event.to_sse_event()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@assistant_router.patch("/sessions/{session_id}")
@_remote_access_policy(_RemoteAccessPolicy.ONLY_SAFE_PROVIDER)
async def patch_session(
    session_id: str, request: SessionPatchRequest, http_request: Request
) -> SessionPatchResponse:
    """
    Update session status.

    Currently supports cancelling an active session, which terminates
    the running assistant process.

    Args:
        session_id: The session ID
        request: SessionPatchRequest with status to set
        http_request: The FastAPI request object, carrying the authenticated user

    Returns:
        SessionPatchResponse indicating success
    """
    session = _load_owned_session(session_id, _current_username(http_request))
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.status == "cancelled":
        # Terminate any associated subprocess. The OpenAI-compatible provider
        # holds no in-process state to release (the turn ends at each prompt).
        # Drop any tool permissions/results so later stream doesn't see stale state.
        session.pending_tool_decisions = {}
        session.pending_client_tool_results = {}
        SessionManager.save(session_id, session)
        # A turn runs either as a host subprocess or (with the sandbox enabled) in a container.
        # Attempt both (do not short-circuit) and report if either was actually terminated.
        proc_terminated = terminate_session_process(session_id)
        # terminate_session_container makes blocking Docker-socket calls; run it off the event
        # loop so a slow/unhealthy Docker daemon can't stall unrelated requests.
        container_terminated = await asyncio.to_thread(terminate_session_container, session_id)
        terminated = proc_terminated or container_terminated
        msg = (
            "Session cancelled and process/sandbox terminated"
            if terminated
            else "Session cancelled"
        )
        return SessionPatchResponse(message=msg)

    # This branch is unreachable due to Literal type, but satisfies type checker
    raise HTTPException(status_code=400, detail=f"Unknown status: {request.status}")


@assistant_router.post("/sessions/{session_id}/permission")
@_remote_access_policy(_RemoteAccessPolicy.ONLY_SAFE_PROVIDER)
async def resolve_permission(
    session_id: str, request: PermissionDecision, http_request: Request
) -> MessageResponse:
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

    session = _load_owned_session(session_id, _current_username(http_request))
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    session.pending_tool_decisions = {request.request_id: request.decision}
    SessionManager.save(session_id, session)

    return MessageResponse(
        session_id=session_id,
        stream_url=_add_static_prefix(
            f"/ajax-api/3.0/mlflow/assistant/sessions/{session_id}/stream"
        ),
    )


@assistant_router.post("/sessions/{session_id}/tool-result")
@_remote_access_policy(_RemoteAccessPolicy.ONLY_SAFE_PROVIDER)
async def resolve_client_tool_result(
    session_id: str, request: ClientToolResult, http_request: Request
) -> MessageResponse:
    """Deliver a client-executed tool's result and resume the paused turn on a new stream.

    Mirrors `resolve_permission`: the result is stored on the session and consumed
    by the next stream, which splices it in as the tool's result message and
    continues the loop.
    """
    try:
        SessionManager.validate_session_id(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session = _load_owned_session(session_id, _current_username(http_request))
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    session.pending_client_tool_results = {
        request.request_id: {"content": request.content, "is_error": request.is_error}
    }
    SessionManager.save(session_id, session)

    return MessageResponse(
        session_id=session_id,
        stream_url=f"/ajax-api/3.0/mlflow/assistant/sessions/{session_id}/stream",
    )


@assistant_router.get("/providers/{provider}/health")
@_remote_access_policy(_RemoteAccessPolicy.ONLY_SAFE_PROVIDER)
async def provider_health_check(provider: str) -> dict[str, str]:
    p = _get_provider(provider)
    if p is None:
        raise HTTPException(status_code=404, detail=f"Provider '{provider}' not found")

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


@assistant_router.get("/providers")
@_remote_access_policy(_RemoteAccessPolicy.NONE)
async def get_providers() -> ProvidersResponse:
    config = AssistantConfig.load()
    providers = list_providers()
    provider_infos = [
        ProviderInfo(
            name=provider.name,
            display_name=provider.display_name,
            description=provider.description,
            available=provider.is_available(),
            selected=bool(config.providers.get(provider.name, None))
            and config.providers[provider.name].selected,
            requires_api_key=False,
            has_api_key=False,
            allows_remote_access=provider.allows_remote_access,
            client_carries_history=provider.client_carries_history,
            client_tool_delivery=provider.client_tool_delivery,
            model_options=[],
        )
        for provider in providers
    ]
    return ProvidersResponse(
        providers=provider_infos,
        resolved=_resolve_assistant_provider(config, providers),
        gateway_vendor_options=_gateway_vendor_options(),
    )


@assistant_router.get("/config")
@_remote_access_policy(_RemoteAccessPolicy.NONE)
async def get_config(request: Request) -> ConfigResponse:
    """
    Get the current assistant configuration.

    Returns:
        Current configuration including providers and projects.
    """
    config = AssistantConfig.load()
    capabilities = {p.name: p.client_carries_history for p in list_providers()}
    providers = {name: p.model_dump() for name, p in config.providers.items()}
    is_remote = not _is_localhost(request)
    selected_provider = _get_selected_provider(config)
    provider = selected_provider or resolve_default_provider(
        remote=is_remote, include_gateway=False
    )
    if selected_provider is None and provider is not None:
        provider_config = config.providers.get(provider.name) or AssistantProviderConfig()
        provider_data = provider_config.model_dump()
        provider_data["selected"] = True
        providers[provider.name] = provider_data
    for name, provider_data in providers.items():
        provider_data["client_carries_history"] = capabilities.get(name, False)
    for provider_data in providers.values():
        provider_data.pop("api_key", None)

    projects = {exp_id: p.model_dump() for exp_id, p in config.projects.items()}
    if not _is_localhost(request):
        for project_data in projects.values():
            project_data.pop("location", None)

    return ConfigResponse(
        providers=providers,
        projects=projects,
        remote_access_allowed=_provider_allows_remote_access(provider),
    )


@assistant_router.put("/config")
@_remote_access_policy(_RemoteAccessPolicy.AUTHENTICATED)
async def update_config(request: ConfigUpdateRequest, http_request: Request) -> ConfigResponse:
    """
    Update the assistant configuration.

    A remote (authenticated) caller may only change their own per-user provider settings (selected
    provider, model, permissions, base URL), which are saved to their own config. Server-level
    changes -- registering project directories, and creating gateway LLM connections (API keys) --
    stay localhost-only, since they affect the whole server rather than one user.

    Args:
        request: Partial configuration update.
        http_request: The FastAPI request object, used to distinguish local from remote callers.

    Returns:
        Updated configuration.
    """
    if not _is_localhost(http_request):
        if request.projects:
            raise HTTPException(
                status_code=403,
                detail="Project directories can only be configured from the MLflow server host.",
            )
        for provider_data in (request.providers or {}).values():
            # `providers` values are typed `Any`, so a malformed remote payload may not be a dict.
            # Reject it up front instead of letting `.get` raise an unhandled 500.
            if not isinstance(provider_data, dict):
                raise HTTPException(status_code=400, detail="Invalid provider configuration.")
            if provider_data.get("api_key") or provider_data.get("gateway_vendor"):
                raise HTTPException(
                    status_code=403,
                    detail="Gateway connections (API keys) can only be configured from the "
                    "MLflow server host.",
                )
            # Full access bypasses all permission checks, so it is host-only like the fields above.
            # The runtime clamp already neutralizes it for remote tool execution; rejecting the
            # write keeps the persisted config honest and the enforcement in one place.
            permissions = provider_data.get("permissions")
            if isinstance(permissions, dict) and permissions.get("full_access"):
                raise HTTPException(
                    status_code=403,
                    detail="Full access can only be enabled from the MLflow server host.",
                )

    config = AssistantConfig.load()

    # Update providers
    if request.providers:
        for name, provider_data in request.providers.items():
            existing = config.providers.get(name)
            model = provider_data.get("model") or (existing.model if existing else "default")
            base_url = provider_data.get("base_url")
            if gateway_model := _store_gateway_api_key(name, provider_data):
                model = gateway_model
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
                config.set_provider(name, model, permissions, base_url=base_url)
            else:
                config.update_provider(
                    name,
                    model=model,
                    permissions=permissions,
                    base_url=base_url,
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

    providers = {name: p.model_dump() for name, p in config.providers.items()}
    for provider_data in providers.values():
        provider_data.pop("api_key", None)

    return ConfigResponse(
        providers=providers,
        projects={exp_id: p.model_dump() for exp_id, p in config.projects.items()},
        remote_access_allowed=_provider_allows_remote_access(_get_selected_provider(config)),
    )


@assistant_router.post("/skills/install")
@_remote_access_policy(_RemoteAccessPolicy.DENY)
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

    # Skills installation has side effects, so it requires an explicit provider
    # selection instead of using _resolve_provider()'s runtime default.
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
@_remote_access_policy(_RemoteAccessPolicy.ONLY_SAFE_PROVIDER)
async def list_provider_models(
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

import base64
import json
import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from flask import Flask, Response, jsonify, make_response, redirect, request
from werkzeug.datastructures import Authorization

from mlflow.environment_variables import MLFLOW_FLASK_SERVER_SECRET_KEY
from mlflow.exceptions import MlflowException
from mlflow.server import app
from mlflow.server.auth import (
    AFTER_REQUEST_HANDLERS,
    WORKSPACE_PARAMETERIZED_BEFORE_REQUEST_VALIDATORS,
    _find_validator,
    _get_workspace_validator,
    create_admin_user,
    is_unprotected_route,
    store,
)
from mlflow.server.auth.oauth.config import read_oauth_config
from mlflow.server.auth.oauth.external_authz import ExternalAuthzClient
from mlflow.server.auth.oauth.login_page import render_login_page
from mlflow.server.auth.oauth.oidc import (
    handle_oidc_callback,
    start_oidc_flow,
    validate_bearer_token,
)
from mlflow.server.auth.oauth.provisioning import UserProvisioner
from mlflow.server.auth.oauth.rate_limit import auth_rate_limiter
from mlflow.server.auth.oauth.routes import (
    AUTH_CALLBACK,
    AUTH_CONFIG,
    AUTH_LOGIN,
    AUTH_LOGOUT,
    AUTH_SAML_ACS,
    AUTH_SAML_SLO,
    AUTH_SESSION,
    AUTH_START,
)
from mlflow.server.auth.oauth.session import SessionManager
from mlflow.server.handlers import catch_mlflow_exception

_logger = logging.getLogger(__name__)

# Module-level state
_oauth_config = None
_session_manager = None
_provisioner = None
_external_authz = None
_cleanup_timer = None

# Routes that don't require authentication
_OAUTH_UNPROTECTED_ROUTES = frozenset(
    {
        "/auth/login",
        "/auth/callback",
        "/auth/saml/acs",
        "/auth/saml/slo",
    }
)


def _is_oauth_unprotected(path: str) -> bool:
    return (
        path in _OAUTH_UNPROTECTED_ROUTES
        or path.startswith("/auth/start/")
        or path.startswith("/auth/admin/")
    )


def authenticate_request_oauth() -> Authorization | Response:
    # 1. Check session cookie
    if session_id := _session_manager.get_session_id_from_cookie(request):
        if session_info := _session_manager.validate_session(session_id):
            claims = session_info.get("id_token_claims") or {}
            if username := claims.get("username", ""):
                # Check if token needs refresh
                if _session_manager.should_refresh_token(session_info):
                    _try_refresh_token(session_id, session_info)
                return Authorization("session", {"username": username})

    # 2. Check Bearer token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        if token_info := validate_bearer_token(token, _oauth_config):
            # Auto-provision on first bearer token use
            username = token_info["username"]
            provider = _oauth_config.get_provider(token_info["provider"].split(":", 1)[-1])
            if provider and _oauth_config.auto_provision_users:
                try:
                    _provisioner.provision_user(
                        username=username,
                        provider_config=provider,
                        groups=token_info.get("groups", []),
                    )
                except Exception:
                    _logger.debug("Bearer token user provisioning skipped", exc_info=True)
            return Authorization("bearer", {"username": username})

    # 3. Check Basic Auth fallback
    if _oauth_config.allow_basic_auth_fallback and request.authorization:
        username = request.authorization.username
        password = request.authorization.password
        if store.authenticate_user(username, password):
            return request.authorization

    # Not authenticated
    accept = request.headers.get("Accept", "")
    if "text/html" in accept:
        next_url = request.url
        return redirect(f"/auth/login?next={next_url}")

    res = make_response("Authentication required")
    res.status_code = 401
    res.headers["WWW-Authenticate"] = 'Bearer realm="mlflow"'
    return res


def _try_refresh_token(session_id: str, session_info: dict[str, object]):
    from mlflow.server.auth.oauth.oidc import refresh_access_token

    provider_str = session_info.get("provider", "")
    if not provider_str.startswith("oidc:"):
        return

    provider_name = provider_str.split(":", 1)[1]
    provider = _oauth_config.oidc_providers.get(provider_name)
    if not provider:
        return

    tokens = _session_manager.get_session_tokens(session_id)
    if not tokens or not tokens.get("refresh_token"):
        return

    new_tokens = refresh_access_token(provider, tokens["refresh_token"])
    if not new_tokens:
        return

    token_expiry = datetime.now(timezone.utc) + timedelta(
        seconds=new_tokens.get("expires_in", 3600)
    )

    _session_manager.update_session_tokens(
        old_session_id=session_id,
        access_token=new_tokens["access_token"],
        refresh_token=new_tokens.get("refresh_token", ""),
        token_expiry=token_expiry,
    )


def _check_external_authz(username: str, resource_type: str, resource_id: str, action: str):
    if not _external_authz or not _oauth_config.external_authz.enabled:
        return None

    session_id = _session_manager.get_session_id_from_cookie(request)
    access_token = ""
    if session_id:
        if tokens := _session_manager.get_session_tokens(session_id):
            access_token = tokens.get("access_token", "")

    session_info = _session_manager.validate_session(session_id) if session_id else None
    claims = (session_info or {}).get("id_token_claims", {}) or {}

    result = _external_authz.check_permission(
        username=username,
        email=claims.get("email", ""),
        provider=claims.get("provider", ""),
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        access_token=access_token,
        ip_address=request.remote_addr or "",
    )

    if result is None:
        # 404 or fallback_to_default - fall through to MLflow RBAC
        return None

    if result.get("is_admin"):
        return True  # Allow
    if not result.get("allowed"):
        return False  # Deny
    if result.get("permission"):
        from mlflow.server.auth.permissions import get_permission

        perm = get_permission(result["permission"])
        # Store the resolved permission so validators can use it
        request._external_authz_permission = perm
        return True

    return None


def _get_idp_default_permission(username: str) -> str | None:
    if not _provisioner:
        return None
    user = store.get_user(username)
    return _provisioner.get_user_default_permission(user.id)


# Endpoint handlers


def _login_page():
    if rate_limited := auth_rate_limiter.is_rate_limited():
        return rate_limited

    next_url = request.args.get("next", "/")
    providers = _oauth_config.get_enabled_providers()

    if not providers:
        return make_response("No authentication providers configured", 500)

    if len(providers) == 1 and _oauth_config.auto_redirect_single_provider:
        return redirect(f"/auth/start/{providers[0]['name']}?next={next_url}")

    return render_login_page(providers, next_url)


def _auth_start(provider_name: str):
    if rate_limited := auth_rate_limiter.is_rate_limited():
        return rate_limited

    next_url = request.args.get("next", "/")

    # Check OIDC providers
    oidc_provider = _oauth_config.oidc_providers.get(provider_name)
    if oidc_provider and oidc_provider.enabled:
        return start_oidc_flow(oidc_provider, _session_manager, next_url)

    # Check SAML providers
    saml_provider = _oauth_config.saml_providers.get(provider_name)
    if saml_provider and saml_provider.enabled:
        from mlflow.server.auth.oauth.saml import start_saml_flow

        return start_saml_flow(saml_provider, _session_manager, next_url)

    return redirect("/auth/login?error=unknown_provider")


def _auth_callback():
    if rate_limited := auth_rate_limiter.is_rate_limited():
        return rate_limited

    return handle_oidc_callback(_oauth_config, _session_manager, _provisioner)


def _saml_acs():
    from mlflow.server.auth.oauth.saml import handle_saml_acs

    return handle_saml_acs(_oauth_config, _session_manager, _provisioner)


def _saml_slo():
    from mlflow.server.auth.oauth.saml import handle_saml_slo

    return handle_saml_slo(_oauth_config, _session_manager)


def _logout():
    session_id = _session_manager.get_session_id_from_cookie(request)
    session_info = None
    if session_id:
        session_info = _session_manager.delete_session(session_id)

        # Audit log
        if session_info:
            from mlflow.server.auth.oauth.audit import log_logout

            claims = (
                json.loads(session_info.get("id_token_claims", "{}"))
                if session_info.get("id_token_claims")
                else {}
            )
            log_logout(
                claims.get("username", "unknown"),
                session_info.get("provider", ""),
                request.remote_addr or "",
            )

        if _external_authz:
            # Invalidate authz cache for this user
            claims = json.loads(session_info.get("id_token_claims", "{}")) if session_info else {}
            if username := claims.get("username", ""):
                _external_authz.invalidate_cache_for_user(username)

    # Build the redirect URL (IdP logout or local login page)
    redirect_url = "/auth/login"
    if session_info:
        provider_str = session_info.get("provider", "")
        if provider_str.startswith("oidc:"):
            provider_name = provider_str.split(":", 1)[1]
            if provider := _oauth_config.oidc_providers.get(provider_name):
                from mlflow.server.auth.oauth.oidc import _get_endpoints

                endpoints = _get_endpoints(provider)
                if end_session_url := endpoints.get("end_session_endpoint"):
                    params = {
                        "post_logout_redirect_uri": f"{request.scheme}://{request.host}/auth/login",
                        "client_id": provider.client_id,
                    }
                    redirect_url = f"{end_session_url}?{urlencode(params)}"

    # Return JSON so the frontend can do a full page navigation (avoids CORS
    # issues when fetch() would otherwise follow a 302 to the IdP).
    response = make_response(jsonify({"redirect_url": redirect_url}))
    _session_manager.clear_session_cookie(response)
    return response


def _get_session_info():
    session_id = _session_manager.get_session_id_from_cookie(request)
    if not session_id:
        return make_response(jsonify({"authenticated": False}), 401)

    session_info = _session_manager.validate_session(session_id)
    if not session_info:
        return make_response(jsonify({"authenticated": False}), 401)

    claims = session_info.get("id_token_claims") or {}
    return jsonify(
        {
            "authenticated": True,
            "username": claims.get("username", ""),
            "display_name": claims.get("display_name", ""),
            "email": claims.get("email", ""),
            "is_admin": claims.get("is_admin", False),
            "provider": session_info.get("provider", ""),
            "expires_at": session_info.get("expires_at", "").isoformat()
            if session_info.get("expires_at")
            else "",
        }
    )


def _list_active_sessions():
    # Admin-only endpoint
    session_id = _session_manager.get_session_id_from_cookie(request)
    if not session_id:
        return make_response(jsonify({"error": "Authentication required"}), 401)

    session_info = _session_manager.validate_session(session_id)
    if not session_info:
        return make_response(jsonify({"error": "Authentication required"}), 401)

    claims = session_info.get("id_token_claims") or {}
    username = claims.get("username", "")
    if not username or not store.get_user(username).is_admin:
        return make_response(jsonify({"error": "Admin access required"}), 403)

    from sqlalchemy.orm import Session as DbSession

    from mlflow.server.auth.oauth.db.models import SqlSession

    with DbSession(store.engine) as db:
        sessions = db.query(SqlSession).all()
        result = []
        for s in sessions:
            s_claims = json.loads(s.id_token_claims) if s.id_token_claims else {}
            result.append(
                {
                    "session_id": s.id[:8] + "...",
                    "username": s_claims.get("username", ""),
                    "provider": s.provider,
                    "created_at": s.created_at.isoformat() if s.created_at else "",
                    "last_accessed_at": s.last_accessed_at.isoformat()
                    if s.last_accessed_at
                    else "",
                    "expires_at": s.expires_at.isoformat() if s.expires_at else "",
                    "ip_address": s.ip_address or "",
                }
            )
        return jsonify({"sessions": result})


def _revoke_session():
    # Admin-only endpoint
    session_id = _session_manager.get_session_id_from_cookie(request)
    if not session_id:
        return make_response(jsonify({"error": "Authentication required"}), 401)

    session_info = _session_manager.validate_session(session_id)
    if not session_info:
        return make_response(jsonify({"error": "Authentication required"}), 401)

    claims = session_info.get("id_token_claims") or {}
    username = claims.get("username", "")
    if not username or not store.get_user(username).is_admin:
        return make_response(jsonify({"error": "Admin access required"}), 403)

    data = request.get_json(silent=True) or {}
    target_username = data.get("username", "")
    if not target_username:
        return make_response(jsonify({"error": "username is required"}), 400)

    if not store.has_user(target_username):
        return make_response(jsonify({"error": "User not found"}), 404)

    target_user = store.get_user(target_username)
    count = _session_manager.delete_user_sessions(target_user.id)
    return jsonify({"revoked_sessions": count})


def _get_auth_config_endpoint():
    providers = _oauth_config.get_enabled_providers()

    # Get current user info if authenticated
    auth_user = None
    if session_id := _session_manager.get_session_id_from_cookie(request):
        if session_info := _session_manager.validate_session(session_id):
            claims = session_info.get("id_token_claims") or {}
            auth_user = {
                "username": claims.get("username", ""),
                "display_name": claims.get("display_name", ""),
                "email": claims.get("email", ""),
                "is_admin": claims.get("is_admin", False),
            }

    return jsonify(
        {
            "auth_type": "oauth",
            "auth_user": auth_user,
            "auth_providers": providers,
        }
    )


def _schedule_cleanup():
    global _cleanup_timer
    if _session_manager:
        try:
            _session_manager.cleanup_expired_sessions()
        except Exception:
            _logger.exception("Session cleanup failed")
    _cleanup_timer = threading.Timer(900, _schedule_cleanup)  # 15 minutes
    _cleanup_timer.daemon = True
    _cleanup_timer.start()


@catch_mlflow_exception
def _before_request():
    if is_unprotected_route(request.path):
        return

    if _is_oauth_unprotected(request.path):
        return

    authorization = authenticate_request_oauth()
    if isinstance(authorization, Response):
        return authorization
    elif not isinstance(authorization, Authorization):
        raise MlflowException(
            f"Unsupported result type from authenticate_request_oauth: "
            f"'{type(authorization).__name__}'",
        )

    # Store username for downstream use
    request.username = authorization.username

    # Step 1: Admins don't need further authorization
    username = authorization.username
    user = store.get_user(username)
    if user.is_admin:
        return

    # Step 2: Check external authz service (if configured)
    # Infer resource type and action from the request path/method
    resource_type, resource_id, action = _infer_resource_context(request)
    if resource_type:
        ext_result = _check_external_authz(username, resource_type, resource_id, action)
        if ext_result is True:
            return  # External authz allowed
        if ext_result is False:
            from mlflow.server.auth.oauth.audit import log_permission_denied

            log_permission_denied(
                username, f"{resource_type}:{resource_id}", action, request.remote_addr or ""
            )
            return make_response("Permission denied", 403)
        # ext_result is None: fall through to MLflow RBAC (steps 3-6)

    # Steps 3-6: Run existing MLflow RBAC validators
    # The validators check per-resource permissions (step 3), workspace permissions (step 4),
    # and fall back to default_permission (step 6).
    # Step 5 (IdP default permission via user_role_overrides) is integrated via
    # _install_idp_default_permission_hook() which patches the fallback behavior.
    if validator := _find_validator(request):
        if not validator():
            return make_response("Permission denied", 403)

    if validator := _get_workspace_validator(request):
        if not validator():
            return make_response("Permission denied", 403)


def _infer_resource_context(req) -> tuple[str, str, str]:
    path = req.path
    method = req.method

    action_map = {
        "GET": "read",
        "POST": "create",
        "PATCH": "update",
        "PUT": "update",
        "DELETE": "delete",
    }
    action = action_map.get(method, "read")

    if "/experiments/" in path:
        resource_type = "experiment"
        try:
            from mlflow.server.auth import _get_request_param

            resource_id = _get_request_param("experiment_id")
        except Exception:
            resource_id = ""
        if "/permissions/" in path:
            action = "manage"
    elif "/registered-models/" in path:
        resource_type = "registered_model"
        try:
            from mlflow.server.auth import _get_request_param

            resource_id = _get_request_param("name")
        except Exception:
            resource_id = ""
        if "/permissions/" in path:
            action = "manage"
    elif "/scorers/" in path:
        resource_type = "scorer"
        resource_id = ""
    elif "/gateway/" in path:
        if "/secrets/" in path:
            resource_type = "gateway_secret"
        elif "/endpoints/" in path:
            resource_type = "gateway_endpoint"
        elif "/model-definitions/" in path:
            resource_type = "gateway_model_definition"
        else:
            resource_type = ""
        resource_id = ""
    elif "/workspaces/" in path:
        resource_type = "workspace"
        resource_id = ""
    else:
        return "", "", ""

    return resource_type, resource_id, action


@catch_mlflow_exception
def _after_request(resp: Response):
    if 400 <= resp.status_code < 600:
        return resp

    # Enrich /server-info response with auth context
    if request.path.endswith("/mlflow/server-info") and request.method == "GET":
        _enrich_server_info_response(resp)
        return resp

    handler = AFTER_REQUEST_HANDLERS.get((request.path, request.method))
    if handler is None and "/workspaces/" in request.path:
        for (path, method), candidate in WORKSPACE_PARAMETERIZED_BEFORE_REQUEST_VALIDATORS.items():
            if method == request.method and path.fullmatch(request.path):
                handler = AFTER_REQUEST_HANDLERS.get((request.path, request.method))
                break
    if handler:
        handler(resp)

    return resp


def _enrich_server_info_response(resp: Response):
    try:
        data = resp.get_json()
        if not isinstance(data, dict):
            return
    except Exception:
        return

    data["auth_type"] = "oauth"
    data["auth_providers"] = _oauth_config.get_enabled_providers()

    # Add current user info if authenticated via session
    if username := getattr(request, "username", None):
        session_id = _session_manager.get_session_id_from_cookie(request)
        session_info = _session_manager.validate_session(session_id) if session_id else None
        claims = (session_info or {}).get("id_token_claims") or {}
        try:
            user = store.get_user(username)
            is_admin = user.is_admin
        except Exception:
            is_admin = False
        data["auth_user"] = {
            "username": username,
            "display_name": claims.get("display_name", ""),
            "email": claims.get("email", ""),
            "is_admin": is_admin,
        }

    resp.set_data(json.dumps(data))
    resp.content_type = "application/json"


def _install_authenticate_request_shortcut():
    import mlflow.server.auth as auth_module

    _original_authenticate_request = auth_module.authenticate_request

    def _patched_authenticate_request() -> Authorization | Response:
        # If _before_request already authenticated the user, return the stored username
        # instead of re-running the full OAuth flow. This is needed because after-request
        # handlers (e.g. filter_search_experiments) call authenticate_request() again.
        if username := getattr(request, "username", None):
            return Authorization("session", {"username": username})
        return _original_authenticate_request()

    auth_module.authenticate_request = _patched_authenticate_request


def _install_idp_default_permission_hook():
    import mlflow.server.auth as auth_module
    from mlflow.server.auth.permissions import get_permission

    def _patched_get_permission_from_store_or_default(
        store_permission_func, workspace_level_permission_func=None
    ):
        from mlflow.exceptions import MlflowException
        from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

        try:
            perm = store_permission_func()
        except MlflowException as e:
            if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
                # Step 4: Workspace-level permission
                if workspace_level_permission_func is not None:
                    workspace_permission = workspace_level_permission_func()
                    if workspace_permission is not None:
                        return workspace_permission

                # Step 5: IdP-derived default permission (NEW)
                if username := getattr(request, "username", None):
                    if idp_perm := _get_idp_default_permission(username):
                        return get_permission(idp_perm)

                # Step 6: Fall back to auth_config.default_permission
                perm = _oauth_config.auth_config.default_permission
            else:
                raise
        return get_permission(perm)

    auth_module._get_permission_from_store_or_default = (
        _patched_get_permission_from_store_or_default
    )


def create_app(app: Flask = app):
    global _oauth_config, _session_manager, _provisioner, _external_authz

    _logger.info("Initializing MLflow OAuth authentication plugin")

    # Read config
    _oauth_config = read_oauth_config()

    # Set up Flask secret key for CSRF
    secret_key = MLFLOW_FLASK_SERVER_SECRET_KEY.get()
    if not secret_key:
        raise MlflowException(
            "A static secret key is required for the OAuth plugin. "
            "Please set the MLFLOW_FLASK_SERVER_SECRET_KEY environment variable."
        )
    app.secret_key = secret_key

    # CSRF protection for form-based endpoints
    app.config["WTF_CSRF_CHECK_DEFAULT"] = False
    from flask_wtf import CSRFProtect

    csrf = CSRFProtect()
    csrf.init_app(app)

    # Initialize database
    store.init_db(_oauth_config.auth_config.database_uri)

    # Run OAuth-specific migrations

    from sqlalchemy import inspect as sa_inspect

    inspector = sa_inspect(store.engine)
    existing_tables = inspector.get_table_names()
    if "sessions" not in existing_tables:
        from mlflow.server.auth.oauth.db.models import Base as OAuthBase

        OAuthBase.metadata.create_all(store.engine, checkfirst=True)

    create_admin_user(
        _oauth_config.auth_config.admin_username,
        _oauth_config.auth_config.admin_password,
    )

    # Initialize session manager
    from sqlalchemy.orm import sessionmaker

    SessionMaker = sessionmaker(bind=store.engine)

    @contextmanager
    def managed_session():
        session = SessionMaker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    _session_manager = SessionManager(managed_session, _oauth_config)

    # Initialize provisioner
    _provisioner = UserProvisioner(store, _oauth_config)

    # Initialize external authz client
    if _oauth_config.external_authz.enabled:
        _external_authz = ExternalAuthzClient(_oauth_config.external_authz)

    # Patch authenticate_request() to reuse the username from _before_request,
    # so after-request handlers don't re-run the full OAuth flow.
    _install_authenticate_request_shortcut()

    # Install IdP default permission hook (step 5 in authz chain)
    _install_idp_default_permission_hook()

    # Register OAuth routes
    app.add_url_rule(AUTH_LOGIN, view_func=_login_page, methods=["GET"])
    app.add_url_rule(AUTH_START, view_func=_auth_start, methods=["GET"])
    app.add_url_rule(AUTH_CALLBACK, view_func=_auth_callback, methods=["GET"])
    app.add_url_rule(AUTH_SAML_ACS, view_func=_saml_acs, methods=["POST"])
    app.add_url_rule(AUTH_SAML_SLO, view_func=_saml_slo, methods=["GET", "POST"])
    app.add_url_rule(AUTH_LOGOUT, view_func=_logout, methods=["POST"])
    app.add_url_rule(AUTH_SESSION, view_func=_get_session_info, methods=["GET"])
    app.add_url_rule(AUTH_CONFIG, view_func=_get_auth_config_endpoint, methods=["GET"])

    # Admin session management endpoints
    app.add_url_rule("/auth/admin/sessions", view_func=_list_active_sessions, methods=["GET"])
    app.add_url_rule("/auth/admin/revoke", view_func=_revoke_session, methods=["POST"])

    # Exempt OAuth callback routes from CSRF
    csrf.exempt(_auth_callback)
    csrf.exempt(_saml_acs)
    csrf.exempt(_saml_slo)

    # Register all existing auth routes from basic-auth plugin
    # (user management, permission management, etc.)
    from mlflow.server.auth import (
        create_experiment_permission,
        create_gateway_endpoint_permission,
        create_gateway_model_definition_permission,
        create_gateway_secret_permission,
        create_registered_model_permission,
        create_scorer_permission,
        create_user,
        delete_experiment_permission,
        delete_gateway_endpoint_permission,
        delete_gateway_model_definition_permission,
        delete_gateway_secret_permission,
        delete_registered_model_permission,
        delete_scorer_permission,
        delete_user,
        delete_workspace_permission,
        get_experiment_permission,
        get_gateway_endpoint_permission,
        get_gateway_model_definition_permission,
        get_gateway_secret_permission,
        get_registered_model_permission,
        get_scorer_permission,
        get_user,
        list_user_workspace_permissions,
        list_users,
        list_workspace_permissions,
        set_workspace_permission,
        update_experiment_permission,
        update_gateway_endpoint_permission,
        update_gateway_model_definition_permission,
        update_gateway_secret_permission,
        update_registered_model_permission,
        update_scorer_permission,
        update_user_admin,
        update_user_password,
    )
    from mlflow.server.auth.routes import (
        AJAX_LIST_USERS,
        CREATE_EXPERIMENT_PERMISSION,
        CREATE_GATEWAY_ENDPOINT_PERMISSION,
        CREATE_GATEWAY_MODEL_DEFINITION_PERMISSION,
        CREATE_GATEWAY_SECRET_PERMISSION,
        CREATE_REGISTERED_MODEL_PERMISSION,
        CREATE_SCORER_PERMISSION,
        CREATE_USER,
        DELETE_EXPERIMENT_PERMISSION,
        DELETE_GATEWAY_ENDPOINT_PERMISSION,
        DELETE_GATEWAY_MODEL_DEFINITION_PERMISSION,
        DELETE_GATEWAY_SECRET_PERMISSION,
        DELETE_REGISTERED_MODEL_PERMISSION,
        DELETE_SCORER_PERMISSION,
        DELETE_USER,
        GET_EXPERIMENT_PERMISSION,
        GET_GATEWAY_ENDPOINT_PERMISSION,
        GET_GATEWAY_MODEL_DEFINITION_PERMISSION,
        GET_GATEWAY_SECRET_PERMISSION,
        GET_REGISTERED_MODEL_PERMISSION,
        GET_SCORER_PERMISSION,
        GET_USER,
        LIST_USER_WORKSPACE_PERMISSIONS,
        LIST_USERS,
        LIST_WORKSPACE_PERMISSIONS,
        UPDATE_EXPERIMENT_PERMISSION,
        UPDATE_GATEWAY_ENDPOINT_PERMISSION,
        UPDATE_GATEWAY_MODEL_DEFINITION_PERMISSION,
        UPDATE_GATEWAY_SECRET_PERMISSION,
        UPDATE_REGISTERED_MODEL_PERMISSION,
        UPDATE_SCORER_PERMISSION,
        UPDATE_USER_ADMIN,
        UPDATE_USER_PASSWORD,
    )

    # User management routes
    app.add_url_rule(CREATE_USER, view_func=create_user, methods=["POST"])
    app.add_url_rule(GET_USER, view_func=get_user, methods=["GET"])
    for rule in [LIST_USERS, AJAX_LIST_USERS]:
        app.add_url_rule(rule, view_func=list_users, methods=["GET"])
    app.add_url_rule(UPDATE_USER_PASSWORD, view_func=update_user_password, methods=["PATCH"])
    app.add_url_rule(UPDATE_USER_ADMIN, view_func=update_user_admin, methods=["PATCH"])
    app.add_url_rule(DELETE_USER, view_func=delete_user, methods=["DELETE"])

    # Experiment permission routes
    app.add_url_rule(
        CREATE_EXPERIMENT_PERMISSION, view_func=create_experiment_permission, methods=["POST"]
    )
    app.add_url_rule(
        GET_EXPERIMENT_PERMISSION, view_func=get_experiment_permission, methods=["GET"]
    )
    app.add_url_rule(
        UPDATE_EXPERIMENT_PERMISSION, view_func=update_experiment_permission, methods=["PATCH"]
    )
    app.add_url_rule(
        DELETE_EXPERIMENT_PERMISSION, view_func=delete_experiment_permission, methods=["DELETE"]
    )

    # Registered model permission routes
    app.add_url_rule(
        CREATE_REGISTERED_MODEL_PERMISSION,
        view_func=create_registered_model_permission,
        methods=["POST"],
    )
    app.add_url_rule(
        GET_REGISTERED_MODEL_PERMISSION,
        view_func=get_registered_model_permission,
        methods=["GET"],
    )
    app.add_url_rule(
        UPDATE_REGISTERED_MODEL_PERMISSION,
        view_func=update_registered_model_permission,
        methods=["PATCH"],
    )
    app.add_url_rule(
        DELETE_REGISTERED_MODEL_PERMISSION,
        view_func=delete_registered_model_permission,
        methods=["DELETE"],
    )

    # Scorer permission routes
    app.add_url_rule(CREATE_SCORER_PERMISSION, view_func=create_scorer_permission, methods=["POST"])
    app.add_url_rule(GET_SCORER_PERMISSION, view_func=get_scorer_permission, methods=["GET"])
    app.add_url_rule(
        UPDATE_SCORER_PERMISSION, view_func=update_scorer_permission, methods=["PATCH"]
    )
    app.add_url_rule(
        DELETE_SCORER_PERMISSION, view_func=delete_scorer_permission, methods=["DELETE"]
    )

    # Gateway permission routes
    app.add_url_rule(
        CREATE_GATEWAY_SECRET_PERMISSION,
        view_func=create_gateway_secret_permission,
        methods=["POST"],
    )
    app.add_url_rule(
        GET_GATEWAY_SECRET_PERMISSION, view_func=get_gateway_secret_permission, methods=["GET"]
    )
    app.add_url_rule(
        UPDATE_GATEWAY_SECRET_PERMISSION,
        view_func=update_gateway_secret_permission,
        methods=["PATCH"],
    )
    app.add_url_rule(
        DELETE_GATEWAY_SECRET_PERMISSION,
        view_func=delete_gateway_secret_permission,
        methods=["DELETE"],
    )
    app.add_url_rule(
        CREATE_GATEWAY_ENDPOINT_PERMISSION,
        view_func=create_gateway_endpoint_permission,
        methods=["POST"],
    )
    app.add_url_rule(
        GET_GATEWAY_ENDPOINT_PERMISSION,
        view_func=get_gateway_endpoint_permission,
        methods=["GET"],
    )
    app.add_url_rule(
        UPDATE_GATEWAY_ENDPOINT_PERMISSION,
        view_func=update_gateway_endpoint_permission,
        methods=["PATCH"],
    )
    app.add_url_rule(
        DELETE_GATEWAY_ENDPOINT_PERMISSION,
        view_func=delete_gateway_endpoint_permission,
        methods=["DELETE"],
    )
    app.add_url_rule(
        CREATE_GATEWAY_MODEL_DEFINITION_PERMISSION,
        view_func=create_gateway_model_definition_permission,
        methods=["POST"],
    )
    app.add_url_rule(
        GET_GATEWAY_MODEL_DEFINITION_PERMISSION,
        view_func=get_gateway_model_definition_permission,
        methods=["GET"],
    )
    app.add_url_rule(
        UPDATE_GATEWAY_MODEL_DEFINITION_PERMISSION,
        view_func=update_gateway_model_definition_permission,
        methods=["PATCH"],
    )
    app.add_url_rule(
        DELETE_GATEWAY_MODEL_DEFINITION_PERMISSION,
        view_func=delete_gateway_model_definition_permission,
        methods=["DELETE"],
    )

    # Workspace permission routes
    app.add_url_rule(
        LIST_WORKSPACE_PERMISSIONS, view_func=list_workspace_permissions, methods=["GET"]
    )
    app.add_url_rule(
        LIST_WORKSPACE_PERMISSIONS, view_func=set_workspace_permission, methods=["POST"]
    )
    app.add_url_rule(
        LIST_WORKSPACE_PERMISSIONS, view_func=delete_workspace_permission, methods=["DELETE"]
    )
    app.add_url_rule(
        LIST_USER_WORKSPACE_PERMISSIONS,
        view_func=list_user_workspace_permissions,
        methods=["GET"],
    )

    # Register hooks
    app.before_request(_before_request)
    app.after_request(_after_request)

    # Start background cleanup
    _schedule_cleanup()

    # Handle FastAPI middleware for uvicorn
    from mlflow.environment_variables import _MLFLOW_SGI_NAME

    if _MLFLOW_SGI_NAME.get() == "uvicorn":
        from mlflow.server.fastapi_app import create_fastapi_app

        fastapi_app = create_fastapi_app(app)
        _add_fastapi_oauth_middleware(fastapi_app)
        return fastapi_app

    return app


def _add_fastapi_oauth_middleware(fastapi_app):
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as StarletteRequest
    from starlette.responses import JSONResponse

    class OAuthFastAPIMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next):
            if is_unprotected_route(request.url.path):
                return await call_next(request)

            if _is_oauth_unprotected(request.url.path):
                return await call_next(request)

            # Authenticate via session cookie or Bearer token
            user = _authenticate_fastapi_request(request)
            if user is None:
                return JSONResponse(
                    {"error": "Authentication required"},
                    status_code=401,
                    headers={"WWW-Authenticate": 'Bearer realm="mlflow"'},
                )

            request.state.username = user.username
            request.state.user_id = user.id

            if user.is_admin:
                return await call_next(request)

            return await call_next(request)

    fastapi_app.add_middleware(OAuthFastAPIMiddleware)


def _authenticate_fastapi_request(request) -> object | None:
    # Check session cookie
    if session_cookie := request.cookies.get(_oauth_config.session_cookie_name, ""):
        if session_info := _session_manager.validate_session(session_cookie):
            claims = session_info.get("id_token_claims") or {}
            username = claims.get("username", "")
            if username and store.has_user(username):
                return store.get_user(username)

    # Check Bearer token
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        token_info = validate_bearer_token(token, _oauth_config)
        if token_info and store.has_user(token_info["username"]):
            return store.get_user(token_info["username"])

    # Check Basic Auth fallback
    if _oauth_config.allow_basic_auth_fallback and "authorization" in request.headers:
        auth = request.headers["authorization"]
        try:
            scheme, credentials = auth.split(None, 1)
            if scheme.lower() == "basic":
                decoded = base64.b64decode(credentials).decode("ascii")
                username, _, password = decoded.partition(":")
                if store.authenticate_user(username, password):
                    return store.get_user(username)
        except Exception:
            pass

    return None

# CI guard for the basic-auth dispatcher: every route the basic-auth server serves must
# resolve to an authorization decision, or be listed in the debt list
# _KNOWN_UNGATED_ROUTE_MARKERS.

import ast
import inspect
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.routing import iter_route_contexts
from flask import Flask
from starlette.routing import Mount, Route

import mlflow
from mlflow.server import app as server_app
from mlflow.server import auth as a
from mlflow.server.auth.routes import CREATE_USER_UI
from mlflow.server.handlers import _not_implemented


@pytest.fixture(scope="module")
def basic_auth_app():
    # Build the app `mlflow server --app-name basic-auth` serves, so the guards walk what is
    # actually registered rather than a hand-kept list. `create_app` gets a fresh Flask app so
    # the auth routes it adds stay off the shared module-level one.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MLFLOW_FLASK_SERVER_SECRET_KEY", "coverage-guard")
        mp.setenv("_MLFLOW_SGI_NAME", "uvicorn")
        # Registering routes needs no auth database. `create_app` also flips the module's
        # `_auth_initialized`; recording it here restores it when the context exits.
        mp.setattr(a, "_init_store_for_app", lambda: None)
        mp.setattr(a, "_auth_initialized", a._auth_initialized)
        # No static folder, so this app adds no static route of its own.
        auth_flask_app = Flask(__name__, static_folder=None)
        fastapi_app = a.create_app(auth_flask_app)
    return auth_flask_app, fastapi_app


@pytest.fixture(scope="module")
def flask_routes(basic_auth_app):
    # The module-level app holds the handler endpoints, the plain `@app.route` views and Flask's
    # static route; the auth app holds the routes `create_app` adds.
    auth_flask_app, _ = basic_auth_app
    routes = set()
    for flask_app in (server_app, auth_flask_app):
        for rule in flask_app.url_map.iter_rules():
            if flask_app.view_functions[rule.endpoint] is _not_implemented:
                continue
            # Flask's own route over the React build directory (/build/<path:filename>). The UI
            # loads those files from /static-files/, so nothing requests it, and it is left to the
            # fail-closed net (see test_flask_static_route_is_left_to_the_fail_closed_net).
            if rule.endpoint == "static":
                continue
            # Flask adds HEAD and OPTIONS to every rule.
            routes.update((rule.rule, method) for method in rule.methods - {"HEAD", "OPTIONS"})
    return sorted(routes)


class _Req:
    def __init__(self, path, method):
        self.path = path
        self.method = method


def _is_covered(path, method):
    req = _Req(path, method)
    return (
        a.is_unprotected_route(path)
        or a._find_validator(req) is not None
        or a._is_proxy_artifact_path(path)
        or a._authorized_outside_before_request(req)
    )


def test_no_new_ungated_routes(flask_routes):
    uncovered = sorted(
        f"{method:6} {path}"
        for path, method in flask_routes
        if not _is_covered(path, method) and not a._is_known_ungated_route(path)
    )
    assert not uncovered, (
        "Routes with no authorization decision that are not in the documented debt "
        "list (_KNOWN_UNGATED_ROUTE_MARKERS). Add a before-request validator, an "
        "after-request filter, a self-authorizing handler + allowlist entry, or an "
        "explicit public-route entry:\n" + "\n".join(uncovered)
    )


def _routes_on_the_deny_fallback(routes):
    # The dataset, job, issue and trace families deny a path they do not recognize.
    # test_no_new_ungated_routes counts that denial as a decision, so a new route that is
    # missing from its family's table passes it and only shows up as a runtime 403.
    return sorted(
        f"{method:6} {path}"
        for path, method in routes
        if a._find_validator(_Req(path, method)) is a._deny_unmatched_route
    )


def test_no_route_falls_through_to_a_family_deny_fallback(flask_routes):
    unresolved = _routes_on_the_deny_fallback(flask_routes)
    assert not unresolved, (
        "Routes that resolve only to the fail-closed fallback of their route family, not to "
        "a real validator. Register a handler for each one in the family's table (for "
        "example DATASET_BEFORE_REQUEST_HANDLERS, ISSUE_BEFORE_REQUEST_HANDLERS, "
        "JOB_BEFORE_REQUEST_VALIDATORS or BEFORE_REQUEST_VALIDATORS):\n" + "\n".join(unresolved)
    )


@pytest.mark.parametrize(
    "path",
    [
        "/api/2.0/mlflow/datasets/<dataset_id>/new-route",
        "/ajax-api/3.0/mlflow/jobs/<job_id>/new-route",
        "/api/3.0/mlflow/issues/<issue_id>/new-route",
        "/api/3.0/mlflow/traces/<trace_id>/new-route",
    ],
)
def test_new_route_missing_from_its_family_table_fails_the_guard(flask_routes, path):
    assert (path, "POST") not in flask_routes
    assert _routes_on_the_deny_fallback([(path, "POST")]) == [f"POST   {path}"]


@pytest.mark.parametrize(
    ("marker", "table"),
    [
        ("/mlflow/datasets/", "DATASET_BEFORE_REQUEST_VALIDATORS"),
        ("/mlflow/jobs/", "JOB_BEFORE_REQUEST_VALIDATORS"),
        ("/mlflow/issues/", "ISSUE_BEFORE_REQUEST_VALIDATORS"),
        ("/mlflow/traces/", "TRACE_PARAMETERIZED_BEFORE_REQUEST_VALIDATORS"),
    ],
)
def test_registered_route_with_its_validator_removed_fails_the_guard(
    flask_routes, monkeypatch, marker, table
):
    family_routes = [(p, m) for p, m in flask_routes if marker in p and "<" in p]
    assert family_routes
    assert not _routes_on_the_deny_fallback(family_routes)

    monkeypatch.setattr(a, table, {})
    # Parameterized trace routes also match their template in the exact-match table.
    monkeypatch.setattr(
        a,
        "BEFORE_REQUEST_VALIDATORS",
        {k: v for k, v in a.BEFORE_REQUEST_VALIDATORS.items() if k not in family_routes},
    )
    assert _routes_on_the_deny_fallback(family_routes) == sorted(
        f"{m:6} {p}" for p, m in family_routes
    )


def test_known_ungated_markers_are_not_stale(flask_routes):
    matched = set()
    for path, method in flask_routes:
        if _is_covered(path, method):
            continue
        for marker in a._KNOWN_UNGATED_ROUTE_MARKERS:
            if marker in path:
                matched.add(marker)
    stale = [m for m in a._KNOWN_UNGATED_ROUTE_MARKERS if m not in matched]
    assert not stale, (
        "Debt markers that no longer match any ungated route (their family is now "
        f"gated — remove them from _KNOWN_UNGATED_ROUTE_MARKERS): {stale}"
    )


def test_gateway_secret_selector_inert_routes_still_drop_the_field():
    """A route is exempt from the `secret_id` gate only while its handler ignores the field.

    `_list_gateway_endpoints` declares `secret_id` in the proto and the store accepts it, but the
    handler forwards only `provider=`, so the selector filters nothing and gating it would refuse a
    request that returns an unfiltered listing. If someone wires it, the oracle becomes real and the
    route needs `_gateway_secret_selector_not_denied` -- fail here rather than leak quietly.
    """
    from mlflow.server import handlers

    for name in a._GATEWAY_SECRET_SELECTOR_INERT_ROUTES:
        source = inspect.getsource(getattr(handlers, name))
        assert "secret_id=" not in source, (
            f"{name} now forwards secret_id to the store, so its selector is real. Gate it with "
            "_gateway_secret_selector_not_denied and drop it from "
            "_GATEWAY_SECRET_SELECTOR_INERT_ROUTES."
        )


# Functions that decrypt a stored gateway secret.
_SECRET_DECRYPTION_FUNCTIONS = frozenset({
    "_decrypt_secret",
    "_get_decrypted_secret",
    "_fetch_provider_credentials",
})

# Every function in ``mlflow/`` allowed to reference one, with the validators that check the
# caller's permission on the secret. Empty for a pass-through layer, whose callers are listed too.
_SECRET_DECRYPTION_CALL_SITES = {
    ("mlflow/store/tracking/sqlalchemy_store.py", "<module>"): (),
    ("mlflow/store/tracking/sqlalchemy_store.py", "_get_decrypted_secret"): (),
    ("mlflow/genai/discovery/job.py", "_fetch_provider_credentials"): (),
    # ``secret_id`` from the request body.
    ("mlflow/server/handlers.py", "_invoke_issue_detection_handler"): (
        "validate_can_invoke_issue_detection",
    ),
    # ``secretId`` from the schedule tag, checked on the routes that store the tag.
    ("mlflow/genai/discovery/scheduler.py", "_provider_credentials"): (
        "validate_can_set_experiment_tag",
        "validate_can_create_experiment",
    ),
    # The gateway decrypts the secret a model definition binds, not one the caller names.
    ("mlflow/store/tracking/gateway/config_resolver.py", "<module>"): (),
    ("mlflow/store/tracking/gateway/config_resolver.py", "get_resource_endpoint_configs"): (
        "validate_can_create_gateway_model_definition",
        "validate_can_update_gateway_model_definition",
    ),
    ("mlflow/store/tracking/gateway/config_resolver.py", "get_endpoint_config"): (
        "validate_can_create_gateway_model_definition",
        "validate_can_update_gateway_model_definition",
        "_validate_gateway_use_permission",
    ),
}


class _SecretDecryptionReferences(ast.NodeVisitor):
    def __init__(self):
        self._scope = ["<module>"]
        self.functions = set()
        self.renamed = []

    def _record(self, name):
        if name in _SECRET_DECRYPTION_FUNCTIONS:
            self.functions.add(self._scope[-1])

    def visit_FunctionDef(self, node):
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Name(self, node):
        self._record(node.id)

    def visit_Attribute(self, node):
        self._record(node.attr)
        self.generic_visit(node)

    def visit_Constant(self, node):
        # e.g. ``getattr(store, "_get_decrypted_secret")``
        if isinstance(node.value, str):
            self._record(node.value)

    def visit_alias(self, node):
        name = node.name.rpartition(".")[2]
        self._record(name)
        if name in _SECRET_DECRYPTION_FUNCTIONS and node.asname not in (None, name):
            self.renamed.append(f"{name} as {node.asname}")


def _secret_decryption_references():
    package = Path(mlflow.__file__).parent
    call_sites = set()
    renamed = []
    for directory, subdirectories, filenames in os.walk(package):
        subdirectories[:] = [d for d in subdirectories if d not in ("js", "node_modules")]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = Path(directory, filename)
            source = path.read_text(encoding="utf-8")
            # Only parse the few files that mention one of the names.
            if not any(name in source for name in _SECRET_DECRYPTION_FUNCTIONS):
                continue
            references = _SecretDecryptionReferences()
            references.visit(ast.parse(source))
            relative_path = path.relative_to(package.parent).as_posix()
            call_sites.update((relative_path, function) for function in references.functions)
            renamed.extend(f"{relative_path}: {entry}" for entry in references.renamed)
    return call_sites, sorted(renamed)


def test_secret_decryption_call_sites_are_all_accounted_for():
    """A gateway secret is decrypted by id with no user attached, so each path that reaches
    decryption must check the caller itself. A new reference fails here until it is listed with
    the check that covers it.
    """
    call_sites, renamed = _secret_decryption_references()

    assert not renamed, (
        "Import these under their own names so this guard can follow them: " + ", ".join(renamed)
    )

    unlisted = sorted(call_sites - _SECRET_DECRYPTION_CALL_SITES.keys())
    assert not unlisted, (
        "New code decrypts a stored gateway secret:\n"
        + "\n".join(f"  {path}::{function}" for path, function in unlisted)
        + "\nIf the secret id can come from a client (a request field, a tag, any stored "
        "config a user can write), require USE on that secret from the user who supplied it -- "
        "at write time if it is used later with no request to authorize -- and add a regression "
        "test. Then list the call site in _SECRET_DECRYPTION_CALL_SITES with that validator."
    )

    stale = sorted(_SECRET_DECRYPTION_CALL_SITES.keys() - call_sites)
    assert not stale, (
        "These no longer reference a decryption function; remove them from "
        f"_SECRET_DECRYPTION_CALL_SITES: {stale}"
    )

    missing = sorted(
        validator
        for validators in _SECRET_DECRYPTION_CALL_SITES.values()
        for validator in validators
        if not callable(getattr(a, validator, None))
    )
    assert not missing, (
        "_SECRET_DECRYPTION_CALL_SITES names validators that no longer exist in "
        f"mlflow.server.auth, so those call sites have no check on record: {missing}"
    )


def test_unknown_issue_subpath_fails_closed():
    # An unrecognized /mlflow/issues/ path resolves to a denying validator (not None), so
    # it is denied even with MLFLOW_BASIC_AUTH_FAIL_CLOSED off, matching the dataset/trace
    # branches, rather than silently falling through.
    v = a._find_validator(_Req("/api/3.0/mlflow/issues/i-1/comments", "POST"))
    assert v is not None
    assert v() is False
    # The invoke route is exempt from that hard-deny: it either has its own validator or is
    # still tracked in the debt list, never the fail-closed fallback.
    invoke = "/ajax-api/3.0/mlflow/issues/invoke"
    assert a._is_known_ungated_route(invoke) or (invoke, "POST") in a.BEFORE_REQUEST_VALIDATORS


def _fastapi_native_routes(basic_auth_app):
    # Every route served directly by FastAPI, with its router prefix applied. The mounted
    # Flask app is authorized by _before_request, so it is out of scope for these guards.
    auth_flask_app, fastapi_app = basic_auth_app
    for route in iter_route_contexts(fastapi_app.routes):
        original = route.original_route
        if isinstance(original, Mount) and getattr(original.app, "app", None) is auth_flask_app:
            continue
        yield route


def test_no_ungated_fastapi_native_routes(basic_auth_app):
    # The FastAPI permission middleware denies a native route with no validator; this guard
    # makes a new one a failing test rather than a 403 first seen in production.
    uncovered = sorted(
        f"{method:6} {route.path}"
        for route in _fastapi_native_routes(basic_auth_app)
        if isinstance(route.original_route, Route)
        for method in route.methods - {"HEAD", "OPTIONS"}
        if a._find_fastapi_validator(route.path, method) is None
        and not a.is_unprotected_route(route.path)
        and not a._matches_route_suffix(
            a._strip_static_prefix(route.path), a._PUBLIC_ROUTE_SUFFIXES
        )
        and not any(m in route.path for m in a._KNOWN_UNGATED_FASTAPI_ROUTE_MARKERS)
    )
    assert not uncovered, (
        "Native FastAPI routes with no authorization validator that are not documented in "
        "_KNOWN_UNGATED_FASTAPI_ROUTE_MARKERS. Add a validator branch in "
        "_find_fastapi_validator, or document the exemption:\n" + "\n".join(uncovered)
    )


def test_fastapi_native_routes_are_http_routes(basic_auth_app):
    # The permission middleware only sees HTTP requests, and its fail-closed check passes
    # through any path no native route matches as if Flask served it. A WebSocket route or
    # another mounted app would therefore be served with no authorization at all.
    unauthorized = sorted(
        f"{type(route.original_route).__name__} {route.path}"
        for route in _fastapi_native_routes(basic_auth_app)
        if not isinstance(route.original_route, Route)
    )
    assert not unauthorized, (
        "Native FastAPI routes that the permission middleware cannot authorize:\n"
        + "\n".join(unauthorized)
    )


def test_jobs_owner_check(monkeypatch):
    # Jobs get/cancel are gated on ownership — the recorded creator must match the
    # caller; a different creator or no creator is denied (admins bypass upstream).
    import mlflow.server.jobs as jobs_mod

    monkeypatch.setattr(a, "authenticate_request", lambda: SimpleNamespace(username="alice"))
    monkeypatch.setattr(a, "_get_request_param", lambda p: "job-1")

    monkeypatch.setattr(jobs_mod, "get_job", lambda jid: SimpleNamespace(creator="alice"))
    assert a.validate_is_job_owner() is True

    monkeypatch.setattr(jobs_mod, "get_job", lambda jid: SimpleNamespace(creator="bob"))
    assert a.validate_is_job_owner() is False

    monkeypatch.setattr(jobs_mod, "get_job", lambda jid: SimpleNamespace(creator=None))
    assert a.validate_is_job_owner() is False


def test_gateway_discovery_config_gating():
    def vname(path, method="GET"):
        v = a._find_validator(_Req(path, method))
        return v.__name__ if v is not None else None

    # Discovery + config back the gateway UI for any signed-in user (no secret material).
    assert vname("/ajax-api/3.0/mlflow/gateway/supported-providers") == "_allow_authenticated"
    assert vname("/ajax-api/3.0/mlflow/gateway/supported-models") == "_allow_authenticated"
    assert vname("/ajax-api/3.0/mlflow/gateway/provider-config") == "_allow_authenticated"
    assert vname("/ajax-api/3.0/mlflow/gateway/secrets/config") == "_allow_authenticated"


# Budget-policy writes are admin-only; reads require visibility of the request workspace.
@pytest.mark.parametrize("prefix", ["/api/3.0", "/ajax-api/3.0"])
@pytest.mark.parametrize(
    ("route", "method", "expected"),
    [
        ("budgets/get", "GET", a.validate_can_view_request_workspace),
        ("budgets/list", "GET", a.validate_can_view_request_workspace),
        ("budgets/windows", "GET", a.validate_can_view_request_workspace),
        ("budgets/create", "POST", a.sender_is_admin),
        ("budgets/update", "POST", a.sender_is_admin),
        ("budgets/delete", "DELETE", a.sender_is_admin),
    ],
)
def test_gateway_budget_policy_routes_resolve_validator(prefix, route, method, expected):
    assert a._find_validator(_Req(f"{prefix}/mlflow/gateway/{route}", method)) is expected


def test_create_gateway_secret_requires_workspace_create_validator():
    v = a._find_validator(_Req("/api/3.0/mlflow/gateway/secrets/create", "POST"))
    assert v is a.validate_can_create_gateway_secret


def test_every_ownership_grant_route_has_a_before_request_validator():
    # An after-request MANAGE grant only records ownership; the create itself must be
    # authorized up front, or the grant would mask the missing check. Scanning
    # AFTER_REQUEST_HANDLERS alone is sufficient because
    # PARAMETERIZED_AFTER_REQUEST_HANDLERS is derived from it.
    grants = {
        h
        for h in a.AFTER_REQUEST_PATH_HANDLERS.values()
        if h.__name__.startswith("set_can_manage_")
    }
    ungated = sorted(
        f"{method:6} {path}"
        for (path, method), handler in a.AFTER_REQUEST_HANDLERS.items()
        if handler in grants and a._find_validator(_Req(path, method)) is None
    )
    assert not ungated


def test_gateway_guardrail_gating():
    def vname(path, method):
        v = a._find_validator(_Req(path, method))
        return v.__name__ if v is not None else None

    base = "/api/3.0/mlflow/gateway/guardrails"
    # Standalone guardrail CRUD -> admin-only.
    assert vname(f"{base}/create", "POST") == "sender_is_admin"
    assert vname(f"{base}/get", "GET") == "sender_is_admin"
    assert vname(f"{base}/list", "GET") == "sender_is_admin"
    assert vname(f"{base}/delete", "DELETE") == "sender_is_admin"
    # Endpoint-attached routes gate on the owning gateway endpoint. add-to-endpoint and
    # update-config additionally veto on the scorer tier, since a Guardrail carries a full
    # ScorerVersion; remove-from-endpoint and list-for-endpoint do not serve one.
    assert (
        vname(f"{base}/add-to-endpoint", "POST") == "validate_can_add_guardrail_to_gateway_endpoint"
    )
    assert (
        vname(f"{base}/remove-from-endpoint", "DELETE")
        == "validate_can_remove_guardrail_from_gateway_endpoint"
    )
    assert (
        vname(f"{base}/update-config", "PATCH")
        == "validate_can_update_gateway_endpoint_guardrail_config"
    )
    assert (
        vname(f"{base}/list-for-endpoint", "GET")
        == "validate_can_read_gateway_endpoint_guardrail_configs"
    )


def test_filter_list_gateway_endpoints_drops_unreadable(monkeypatch):
    # Cross-resource gateway list endpoints are filtered after-request to rows the
    # caller can read.
    monkeypatch.setattr(a, "sender_is_admin", lambda: False)
    monkeypatch.setattr(a, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        a, "_role_based_read_predicate", lambda username, rt: lambda rid: rid == "ep-allowed"
    )
    resp = SimpleNamespace(
        json={"endpoints": [{"endpoint_id": "ep-allowed"}, {"endpoint_id": "ep-denied"}]},
        data=None,
    )
    a.filter_list_gateway_endpoints(resp)
    ids = [e["endpoint_id"] for e in json.loads(resp.data)["endpoints"]]
    assert ids == ["ep-allowed"]


def test_filter_list_gateway_model_definitions_drops_unreadable(monkeypatch):
    monkeypatch.setattr(a, "sender_is_admin", lambda: False)
    monkeypatch.setattr(a, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        a, "_role_based_read_predicate", lambda username, rt: lambda rid: rid == "md-allowed"
    )
    resp = SimpleNamespace(
        json={
            "model_definitions": [
                {"model_definition_id": "md-allowed"},
                {"model_definition_id": "md-denied"},
            ]
        },
        data=None,
    )
    a.filter_list_gateway_model_definitions(resp)
    ids = [m["model_definition_id"] for m in json.loads(resp.data)["model_definitions"]]
    assert ids == ["md-allowed"]


def test_filter_list_gateway_secrets_drops_unreadable(monkeypatch):
    monkeypatch.setattr(a, "sender_is_admin", lambda: False)
    monkeypatch.setattr(a, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        a, "_role_based_read_predicate", lambda username, rt: lambda rid: rid == "sec-allowed"
    )
    resp = SimpleNamespace(
        json={"secrets": [{"secret_id": "sec-allowed"}, {"secret_id": "sec-denied"}]},
        data=None,
    )
    a.filter_list_gateway_secrets(resp)
    ids = [s["secret_id"] for s in json.loads(resp.data)["secrets"]]
    assert ids == ["sec-allowed"]


def test_secrets_config_redaction_registered():
    # The redaction filter must be wired for the secrets/config GET so _after_request runs it
    # (the path is otherwise excluded from the auto-built after-request handlers).
    key = (a.GATEWAY_SECRETS_CONFIG, "GET")
    assert a.AFTER_REQUEST_HANDLERS.get(key) is a.redact_secrets_config_for_non_admins


def test_secrets_config_redacts_passphrase_for_non_admin(monkeypatch):
    monkeypatch.setattr(a, "sender_is_admin", lambda: False)
    resp = SimpleNamespace(
        json={"secrets_available": True, "using_default_passphrase": True}, data=None
    )
    a.redact_secrets_config_for_non_admins(resp)
    body = json.loads(resp.data)
    assert body == {"secrets_available": True}
    assert "using_default_passphrase" not in body


def test_secrets_config_keeps_passphrase_for_admin(monkeypatch):
    monkeypatch.setattr(a, "sender_is_admin", lambda: True)
    resp = SimpleNamespace(
        json={"secrets_available": True, "using_default_passphrase": True}, data=None
    )
    a.redact_secrets_config_for_non_admins(resp)
    # Admin response is left untouched (data not rewritten).
    assert resp.data is None


def test_metric_history_bulk_interval_rest_prefix_is_gated():
    # The /api/2.0 twin was ungated when only the /ajax-api path had a validator.
    req = _Req("/api/2.0/mlflow/metrics/get-history-bulk-interval", "GET")
    assert a._find_validator(req) is not None


def test_demo_routes_gating():
    # generate is authenticated-open; delete hard-deletes the shared demo -> admin-only.
    gen = a._find_validator(_Req("/ajax-api/3.0/mlflow/demo/generate", "POST"))
    dele = a._find_validator(_Req("/ajax-api/3.0/mlflow/demo/delete", "POST"))
    assert gen is a._allow_authenticated
    assert dele is a.sender_is_admin


def test_signup_form_submit_takes_the_create_user_check():
    # /signup renders a form that posts to create-ui. Without the same check as the page and the
    # JSON create route, a workspace admin could open the form but never submit it.
    assert a._find_validator(_Req("/signup", "GET")) is a.validate_can_create_user
    assert a._find_validator(_Req(CREATE_USER_UI, "POST")) is a.validate_can_create_user


def test_flask_static_route_is_left_to_the_fail_closed_net():
    # The guards skip Flask's static route on the premise that nothing authorizes it, so the
    # fail-closed net denies it to non-admins. A decision keyed on its /build/ prefix would also
    # catch every route under `--static-prefix /build`.
    assert server_app.static_url_path == "/build"
    assert not _is_covered("/build/index.html", "GET")


def _graphql_schema_field_names():
    # The GraphQL schema is autogenerated from the protos, so a new field can appear without
    # anyone touching the auth module. Graphene exposes the camelCase name the middleware sees.
    from graphene.utils.str_converters import to_camel_case

    from mlflow.server.graphql import autogenerated_graphql_schema as schema

    return {
        to_camel_case(attr)
        for type_name in ("QueryType", "MutationType")
        for attr in vars(getattr(schema, type_name))
        if attr.startswith("mlflow")
    }


def test_graphql_protected_fields_match_the_schema():
    """/graphql resolves to no before- or after-request handler: it is listed in
    _HANDLER_INTERNAL_AUTHZ_SUFFIXES, so test_no_new_ungated_routes skips it, and it is excluded
    from AFTER_REQUEST_HANDLERS. Its only gate is the middleware matching GraphQL field names as
    plain strings, which fails OPEN -- an unlisted field resolves unauthorized. Nothing else
    checks that the list keeps up with the autogenerated schema.
    """
    middleware = a.GraphQLAuthorizationMiddleware
    schema_fields = _graphql_schema_field_names()
    unprotected = sorted(schema_fields - middleware.PROTECTED_FIELDS)
    assert not unprotected, (
        "GraphQL fields with no entry in GraphQLAuthorizationMiddleware.PROTECTED_FIELDS. The "
        "middleware passes an unlisted field straight through, so these resolve with no "
        "authorization at all:\n" + "\n".join(unprotected)
    )
    stale = sorted(middleware.PROTECTED_FIELDS - schema_fields)
    assert not stale, (
        "PROTECTED_FIELDS entries that match no schema field. A renamed field leaves a dead "
        "string here and an ungated field in the schema:\n" + "\n".join(stale)
    )


def test_every_protected_graphql_field_has_a_gate():
    """A field can sit in PROTECTED_FIELDS while _check_authorization has no branch for it: the
    chain ends in `return True`, so the field is routed through the middleware and then allowed.
    That is how mlflowSearchModelVersions came to be row-filtered but never gated -- which is why
    this checks the GATE alone. Accepting a mention in _post_resolve instead would have passed on
    that bug, because the row filter was present the whole time.
    """
    middleware = a.GraphQLAuthorizationMiddleware
    gate = inspect.getsource(middleware._check_authorization)
    ungated = sorted(field for field in middleware.PROTECTED_FIELDS if f'"{field}"' not in gate)
    assert not ungated, (
        "Fields listed in PROTECTED_FIELDS with no branch in _check_authorization. The middleware "
        "routes them and then falls through to `return True`:\n" + "\n".join(ungated)
    )
    # A nested field is filter-only by design: it takes no input of its own to gate on, so its
    # enforcement is the row filter.
    post_resolve = inspect.getsource(middleware._post_resolve)
    for parent_type, field in middleware.PROTECTED_NESTED_FIELDS:
        assert f'"{field}"' in post_resolve, (
            f"{parent_type}.{field} is routed through the middleware but never filtered"
        )


def test_every_after_request_handler_is_dispatchable():
    """A response filter on a parameterized rule is silently unreachable by exact-path lookup.

    AFTER_REQUEST_HANDLERS is keyed by the Flask rule, so ".../traces/<trace_id>" can never equal
    a concrete request path. ``_after_request`` therefore falls back to
    PARAMETERIZED_AFTER_REQUEST_HANDLERS, and a rule missing from that table is registered but
    never runs. That failure mode is invisible from the outside: authorization still passes, so the
    route answers 200 carrying exactly the fields the filter exists to remove. This caught
    PARAMETERIZED_AFTER_REQUEST_HANDLERS being scoped to "/workspaces/", which left the trace,
    logged-model and issue redactors inert.
    """
    unreachable = sorted(
        f"{method:6} {path} -> {handler.__name__}"
        for (path, method), handler in a.AFTER_REQUEST_HANDLERS.items()
        if "<" in path
        and not any(
            m == method and pat.fullmatch(re.sub(r"<[^>]+>", "sample-id", path))
            for (pat, m) in a.PARAMETERIZED_AFTER_REQUEST_HANDLERS
        )
    )
    assert not unreachable, (
        "After-request handlers registered on a parameterized route with no entry in "
        "PARAMETERIZED_AFTER_REQUEST_HANDLERS. They never run:\n" + "\n".join(unreachable)
    )


def test_parameterized_trace_route_redacts_assessments():
    """The UI reads a trace by path parameter, not the SDK's ``traces/get?trace_id=``.

    Both resolve to GetTraceInfoV3, but only the query-string form matches
    AFTER_REQUEST_HANDLERS by exact path, so a redactor reachable from the SDK route can still be
    missing from the one a browser uses. Pinned per prefix because the UI calls the /ajax-api twin.
    """
    for prefix in ("/api", "/ajax-api"):
        path = f"{prefix}/3.0/mlflow/traces/tr-0123456789abcdef"
        matched = [
            handler
            for (pat, method), handler in a.PARAMETERIZED_AFTER_REQUEST_HANDLERS.items()
            if method == "GET" and pat.fullmatch(path)
        ]
        assert matched == [a.redact_trace_info_v3_assessments], (
            f"GET {path} dispatches to {[h.__name__ for h in matched]}, "
            "so assessments ride out unredacted"
        )


@pytest.mark.parametrize("prefix", ["/api/3.0", "/ajax-api/3.0"])
@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("GET", a.validate_can_view_workspace),
        ("PATCH", a.sender_is_admin),
        ("DELETE", a.sender_is_admin),
    ],
)
def test_workspace_routes_resolve_validator_for_concrete_name(prefix, method, expected):
    assert a._find_validator(_Req(f"{prefix}/mlflow/workspaces/ws", method)) is expected

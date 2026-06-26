# Test Plan: Removing Flask from the MLflow Server

## The one invariant

**Identical HTTP behavior before and after.** For the same request, the
de-Flasked server must return the same status code, the same contractual
headers, and a byte-equivalent response body. Every test below exists to defend
that invariant. The migration is a refactor of *transport*, not of API
semantics.

## Current architecture (context)

Today FastAPI (`mlflow/server/fastapi_app.py`) mounts the entire Flask app
(`mlflow/server/__init__.py`) at `/` via a custom WSGI bridge
(`_EfficientWSGIMiddleware`), with native FastAPI routers (otel, job, gateway,
assistant) registered in front of it. Serving splits by server:

- `uvicorn` (default) -> FastAPI app (ASGI)
- `gunicorn` / `waitress` -> Flask app directly (WSGI)

So both apps must work today. ~390 endpoint rules / 338 unique paths / 179
handlers come from protobuf service definitions, registered onto Flask via
`add_url_rule`.

## What we already have (coverage baseline)

| Asset | What it covers | File |
|---|---|---|
| `ServerThread(app)` running `fastapi_app:app` in-process | The **entire** REST functional suite already runs through FastAPI->WSGI->Flask | `test_rest_tracking.py:79,144,174` |
| `_init_server(server_type="flask"\|"fastapi")` | Subprocess launcher, parametrized; defaults fastapi | `integration_test_utils.py:40-89` |
| `mlflow_app_client` (werkzeug `Client` over fresh Flask) / `fastapi_client` (Starlette `TestClient`) | Unit-level request fixtures, both already exist | `tests/server/conftest.py:40,56` |
| `test_handlers.py` via `app.test_request_context()` | Handler unit behavior | `tests/server/test_handlers.py` |
| `test_security.py`, `test_security_integration.py` | Host validation, CORS, headers (Flask + fastapi_client) | `tests/server/` |
| `test_workspace_endpoints.py`, `test_workspace_middleware.py` | Workspace routing/middleware (Flask + FastAPI variants) | `tests/server/` |
| `auth/test_auth.py` (`client` flask + `fastapi_client`), `test_auth_workspace.py` | Basic-auth, RBAC, workspace auth | `tests/server/auth/` |
| `test_otel_api.py`, `test_gateway_api.py`, `jobs/test_endpoint.py`, `assistant/test_api.py` | Native FastAPI routers (already Flask-free) | `tests/server/` |
| `test_init.py`, `test_prometheus_exporter.py` | Serving command builders, metrics | `tests/server/` |

**Implication:** functional regression is largely *already* in place via the
FastAPI app. The genuinely new work is (a) a **parity gate** proving
native-FastAPI == Flask-via-WSGI byte-for-byte, and (b) migrating the **unit
fixtures** off Flask test clients.

## Test strategy: three layers

### Layer A: Endpoint parity harness (new; build first, Phase 0)

The broad net. Proves the FastAPI-native path produces identical responses to
the Flask path across all 338 paths, including error paths, **without** needing
a fully populated store.

**Construction** - two in-process apps over the *same* backend + same
`handlers`:

- `flask_client` = `werkzeug.test.Client` over a fresh Flask app built from
  `get_endpoints()` (exactly `conftest.py:40`'s pattern).
- `fastapi_client` = `starlette.testclient.TestClient(fastapi_app)`.
- Both in-process (no sockets), same SQLAlchemy store, auth off and on.

**Request corpus** - two sources:

1. **Synthetic enumeration** - iterate `get_endpoints()`; for every
   `(path, methods)` fire: an empty body, a malformed-JSON body, a
   wrong-content-type body, and an unsupported method. This catches routing,
   400/404/405/415, and error-serialization parity with zero entity setup.
2. **Golden recorded corpus** - capture real request/response pairs from a
   short `test_rest_tracking` flow (create/get/search/log with concrete IDs)
   and replay against both apps.

**Diff function** - compare `status_code`, order-normalized JSON body, and
*contractual* headers only (`Content-Type`, `Location`, `Content-Disposition`,
gateway timing headers, security headers). Explicitly ignore
`Date`/`Server`/`Content-Length`/connection. Store golden snapshots so the
harness also catches *both* apps drifting together.

This harness becomes the gate that runs in **every** subsequent migration PR.

### Layer B: Functional integration (existing, re-pointed)

- `test_rest_tracking.py` already runs on FastAPI `ServerThread(app)` - keep it;
  it's the primary regression and a required pre-commit gate (per CLAUDE.md).
- Run the full surface against the Flask-free server: model registry,
  artifacts, gateway, jobs, otel, assistant, auth RBAC/workspace, webhooks.
- Collapse the `server_type` parametrization to fastapi-only as Flask routes are
  deleted (drop `server_type="flask"` at `test_rest_tracking.py:2243` and the
  auth `client` fixtures last).
- **New: client/server compatibility** - pip-install a pinned older MLflow
  client, point it at the new server, exercise core flows. Proves the wire
  contract didn't move.

### Layer C: Unit/contract migration (per area, as code moves)

| Existing (Flask) | Target |
|---|---|
| `test_handlers.py` `app.test_request_context()` | A `request_context(method, path, json, query, headers)` builder that sets the new sync request-shim contextvar; same assertions |
| `conftest.py:mlflow_app_client` (werkzeug) | Add `mlflow_fastapi_client` over the native router; keep both during migration, delete Flask one at the end |
| `test_security*.py` (werkzeug Client) | Make `fastapi_client` assertions canonical; port DNS-rebind / CORS preflight / header tests |
| `test_workspace_endpoints.py`, `test_workspace_middleware.py` | FastAPI `TestClient` variants (middleware already exists) |
| `auth/test_auth.py` | Flip `fastapi_client` to canonical |
| `auth/test_auth_workspace.py` `test_request_context()` | Shim builder; **add** Basic-auth-parse + CSRF unit tests |
| `test_prometheus_exporter.py` `app.test_client()` | ASGI `/metrics` endpoint test |
| `test_init.py` command builders | uvicorn unchanged; gunicorn -> `-k uvicorn.workers.UvicornWorker`; drop waitress builder; add Windows-uvicorn assertion |

## Hard cases requiring dedicated parity tests

The naive harness misses these - they're exactly where Flask and Starlette
semantics genuinely differ, so each gets an explicit before/after parity test:

1. **File upload** (`upload-artifact`, multipart) - Flask `request.files` ->
   Starlette `UploadFile`; streaming to disk.
2. **File download / artifact serving** (`get-artifact`,
   `model-versions/get-artifact`, `get-trace-artifact`, logged-model artifacts,
   `static-files`) - `send_file`/`send_from_directory` -> `FileResponse`: Range
   requests, `Content-Disposition`, and caching (`max_age`/`Cache-Control`).
3. **Streaming gateway responses** (`/gateway/...`) - SSE/streamed bodies;
   assert chunk sequence + `MLFLOW_GATEWAY_DURATION_HEADER`/overhead headers
   (byte-diff impossible, so compare structure).
4. **Repeated query params** - `request.args.getlist()` vs
   `query_params.getlist()` (e.g. `view_type`, array fields in GET->protobuf at
   `handlers.py:1007`).
5. **Content-type / empty-body handling** - `get_json(force=True, silent=True)`;
   415 on wrong type.
6. **Path-param encoding** - `<model_id>`, names with slashes/encoded chars; the
   `_convert_path_parameter_to_flask_format` cases (`handlers.py:6696`).
7. **Error serialization** - `MlflowException.serialize_as_json()` + HTTP status
   mapping must be byte-identical.
8. **Auth Basic header parsing** - `werkzeug Authorization` -> manual parse;
   malformed/missing creds -> 401 + `WWW-Authenticate`.
9. **CSRF on signup** - Flask-WTF token issue/validate flow -> replacement.
10. **`flask.g` -> `request.state`** - authenticated-user stamping that handlers
    read (owner attribution, `handlers.py:4653`).
11. **Security middleware ordering** - Host validation + CORS preflight
    (`OPTIONS`) + X-Frame-Options, on health vs API endpoints.
12. **Large-body / concurrency** - the O(n^2) buffering the WSGI bridge worked
    around; add a large-payload + concurrent-request parity case so the
    threadpool body-read path doesn't regress.

## Exit criteria

- Parity harness: 100% of `get_endpoints()` paths exercised (synthetic) + golden
  corpus replays, **zero diffs**.
- All `tests/server/` suites green against the Flask-free app; Flask/werkzeug
  imports gone from `tests/server/` (ideally zero).
- `test_rest_tracking.py` green in-process **and** subprocess.
- Client/server compat test green (old client -> new server).
- Grep gate: no `flask`/`werkzeug`/`flask_wtf`/`flask_cors`/
  `prometheus_flask_exporter` in `mlflow/` except the guarded
  `inference_table.py` import.
- Windows path covered by a Windows CI runner (uvicorn replacing waitress);
  gunicorn-UvicornWorker multi-worker run covered, including prometheus
  multiprocess metric aggregation.

## Sequencing

Phase 0 lands the parity harness; it then runs on every PR. Each migration PR
(handlers -> routes -> auth -> serving) must keep **parity +
`test_rest_tracking.py`** green. The `server_type` parametrization stays alive
until Flask routes are physically deleted, then collapses to fastapi-only.

## Decisions baked in (from earlier discussion)

- gunicorn -> `gunicorn -k uvicorn.workers.UvicornWorker` (keep gunicorn as
  process manager, ASGI via uvicorn workers).
- Flask removed from core deps; `inference_table.py` keeps its lazy
  `import flask` + `has_request_context()` guard (reads a *user's* Flask context
  in Databricks Model Serving, not our server).
- Windows: recommend uvicorn replacing waitress (pending final confirmation).

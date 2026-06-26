"""Phase 0 parity harness for the Flask -> FastAPI server migration.

Standalone, runnable script (NOT wired into pytest). It stands up two in-process
apps over the SAME backend store and diffs their HTTP responses across every
endpoint returned by ``handlers.get_endpoints()``:

  * flask_client   -- werkzeug Client over a fresh Flask app built from
                      get_endpoints() (mirrors tests/server/conftest.py).
  * fastapi_client -- Starlette TestClient over mlflow.server.fastapi_app.app
                      (FastAPI, currently wrapping Flask via the WSGI bridge).

Today both paths funnel into the same handlers, so the harness should report
ZERO diffs. That is the point: it captures the current wire contract as a
reference. As native FastAPI routes replace the Flask mount, this harness is the
gate that proves the responses did not move.

Usage:
    uv run --with httpx python integ-test/flask_parity_harness.py
    uv run --with httpx python integ-test/flask_parity_harness.py --snapshot  # golden
    uv run --with httpx python integ-test/flask_parity_harness.py --verbose

Exit code is non-zero if any unexpected diff is found.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Env must be set BEFORE importing mlflow.server so the modules pick it up.
# Disable security middleware on both apps: Host/CORS parity is a Flask-vs-Starlette
# concern covered by a dedicated parity case, not this broad handler sweep.
os.environ.setdefault("MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE", "true")

_SNAPSHOT_PATH = Path(__file__).with_name("flask_parity_snapshot.json")

# Headers that legitimately vary between WSGI/ASGI servers and request runs.
# Everything else is contractual and must match.
_VOLATILE_HEADERS = {
    "date",
    "server",
    "content-length",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "vary",
}

# Paths whose responses are inherently stateful (mutating singletons);
# order-dependent across two apps sharing one store.
_SKIP_PATHS = {
    "/ajax-api/3.0/mlflow/demo/generate",
    "/ajax-api/3.0/mlflow/demo/delete",
}

# Regex patterns for values that are non-deterministic across runs:
# UUIDs, hex IDs, random run names, temp directory paths, timestamps.
_SCRUB_PATTERNS = [
    (re.compile(r"[0-9a-f]{32}"), "<UUID>"),
    (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"), "<UUID>"),
    (re.compile(r"'[a-z]+-[a-z]+-\d+'"), "'<RUN_NAME>'"),
    (re.compile(r"/tmp/[^\s'\"]+|/var/folders/[^\s'\"]+"), "<TMPDIR>"),
]


def _scrub(value: Any) -> Any:
    """Normalize non-deterministic fragments so body comparisons ignore them."""
    if isinstance(value, str):
        for pat, repl in _SCRUB_PATTERNS:
            value = pat.sub(repl, value)
        return value
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


@dataclass
class Case:
    """One synthetic request fired at both apps."""

    label: str
    method: str
    path: str
    json_body: Any | None = None
    raw_body: bytes | None = None
    content_type: str | None = None


@dataclass
class Captured:
    status: int
    content_type: str | None
    headers: dict[str, str]
    body: Any  # parsed JSON when possible, else decoded text


@dataclass
class Diff:
    case: str
    field: str
    flask: Any
    fastapi: Any


@dataclass
class Report:
    diffs: list[Diff] = field(default_factory=list)
    n_cases: int = 0
    n_paths: int = 0


def _substitute_path_params(path: str) -> str:
    """Replace Flask-style <param> / <path:param> placeholders with dummy values."""
    # <path:foo> -> dummy/segment ; <foo> -> dummy
    path = re.sub(r"<path:[^>]+>", "dummy/segment", path)
    path = re.sub(r"<[^>]+>", "dummy", path)
    return path


def _build_cases(endpoints) -> list[Case]:
    cases: list[Case] = []
    seen: set[tuple[str, str]] = set()
    for http_path, _handler, methods in endpoints:
        if http_path in _SKIP_PATHS:
            continue
        # graphql/internal handlers can carry odd path tokens; the placeholder
        # substitution handles <...>; skip anything still templated.
        concrete = _substitute_path_params(http_path)
        for method in methods:
            key = (method, concrete)
            if key in seen:
                continue
            seen.add(key)

            if method == "GET":
                cases.append(
                    Case(label=f"GET {concrete}", method="GET", path=concrete)
                )
            else:
                # Three error-path probes per write endpoint: empty JSON, malformed
                # JSON, and wrong content-type. These exercise routing + validation
                # parity without needing any entities to exist.
                cases.append(
                    Case(
                        label=f"{method} {concrete} [empty-json]",
                        method=method,
                        path=concrete,
                        json_body={},
                    )
                )
                cases.append(
                    Case(
                        label=f"{method} {concrete} [malformed-json]",
                        method=method,
                        path=concrete,
                        raw_body=b"{not-json",
                        content_type="application/json",
                    )
                )
                cases.append(
                    Case(
                        label=f"{method} {concrete} [wrong-content-type]",
                        method=method,
                        path=concrete,
                        raw_body=b"hello",
                        content_type="text/plain",
                    )
                )
        # Also probe an unsupported method to compare 404/405 behavior.
        concrete_405 = concrete
        unsupported = "DELETE" if "DELETE" not in methods else "PATCH"
        key = (unsupported, concrete_405)
        if key not in seen:
            seen.add(key)
            cases.append(
                Case(
                    label=f"{unsupported} {concrete_405} [unsupported-method]",
                    method=unsupported,
                    path=concrete_405,
                )
            )
    return cases


def _normalize_body(text: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def _contractual_headers(headers) -> dict[str, str]:
    return {
        k.lower(): v
        for k, v in headers.items()
        if k.lower() not in _VOLATILE_HEADERS
    }


def _capture_flask(client, case: Case) -> Captured:
    kwargs: dict[str, Any] = {}
    if case.raw_body is not None:
        kwargs["data"] = case.raw_body
        kwargs["content_type"] = case.content_type
    elif case.json_body is not None:
        kwargs["data"] = json.dumps(case.json_body)
        kwargs["content_type"] = "application/json"
    resp = client.open(case.path, method=case.method, **kwargs)
    text = resp.get_data(as_text=True)
    return Captured(
        status=resp.status_code,
        content_type=resp.headers.get("Content-Type"),
        headers=_contractual_headers(resp.headers),
        body=_normalize_body(text),
    )


def _capture_fastapi(client, case: Case) -> Captured:
    kwargs: dict[str, Any] = {}
    if case.raw_body is not None:
        kwargs["content"] = case.raw_body
        kwargs["headers"] = {"content-type": case.content_type}
    elif case.json_body is not None:
        kwargs["json"] = case.json_body
    resp = client.request(case.method, case.path, **kwargs)
    return Captured(
        status=resp.status_code,
        content_type=resp.headers.get("content-type"),
        headers=_contractual_headers(resp.headers),
        body=_normalize_body(resp.text),
    )


def _diff_case(case: Case, f: Captured, a: Captured) -> list[Diff]:
    diffs: list[Diff] = []
    if f.status != a.status:
        diffs.append(Diff(case.label, "status", f.status, a.status))
    # Compare content-type up to the leading media type (ignore charset noise).
    f_ct = (f.content_type or "").split(";")[0].strip()
    a_ct = (a.content_type or "").split(";")[0].strip()
    if f_ct != a_ct:
        diffs.append(Diff(case.label, "content_type", f_ct, a_ct))
    f_body = _scrub(f.body)
    a_body = _scrub(a.body)
    if f_body != a_body:
        diffs.append(Diff(case.label, "body", f_body, a_body))
    return diffs


def _setup_stores() -> None:
    from mlflow.server import handlers

    tmp = Path(tempfile.mkdtemp(prefix="flask-parity-"))
    backend_uri = f"sqlite:///{tmp / 'mlflow.db'}"
    artifact_root = (tmp / "artifacts").as_uri()
    handlers._tracking_store = None
    handlers._model_registry_store = None
    handlers.initialize_backend_stores(backend_uri, default_artifact_root=artifact_root)


def _build_flask_client():
    from flask import Flask
    from werkzeug.test import Client

    from mlflow.server import handlers

    app = Flask(__name__)
    for http_path, handler, methods in handlers.get_endpoints():
        app.add_url_rule(http_path, handler.__name__, handler, methods=methods)
    return Client(app)


def _build_fastapi_client():
    from starlette.testclient import TestClient

    from mlflow.server.fastapi_app import app

    return TestClient(app, raise_server_exceptions=False)


def run(verbose: bool, write_snapshot: bool) -> int:
    _setup_stores()

    from mlflow.server import handlers

    endpoints = handlers.get_endpoints()
    cases = _build_cases(endpoints)

    flask_client = _build_flask_client()
    fastapi_client = _build_fastapi_client()

    report = Report(n_cases=len(cases), n_paths=len({p for p, _, _ in endpoints}))
    snapshot: dict[str, Any] = {}

    for case in cases:
        f = _capture_flask(flask_client, case)
        a = _capture_fastapi(fastapi_client, case)
        report.diffs.extend(_diff_case(case, f, a))
        if write_snapshot:
            snapshot[case.label] = {
                "status": f.status,
                "content_type": (f.content_type or "").split(";")[0].strip(),
                "body": f.body,
            }
        if verbose:
            print(f"  {case.label}: flask={f.status} fastapi={a.status}")

    print(
        f"\nParity sweep: {report.n_paths} paths, {report.n_cases} cases, "
        f"{len(report.diffs)} diffs"
    )

    if write_snapshot:
        _SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
        print(f"Wrote golden snapshot -> {_SNAPSHOT_PATH}")

    if report.diffs:
        print("\nDIVERGENCES (flask vs fastapi):")
        for d in report.diffs[:50]:
            print(f"  [{d.case}] {d.field}:")
            print(f"      flask:   {d.flask!r}")
            print(f"      fastapi: {d.fastapi!r}")
        if len(report.diffs) > 50:
            print(f"  ... and {len(report.diffs) - 50} more")
        return 1

    print("OK: FastAPI and Flask responses are identical across all probed cases.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="print every case")
    parser.add_argument(
        "--snapshot", action="store_true", help="write the Flask golden snapshot"
    )
    args = parser.parse_args()
    return run(verbose=args.verbose, write_snapshot=args.snapshot)


if __name__ == "__main__":
    sys.exit(main())

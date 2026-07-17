"""
Security-relevant validation shared between the `--static-prefix` CLI option
(`mlflow/cli/__init__.py`) and `create_fastapi_app()`
(`mlflow/server/fastapi_app.py`).

`_MLFLOW_STATIC_PREFIX` can be set directly, bypassing the CLI's validation
callback (e.g. running an ASGI server against `mlflow.server.fastapi_app:app`
directly instead of via `mlflow server --static-prefix`), so
`create_fastapi_app()` must independently reject unsafe values to fail closed.
This module has no dependency on `mlflow.server`/`mlflow.server.auth` (both of
which pull in Flask/FastAPI app construction) so the CLI can import it without
paying that cost just to parse arguments.
"""

# Route roots a `--static-prefix` must not overlap.
#
# The last two are load-bearing: `artifact_router`'s native routes are the one set
# `create_fastapi_app` does NOT register under the prefix, so a prefix overlapping them
# would let their unprefixed routes win Starlette's first-match-wins routing over a
# prefixed route with an identically-shaped path, running one route's handler under
# another's (weaker) permission check.
#
# The first four are the prefix-aware route roots themselves. Registering those routers
# only under the prefix already removes the ambiguity, so these are kept as defense in
# depth against confusing `<prefix>/<same-root>` paths.
RESERVED_STATIC_PREFIX_ROOTS = (
    "/gateway",
    "/v1/traces",
    "/ajax-api/3.0/jobs",
    "/ajax-api/3.0/mlflow/assistant",
    "/api/2.0/mlflow-artifacts/artifacts",
    "/ajax-api/2.0/mlflow-artifacts/artifacts",
)

# `is_unprotected_route()` (mlflow/server/auth/__init__.py) matches these lexically,
# not segment-boundary aware (the real static-files route is `/static-files/<path>`,
# not `/static`). A `--static-prefix` that itself lexically starts with one of these
# would make every route served under it skip authentication entirely.
UNPROTECTED_STATIC_PREFIX_ROOTS = ("/health", "/static", "/favicon.ico")


def _path_segments_overlap(a: str, b: str) -> bool:
    return a == b or a.startswith(f"{b}/") or b.startswith(f"{a}/")


def validate_static_prefix_security(value: str) -> None:
    """
    Raise `ValueError` if `value` would be unsafe to serve routes under (see
    `create_fastapi_app`). Only covers the checks that affect authentication/routing
    correctness; format checks (leading/trailing slash) are CLI-input concerns handled
    separately by `mlflow.cli._validate_static_prefix`.
    """
    # `create_fastapi_app` passes this value as the `prefix` argument to FastAPI's
    # `include_router()`, which interprets "{"/"}" as Starlette path-parameter
    # template syntax (e.g. "/{user}" matches "/alice", "/bob", ...). The auth checks
    # compare against the literal configured value, so any concrete request matching
    # such a template resolves to no validator at all: an unauthenticated bypass.
    if "{" in value or "}" in value:
        raise ValueError("must not contain '{' or '}'.")

    # See `RESERVED_STATIC_PREFIX_ROOTS` for why overlapping these can route a request
    # to one handler while a different route's permission check is applied.
    for reserved in RESERVED_STATIC_PREFIX_ROOTS:
        if _path_segments_overlap(value, reserved):
            raise ValueError(
                f"conflicts with the reserved route '{reserved}'. Choose a prefix "
                f"that does not overlap {', '.join(RESERVED_STATIC_PREFIX_ROOTS)}."
            )

    if value.startswith(UNPROTECTED_STATIC_PREFIX_ROOTS):
        raise ValueError(
            f"conflicts with the unauthenticated routes "
            f"{', '.join(UNPROTECTED_STATIC_PREFIX_ROOTS)}. Choose a prefix that "
            f"does not start with any of these."
        )

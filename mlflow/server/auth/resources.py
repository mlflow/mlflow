"""The auth layer's single path for reading a tracking/registry resource.

Two jobs, and the second is why this module exists:

1. ``fetch_*`` -- load an entity, returning ``None`` rather than raising when it does
   not exist, so a missing resource denies instead of 404ing (the response must not be
   an oracle for which ids exist).
2. ``attrs_for`` / ``attrs_for_bulk`` -- project an entity's condition-relevant state
   into a :class:`~mlflow.server.auth.conditions.ResourceValues`.

Both are memoized per request. The memo is what makes condition enforcement close to
free on the paths that already load the entity for their base authorization check: the
validator's fetch populates the cache, and the condition gate reads it back instead of
issuing a second query.

**Scope: only the auth module's own resource reads.** The ~46 auth-*store* reads
(``store.get_user``, ``store.list_grants``, role and assignment queries) are a
different database and a different concern, and do not come through here.

**Not routed through here either: the ``workspace_fetcher=`` sites.**
``_get_resource_workspace`` consults a cross-request TTL cache *before* calling its
fetcher and discards the entity even on a miss, so it cannot supply condition
attributes. Making it retain entities would mean either caching mutable tags across
requests (unsafe -- see the cache-key note below) or bypassing a hot-path cache on
every call. So ``experiment`` reads here at evaluation time like the version types,
and only ``run``, ``trace``, ``logged_model`` and ``registered_model``/``prompt`` get
a free memo hit from their base check.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from typing import Any
from urllib.parse import quote, unquote

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.server.auth.conditions import (
    ALIAS_OWNING_RESOURCE_TYPES,
    ResourceValues,
    resource_values_shape,
)
from mlflow.utils import workspace_context

# ---------------------------------------------------------------------------
# Storage kinds
# ---------------------------------------------------------------------------

#: Maps an auth resource type onto the *storage* identity behind it.
#:
#: A prompt IS a registered model -- same table, same row, same tags -- so both auth
#: types must share one cache entry. This is required rather than an optimisation:
#: the fetch **precedes** the classification, because the routes that distinguish the
#: two families do so by fetching the row and reading ``rm._is_prompt()``. At fetch
#: time the auth type is not yet known, so there is nothing family-specific to key on.
#:
#: The sequence is: fetch under the storage key -> classify the family -> load
#: conditions under the auth ``resource_type``. Conditions stay keyed on the auth type,
#: so prompt/model parity (D2) is unaffected; ``storage_kind`` never escapes this
#: module.
_STORAGE_KIND = {
    "registered_model": "registry_entry",
    "prompt": "registry_entry",
    "registered_model_version": "registry_version",
    "prompt_version": "registry_version",
    "experiment": "experiment",
    "run": "run",
    "trace": "trace",
    "logged_model": "logged_model",
    "mcp_server": "mcp_server",
    "mcp_server_version": "mcp_server_version",
}


def storage_kind(resource_type: str) -> str:
    return _STORAGE_KIND.get(resource_type, resource_type)


def version_resource_id(name: str, version: str) -> str:
    """Compose the opaque id for a model/prompt version.

    Percent-encoding the name keeps the separator unambiguous for names containing
    ``/``, following the existing ``_scorer_pattern`` precedent in the auth store.
    """
    return f"{quote(name, safe='')}/{version}"


def _split_version_resource_id(resource_id: str) -> tuple[str, str]:
    name, _, version = resource_id.rpartition("/")
    return unquote(name), version


# ---------------------------------------------------------------------------
# Request-scoped caches
# ---------------------------------------------------------------------------

# Where the caches live, and why it is two mechanisms rather than one.
#
# When a Flask request context is active, the caches hang off ``g``. Flask tears ``g``
# down with the request context, so the cache cannot outlive the request even if no
# explicit clear runs -- which matters because validators are also invoked directly
# (outside ``_before_request``) by tests and by the GraphQL path, and a cache that
# straddled two such invocations would answer the second with the first's tags.
#
# The FastAPI middleware bypasses Flask's hooks and has no ``g`` outside the
# custom-auth bridge, so there the caches fall back to ContextVars, following
# ``workspace_context``. Unlike ``g``, a ContextVar is NOT torn down for us: an
# uncleared entry persists on the worker thread and would be read by the *next* request
# served there. Hence ``clear_cache()`` in a ``finally`` on that funnel.
#
# Deliberately NOT modelled on ``_RESOURCE_WORKSPACE_CACHE``, whose cross-request
# TTLCache is justified by an explicit immutability comment ("the relationship ... is
# immutable, so caching is safe"). Tags and aliases are mutable, so caching them beyond
# one request would be wrong.
_ENTITY_CACHE: ContextVar[dict[tuple[str, str | None, str], Any] | None] = ContextVar(
    "mlflow_auth_resource_entities", default=None
)
_ATTRS_CACHE: ContextVar[dict[tuple[str, str | None, str], ResourceValues] | None] = ContextVar(
    "mlflow_auth_resource_attrs", default=None
)

_G_ENTITY_ATTR = "_mlflow_auth_resource_entities"
_G_ATTRS_ATTR = "_mlflow_auth_resource_attrs"


def _cache_key(resource_type: str, resource_id: str) -> tuple[str, str | None, str]:
    """``(storage_kind, workspace, resource_id)``.

    ``workspace`` is included because a resource id is not globally unique -- for a
    registered model the id *is* the name, and names are unique per workspace. Within
    one request the active workspace is constant, so strictly it is redundant under
    request scoping; it costs nothing and documents the invariant.
    """
    return (
        storage_kind(resource_type),
        workspace_context.get_request_workspace(),
        resource_id,
    )


def _in_flask_request() -> bool:
    try:
        from flask import has_request_context

        return has_request_context()
    except Exception:
        return False


def _cache(g_attr: str, var: ContextVar) -> dict[str, Any]:
    if _in_flask_request():
        from flask import g

        cache = getattr(g, g_attr, None)
        if cache is None:
            cache = {}
            setattr(g, g_attr, cache)
        return cache
    cache = var.get()
    if cache is None:
        cache = {}
        var.set(cache)
    return cache


def _entities() -> dict[str, Any]:
    return _cache(_G_ENTITY_ATTR, _ENTITY_CACHE)


def _attrs() -> dict[str, Any]:
    return _cache(_G_ATTRS_ATTR, _ATTRS_CACHE)


_CONDITION_DENIAL: ContextVar[bool] = ContextVar("mlflow_auth_condition_denial", default=False)

_G_CONDITION_DENIAL_ATTR = "_mlflow_auth_condition_denial"


def note_condition_denial() -> None:
    """Record that a mutation condition -- not a missing grant -- refused this request.

    Lives here, beside the resource memo, because it has exactly the same lifetime and the
    same hazard: left set, it would label the NEXT request's grant denial as a condition
    denial. Putting it behind ``clear_cache()`` means every funnel that already clears
    covers it too, so the "forgot to clear" failure mode cannot be reintroduced by adding a
    funnel.
    """
    _CONDITION_DENIAL.set(True)
    if _in_flask_request():
        from flask import g

        setattr(g, _G_CONDITION_DENIAL_ATTR, True)


def condition_denied() -> bool:
    """Whether a condition refused something during this request.

    Read only to choose between two 403 messages, so it can never widen access. It says a
    condition failed somewhere in this request's authorization, not that a condition was the
    sole reason: a validator evaluating a disjunction can have one branch refused by a
    condition and another by a grant.
    """
    if _in_flask_request():
        from flask import g

        if getattr(g, _G_CONDITION_DENIAL_ATTR, False):
            return True
    return _CONDITION_DENIAL.get()


def clear_cache() -> None:
    """Drop both caches and the denial reason. Call from a ``finally`` when before-request
    work completes.

    Mandatory on the FastAPI funnel, where the caches live in ContextVars that survive
    the request: without this a pooled worker thread would serve the next request stale
    tags. On the Flask path ``g`` already discards them, so this is belt-and-braces --
    but it is also what narrows the window to before-request rather than the whole
    request context.
    """
    _ENTITY_CACHE.set(None)
    _ATTRS_CACHE.set(None)
    _CONDITION_DENIAL.set(False)
    if _in_flask_request():
        from flask import g

        for attr in (_G_ENTITY_ATTR, _G_ATTRS_ATTR, _G_CONDITION_DENIAL_ATTR):
            if hasattr(g, attr):
                delattr(g, attr)


def cache_sizes() -> tuple[int, int]:
    """``(entities, attrs)`` currently cached. For tests and diagnostics."""
    if _in_flask_request():
        from flask import g

        return (
            len(getattr(g, _G_ENTITY_ATTR, None) or {}),
            len(getattr(g, _G_ATTRS_ATTR, None) or {}),
        )
    return len(_ENTITY_CACHE.get() or {}), len(_ATTRS_CACHE.get() or {})


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _fetch_or_none(fetch: Callable[[str], Any], resource_id: str):
    """Missing -> ``None``; every other failure propagates.

    Swallowing all exceptions would report an outage as "permission denied" and send
    operators after the wrong fault.
    """
    try:
        return fetch(resource_id)
    except MlflowException as e:
        if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return None
        raise


def _memoized(resource_type: str, resource_id: str, loader: Callable[[], Any]):
    cache = _entities()
    key = _cache_key(resource_type, resource_id)
    if key in cache:
        return cache[key]
    entity = loader()
    cache[key] = entity
    return entity


def _tracking_store():
    """Resolve the tracking store **through the auth module**, not from
    ``mlflow.server.handlers`` directly.

    The auth module re-exports ``_get_tracking_store`` /
    ``_get_model_registry_store``, and those module attributes are the established
    seam: existing tests substitute fake stores by patching them. Importing the
    originals here would bypass every such patch and silently run authorization
    against the real store.
    """
    from mlflow.server import auth as auth_module

    return auth_module._get_tracking_store()


def _registry_store():
    """See :func:`_tracking_store` -- resolved through the auth module's seam."""
    from mlflow.server import auth as auth_module

    return auth_module._get_model_registry_store()


def fetch_experiment(experiment_id: str):
    return _memoized(
        "experiment",
        experiment_id,
        lambda: _fetch_or_none(_tracking_store().get_experiment, experiment_id),
    )


def fetch_run(run_id: str):
    return _memoized("run", run_id, lambda: _fetch_or_none(_tracking_store().get_run, run_id))


def fetch_trace_info(trace_id: str):
    return _memoized(
        "trace", trace_id, lambda: _fetch_or_none(_tracking_store().get_trace_info, trace_id)
    )


def fetch_logged_model(model_id: str):
    return _memoized(
        "logged_model",
        model_id,
        lambda: _fetch_or_none(_tracking_store().get_logged_model, model_id),
    )


def fetch_mcp_server(name: str):
    """Fetch an MCP server entry.

    The name is the ``namespace/slug`` pair the routes address it by, which is already the
    store's key, so no composition is needed -- unlike a registry version, whose id this
    layer has to build.
    """
    return _memoized(
        "mcp_server",
        name,
        lambda: _fetch_or_none(_tracking_store().get_mcp_server, name),
    )


def fetch_mcp_server_version(name: str, version: str):
    """Fetch one version of an MCP server.

    Keyed on the composed id rather than the pair, so it shares the memo shape every other
    version fetch uses.
    """

    def load():
        try:
            return _tracking_store().get_mcp_server_version(name, version)
        except MlflowException as e:
            if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
                return None
            raise

    return _memoized("mcp_server_version", version_resource_id(name, version), load)


# A cascade's children have to be enumerated to judge each one, and that enumeration is
# unbounded in principle: an experiment can hold any number of runs. The cap bounds the work,
# and exceeding it is reported as "cannot enumerate" rather than as "no children" -- the gate
# then refuses, because a condition that cannot be evaluated must never pass vacuously.
MAX_CASCADE_CHILDREN = 2000


def _collect_ids(fetch_page, id_of) -> "tuple[str, ...] | None":
    """Page through a search, returning ids -- or ``None`` when there are too many.

    ``None`` and ``()`` mean different things to the caller and must not be conflated: ``()``
    is "this parent genuinely has no children", which lets the cascade proceed, while ``None``
    is "the children could not be enumerated", which must deny.
    """
    ids: list[str] = []
    token = None
    while True:
        page = fetch_page(token)
        ids.extend(id_of(entity) for entity in page)
        if len(ids) > MAX_CASCADE_CHILDREN:
            return None
        token = getattr(page, "token", None)
        if not token:
            return tuple(ids)


def runs_of_experiment(experiment_id: str) -> "tuple[str, ...] | None":
    from mlflow.entities import ViewType

    store = _tracking_store()
    return _collect_ids(
        lambda token: store.search_runs(
            [experiment_id],
            None,
            # ALL, not ACTIVE_ONLY: delete transitions the active children and restore
            # transitions the deleted ones, and this one helper serves both. Judging the
            # wider set can only deny more, never less.
            ViewType.ALL,
            max_results=500,
            page_token=token,
        ),
        lambda run: run.info.run_id,
    )


def traces_of_experiment(experiment_id: str) -> "tuple[str, ...] | None":
    store = _tracking_store()
    return _collect_ids(
        lambda token: store.search_traces(
            experiment_ids=[experiment_id], max_results=500, page_token=token
        ),
        lambda trace: trace.request_id,
    )


def logged_models_of_experiment(experiment_id: str) -> "tuple[str, ...] | None":
    store = _tracking_store()
    return _collect_ids(
        lambda token: store.search_logged_models(
            experiment_ids=[experiment_id], max_results=500, page_token=token
        ),
        lambda model: model.model_id,
    )


def versions_of_registered_model(name: str) -> "tuple[str, ...] | None":
    """Every version of a registered model, or ``None`` when they cannot be established.

    The name has to travel inside a search filter string, and a name containing a quote is the
    dangerous case: a filter that parses but matches nothing returns no rows, which the gate
    would read as "this model has no versions" and let the cascade through -- a silent
    fail-OPEN driven by the resource's own name. So a name that cannot be quoted unambiguously
    is reported as unenumerable instead, and the cascade is refused.

    The quoting mirrors MLflow's own convention (see `mlflow.genai.datasets`): prefer double
    quotes, fall back to single. The case that convention handles by doubling the quote is
    NOT used here, because the search parser does not unescape it -- the filter would then
    match nothing, which is exactly the fail-open above.

    Results are additionally checked against the requested name, so an over-broad filter
    cannot quietly widen the set either.
    """
    if '"' not in name:
        filter_string = f'name = "{name}"'
    elif "'" not in name:
        filter_string = f"name = '{name}'"
    else:
        return None

    store = _registry_store()

    def page(token):
        found = store.search_model_versions(filter_string, max_results=500, page_token=token)
        kept = [version for version in found if version.name == name]
        return _same_paging(found, kept)

    return _collect_ids(
        page,
        # The composed id, so the projection can fetch the version back.
        lambda version: version_resource_id(version.name, str(version.version)),
    )


def _same_paging(original, items):
    """`items` carrying `original`'s continuation token, so filtering a page keeps paging."""

    class _Filtered(list):
        token = getattr(original, "token", None)

    return _Filtered(items)


def versions_of_mcp_server(name: str) -> "tuple[str, ...] | None":
    store = _tracking_store()
    return _collect_ids(
        lambda token: store.search_mcp_server_versions(name, max_results=500, page_token=token),
        lambda version: version_resource_id(name, str(version.version)),
    )


def fetch_registered_model(name: str):
    """Fetch a registry entry, whichever family it turns out to be.

    Cached under the family-agnostic ``registry_entry`` kind, so a subsequent
    ``attrs_for("prompt", name)`` or ``attrs_for("registered_model", name)`` both hit
    this one row.
    """
    return _memoized(
        "registered_model",
        name,
        lambda: _fetch_or_none(_registry_store().get_registered_model, name),
    )


def fetch_model_version(name: str, version: str):
    def load():
        try:
            return _registry_store().get_model_version(name, version)
        except MlflowException as e:
            if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
                return None
            raise

    return _memoized("registered_model_version", version_resource_id(name, version), load)


def _fetch_for(resource_type: str, resource_id: str):
    if resource_type == "experiment":
        return fetch_experiment(resource_id)
    if resource_type == "run":
        return fetch_run(resource_id)
    if resource_type == "trace":
        return fetch_trace_info(resource_id)
    if resource_type == "logged_model":
        return fetch_logged_model(resource_id)
    if resource_type in ("registered_model", "prompt"):
        return fetch_registered_model(resource_id)
    if resource_type in ("registered_model_version", "prompt_version"):
        return fetch_model_version(*_split_version_resource_id(resource_id))
    if resource_type == "mcp_server":
        return fetch_mcp_server(resource_id)
    if resource_type == "mcp_server_version":
        return fetch_mcp_server_version(*_split_version_resource_id(resource_id))
    return None


def require(entity, label: str, resource_id: str):
    """Raise ``RESOURCE_DOES_NOT_EXIST`` for a ``None`` from a ``fetch_*``.

    The memo stores entity-or-``None`` so one cache serves both styles of call site.
    Most auth call sites want the ``None`` (a missing resource denies rather than
    404ing, so the response is not an oracle for which ids exist), but a few
    deliberately let the store's exception propagate, and their callers catch it. This
    reproduces that exception rather than letting ``None`` surface later as an
    ``AttributeError``.
    """
    if entity is None:
        raise MlflowException(
            f"{label} '{resource_id}' does not exist.",
            RESOURCE_DOES_NOT_EXIST,
        )
    return entity


def fetch_run_strict(run_id: str):
    """Memoized ``get_run`` that raises on a missing run, as the store does."""
    return require(fetch_run(run_id), "Run", run_id)


def fetch_registered_model_strict(name: str):
    """Memoized registry fetch that raises on a missing entry, as the store does."""
    return require(fetch_registered_model(name), "Registered model", name)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def _as_str_mapping(mapping) -> Mapping[str, str]:
    """Coerce a mapping's keys and values to ``str``.

    The condition vocabulary compares strings only -- which is also why the comparator
    allowlist excludes the ordering operators -- so a value arriving as another type can
    never satisfy a clause, and on the resource side that denies rather than allows.
    """
    return {str(k): str(v) for k, v in dict(mapping).items()}


def _tags_of(entity) -> Mapping[str, str]:
    """Project tags from ``._tags``, not the public ``tags`` property.

    ``SearchUtils`` reads ``._tags`` with the reasoning in a comment -- "consider all tags
    including reserved ones" -- and matching it is what keeps a resource condition selecting
    exactly the resources the same filter string would return in a search box. The public
    ``RegisteredModel.tags`` **strips** the prompt marker ("should not be user-facing"), so
    reading the property would diverge from search on an entity carrying it.

    D4 now rejects a ``tags.mlflow.*`` clause at authoring in both namespaces, so no stored
    clause can name a reserved key and the two projections agree on every key a condition
    can actually reach. ``._tags`` is kept anyway: search parity is the invariant, and it
    should not depend on which keys the authoring layer happens to allow today.

    Keys and values are coerced to ``str`` for the reason given in :func:`_as_str_mapping`.
    """
    if (private := getattr(entity, "_tags", None)) is not None:
        return _as_str_mapping(private)
    tags = getattr(entity, "tags", None)
    if isinstance(tags, Mapping):
        return _as_str_mapping(tags)
    if tags:
        # A repeated proto field or a list of tag entities.
        return {str(t.key): str(t.value) for t in tags}
    return {}


def _aliases_of(entity, resource_type: str) -> Mapping[str, str]:
    """Project aliases, but only for the type that *owns* them (D18).

    ``RegisteredModel.aliases`` is a mapping ``alias -> version``;
    ``ModelVersion.aliases`` is a ``list[str]`` naming aliases stored on the *parent*.
    Treating the version's list as its own state would let one alias be governed under
    two different resource types, so a version's ``ResourceValues.aliases`` is always
    empty and an alias clause is keyed on the registry entry.

    The version is coerced to ``str`` because the store returns it as an ``int`` while
    every comparison in a condition is a string comparison. Left as an ``int`` it would
    match no clause an admin could write -- ``aliases.champion = '1'`` would compare
    ``1`` against ``'1'`` and fail -- which on the resource side denies every mutation of
    the type. The same failure that :func:`validate_condition`'s type cross-check exists
    to prevent, arriving by a different route.
    """
    if resource_type not in ALIAS_OWNING_RESOURCE_TYPES:
        return {}
    aliases = getattr(entity, "aliases", None)
    if isinstance(aliases, Mapping):
        return {str(alias): str(version) for alias, version in aliases.items()}
    return {}


def values_for_entity(resource_type: str, resource_id: str, entity) -> ResourceValues:
    """Project an entity into the values shape its resource type declares.

    The shape decides what gets projected, so a type that owns no alias produces a value
    object with no alias field at all -- rather than one carrying an empty mapping that
    reads the same as "has no aliases right now". The distinction matters because absence
    fails on the resource side (D20): an empty field and a missing field would deny
    identically here, but only the missing field says the type could never have one.
    """
    shape = resource_values_shape(resource_type)
    if "aliases" in shape._fields:
        return shape(
            resource_id=resource_id,
            tags=_tags_of(entity),
            aliases=_aliases_of(entity, resource_type),
        )
    return shape(resource_id=resource_id, tags=_tags_of(entity))


def attrs_for(resource_type: str, resource_id: str) -> ResourceValues | None:
    """The resource's condition-relevant state, or ``None`` if it does not exist.

    ``None`` means deny: a condition cannot be satisfied by a resource that is not
    there, and denying rather than 404ing keeps the response from revealing which ids
    exist.

    Sharing one cache entry across the two auth types of a storage kind is safe because
    the projection is identical within a kind -- ``registered_model`` and ``prompt`` are
    both alias-owning, and neither version type is -- so the same row yields the same
    ``ResourceValues`` whichever family the caller names.
    """
    cache = _attrs()
    key = _cache_key(resource_type, resource_id)
    if key in cache:
        return cache[key]
    entity = _fetch_for(resource_type, resource_id)
    if entity is None:
        return None
    values = values_for_entity(resource_type, resource_id, entity)
    cache[key] = values
    return values


def attrs_for_bulk(
    resource_type: str, resource_ids: Iterable[str]
) -> dict[str, ResourceValues | None]:
    """Attributes for many ids, filling misses with one bulk call where possible.

    Traces have ``batch_get_trace_infos`` on the abstract store -- it batches
    internally, skips spans, and returns tags -- so N ids cost one call with no new
    store method and no cap needed. Everything else falls back to per-id fetches,
    which are already memoized.
    """
    ids = list(dict.fromkeys(resource_ids))
    resolved: dict[str, ResourceValues | None] = {}
    missing: list[str] = []

    attrs_cache = _attrs()
    for resource_id in ids:
        key = _cache_key(resource_type, resource_id)
        if key in attrs_cache:
            resolved[resource_id] = attrs_cache[key]
        else:
            missing.append(resource_id)

    if missing and resource_type == "trace":
        _prefetch_traces(missing)

    for resource_id in missing:
        resolved[resource_id] = attrs_for(resource_type, resource_id)
    return resolved


def _prefetch_traces(trace_ids: list[str]) -> None:
    """Populate the entity cache for many traces in one store call."""
    store = _tracking_store()
    batch = getattr(store, "batch_get_trace_infos", None)
    if batch is None:
        return
    entity_cache = _entities()
    uncached = [t for t in trace_ids if _cache_key("trace", t) not in entity_cache]
    if not uncached:
        return
    try:
        infos = batch(uncached)
    except MlflowException:
        # Fall back to per-id fetches; a bulk failure must not deny the request on its
        # own, because the per-id path reaches the same decision.
        return
    found = set()
    for info in infos or []:
        trace_id = getattr(info, "trace_id", None) or getattr(info, "request_id", None)
        if trace_id is None:
            continue
        entity_cache[_cache_key("trace", trace_id)] = info
        found.add(trace_id)
    # A trace the batch did not return does not exist. Cache the negative so the
    # per-id loop below does not re-query for it.
    for trace_id in uncached:
        if trace_id not in found:
            entity_cache[_cache_key("trace", trace_id)] = None

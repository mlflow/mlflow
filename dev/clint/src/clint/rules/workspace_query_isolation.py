import ast
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from clint.rules.base import Rule

# Map each base store path to its workspace-aware counterpart.
_TRACKING = "mlflow/store/tracking"
_REGISTRY = "mlflow/store/model_registry"
_JOBS = "mlflow/store/jobs"
WORKSPACE_STORE_PATHS = {
    f"{_TRACKING}/sqlalchemy_store.py": f"{_TRACKING}/sqlalchemy_workspace_store.py",
    f"{_REGISTRY}/sqlalchemy_store.py": f"{_REGISTRY}/sqlalchemy_workspace_store.py",
    f"{_JOBS}/sqlalchemy_store.py": f"{_JOBS}/sqlalchemy_workspace_store.py",
}
BASE_STORE_PATHS = set(WORKSPACE_STORE_PATHS)
# Methods that intentionally don't need workspace isolation (e.g., DB setup).
ALLOWLISTED_METHODS = {"_initialize_store_state"}

# Per-model coverage for workspace helpers.  When a method calls one of these
# helpers via ``self``, only queries on the listed models are trusted.
# New workspace helpers must be added here to be recognized by the linter.
#
# Child models (e.g. SqlMetric, SqlTag) that derive workspace scope from a
# validated parent are listed under the helper that validates the parent.
HELPER_MODEL_COVERAGE: dict[str, frozenset[str]] = {
    # --- Experiment-level validation ---
    "get_experiment": frozenset(
        {"SqlExperiment", "SqlExperimentTag", "SqlScorer", "SqlScorerVersion"}
    ),
    "get_experiment_by_name": frozenset({"SqlExperiment"}),
    "_get_experiment": frozenset({"SqlExperiment", "SqlExperimentTag"}),
    "_experiment_where_clauses": frozenset({"SqlExperiment"}),
    "_filter_experiment_ids": frozenset({"SqlExperiment", "SqlDataset", "SqlInputTag"}),
    # --- Run-level validation ---
    "_validate_run_accessible": frozenset(
        {"SqlRun", "SqlMetric", "SqlTag", "SqlEntityAssociation"}
    ),
    "_get_run": frozenset(
        {
            "SqlRun",
            "SqlMetric",
            "SqlLatestMetric",
            "SqlTag",
            "SqlInput",
            "SqlInputTag",
            "SqlDataset",
            "SqlLoggedModelMetric",
        }
    ),
    # --- Trace-level validation ---
    "_validate_trace_accessible": frozenset(
        {
            "SqlTraceInfo",
            "SqlAssessments",
            "SqlTraceTag",
            "SqlTraceMetadata",
            "SqlEntityAssociation",
        }
    ),
    "_trace_query": frozenset({"SqlTraceInfo", "SqlTraceMetadata"}),
    "_get_sql_assessment": frozenset({"SqlAssessments"}),
    # --- Dataset-level validation ---
    "_validate_dataset_accessible": frozenset(
        {"SqlEvaluationDataset", "SqlEvaluationDatasetRecord", "SqlEvaluationDatasetTag"}
    ),
    "_dataset_query": frozenset(
        {
            "SqlEvaluationDataset",
            "SqlEvaluationDatasetRecord",
            "SqlEvaluationDatasetTag",
            "SqlEntityAssociation",
        }
    ),
    # --- Logged model validation ---
    "_get_logged_model_record": frozenset(
        {"SqlLoggedModel", "SqlLoggedModelTag", "SqlLoggedModelParam"}
    ),
    # --- Entity ID / association filtering (workspace overrides) ---
    "_filter_entity_ids": frozenset({"SqlMetric", "SqlEntityAssociation"}),
    "_filter_association_query": frozenset({"SqlEntityAssociation"}),
    # --- Endpoint-level validation ---
    "_filter_endpoint_binding_query": frozenset({"SqlGatewayEndpointBinding"}),
    # --- Webhook validation ---
    "_get_webhook_by_id": frozenset({"SqlWebhook", "SqlWebhookEvent"}),
}

# Model definition files to discover WorkspaceIsolatedModel subclasses.
MODEL_FILE_PATHS = [
    "mlflow/store/tracking/dbmodels/models.py",
    "mlflow/store/model_registry/dbmodels/models.py",
]


@lru_cache(maxsize=None)
def _get_workspace_overrides(base_store_posix: str) -> frozenset[str]:
    """Discover methods overridden by the corresponding workspace store.

    For a base store at ``mlflow/store/tracking/sqlalchemy_store.py``, this
    reads ``mlflow/store/tracking/sqlalchemy_workspace_store.py`` and returns
    the set of method names defined in the workspace-aware class.  These
    methods are "workspace extension points" — the workspace store replaces
    them at runtime to add filtering.
    """
    tree = _parse_file(Path(WORKSPACE_STORE_PATHS[base_store_posix]))
    methods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.add(item.name)
    return frozenset(methods)


@lru_cache(maxsize=None)
def _get_workspace_isolated_models() -> frozenset[str]:
    """Discover models that require workspace-aware queries.

    Includes both ``WorkspaceIsolatedModel`` subclasses (which go through
    ``_get_query``) and models listed in ``HELPER_MODEL_COVERAGE`` (child
    models that inherit workspace scope through parent validation).
    """
    models: set[str] = set()
    for path_str in MODEL_FILE_PATHS:
        tree = _parse_file(Path(path_str))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _has_workspace_isolated_base(node):
                models.add(node.name)
    if not models:
        raise RuntimeError(
            "No WorkspaceIsolatedModel subclasses found. "
            "Has the mixin or model file structure been refactored?"
        )
    for covered_models in HELPER_MODEL_COVERAGE.values():
        models.update(covered_models)
    return frozenset(models)


def _has_workspace_isolated_base(node: ast.ClassDef) -> bool:
    """Return True if *node* lists ``WorkspaceIsolatedModel`` in its bases."""
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "WorkspaceIsolatedModel":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "WorkspaceIsolatedModel":
            return True
    return False


@lru_cache(maxsize=None)
def _parse_file(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


class WorkspaceQueryIsolation(Rule):
    def _message(self) -> str:
        return (
            "Direct session.query() on a workspace-isolated model must go through "
            "self._get_query(...) or a workspace-aware helper for workspace filtering."
        )

    @staticmethod
    def check(
        node: ast.Call,
        *,
        path: Path,
        function_name: str | None,
        ancestors: list[ast.AST],
    ) -> bool:
        if path.as_posix() not in BASE_STORE_PATHS:
            return False
        if not function_name:
            return False
        # For a chain like session.query(X).filter(...).all(), only check the
        # outermost call (.all()) to avoid reporting the same chain multiple times.
        if not _is_outermost_call(node, ancestors):
            return False
        # Chains rooted in self._get_query() or self._trace_query() are already
        # workspace-aware — only flag direct session.query() chains.
        if not _is_direct_session_query(node):
            return False

        workspace_models = _get_workspace_isolated_models()
        queried_models = _get_queried_models(node)
        isolated = queried_models & workspace_models
        if not isolated:
            return False

        method_name, method_node = _get_enclosing_class_method(ancestors)
        if not method_name:
            return False
        if method_name in ALLOWLISTED_METHODS:
            return False

        workspace_overrides = _get_workspace_overrides(path.as_posix())
        # The workspace store overrides this method, so its base implementation
        # is replaced at runtime with a workspace-filtered version.
        if method_name in workspace_overrides:
            return False
        # Per-model trust: check which models the method's workspace helpers
        # actually protect, and only trust queries on those specific models.
        safe_methods = workspace_overrides | frozenset(HELPER_MODEL_COVERAGE)
        if method_node is not None:
            protected = _get_protected_models(id(method_node), method_node, safe_methods)
            if isolated <= protected:
                return False

        return True


def _is_outermost_call(node: ast.Call, ancestors: list[ast.AST]) -> bool:
    """Return False if this call is an inner part of a larger method chain."""
    if len(ancestors) < 3:
        return True
    parent = ancestors[-2]
    grandparent = ancestors[-3]
    return not (
        isinstance(parent, ast.Attribute)
        and parent.value is node
        and isinstance(grandparent, ast.Call)
        and grandparent.func is parent
    )


def _iter_call_chain(node: ast.Call) -> Iterator[ast.Call]:
    """Yield calls in a method chain like ``session.query(...).filter(...).all()``."""
    current: ast.Call | None = node
    while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        yield current
        current = current.func.value if isinstance(current.func.value, ast.Call) else None


def _is_direct_session_query(node: ast.Call) -> bool:
    """Return True if this chain originates from a direct ``.query()`` call.

    Returns False for chains starting with ``self._get_query(...)`` or other
    ``self.<method>(...)`` calls since those are workspace-aware helpers.
    """
    for call in _iter_call_chain(node):
        if not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr == "query":
            # .query() on self (e.g. self.query()) is not a session query
            if isinstance(call.func.value, ast.Name) and call.func.value.id == "self":
                return False
            return True
    return False


def _get_queried_models(node: ast.Call) -> frozenset[str]:
    """Extract all ``Sql*`` model names from the ``session.query()`` in the chain."""
    models: set[str] = set()
    for call in _iter_call_chain(node):
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "query":
            continue
        for arg in call.args:
            for inner in ast.walk(arg):
                if isinstance(inner, ast.Name) and inner.id.startswith("Sql"):
                    models.add(inner.id)
    return frozenset(models)


def _get_enclosing_class_method(
    ancestors: list[ast.AST],
) -> tuple[str | None, ast.FunctionDef | ast.AsyncFunctionDef | None]:
    """Find the enclosing method defined directly in a class."""
    for i in range(len(ancestors) - 1, 0, -1):
        node = ancestors[i]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(
            ancestors[i - 1], ast.ClassDef
        ):
            return node.name, node
    return None, None


@lru_cache(maxsize=None)
def _get_protected_models(
    _method_id: int,
    method_node: ast.FunctionDef | ast.AsyncFunctionDef,
    safe_methods: frozenset[str],
) -> frozenset[str]:
    """Return models protected by workspace helpers called via ``self``.

    Returns an empty frozenset if no recognized workspace helpers are called.
    Results are cached per method node to avoid repeated AST walks.
    ``_method_id`` (``id(method_node)``) is safe as a cache key because AST
    nodes live for the entire file traversal, so ids are not recycled.
    """
    protected: set[str] = set()
    for node in ast.walk(method_node):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr in safe_methods
        ):
            continue
        helper_name = node.func.attr
        if helper_name in HELPER_MODEL_COVERAGE:
            protected.update(HELPER_MODEL_COVERAGE[helper_name])
        elif helper_name == "_get_query" and len(node.args) >= 2:
            model_arg = node.args[1]
            if isinstance(model_arg, ast.Name) and model_arg.id.startswith("Sql"):
                protected.add(model_arg.id)
        # Helpers not in HELPER_MODEL_COVERAGE and not _get_query are ignored —
        # they don't contribute any model coverage.  This is intentionally
        # strict: new helpers must be added to the mapping to be recognized.
    return frozenset(protected)

import ast
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
# Base store methods that internally delegate to workspace overrides (e.g.,
# get_experiment calls _get_experiment which calls self._get_query).  Calling
# these via self makes the caller workspace-aware.
WORKSPACE_SAFE_CALLERS = {"get_experiment", "get_experiment_by_name"}


@lru_cache(maxsize=None)
def _get_workspace_overrides(base_store_posix: str) -> frozenset[str]:
    """Discover methods overridden by the corresponding workspace store.

    For a base store at ``mlflow/store/tracking/sqlalchemy_store.py``, this
    reads ``mlflow/store/tracking/sqlalchemy_workspace_store.py`` and returns
    the set of method names defined in the workspace-aware class.  These
    methods are "workspace extension points" — the workspace store replaces
    them at runtime to add filtering.
    """
    tree = _parse_workspace_store(Path(WORKSPACE_STORE_PATHS[base_store_posix]))
    methods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.add(item.name)
    return frozenset(methods)


@lru_cache(maxsize=None)
def _get_workspace_isolated_models(base_store_posix: str) -> frozenset[str]:
    """Discover models that the workspace store handles in ``_get_query``.

    Only queries on these models are checked by the linter.  Models *not*
    listed here (e.g. ``SqlTag``, ``SqlMetric``) are implicitly workspace-
    scoped through their parent model relationships and do not need direct
    checks.
    """
    tree = _parse_workspace_store(Path(WORKSPACE_STORE_PATHS[base_store_posix]))

    # First pass: collect module-level tuple/set variables that hold Sql* names
    # so the second pass can resolve comparisons like ``model in MY_MODELS``.
    # The model registry workspace store uses this pattern because its foreign
    # keys carry the workspace column directly on each model.
    var_models: dict[str, set[str]] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names: set[str] = set()
                    _collect_sql_names(node.value, names)
                    if names:
                        var_models[target.id] = names

    # Second pass: extract model names from ``_get_query`` comparisons.
    models: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "_get_query":
                        _collect_compared_model_names(item, models, var_models)
    return frozenset(models)


@lru_cache(maxsize=None)
def _parse_workspace_store(ws_path: Path) -> ast.Module:
    return ast.parse(ws_path.read_text())


def _collect_sql_names(node: ast.AST, names: set[str]) -> None:
    """Extract all ``Sql*`` Name identifiers reachable from *node*."""
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and inner.id.startswith("Sql"):
            names.add(inner.id)


def _collect_compared_model_names(
    node: ast.AST,
    models: set[str],
    var_models: dict[str, set[str]] | None = None,
) -> None:
    """Extract ``Sql*`` model names from comparisons in ``_get_query``.

    Handles direct references (``model is SqlFoo``, ``model in (SqlFoo, SqlBar)``)
    and variable references (``model in MY_MODELS``) resolved via *var_models*.
    """
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Compare):
            continue
        for comp in inner.comparators:
            _collect_sql_names(comp, models)
            # Resolve variable references to their Sql* contents
            if var_models and isinstance(comp, ast.Name) and comp.id in var_models:
                models.update(var_models[comp.id])


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
        # Only lint the base store files, not workspace stores or other files.
        if path.as_posix() not in BASE_STORE_PATHS:
            return False
        # Module-level code (imports, constants) is not a query concern.
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

        # Only flag queries on models that _get_query handles. Child models
        # (e.g. SqlTag, SqlMetric) are implicitly workspace-scoped through
        # their parent model relationships.
        workspace_models = _get_workspace_isolated_models(path.as_posix())
        if not workspace_models:
            # If discovery found zero models, the workspace store's _get_query
            # was likely refactored.  Fail loudly so the linter gets updated.
            raise RuntimeError(
                f"No workspace-isolated models found for {path}. "
                "Has the workspace store's _get_query been refactored?"
            )
        if not _targets_workspace_model(node, workspace_models):
            return False

        method_name, method_node = _get_enclosing_class_method(ancestors)
        # Not inside a class method (e.g. module-level helper); not our concern.
        if not method_name:
            return False
        # Some methods intentionally skip workspace filtering (e.g. DB setup).
        if method_name in ALLOWLISTED_METHODS:
            return False

        workspace_overrides = _get_workspace_overrides(path.as_posix())
        # The workspace store overrides this method, so its base implementation
        # is replaced at runtime with a workspace-filtered version.
        if method_name in workspace_overrides:
            return False
        # The method already calls a workspace-aware helper (e.g.
        # self._get_query(), self._validate_run_accessible()), so it is
        # workspace-aware and its session.query() calls are trusted.
        # Note: this is a method-level heuristic — a method could validate one
        # entity but miss another.  Catching that would require data-flow
        # analysis, so we accept the tradeoff of trusting the method once it
        # demonstrates workspace awareness.
        safe_methods = workspace_overrides | WORKSPACE_SAFE_CALLERS
        if method_node is not None and _calls_workspace_helper(
            id(method_node), method_node, safe_methods
        ):
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


def _get_call_chain(node: ast.Call) -> list[ast.Call]:
    """Walk a method chain like ``session.query(...).filter(...).all()``."""
    chain: list[ast.Call] = []
    current: ast.Call | None = node
    while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        chain.append(current)
        current = current.func.value if isinstance(current.func.value, ast.Call) else None
    return chain


def _is_direct_session_query(node: ast.Call) -> bool:
    """Return True if this chain originates from a direct ``.query()`` call.

    Returns False for chains starting with ``self._get_query(...)`` or other
    ``self.<method>(...)`` calls since those are workspace-aware helpers.
    """
    for call in _get_call_chain(node):
        if not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr == "query":
            # .query() on self (e.g. self.query()) is not a session query
            if isinstance(call.func.value, ast.Name) and call.func.value.id == "self":
                return False
            return True
    return False


def _targets_workspace_model(node: ast.Call, workspace_models: frozenset[str]) -> bool:
    """Return True if the query chain references a workspace-isolated model."""
    for call in _get_call_chain(node):
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "query":
            continue
        for arg in call.args:
            for inner in ast.walk(arg):
                if isinstance(inner, ast.Name) and inner.id in workspace_models:
                    return True
    return False


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
def _calls_workspace_helper(
    _method_id: int,
    method_node: ast.FunctionDef | ast.AsyncFunctionDef,
    safe_methods: frozenset[str],
) -> bool:
    """Return True if the method calls any workspace-aware helper via ``self``.

    Results are cached per method node (keyed by ``id(method_node)``) so the
    AST walk is not repeated for every ``session.query()`` call in the same
    method.
    """
    for node in ast.walk(method_node):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr in safe_methods
        ):
            return True
    return False

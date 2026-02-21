import ast
import logging
from dataclasses import dataclass, field
from enum import Enum

from alembic.script import ScriptDirectory

_logger = logging.getLogger(__name__)


class MigrationSafety(Enum):
    SAFE = "safe"
    CAUTIOUS = "cautious"
    BREAKING = "breaking"


@dataclass
class MigrationOperation:
    name: str
    safety: MigrationSafety
    detail: str = ""


@dataclass
class MigrationAnalysis:
    revision: str
    safety: MigrationSafety
    operations: list[MigrationOperation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# Manual overrides for migrations that the AST parser can't accurately classify.
# Maps revision -> (safety, reason)
_MANUAL_OVERRIDES: dict[str, tuple[MigrationSafety, str]] = {
    # VARCHAR widening is safe on all supported DB backends (SQLite, MySQL, PostgreSQL)
    "cc1f77228345": (MigrationSafety.SAFE, "VARCHAR widening (250->500) is safe on all backends"),
    "7ac759974ad8": (MigrationSafety.SAFE, "VARCHAR widening (250->5000) is safe on all backends"),
    "2d6e25af4d3e": (MigrationSafety.SAFE, "VARCHAR widening for param values is safe"),
    "f5a4f2784254": (MigrationSafety.SAFE, "VARCHAR widening for run tag values is safe"),
    "bda7b8c39065": (
        MigrationSafety.SAFE,
        "VARCHAR widening for model version tag values is safe",
    ),
    "4465047574b1": (MigrationSafety.SAFE, "VARCHAR widening for dataset schema is safe"),
    # Constraint cleanup that gracefully handles failures
    "0a8213491aaa": (
        MigrationSafety.SAFE,
        "Drops duplicate check constraint with graceful error handling",
    ),
    # FK cascade changes are safe - they only affect future DELETE behavior
    "0584bdc529eb": (
        MigrationSafety.CAUTIOUS,
        "Replaces FK constraint to add ON DELETE CASCADE",
    ),
    "5b0e9adcef9c": (
        MigrationSafety.CAUTIOUS,
        "Adds CASCADE deletion to trace tables FK",
    ),
}


def _get_script_directory() -> ScriptDirectory:
    from mlflow.store.db.utils import _get_alembic_config

    config = _get_alembic_config(db_url="")
    return ScriptDirectory.from_config(config)


def _get_migration_source(revision: str) -> str:
    from alembic.util.exc import CommandError

    script_dir = _get_script_directory()
    try:
        script = script_dir.get_revision(revision)
    except CommandError as e:
        raise ValueError(f"Revision {revision!r} not found in migration directory") from e
    if script is None:
        raise ValueError(f"Revision {revision!r} not found in migration directory")
    source_path = script.path
    with open(source_path) as f:
        return f.read()


def _classify_ast_call(node: ast.Call, in_batch: bool) -> MigrationOperation | None:
    """Classify a single AST function call node as a migration operation."""
    func_name = _get_call_name(node)
    if func_name is None:
        return None

    if func_name == "create_table":
        table_name = _get_first_str_arg(node)
        return MigrationOperation(
            name="create_table",
            safety=MigrationSafety.SAFE,
            detail=f"table={table_name}" if table_name else "",
        )

    if func_name == "create_index":
        idx_name = _get_first_str_arg(node)
        return MigrationOperation(
            name="create_index",
            safety=MigrationSafety.SAFE,
            detail=f"index={idx_name}" if idx_name else "",
        )

    if func_name == "add_column":
        col_name, nullable = _get_add_column_info(node)
        safety = MigrationSafety.SAFE if nullable else MigrationSafety.CAUTIOUS
        return MigrationOperation(
            name="add_column",
            safety=safety,
            detail=f"column={col_name}, nullable={nullable}",
        )

    if func_name == "alter_column":
        return _classify_alter_column(node)

    if func_name == "drop_column":
        col_name = _get_first_str_arg(node)
        return MigrationOperation(
            name="drop_column",
            safety=MigrationSafety.BREAKING,
            detail=f"column={col_name}" if col_name else "",
        )

    if func_name == "drop_table":
        table_name = _get_first_str_arg(node)
        return MigrationOperation(
            name="drop_table",
            safety=MigrationSafety.BREAKING,
            detail=f"table={table_name}" if table_name else "",
        )

    if func_name == "drop_constraint":
        constraint_name = _get_first_str_arg(node)
        return MigrationOperation(
            name="drop_constraint",
            safety=MigrationSafety.CAUTIOUS,
            detail=f"constraint={constraint_name}" if constraint_name else "",
        )

    if func_name == "create_foreign_key":
        fk_name = _get_first_str_arg(node)
        return MigrationOperation(
            name="create_foreign_key",
            safety=MigrationSafety.CAUTIOUS,
            detail=f"fk={fk_name}" if fk_name else "",
        )

    if func_name == "drop_index":
        idx_name = _get_first_str_arg(node)
        return MigrationOperation(
            name="drop_index",
            safety=MigrationSafety.CAUTIOUS,
            detail=f"index={idx_name}" if idx_name else "",
        )

    if func_name == "execute":
        sql_text = _get_first_str_arg(node)
        return MigrationOperation(
            name="execute",
            safety=MigrationSafety.CAUTIOUS,
            detail=(
                f"sql={sql_text[:80]}..."
                if sql_text and len(sql_text) > 80
                else f"sql={sql_text}"
            ),
        )

    if func_name == "rename_table":
        return MigrationOperation(
            name="rename_table",
            safety=MigrationSafety.BREAKING,
            detail="",
        )

    return None


def _classify_alter_column(node: ast.Call) -> MigrationOperation:
    """Classify an alter_column call based on its keyword arguments."""
    col_name = _get_first_str_arg(node)
    kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}

    # Column rename is breaking
    if "new_column_name" in kwargs:
        new_name = _get_const_value(kwargs["new_column_name"])
        return MigrationOperation(
            name="alter_column",
            safety=MigrationSafety.BREAKING,
            detail=f"rename {col_name} -> {new_name}",
        )

    # Making a column nullable is safe; making it NOT NULL is cautious
    if "nullable" in kwargs:
        nullable_val = _get_const_value(kwargs["nullable"])
        if nullable_val is True:
            return MigrationOperation(
                name="alter_column",
                safety=MigrationSafety.SAFE,
                detail=f"column={col_name}, set nullable=True",
            )
        elif nullable_val is False:
            return MigrationOperation(
                name="alter_column",
                safety=MigrationSafety.CAUTIOUS,
                detail=f"column={col_name}, set nullable=False",
            )

    # Type change: widening VARCHAR is safe, narrowing or changing type is cautious
    if "type_" in kwargs:
        return MigrationOperation(
            name="alter_column",
            safety=MigrationSafety.CAUTIOUS,
            detail=f"column={col_name}, type change",
        )

    # Adding/changing server_default is generally safe
    if "server_default" in kwargs:
        return MigrationOperation(
            name="alter_column",
            safety=MigrationSafety.SAFE,
            detail=f"column={col_name}, server_default change",
        )

    return MigrationOperation(
        name="alter_column",
        safety=MigrationSafety.CAUTIOUS,
        detail=f"column={col_name}, unrecognized alteration",
    )


def _get_call_name(node: ast.Call) -> str | None:
    """Extract the function name from a Call node (handles op.X and batch_op.X)."""
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _get_first_str_arg(node: ast.Call) -> str | None:
    """Get the first positional string argument from a call."""
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return node.args[0].value
    return None


def _get_const_value(node: ast.expr):
    """Extract a constant value from an AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _get_add_column_info(node: ast.Call) -> tuple[str | None, bool]:
    """Extract column name and nullable status from an add_column call."""
    col_name = None
    nullable = True  # default in most DBs

    # For op.add_column("table", sa.Column("name", ...))
    # or batch_op.add_column(sa.Column("name", ...))
    for arg in node.args:
        if isinstance(arg, ast.Call):
            inner_name = _get_call_name(arg)
            if inner_name == "Column":
                col_name = _get_first_str_arg(arg)
                for kw in arg.keywords:
                    if kw.arg == "nullable":
                        val = _get_const_value(kw.value)
                        if val is not None:
                            nullable = val

    # Check direct keywords for nullable
    for kw in node.keywords:
        if kw.arg == "nullable":
            val = _get_const_value(kw.value)
            if val is not None:
                nullable = val

    return col_name, nullable


def _has_orm_usage(tree: ast.Module) -> bool:
    """Check if upgrade() uses ORM queries (session.query, op.get_bind + Session)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = _get_call_name(node)
            if func_name in ("query", "get_bind"):
                return True
        if isinstance(node, ast.Attribute):
            if node.attr in ("query", "get_bind"):
                return True
    return False


def _find_upgrade_function(tree: ast.Module) -> ast.FunctionDef | None:
    """Find the upgrade() function in the AST."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "upgrade":
            return node
    return None


def classify_migration(revision: str) -> MigrationAnalysis:
    """Classify a single migration revision as SAFE, CAUTIOUS, or BREAKING.

    Uses AST parsing to detect Alembic operations in the upgrade() function
    and classify them by their impact on running applications.
    """
    # Check manual overrides first
    if revision in _MANUAL_OVERRIDES:
        safety, reason = _MANUAL_OVERRIDES[revision]
        return MigrationAnalysis(
            revision=revision,
            safety=safety,
            operations=[],
            notes=[f"Manual override: {reason}"],
        )

    source = _get_migration_source(revision)
    tree = ast.parse(source)

    upgrade_func = _find_upgrade_function(tree)
    if upgrade_func is None:
        return MigrationAnalysis(
            revision=revision,
            safety=MigrationSafety.CAUTIOUS,
            notes=["No upgrade() function found"],
        )

    # Check for empty upgrade
    body_stmts = [
        s for s in upgrade_func.body if not isinstance(s, (ast.Pass, ast.Expr))
        or (isinstance(s, ast.Expr) and not isinstance(s.value, (ast.Constant, ast.Str)))
    ]
    if not body_stmts:
        return MigrationAnalysis(
            revision=revision,
            safety=MigrationSafety.SAFE,
            notes=["Empty upgrade function"],
        )

    # Check for ORM/data migration patterns
    if _has_orm_usage(upgrade_func):
        return MigrationAnalysis(
            revision=revision,
            safety=MigrationSafety.BREAKING,
            notes=["Contains ORM/data migration operations"],
        )

    operations: list[MigrationOperation] = []
    notes: list[str] = []

    # Walk the upgrade function's AST to find all alembic operation calls
    for node in ast.walk(upgrade_func):
        if not isinstance(node, ast.Call):
            continue

        # Detect batch_alter_table context manager usage
        op = _classify_ast_call(node, in_batch=False)
        if op is not None:
            operations.append(op)

    if not operations:
        notes.append("No recognized Alembic operations found; classified as cautious")
        return MigrationAnalysis(
            revision=revision,
            safety=MigrationSafety.CAUTIOUS,
            operations=operations,
            notes=notes,
        )

    # Overall safety is the worst of any individual operation
    worst = MigrationSafety.SAFE
    for op in operations:
        if op.safety == MigrationSafety.BREAKING:
            worst = MigrationSafety.BREAKING
            break
        if op.safety == MigrationSafety.CAUTIOUS and worst == MigrationSafety.SAFE:
            worst = MigrationSafety.CAUTIOUS

    return MigrationAnalysis(
        revision=revision,
        safety=worst,
        operations=operations,
        notes=notes,
    )


def classify_range(from_rev: str, to_rev: str) -> list[MigrationAnalysis]:
    """Classify all migrations in the range (from_rev, to_rev].

    Returns a list of MigrationAnalysis for each migration that would be applied
    when upgrading from from_rev to to_rev.
    """
    script_dir = _get_script_directory()
    revisions = []
    for script in script_dir.walk_revisions(base=from_rev, head=to_rev):
        if script.revision == from_rev:
            continue
        revisions.append(script.revision)

    # walk_revisions returns newest-first, reverse for chronological order
    revisions.reverse()

    return [classify_migration(rev) for rev in revisions]


def is_range_online_safe(from_rev: str, to_rev: str) -> bool:
    """Check if all migrations in the range (from_rev, to_rev] are online-safe.

    Returns True if all pending migrations are classified as SAFE.
    Returns False if any migration is CAUTIOUS or BREAKING.
    """
    analyses = classify_range(from_rev, to_rev)
    return all(a.safety == MigrationSafety.SAFE for a in analyses)


def get_range_worst_safety(from_rev: str, to_rev: str) -> MigrationSafety:
    """Get the worst safety classification across all migrations in a range."""
    analyses = classify_range(from_rev, to_rev)
    if not analyses:
        return MigrationSafety.SAFE

    worst = MigrationSafety.SAFE
    for a in analyses:
        if a.safety == MigrationSafety.BREAKING:
            return MigrationSafety.BREAKING
        if a.safety == MigrationSafety.CAUTIOUS:
            worst = MigrationSafety.CAUTIOUS
    return worst

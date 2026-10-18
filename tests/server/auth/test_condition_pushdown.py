# Parity between pushed-down and in-memory evaluation of a target condition.
#
# ``AbstractStore.find_satisfying_resources`` lets a store answer "which of these
# ids satisfy every clause" without loading the resources. That makes **two**
# implementations of one semantic: the store's SQL predicate and the auth layer's
# :func:`evaluate_resource`. Nothing in the type system forces them to agree, and a
# disagreement is not symmetric -- a pushdown that admits a resource the matcher
# would reject **grants a mutation the condition forbids**.
#
# The cascade half no longer has this exposure: it evaluates children in Python
# through the same :func:`evaluate_resource`, so there is one evaluator rather than
# two. Only the named-id half is still SQL, which is why that is what is pinned here.
#
# So every comparator is checked against every tag state here, and the state that
# matters most is the third one: a resource with *no* such tag. D20 makes absence
# fail on the target side, so ``!=`` and ``NOT IN`` must EXCLUDE an untagged
# resource. That is the reading a hand-written SQL predicate gets wrong by
# default, because ``NOT (value = 'x')`` over a join quietly drops to "no row
# matched" and lets the untagged resource through.

import tempfile
from types import SimpleNamespace

import pytest
from sqlalchemy import event

from mlflow.entities import RunTag, ViewType
from mlflow.entities.model_registry import ModelVersionTag
from mlflow.server import auth as auth_module
from mlflow.server.auth import resources as auth_resources
from mlflow.server.auth.conditions import (
    CONDITION_TYPE_TARGET,
    MAX_CLAUSE_VALUES,
    NAMESPACE_RESOURCE,
    Clause,
    ConditionSpec,
    evaluate_resource,
    parse_condition,
)
from mlflow.store.model_registry.sqlalchemy_store import (
    SqlAlchemyStore as RegistrySqlAlchemyStore,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

# Both halves of the refusal: a SQL store says "cannot express a target condition on
# <entity>", the abstract default says "cannot answer a target condition". Matching either
# keeps these assertions from passing on an unrelated NotImplementedError.
_CANNOT_ANSWER = r"cannot (express|answer) a target condition"

TAG_KEY = "lifecycle"

# MCP names are reverse-DNS and so contain "/" -- the separator
# ``version_resource_id`` percent-encodes. Using realistic names here is what makes
# the composite-id tests meaningful: a store that parsed a joined id would split
# these in the wrong place.
PROD_SERVER = "com.example/svc-prod"
DEV_SERVER = "com.example/svc-dev"
BARE_SERVER = "com.example/svc-bare"

# (comparator, pushdown value, the equivalent condition filter text)
COMPARATORS = [
    ("=", "prod", f"tags.{TAG_KEY} = 'prod'"),
    ("!=", "prod", f"tags.{TAG_KEY} != 'prod'"),
    ("LIKE", "pro%", f"tags.{TAG_KEY} LIKE 'pro%'"),
    ("ILIKE", "PRO%", f"tags.{TAG_KEY} ILIKE 'PRO%'"),
    ("IN", ("prod", "staging"), f"tags.{TAG_KEY} IN ('prod','staging')"),
    ("NOT IN", ("prod", "staging"), f"tags.{TAG_KEY} NOT IN ('prod','staging')"),
]

_DIALECTS = ("sqlite", "postgresql", "mysql", "mssql")


@pytest.fixture
def store_with_runs(monkeypatch):
    """Three runs covering the tag states a condition can meet.

    ``untagged`` is the one that distinguishes the two readings of a negative
    comparator, so it is not an edge case here -- it is the point.
    """
    d = tempfile.mkdtemp()
    store = SqlAlchemyStore(f"sqlite:///{d}/mlflow.db", d)
    monkeypatch.setattr(auth_resources, "_tracking_store", lambda: store)
    experiment_id = store.create_experiment("parity")
    ids = {}
    for label, tag in (("matching", "prod"), ("other", "dev"), ("untagged", None)):
        run_id = store.create_run(experiment_id, "u", 0, [], label).info.run_id
        ids[label] = run_id
        if tag is not None:
            store.set_tag(run_id, RunTag(TAG_KEY, tag))
    return store, experiment_id, ids


def _store_with(monkeypatch, runs):
    """A fresh store holding one experiment with the given ``(label, tag)`` runs."""
    d = tempfile.mkdtemp()
    store = SqlAlchemyStore(f"sqlite:///{d}/mlflow.db", d)
    monkeypatch.setattr(auth_resources, "_tracking_store", lambda: store)
    experiment_id = store.create_experiment("cascade")
    for label, tag in runs:
        run_id = store.create_run(experiment_id, "u", 0, [], label).info.run_id
        if tag is not None:
            store.set_tag(run_id, RunTag(TAG_KEY, tag))
    return store, experiment_id


def _run_ids(store, experiment_id):
    """Every run id under the experiment, matching the enumerator's ViewType.ALL."""
    return [
        run.info.run_id
        for run in store.search_runs([experiment_id], None, ViewType.ALL, max_results=500)
    ]


def _in_memory(filter_text, ids):
    """Evaluate the condition the way the gate does when it cannot push down."""
    clauses = parse_condition(filter_text, NAMESPACE_RESOURCE)
    matched = set()
    for label, run_id in ids.items():
        auth_resources.clear_cache()
        if evaluate_resource(clauses, auth_resources.attrs_for("run", run_id)):
            matched.add(label)
    return matched


def _matches_in_memory(value, comparator, operand):
    """The in-memory semantic the pushdown must agree with, stated independently.

    Deliberately not a call into the condition evaluator: if both sides of a
    parity test route through the same code, the test proves only that the code
    equals itself. Absence returns ``False`` for every comparator (D20).
    """
    if value is None:
        return False
    if comparator == "=":
        return value == operand
    if comparator == "!=":
        return value != operand
    if comparator == "LIKE":
        return value.startswith(operand.rstrip("%"))
    if comparator == "ILIKE":
        return value.lower().startswith(operand.rstrip("%").lower())
    if comparator == "IN":
        return value in operand
    if comparator == "NOT IN":
        return value not in operand
    raise AssertionError(f"unhandled comparator {comparator}")


@staticmethod
def _pushed(filter_text):
    from mlflow.server.auth import _pushable_clauses

    return _pushable_clauses(parse_condition(filter_text, NAMESPACE_RESOURCE))


def test_tag_clauses_convert():
    assert _pushed(f"tags.{TAG_KEY} != 'prod'") == [("tags", TAG_KEY, "!=", "prod")]


def test_several_tag_clauses_convert_in_order():
    pushed = _pushed(f"tags.{TAG_KEY} != 'prod' AND tags.team = 'ml'")
    assert pushed == [("tags", TAG_KEY, "!=", "prod"), ("tags", "team", "=", "ml")]


def test_an_alias_clause_converts():
    assert _pushed("aliases.production = 'yes'") == [("aliases", "production", "=", "yes")]


def test_a_row_mixing_tags_and_aliases_converts_whole():
    # Both halves in one call, so the conjunction is never split.
    pushed = _pushed(f"tags.{TAG_KEY} != 'prod' AND aliases.production = 'yes'")
    assert pushed == [
        ("tags", TAG_KEY, "!=", "prod"),
        ("aliases", "production", "=", "yes"),
    ]


def test_an_unrecognised_identifier_declines_the_whole_row():
    """The guard that outlives any particular namespace list.

    Constructed directly rather than parsed, because the parser rejects an
    unknown prefix at authoring time -- so this pins the helper's own refusal
    rather than relying on the parser never changing.
    """
    from mlflow.server.auth import _pushable_clauses

    rows = [
        Clause(identifier="tags", key=TAG_KEY, comparator="!=", value="prod"),
        Clause(identifier="something_new", key="x", comparator="=", value="y"),
    ]
    assert _pushable_clauses(rows) is None


def test_a_clause_with_no_key_declines():
    # A flat request-style clause has ``key is None`` and names no column.
    from mlflow.server.auth import _pushable_clauses

    rows = [Clause(identifier="tags", key=None, comparator="=", value="v")]
    assert _pushable_clauses(rows) is None


@pytest.fixture
def mcp_store(monkeypatch):
    d = tempfile.mkdtemp()
    store = SqlAlchemyStore(f"sqlite:///{d}/mlflow.db", d)
    monkeypatch.setattr(auth_resources, "_tracking_store", lambda: store)
    for name, tag in (
        (PROD_SERVER, "prod"),
        (DEV_SERVER, "dev"),
        (BARE_SERVER, None),
    ):
        store.create_mcp_server(name)
        store.create_mcp_server_version({"name": name, "version": "1.0.0"})
        if tag is not None:
            store.set_mcp_server_tag(name, TAG_KEY, tag)
            store.set_mcp_server_version_tag(name, "1.0.0", TAG_KEY, tag)
    store.set_mcp_server_alias(PROD_SERVER, "champion", "1.0.0")
    return store


@pytest.fixture
def pushdown_registry(monkeypatch):
    from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
    from mlflow.store.model_registry.sqlalchemy_store import (
        SqlAlchemyStore as RegistrySqlAlchemyStore,
    )

    d = tempfile.mkdtemp()
    store = RegistrySqlAlchemyStore(f"sqlite:///{d}/pushdown_registry.db")
    for name, tag in (("m-prod", "prod"), ("m-dev", "dev"), ("m-bare", None)):
        store.create_registered_model(name)
        store.create_model_version(name, source="s")
        if tag is not None:
            store.set_registered_model_tag(name, RegisteredModelTag(TAG_KEY, tag))
            store.set_model_version_tag(name, "1", ModelVersionTag(TAG_KEY, tag))
    store.set_registered_model_alias("m-prod", "champion", "1")
    return store


def test_an_integer_column_is_cast_to_text():
    from mlflow.store import condition_pushdown
    from mlflow.store.model_registry.dbmodels.models import SqlRegisteredModelAlias

    compiled = str(condition_pushdown.comparable(SqlRegisteredModelAlias.version))
    assert "CAST" in compiled.upper(), (
        "an INTEGER version must be compared as text, or a strictly-typed "
        "backend matches nothing and denies every mutation of the type"
    )


def test_a_text_column_is_left_alone():
    # No gratuitous cast: it would defeat an index for no benefit.
    from mlflow.store import condition_pushdown
    from mlflow.store.model_registry.dbmodels.models import SqlRegisteredModelTag

    assert condition_pushdown.comparable(SqlRegisteredModelTag.value) is (
        SqlRegisteredModelTag.value
    )


def test_a_composite_id_predicate_casts_its_integer_part():
    # The id side needs it too, not just the compared value.
    from mlflow.store import condition_pushdown
    from mlflow.store.model_registry.dbmodels.models import SqlModelVersionTag

    predicate = condition_pushdown.id_predicate(
        (SqlModelVersionTag.name, SqlModelVersionTag.version), [("m", "1")]
    )
    assert "CAST" in str(predicate).upper()


def test_the_tracking_tag_tables_need_no_cast():
    # Every tracking tag value is already text, so nothing is wrapped there.
    from mlflow.store import condition_pushdown
    from mlflow.store.tracking.dbmodels.models import SqlTag

    assert condition_pushdown.comparable(SqlTag.value) is SqlTag.value


def _row(resource_pattern):
    # The stray `@staticmethod` this carried was meaningless on a module-level function
    # and only callable at all on Python 3.10+, where staticmethod objects gained
    # __call__. Removed with the conversion rather than left as a trap.
    return ConditionSpec(
        resource_type="run",
        condition_type=CONDITION_TYPE_TARGET,
        filter=f"tags.{TAG_KEY} != 'prod'",
        resource_pattern=resource_pattern,
    )


def test_an_id_pattern_governs_only_the_resource_it_names():
    """Charging it against a sibling would deny a mutation on a resource the admin
    never pointed the condition at -- over-restriction rather than a hole, but still
    not what was written.
    """
    row = _row("abc")
    assert auth_module._row_governs(row, "abc") is True
    assert auth_module._row_governs(row, "xyz") is False
    # Nothing to judge against: a create, or a cascade before resolution.
    assert auth_module._row_governs(row, None) is False


def test_a_wildcard_pattern_governs_every_resource_including_unnamed():
    # The pre-scope behaviour, and what makes restricting a create possible at all.
    row = _row("*")
    for asked in ("abc", "xyz", None):
        assert auth_module._row_governs(row, asked) is True


@pytest.mark.parametrize(
    "store",
    [SqlAlchemyStore, RegistrySqlAlchemyStore],
    ids=["tracking", "registry"],
)
def test_every_namespace_mapping_names_real_columns(store):
    problems = []
    for entity, namespaces in store._PUSHDOWN_NAMESPACES.items():
        for namespace, (model_name, id_names, key_name, value_name) in namespaces.items():
            model = store._PUSHDOWN_MODELS[model_name]
            for column in (*id_names, key_name, value_name):
                if not hasattr(model, column):
                    actual = [c.name for c in model.__table__.columns]
                    problems.append(
                        f"{entity}.{namespace}: {model_name} has no {column!r} "
                        f"(actual columns: {actual})"
                    )
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "store",
    [SqlAlchemyStore, RegistrySqlAlchemyStore],
    ids=["tracking", "registry"],
)
def test_every_cascade_entity_names_real_columns(store):
    """The cascade path does not consult ``_PUSHDOWN_NAMESPACES``.

    It has its own table, so fixing the namespace mapping alone left this path still
    broken for the same entity. Both are checked, independently, on both stores.
    """
    problems = []
    for entity, mapping in store._CASCADE_PUSHDOWN_ENTITIES.items():
        (
            child_model,
            child_id_names,
            parent_name,
            tag_model,
            tag_id_names,
            tag_key_name,
            tag_value_name,
            tag_parent_name,
        ) = mapping
        columns = [
            *((child_model, name) for name in child_id_names),
            (child_model, parent_name),
            *((tag_model, name) for name in tag_id_names),
            (tag_model, tag_key_name),
            (tag_model, tag_value_name),
        ]
        if tag_parent_name is not None:
            columns.append((tag_model, tag_parent_name))
        for model, column in columns:
            if not hasattr(model, column):
                actual = [c.name for c in model.__table__.columns]
                problems.append(
                    f"{entity}: {model.__name__} has no {column!r} (actual columns: {actual})"
                )
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "store",
    [SqlAlchemyStore, RegistrySqlAlchemyStore],
    ids=["tracking", "registry"],
)
def test_a_composite_keyed_child_scopes_its_tag_subquery(store):
    """A child whose id is not globally unique MUST declare the tag parent column.

    This is the invariant the whole version cascade turns on. Without it the
    satisfying set holds bare discriminators -- version numbers -- and a sibling
    parent's satisfying version acquits this parent's failing one, which is the
    fail-open direction. Derived from the mapping rather than trusted, so a future
    composite-keyed entity cannot be added without it.
    """
    problems = []
    for entity, mapping in store._CASCADE_PUSHDOWN_ENTITIES.items():
        (_, child_id_names, parent_name, _, tag_id_names, _, _, tag_parent_name) = mapping
        composite = len(child_id_names) > 1
        if composite and tag_parent_name is None:
            problems.append(
                f"{entity}: child id {child_id_names} is composite, so its "
                f"discriminator repeats across parents and the tag subquery must be "
                f"scoped -- but tag_parent_name is None"
            )
        if not composite and parent_name in child_id_names:
            problems.append(
                f"{entity}: single-column child id {child_id_names} must not be the "
                f"parent column {parent_name!r}, or the discriminator is empty"
            )
    assert not problems, "\n".join(problems)


@pytest.fixture
def cascade_registry():
    from mlflow.entities.model_registry import ModelVersionTag

    d = tempfile.mkdtemp()
    store = RegistrySqlAlchemyStore(f"sqlite:///{d}/cascade_registry.db")
    # Version NUMBERS collide across models on purpose: every model here has a
    # version 1, and ``m-prod``'s disagrees with the others'. That collision is
    # what makes the subquery's parent filter observable.
    for name, tags in (
        ("m-prod", ["prod"]),
        ("m-dev", ["dev"]),
        ("m-mixed", ["dev", "prod"]),
        ("m-bare", [None]),
    ):
        store.create_registered_model(name)
        for version, tag in enumerate(tags, start=1):
            store.create_model_version(name, source="s")
            if tag is not None:
                store.set_model_version_tag(name, str(version), ModelVersionTag(TAG_KEY, tag))
    return store


def test_the_sentinel_is_gone():
    # Nothing may reintroduce a third verdict: the method answers or raises.
    import mlflow.store.condition_pushdown as cp

    assert not hasattr(cp, "DECLINED")
    assert not hasattr(cp, "Declined")


def _dialect(name):
    from sqlalchemy.dialects import mssql, mysql, postgresql, sqlite

    return {"sqlite": sqlite, "postgresql": postgresql, "mysql": mysql, "mssql": mssql}[
        name
    ].dialect()


def _compiled(clause, dialect_name):
    return str(
        clause.compile(dialect=_dialect(dialect_name), compile_kwargs={"literal_binds": True})
    )


@pytest.mark.parametrize("dialect", _DIALECTS)
@pytest.mark.parametrize("comparator", ["=", "!=", "LIKE", "ILIKE", "IN", "NOT IN"])
def test_a_cast_column_compares_on_every_dialect(dialect, comparator):
    """A clause whose column needs casting must be expressible on every backend.

    An alias clause compares against ``version``, an ``INTEGER``, so ``comparable``
    hands the comparison a ``Cast``. A ``Cast`` reports a ``String`` type and so passes
    the dialect helper's type guard, but has no ``class_`` for its MySQL branch to read
    a table name from -- which raised ``AttributeError`` and failed the mutation with an
    internal error instead of evaluating the clause.
    """
    from mlflow.store.condition_pushdown import comparable, compare_value
    from mlflow.store.model_registry.dbmodels.models import SqlRegisteredModelAlias

    column = comparable(SqlRegisteredModelAlias.version)
    value = ("1", "2") if comparator in ("IN", "NOT IN") else "1"
    # Compiles, rather than raising, is the whole assertion.
    assert _compiled(compare_value(column, comparator, value, dialect), dialect)


@pytest.mark.parametrize("dialect", _DIALECTS)
def test_the_key_comparison_is_case_sensitive(dialect):
    """The key a clause names must match exactly, on every backend.

    MySQL and MSSQL collate case-insensitively by default, so a plain ``==`` let a row
    keyed ``Lifecycle`` answer a clause about ``lifecycle``. On the target side that is
    fail-open: the key the condition names is absent, which must fail, but the
    near-miss row joins the satisfying set and acquits the mutation.
    """
    from mlflow.store.condition_pushdown import key_comparison
    from mlflow.store.model_registry.dbmodels.models import SqlRegisteredModelTag

    sql_text = _compiled(key_comparison(dialect)(SqlRegisteredModelTag.key, "lifecycle"), dialect)
    if dialect == "mysql":
        assert "BINARY" in sql_text
    elif dialect == "mssql":
        assert "COLLATE" in sql_text
        assert "_CS_" in sql_text
    else:
        # Already case-sensitive, so no decoration is expected or wanted.
        assert "BINARY" not in sql_text
        assert "COLLATE" not in sql_text


@pytest.mark.parametrize("dialect", _DIALECTS)
def test_a_composite_id_predicate_avoids_row_values_on_mssql(dialect):
    """SQL Server has no row-value constructor, so ``(a, b) IN ((..), (..))`` is a syntax
    error there. Every named target condition on a model, prompt or MCP server version
    takes this branch, so those mutations failed outright rather than being evaluated.
    """
    from mlflow.store.condition_pushdown import id_predicate
    from mlflow.store.model_registry.dbmodels.models import SqlModelVersion

    columns = (SqlModelVersion.name, SqlModelVersion.version)
    sql_text = _compiled(id_predicate(columns, [("m", "1"), ("m", "2")], dialect), dialect)
    if dialect == "mssql":
        # Distributed into an OR of per-key ANDs instead.
        assert ") IN (" not in sql_text
        assert " OR " in sql_text
        assert " AND " in sql_text
    else:
        assert " IN (" in sql_text


def test_a_composite_id_predicate_on_mssql_groups_its_or():
    # ``AND`` binds tighter than ``OR``, so the distributed form is only correct while it
    # stays parenthesised once ``filter()`` conjoins it with the key and value clauses.
    # Unparenthesised, ``key = 'x' AND a AND b OR c AND d`` would match rows the
    # condition never selected.
    from sqlalchemy.orm import Session

    from mlflow.store.condition_pushdown import comparable, compare_value, id_predicate
    from mlflow.store.model_registry.dbmodels.models import SqlModelVersionTag

    columns = (SqlModelVersionTag.name, SqlModelVersionTag.version)
    query = (
        Session()
        .query(SqlModelVersionTag)
        .with_entities(*columns)
        .filter(
            id_predicate(columns, [("m", "1"), ("m", "2")], "mssql"),
            SqlModelVersionTag.key == "lifecycle",
            compare_value(comparable(SqlModelVersionTag.value), "=", "dev", "mssql"),
        )
    )
    where = str(
        query.statement.compile(dialect=_dialect("mssql"), compile_kwargs={"literal_binds": True})
    ).split("WHERE", 1)[1]
    assert where.strip().startswith("(")


@pytest.mark.parametrize("dialect", _DIALECTS)
def test_a_single_column_id_predicate_is_unchanged(dialect):
    # Only the composite branch needed a dialect, so the common case must not have moved.
    from mlflow.store.condition_pushdown import id_predicate
    from mlflow.store.model_registry.dbmodels.models import SqlModelVersion

    sql_text = _compiled(id_predicate((SqlModelVersion.name,), ["a", "b"], dialect), dialect)
    assert " IN (" in sql_text
    assert " OR " not in sql_text


def _cascade_pushdown(monkeypatch, rows):
    """Ask the cascade selector with these rows, recording whether the store was asked."""
    from mlflow.server.auth.conditions import ConditionScope, context_for

    asked = []

    def _iter_children(*args, **kwargs):
        asked.append(kwargs)
        # No children: the cascade is vacuously allowed, which isolates the scope
        # refusal from any verdict about a child.
        return iter(())

    fake = SimpleNamespace(iter_child_resource_tags=_iter_children)
    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: fake)
    context = context_for("run", None, ConditionScope.MUTATE, None, parent_resource_id="e-1")

    def ask():
        return auth_module._cascade_target_pushdown(context, rows, parent_id="e-1")

    # `asked` is returned separately from the call so a test can still inspect it when
    # the call raises -- an unpushable row must not reach the store at all.
    return ask, asked


def test_an_id_scoped_row_refuses_the_cascade_pushdown(monkeypatch):
    """A row naming a single resource cannot be charged against a parent's other children,
    and there is no fallback that judges each child against only the rows governing it, so
    this must RAISE rather than quietly skip the row.

    Unreachable in practice: every cascade-reachable type (run, trace, logged model, and
    the three version types) is wildcard-only, so ``normalize_condition_scope`` refuses an
    id pattern for all six and such a row cannot be stored. Kept because a future tier
    with id-grain patterns would otherwise fail silently.

    Under OR that silence would fail CLOSED, not open: conditions of a type are
    alternatives, so dropping one removes a way to pass and narrows what is permitted.
    That is the opposite direction from the AND semantics this guard was first written
    for, and it is still wrong -- it would deny operations the configuration allows.
    """
    ask, asked = _cascade_pushdown(monkeypatch, [_row("abc")])
    with pytest.raises(NotImplementedError, match=r"scoped"):
        ask()
    assert asked == [], "a scoped row must not be pushed at all, not pushed and ignored"


def test_an_unscoped_cascade_row_is_still_pushed(monkeypatch):
    """The positive control: the refusal must be keyed on the pattern, not on having rows.

    Without this, raising unconditionally would pass the test above while breaking
    every cascade.
    """
    ask, asked = _cascade_pushdown(monkeypatch, [_row("*")])
    assert ask() is None
    assert len(asked) == 1, "an unscoped row must reach the store"


def test_one_scoped_row_among_unscoped_ones_still_refuses(monkeypatch):
    # Not because rows conjoin -- conditions of a type are alternatives now. The whole
    # context is refused because the set cannot be evaluated faithfully: the pushdown
    # judges every child against one clause set, so a row that governs only one child
    # has no faithful place in it, and guessing either way misreports the verdict.
    ask, _ = _cascade_pushdown(monkeypatch, [_row("*"), _row("abc")])
    with pytest.raises(NotImplementedError, match=r"scoped"):
        ask()


# ---- Bind-parameter ceiling on the named-id path ----
#
# The named-id half is the only part of target evaluation still in SQL, so it is the
# only part with a bind-parameter budget. One statement carries a clause's own
# parameters plus one per id -- more for a composite id -- and `satisfying_named` sizes
# each id chunk as `(chunk - clause_params) // len(id_columns)` so the total lands on
# the chunk constant regardless of how wide the clause is.
#
# That arithmetic has a `max(1, ...)` floor, so the bound holds only while the authoring
# cap on values sits BELOW the chunk. Raising `MAX_CLAUSE_VALUES` past it would silently
# push a statement over MSSQL's 2,100 -- and an `IN` list compiles to POSTCOMPILE, so
# nothing about that is visible in a compile-time test. These tests therefore execute
# and count what the driver was actually handed.


_MSSQL_PARAMETER_CAP = 2100


def _max_statement_parameters(store, call):
    """Run `call` and return the largest bind-parameter count any statement carried."""
    counts = []

    @event.listens_for(store.engine, "before_cursor_execute")
    def record(conn, cursor, statement, parameters, context, executemany):
        if parameters:
            counts.append(len(parameters))

    try:
        result = call()
    finally:
        event.remove(store.engine, "before_cursor_execute", record)
    assert counts, "the call issued no parameterised statement, so nothing was measured"
    return max(counts), result


def _widest_clause_filter():
    """A filter at the widest single clause authoring permits."""
    values = ",".join(f"'v{i}'" for i in range(MAX_CLAUSE_VALUES))
    return f"tags.{TAG_KEY} IN ({values})"


def test_widest_clause_stays_under_the_mssql_parameter_cap(monkeypatch):
    """The worst case an admin can author must fit in one statement on every backend.

    Asking about more ids than a chunk holds is the point: the id list is what grows
    without bound, so if chunking did not subtract the clause's own parameters the
    statement would exceed the cap exactly when a condition is widest.
    """
    store, experiment_id = _store_with(monkeypatch, [("matching", "v1")])
    ids = [
        run.info.run_id
        for run in store.search_runs([experiment_id], None, ViewType.ALL, max_results=10)
    ]
    ids += [f"absent{i:04d}" for i in range(1000)]
    clauses = parse_condition(_widest_clause_filter(), NAMESPACE_RESOURCE)

    peak, satisfying = _max_statement_parameters(
        store, lambda: store.find_satisfying_resources("run", clauses, ids)
    )

    assert peak <= store._TAG_PUSHDOWN_ID_CHUNK, (
        f"a statement carried {peak} parameters, over the {store._TAG_PUSHDOWN_ID_CHUNK} "
        "chunk: id chunking is no longer subtracting the clause's own parameters"
    )
    assert peak < _MSSQL_PARAMETER_CAP
    # The answer still has to be right -- a bound met by querying nothing is no bound.
    assert len(satisfying) == 1


def test_a_composite_id_halves_the_chunk_rather_than_doubling_the_parameters(tmp_path):
    """A model version binds two parameters per id, which must shrink the chunk.

    The failure this guards against is arithmetic that counts ids rather than
    parameters: with two id columns that doubles a statement's width, and a cap breach
    would appear only on the registry store and only for versions.
    """
    store = RegistrySqlAlchemyStore(f"sqlite:///{tmp_path}/version_budget.db")
    store.create_registered_model("m")
    store.create_model_version("m", "source", "run")
    store.set_model_version_tag("m", "1", ModelVersionTag(TAG_KEY, "v1"))
    ids = [("m", "1")] + [("m", str(i)) for i in range(2, 600)]
    clauses = parse_condition(_widest_clause_filter(), NAMESPACE_RESOURCE)

    peak, satisfying = _max_statement_parameters(
        store,
        lambda: store.find_satisfying_resources("registered_model_version", clauses, ids),
    )

    assert peak <= store._PUSHDOWN_ID_CHUNK, (
        f"a statement carried {peak} parameters, over the {store._PUSHDOWN_ID_CHUNK} "
        "chunk: a composite id is being counted as one parameter"
    )
    assert peak < _MSSQL_PARAMETER_CAP
    assert satisfying == {("m", "1")}


def test_the_value_cap_must_stay_below_the_id_chunk():
    """The relationship the two tests above depend on, asserted directly.

    `satisfying_named` floors its chunk at one id, so a clause wider than the chunk
    produces a statement wider than the chunk -- measured at 2,502 parameters for a
    2,500-value clause, past MSSQL's cap. Nothing in the authoring path can reach that
    today because values are capped at 500, which is the only reason the budget holds.
    Raising either constant without the other is the regression, and it would surface as
    a driver error on MSSQL rather than as a failure here, so it is pinned here instead.
    """
    for chunk in (
        SqlAlchemyStore._TAG_PUSHDOWN_ID_CHUNK,
        RegistrySqlAlchemyStore._PUSHDOWN_ID_CHUNK,
    ):
        assert MAX_CLAUSE_VALUES < chunk, (
            f"MAX_CLAUSE_VALUES ({MAX_CLAUSE_VALUES}) must stay below the id chunk "
            f"({chunk}); above it, one clause alone exceeds a statement's budget"
        )
        assert chunk < _MSSQL_PARAMETER_CAP

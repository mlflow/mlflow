"""Shared SQL for pushing a conjunctive tag/alias predicate into the database.

Both the tracking store and the model-registry store answer the same question --
"of these ids, which satisfy every clause?" -- over tables with the same shape: a
tag row is ``(id..., key, value)`` and an alias row is ``(name, alias, version)``.
The query is built here once so the two stores share a single implementation.

That matters more than the saved lines. The predicate already has an in-memory
twin in the authorization layer, which evaluates the same clauses against loaded
resources whenever a store declines to push down. Two implementations of one
semantic is the most that can be allowed, because any drift between them is a
difference in *who may write what*, and the drift is in the fail-open direction
whenever SQL is the more permissive of the two. A third copy, per store, would
make that drift twice as likely and no easier to see.

The two things this module gets right on behalf of both stores:

Absence satisfies nothing. A clause asks which ids *have* a row under that key
whose value compares true, so an id with no such row is simply not in the result
-- for ``!=`` and ``NOT IN`` exactly as for ``=``. This is the shape of the query
rather than a special case, which is why it cannot be forgotten.

Comparison is textual. Every value in a condition is a string, and the
authorization layer's in-memory projection coerces what it loads to ``str`` for
that reason. A registry version is stored as an ``INTEGER``, so comparing it
without a cast would compare ``1`` against ``'1'`` -- matching nothing, which on
the resource side denies every mutation of the type. :func:`comparable` casts
such a column to text so SQL and memory agree.
"""

import sqlalchemy

from mlflow.store.db.db_types import MSSQL
from mlflow.utils.search_utils import SearchUtils


def cannot_express(store, entity, reason: str) -> NotImplementedError:
    """The error a store raises when it cannot answer a predicate.

    Returned rather than raised so the call site reads ``raise cannot_express(...)`` and
    the traceback points at the guard instead of at this function.

    There used to be a ``DECLINED`` sentinel here, and a caller that fell back to loading
    every resource and evaluating the clauses in Python. Every conditionable type now has
    an id-selector mapping and every cascade tier has a cascade mapping, so no SQL store
    can decline for anything authorable -- the fallback only ever served a non-SQL
    backend, at the price of a second evaluator of the same semantic.

    Every reachable case is therefore a wiring bug: a new conditionable type whose mapping
    was forgotten, or a clause the authoring layer should have rejected. Both must be loud.
    Silence here is the fail-open direction, because with no fallback left, "cannot answer"
    read as "nothing failed" would permit the mutation.
    """
    return NotImplementedError(
        f"{type(store).__name__} cannot express a target condition on {entity!r}: {reason}. "
        "This is a wiring bug, not a configuration problem -- every conditionable type is "
        "expected to have a mapping, and the authoring layer is expected to have rejected "
        "any clause that has no table to answer from."
    )


def as_pushdown_key(value):
    """Normalise an id to the key the predicate compares on.

    A single-column entity is keyed by its id string; a composite-keyed one -- a
    model or prompt version, whose name and version live in separate columns --
    by a tuple of its parts.

    The same function normalises both the caller's ids and the rows coming back,
    so the two cannot drift. The ``str`` coercion is what lets an ``INTEGER``
    version column round-trip as the string the caller supplied.
    """
    if isinstance(value, (tuple, list)):
        parts = tuple(str(part) for part in value)
        return parts[0] if len(parts) == 1 else parts
    return value


def comparable(column):
    """Return the column as something that compares like a condition value.

    A condition's values are always strings, so a non-text column must be cast
    rather than compared directly -- see the module docstring.
    """
    if isinstance(column.type, sqlalchemy.Integer):
        return sqlalchemy.cast(column, sqlalchemy.String)
    return column


def key_comparison(dialect):
    """Case-sensitive equality for the key a clause names.

    MySQL and MSSQL collate case-insensitively by default, so a plain ``==`` lets a
    row keyed ``Lifecycle`` answer a clause about ``lifecycle``. On the target side
    that is the fail-open direction: the key the condition names is absent, which
    must fail, but the near-miss row lands in the satisfying set and acquits the
    mutation. The same slip on the request side would compare a submitted key
    against the wrong clause.

    This is how the filter queries beside us compare a key already --
    ``_get_search_registered_model_filter_query``,
    ``_get_search_model_versions_filter_clauses`` and
    ``_get_sqlalchemy_filter_clauses`` all route theirs through this function -- so
    the condition path is held to the same standard rather than a new one.
    """
    return SearchUtils.get_sql_comparison_func("=", dialect)


def compare_value(column, comparator, value, dialect):
    """Compare a clause's value against ``column``, tolerating a cast column.

    A text column goes through ``get_sql_comparison_func``, which restores
    case-sensitive comparison on the two case-insensitive backends.

    :func:`comparable` may have handed us a ``Cast`` instead: a condition value is
    always a string, so a non-text column is rendered as text to compare against one
    -- an alias clause compares against ``version``, an ``INTEGER``. A ``Cast``
    reports a ``String`` type and so passes that function's type guard, but it has no
    ``class_``, and the MySQL branch builds its predicate textually from
    ``column.class_.__tablename__``. The result was ``AttributeError`` -- the
    mutation failed with an internal error instead of the clause being evaluated, and
    only on MySQL, and only for the alias clauses nothing else exercises.

    A cast integer renders as digits, which have no case, so the case-sensitive and
    plain comparisons agree on it. Taking the dialect-agnostic path for a cast column
    is therefore not a relaxation -- there is no case for it to be insensitive to.
    """
    if isinstance(column, sqlalchemy.Cast):
        if comparator == "LIKE":
            return column.like(value)
        if comparator == "ILIKE":
            return column.ilike(value)
        if comparator == "IN":
            return column.in_(value)
        if comparator == "NOT IN":
            return ~column.in_(value)
        return SearchUtils.get_comparison_func(comparator)(column, value)
    return SearchUtils.get_sql_comparison_func(comparator, dialect)(column, value)


def id_predicate(id_columns, keys, dialect=None):
    """Match a chunk of ids: by column for a single key, by row value for a composite.

    ``dialect`` selects how a composite key is expressed. A row-value ``IN`` --
    ``(name, version) IN (('m', '1'), ...)`` -- is the compact form and the one
    MySQL, PostgreSQL and SQLite accept, but SQL Server has no row-value constructor
    and rejects it as a syntax error. Every named target condition on a model, prompt
    or MCP server version takes this branch, so on MSSQL those mutations failed
    outright rather than being evaluated.

    The MSSQL form is the same predicate distributed: an ``OR`` of per-key ``AND``\\ s.
    It binds the same number of parameters -- one per column per key -- so the
    chunking arithmetic that keeps a statement under the backend's parameter cap is
    unaffected.

    Omitting ``dialect`` keeps the row-value form, which is what every backend except
    SQL Server wants.
    """
    if len(id_columns) == 1:
        return id_columns[0].in_(keys)
    casted = tuple(comparable(column) for column in id_columns)
    if dialect == MSSQL:
        return sqlalchemy.or_(*[
            sqlalchemy.and_(*[column == part for column, part in zip(casted, tuple(key))])
            for key in keys
        ])
    return sqlalchemy.sql.tuple_(*casted).in_([tuple(key) for key in keys])


def resolve_clauses(namespaces, models, clauses):
    """Bind clauses to columns, or return ``None`` if any cannot be expressed.

    A namespace this entity does not expose declines the **whole** call, never
    just its own clause: the clauses are conjunctive, so answering from a subset
    of them is the fail-open direction -- the dropped clause is the one that
    would have denied.
    """
    resolved = []
    for namespace, key, comparator, value in clauses:
        mapping = namespaces.get(namespace)
        if mapping is None:
            return None
        model_name, id_names, key_name, value_name = mapping
        model = models[model_name]
        resolved.append((
            model,
            tuple(getattr(model, name) for name in id_names),
            getattr(model, key_name),
            comparable(getattr(model, value_name)),
            key,
            comparator,
            value,
        ))
    return resolved


def find_failing_child(store, mapping, parent_id, clauses, extra_filters=()):
    """Find one child of ``parent_id`` that fails the clauses, without enumerating them.

    Answers the question a cascading mutation actually asks -- "may I touch all of
    them?" -- in one ``LIMIT 1`` query, so cost depends on the number of clauses
    rather than on the number of children.

    A child fails if it fails *any* clause, since clauses are conjunctive, so the
    predicate is a disjunction of ``NOT IN (satisfies)`` subqueries. There is no
    join here and there must not be one: with absence failing on the target side,
    the complement of ``!= 'x'`` is not ``= 'x'`` -- an untagged child satisfies
    neither and must still fail. Asking "which children satisfy" and negating that
    set membership is the only formulation that keeps absence failing.

    Identity is why this is shared rather than written twice. The subquery yields
    the child's **discriminator** -- its id columns minus the parent, which the
    outer query has already fixed -- so a single column suffices even for a
    composite-keyed child. For a run that discriminator is ``run_uuid``, globally
    unique, and the parent is irrelevant to the subquery. For a version it is
    ``version``, unique only *within* one ``name``, so the subquery must carry the
    parent filter too. That filter is load-bearing, not redundant: without it the
    satisfying set holds bare version numbers and a sibling model's satisfying
    version 1 would acquit this model's failing version 1.

    The query selects the child's full id, so naming the offending child in a
    denial costs nothing -- it is already the row being tested for existence.

    ``extra_filters`` narrows the population to the children a mutation will actually
    reach. A cascade usually reaches all of them, but a predicate-mode mutation reaches a
    slice -- ``DeleteTraces`` deletes traces at or before a timestamp -- and judging it
    against the whole parent refuses deletes over windows containing nothing objectionable.
    The filters belong on the OUTER query only: they describe which children are at stake,
    not which satisfy the condition, and adding them to the satisfying subquery would
    shrink that set and so fail children the mutation never touches.
    """
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
    if not clauses:
        return None

    child_ids = tuple(getattr(child_model, name) for name in child_id_names)
    parent_column = getattr(child_model, parent_name)
    child_discriminator = tuple(
        getattr(child_model, name) for name in child_id_names if name != parent_name
    )
    tag_discriminator = tuple(
        getattr(tag_model, name) for name in tag_id_names if name != tag_parent_name
    )
    if len(child_discriminator) != len(tag_discriminator):
        raise ValueError(
            f"cascade mapping for {child_model.__name__} is inconsistent: the child "
            f"discriminator {child_id_names}-{parent_name!r} and the tag discriminator "
            f"{tag_id_names}-{tag_parent_name!r} must have the same width"
        )

    # No cascade child owns aliases (D18) -- a version's alias list names aliases stored
    # on its parent -- so an alias clause here has no table to answer from. Raise rather
    # than ignore it: an ignored clause is a conjunction judged on a subset of itself,
    # which is the fail-open direction. Unreachable today, since authoring rejects an
    # `aliases.*` clause on a type that does not own aliases.
    #
    # Checked BEFORE the session opens. It is a static property of the clauses, needs no
    # database, and a raise inside ``ManagedSessionMaker`` is wrapped into an
    # ``MlflowException`` -- which would disguise a wiring bug as a store error.
    for namespace, _key, _comparator, _value in clauses:
        if namespace != "tags":
            raise cannot_express(store, "a cascade child", f"no child table owns {namespace!r}")

    dialect = store._get_dialect()
    with store.ManagedSessionMaker() as session:
        fails_a_clause = []
        equal_key = key_comparison(dialect)
        for namespace, key, comparator, value in clauses:
            filters = [
                equal_key(getattr(tag_model, tag_key_name), key),
                compare_value(
                    comparable(getattr(tag_model, tag_value_name)), comparator, value, dialect
                ),
            ]
            if tag_parent_name is not None:
                filters.append(getattr(tag_model, tag_parent_name) == parent_id)
            # Built through ``_get_query`` like the outer query, so a workspace-aware
            # store scopes the satisfying set too. Scoping only the outer half is the
            # dangerous direction: children would be read in-scope but judged against
            # out-of-scope tag rows. It matters most on the registry, whose tables are
            # keyed by *name* and so are not unique across workspaces.
            satisfies = (
                store
                ._get_query(session, tag_model)
                .with_entities(*tag_discriminator)
                .filter(*filters)
                .scalar_subquery()
            )
            fails_a_clause.append(~_member_of(child_discriminator, satisfies))
        found = (
            store
            ._get_query(session, child_model)
            .with_entities(*child_ids)
            .filter(parent_column == parent_id, *extra_filters, sqlalchemy.or_(*fails_a_clause))
            .limit(1)
            .first()
        )
    return None if found is None else as_pushdown_key(tuple(found))


def _member_of(columns, subquery):
    """Membership of a discriminator in a subquery: by column, or by row value."""
    if len(columns) == 1:
        return columns[0].in_(subquery)
    return sqlalchemy.sql.tuple_(*columns).in_(subquery)

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


def satisfying_named(store, resolved, ids, id_chunk):
    """Which of these ids satisfy EVERY resolved clause?

    The named-id half of the OR pushdown, and the piece a caller needs once conditions
    are combined with OR rather than AND. The removed AND-era probe answered "is there
    an id failing this one condition?", which cannot be composed into
    "is there an id that no condition admits": it reports the first failure rather than the
    failing set, so per-condition answers cannot be intersected. Returning the satisfying
    set instead lets the caller union across conditions and subtract once.

    The union is left to the caller deliberately. Which conditions apply is decided per id
    -- a condition scoped to a resource pattern governs only some of the ids in play -- so
    the caller is the only layer that knows, for each id, which sets it was eligible for.
    A store combining them would have to be told the pattern grain, which is an
    authorization concept and does not belong here.

    Clauses are conjunctive within a condition, so this intersects across them. Absence
    satisfies nothing: an id with no row under a clause's key is simply not in that
    clause's matched set, and so drops out of the intersection -- including for ``!=`` and
    ``NOT IN``, where an untagged id must fail rather than be vacuously admitted. That
    falls out of asking which ids match rather than negating a comparator, which is the
    same reason the cascade probe asks it that way.

    Every query goes through ``store._get_query``, so a workspace-aware store scopes the
    satisfying set. Scoping only part of the question is the dangerous direction: ids
    judged against out-of-scope tag rows would be admitted by another workspace's tags.

    Returns:
        The subset of ``ids``, as pushdown keys, satisfying every clause. An id absent from
        the store satisfies nothing and so is never in the result -- indistinguishable from
        one that failed a clause, which the caller relies on: a 404 would reveal which ids
        exist to someone who may not read them.
    """
    requested = {as_pushdown_key(i) for i in ids}
    if not requested:
        return set()
    if not resolved:
        # A condition with no clauses constrains nothing, so every id satisfies it. Not
        # reachable through authoring, which rejects an empty filter precisely because
        # under OR it would be an unconditional pass nullifying its siblings.
        return requested

    dialect = store._get_dialect()
    satisfying = set(requested)
    with store.ManagedSessionMaker() as session:
        equal_key = key_comparison(dialect)
        for model, id_columns, key_column, value_column, key, comparator, value in resolved:
            # Ids bind one parameter each -- more for a composite -- and the clause binds
            # the key plus one per compared value, which for ``IN`` is the whole list.
            # Subtract those so the backend's cap stays a property of the statement rather
            # than a limit on how many ids a caller may ask about.
            clause_params = 1 + (len(value) if isinstance(value, tuple) else 1)
            chunk_size = max(1, (id_chunk - clause_params) // len(id_columns))
            # Only ids still in the running are worth asking about: one already dropped by
            # an earlier clause cannot re-enter the intersection.
            ordered = sorted(satisfying)
            matched = set()
            for start in range(0, len(ordered), chunk_size):
                chunk = ordered[start : start + chunk_size]
                rows = (
                    store
                    ._get_query(session, model)
                    .with_entities(*id_columns)
                    .filter(
                        id_predicate(id_columns, chunk, dialect),
                        equal_key(key_column, key),
                        compare_value(value_column, comparator, value, dialect),
                    )
                    .all()
                )
                matched.update(as_pushdown_key(tuple(r)) for r in rows)
            satisfying &= matched
            if not satisfying:
                # Nothing satisfies this condition, and the intersection cannot grow.
                break
    return satisfying


def iter_child_tag_batches(store, mapping, parent_id, chunk_size, extra_filters=()):
    """Yield ``(child_key, tags)`` for a parent's children, a chunk at a time.

    The cascade half of target evaluation once conditions are alternatives. A child is
    only a violator when *no* applicable condition admits it, and that question does not
    decompose into one query per condition: with children A failing only C1 and B failing
    only C2, a per-condition probe finds a victim for each while no child fails both, so
    the cascade should be allowed and per-condition probes would refuse it. Asking it as a
    single combined predicate instead makes one statement carry every condition's bind
    parameters -- ``sum(clauses + values)`` across all of them, against MSSQL's 2,100 --
    where the AND form bounded each statement by one condition. So the combination is done
    in Python over loaded children, and this is the loader.

    Evaluating in Python also collapses the two evaluators into one. The SQL form and
    :func:`~mlflow.server.auth.conditions.evaluate_resource` had to agree on absence,
    comparator semantics, ``LIKE``/``ILIKE`` and the MSSQL case-sensitive collation; a
    divergence between them would be a difference in who may write what. Here only the
    Python evaluator decides, and absence fails natively because a missing resource
    attribute fails every comparator including ``!=``.

    Cost is proportional to the number of children, which the single-query form avoided.
    That is accepted deliberately: the delete this authorizes already enumerates --
    ``delete_experiment`` loads every run as a full ORM entity and marks each one
    individually -- so authorization does not change the order of cost, and this loads one
    id column plus tags rather than whole entities. Chunking bounds the memory held at
    once, not the total work, and the caller is expected to stop at the first violator, so
    a cascade that fails early never loads the rest.

    Ids are fetched a chunk at a time ordered by the child's id columns, so an early exit
    stops the id scan too rather than only the tag loads.

    ``extra_filters`` narrows the population to the children the mutation reaches, exactly
    as in the cascade loader -- a predicate-mode delete reaches a slice, and
    judging it against the whole parent would refuse deletes over windows holding nothing
    objectionable. Here there is no satisfying subquery to keep them off, so they apply to
    the child query directly.

    Yields:
        Lists of ``(child_key, tags)``, where ``child_key`` is the child's
        :func:`as_pushdown_key` identity and ``tags`` is a plain ``{key: value}`` dict.
        A child with no tags yields an empty dict rather than being skipped -- it still has
        to be judged, and under target semantics it fails every clause.
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
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1, got {chunk_size!r}")

    child_ids = tuple(getattr(child_model, name) for name in child_id_names)
    parent_column = getattr(child_model, parent_name)
    tag_ids = tuple(getattr(tag_model, name) for name in tag_id_names if name != tag_parent_name)
    child_discriminator_names = tuple(n for n in child_id_names if n != parent_name)
    if len(tag_ids) != len(child_discriminator_names):
        raise ValueError(
            f"cascade mapping for {child_model.__name__} is inconsistent: the child "
            f"discriminator {child_id_names}-{parent_name!r} and the tag discriminator "
            f"{tag_id_names}-{tag_parent_name!r} must have the same width"
        )

    with store.ManagedSessionMaker() as session:
        offset = 0
        while True:
            rows = (
                store
                ._get_query(session, child_model)
                .with_entities(*child_ids)
                .filter(parent_column == parent_id, *extra_filters)
                .order_by(*child_ids)
                .limit(chunk_size)
                .offset(offset)
                .all()
            )
            if not rows:
                return
            offset += len(rows)

            # The discriminator is what the tag table is keyed by: the child's id minus the
            # parent, which this query has already fixed. For a run that is run_uuid; for a
            # version it is the bare version number, unique only within one name, which is
            # why the tag query carries the parent filter too.
            keyed = {}
            for row in rows:
                full = tuple(row)
                disc = tuple(v for n, v in zip(child_id_names, full) if n != parent_name)
                keyed[disc] = as_pushdown_key(full)

            tag_filters = [_member_of_values(tag_ids, list(keyed))]
            if tag_parent_name is not None:
                tag_filters.append(getattr(tag_model, tag_parent_name) == parent_id)
            tags_by_child = {disc: {} for disc in keyed}
            tag_rows = (
                store
                ._get_query(session, tag_model)
                .with_entities(
                    *tag_ids, getattr(tag_model, tag_key_name), getattr(tag_model, tag_value_name)
                )
                .filter(*tag_filters)
                .all()
            )
            for tag_row in tag_rows:
                disc = tuple(tag_row[: len(tag_ids)])
                key = tag_row[len(tag_ids)]
                value = tag_row[len(tag_ids) + 1]
                tags_by_child.setdefault(disc, {})[key] = value

            yield [(keyed[disc], tags_by_child[disc]) for disc in keyed]
            if len(rows) < chunk_size:
                return


def _member_of_values(columns, keys):
    """``columns IN keys`` for a single column, or a row-value IN for a composite."""
    if len(columns) == 1:
        return columns[0].in_([k[0] for k in keys])
    return sqlalchemy.tuple_(*columns).in_([tuple(k) for k in keys])


def _member_of(columns, subquery):
    """Membership of a discriminator in a subquery: by column, or by row value."""
    if len(columns) == 1:
        return columns[0].in_(subquery)
    return sqlalchemy.sql.tuple_(*columns).in_(subquery)

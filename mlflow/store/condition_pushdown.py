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

from mlflow.utils.search_utils import SearchUtils


class Declined:
    """The sentinel a store returns when it cannot answer a predicate.

    Kept distinct from ``None`` because the two outcomes are not interchangeable and
    confusing them is one-directional: ``None`` means every resource satisfied every
    clause, so reading "I cannot answer" as ``None`` lets every mutation through
    unjudged. A boolean test is how that mistake gets written -- ``if failing:`` looks
    reasonable and would treat a decline as a pass -- so this refuses to be one.
    """

    __slots__ = ()

    def __repr__(self):
        return "DECLINED"

    def __bool__(self):
        raise TypeError(
            "DECLINED is not a verdict: compare it with `is DECLINED` rather than "
            "testing it for truth. Treated as falsy it would read as 'nothing failed', "
            "which is the one direction this predicate must never fail in."
        )


DECLINED = Declined()


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


def id_predicate(id_columns, keys):
    """Match a chunk of ids: by column for a single key, by row value for a composite."""
    if len(id_columns) == 1:
        return id_columns[0].in_(keys)
    casted = tuple(comparable(column) for column in id_columns)
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


def find_failing_child(store, mapping, parent_id, clauses):
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

    dialect = store._get_dialect()
    with store.ManagedSessionMaker() as session:
        fails_a_clause = []
        for namespace, key, comparator, value in clauses:
            # No cascade child owns aliases (D18) -- a version's alias list names
            # aliases stored on its parent -- so an alias clause here has no table to
            # answer from. Decline rather than ignore it: an ignored clause is a
            # conjunction judged on a subset of itself, which is the fail-open
            # direction.
            if namespace != "tags":
                return DECLINED
            comparison = SearchUtils.get_sql_comparison_func(comparator, dialect)
            filters = [
                getattr(tag_model, tag_key_name) == key,
                comparison(comparable(getattr(tag_model, tag_value_name)), value),
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
            .filter(parent_column == parent_id, sqlalchemy.or_(*fails_a_clause))
            .limit(1)
            .first()
        )
    return None if found is None else as_pushdown_key(tuple(found))


def _member_of(columns, subquery):
    """Membership of a discriminator in a subquery: by column, or by row value."""
    if len(columns) == 1:
        return columns[0].in_(subquery)
    return sqlalchemy.sql.tuple_(*columns).in_(subquery)

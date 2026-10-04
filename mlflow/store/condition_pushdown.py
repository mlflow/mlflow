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

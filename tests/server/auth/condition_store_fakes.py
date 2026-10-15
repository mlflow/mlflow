"""Store fakes for the condition gate's target half.

A target condition is answered by the store or not at all -- there is no in-memory
fallback. So a test that wants to control *resource state* has to do it where the real
server does: inside the store's answer.

``answering_store`` is the faithful stand-in. It evaluates the pushed clauses with the
REAL :func:`evaluate_resource`, so a test keeps exercising the production semantics
(absence fails, conjunction over clauses, comparator handling) rather than a
reimplementation that could drift from it. It reproduces the behaviour the deleted Python
fallback had, including denying for a resource whose state is absent.

Before Phase B these tests used a store that returned ``DECLINED`` plus a patched
``attrs_for_bulk``, which routed the gate into that fallback. Both are gone.
"""

from types import SimpleNamespace

from mlflow.server.auth.conditions import Clause, evaluate_resource


def _clauses_of(pushed):
    """Store tuples back into the real clause objects.

    The store receives ``(namespace, key, comparator, value)`` tuples; the evaluator takes
    :class:`Clause` objects. Converting rather than re-implementing the comparison is the
    whole point of this fake.
    """
    return [
        Clause(identifier=namespace, key=key, comparator=comparator, value=value)
        for namespace, key, comparator, value in pushed
    ]


def answering_store(values=None, *, children=None, calls=None):
    """A store that answers every selector.

    Args:
        values: either a ``{(resource_type, resource_id): ResourceValues}`` mapping or a
            callable ``(resource_type, resource_id) -> ResourceValues | None``. An id that
            resolves to ``None`` is treated as a resource that is not there, which denies --
            matching the real gate, where a missing resource must not read as a vacuous
            pass. The callable form is the faithful stand-in for the deleted
            ``attrs_for_bulk`` patch, which answered for any id a test asked about.
        children: what the CASCADE loader yields -- either a
            ``{(resource_type, parent_id): [(child_id, tags), ...]}`` mapping or a callable
            ``(resource_type, parent_id) -> iterable``. This is the state the gate reaches
            its own verdict from, which replaced a precomputed one: under OR a single
            scripted answer cannot express a verdict, because whether a child is a violator
            depends on whether ANY condition admits it, so the tags have to be real.
        calls: optional list, appended to with ``("ids", entity, ids)``,
            ``("satisfying", entity, ids)``,
            ``("parent", entity, parent_id, max_timestamp_ms, stage)`` or
            ``("children", entity, parent_id, max_timestamp_ms, stage)`` so a test can
            assert which selector was used and how often.
    """
    if callable(values):
        resolve = values
    else:
        state = dict(values or {})
        resolve = lambda entity, resource_id: state.get((entity, resource_id))  # noqa: E731

    if callable(children):
        resolve_children = children
    else:
        kids = dict(children or {})
        # A flat ``[(child, tags), ...]`` list is the easy mistake here, and ``dict()``
        # accepts it silently -- the lookup then misses and the parent reads as EMPTY,
        # which is the permissive answer. A cascade test would pass while asserting
        # nothing. Rejecting the shape turns that into a loud failure.
        if bad := [k for k in kids if not (isinstance(k, tuple) and len(k) == 2)]:
            raise TypeError(
                "children must be keyed by (resource_type, parent_id) or be a callable; "
                f"got key(s) {bad!r}. A flat [(child, tags), ...] list looks like a "
                "mapping to dict() but yields an empty population, which ALLOWS."
            )
        resolve_children = lambda entity, parent_id: kids.get((entity, parent_id), ())  # noqa: E731

    def find_satisfying_resources(entity, clauses, ids):
        """Which ids satisfy EVERY clause -- the named-id half under OR.

        Answered with the real :func:`evaluate_resource`, like
        the real evaluator, so a test keeps exercising production semantics.
        An id whose state resolves to ``None`` is absent and satisfies nothing, which is
        how the real store behaves: it is simply not in the satisfying set, and is
        indistinguishable from one that failed a clause.
        """
        if calls is not None:
            calls.append(("satisfying", entity, tuple(ids)))
        parsed = _clauses_of(clauses)
        satisfying = set()
        for resource_id in ids:
            resolved = resolve(entity, resource_id)
            if resolved is not None and evaluate_resource(parsed, resolved):
                satisfying.add(resource_id)
        return satisfying

    def iter_child_resource_tags(
        entity, parent_id, *, chunk_size, max_timestamp_ms=None, stage=None
    ):
        """Yield a parent's children with their tags, a chunk at a time.

        Unlike the removed AND-era parent probe, this cannot be
        scripted with a single failing child: the gate combines conditions with OR, so the
        verdict depends on each child's actual tags rather than on one precomputed answer.
        A test that cares states the children.

        The window and stage are recorded, not applied -- these fakes hold no timestamps --
        which is what lets a test assert the bound reached the store at all, the whole
        point of narrowing a predicate-mode cascade.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if calls is not None:
            calls.append(("children", entity, parent_id, max_timestamp_ms, stage))
        rows = list(resolve_children(entity, parent_id))
        for start in range(0, len(rows), chunk_size):
            yield rows[start : start + chunk_size]

    return SimpleNamespace(
        find_satisfying_resources=find_satisfying_resources,
        iter_child_resource_tags=iter_child_resource_tags,
    )


def permissive_store():
    """A store for which nothing ever fails a target condition.

    The autouse default: a test about the REQUEST half, or about which conditions are
    selected at all, should not have to model resource state. Previously this role was
    played by a declining store, which reached the same outcome by a different route.

    Every id satisfies, so the gate's ``requested - union(satisfying)`` is empty, and a
    parent holds no children, so a cascade finds no violator. Both are the permissive
    answer rather than the empty one -- returning no satisfying ids would deny everything.
    """
    return SimpleNamespace(
        find_satisfying_resources=lambda entity, clauses, ids: set(ids),
        iter_child_resource_tags=lambda *a, **k: iter(()),
    )

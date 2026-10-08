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


def answering_store(values=None, *, failing_child=None, calls=None):
    """A store that answers both selectors.

    Args:
        values: either a ``{(resource_type, resource_id): ResourceValues}`` mapping or a
            callable ``(resource_type, resource_id) -> ResourceValues | None``. An id that
            resolves to ``None`` is treated as a resource that is not there, which denies --
            matching the real gate, where a missing resource must not read as a vacuous
            pass. The callable form is the faithful stand-in for the deleted
            ``attrs_for_bulk`` patch, which answered for any id a test asked about.
        failing_child: what the PARENT selector returns -- a child id for "this one fails",
            or ``None`` for "every child satisfies every clause". The real store answers
            this with one query and never enumerates, so there is nothing to derive it
            from; a test that cares states it directly.
        calls: optional list, appended to with ``("ids", entity, ids)`` or
            ``("parent", entity, parent_id, max_timestamp_ms)`` so a
            test can assert which selector was used and how often.
    """
    if callable(values):
        resolve = values
    else:
        state = dict(values or {})
        resolve = lambda entity, resource_id: state.get((entity, resource_id))  # noqa: E731

    def find_failing_resource(
        entity, clauses, *, ids=None, parent_id=None, max_timestamp_ms=None, stage=None
    ):
        if (ids is None) == (parent_id is None):
            raise ValueError("exactly one of ids or parent_id")
        if parent_id is not None:
            # The window is recorded, not applied: these fakes answer with a scripted
            # failing child rather than holding timestamps. Recording it is what lets a
            # test assert the bound actually reached the store, which is the whole point
            # of narrowing a predicate-mode cascade.
            if calls is not None:
                calls.append(("parent", entity, parent_id, max_timestamp_ms, stage))
            return failing_child
        if max_timestamp_ms is not None:
            raise ValueError("max_timestamp_ms is only valid with parent_id")
        if stage is not None:
            raise ValueError("stage is only valid with parent_id")
        if calls is not None:
            calls.append(("ids", entity, tuple(ids)))
        parsed = _clauses_of(clauses)
        for resource_id in ids:
            resolved = resolve(entity, resource_id)
            if resolved is None:
                return resource_id
            if not evaluate_resource(parsed, resolved):
                return resource_id
        return None

    return SimpleNamespace(find_failing_resource=find_failing_resource)


def permissive_store():
    """A store for which nothing ever fails a target condition.

    The autouse default: a test about the REQUEST half, or about which conditions are
    selected at all, should not have to model resource state. Previously this role was
    played by a declining store, which reached the same outcome by a different route.
    """
    return SimpleNamespace(find_failing_resource=lambda *a, **k: None)

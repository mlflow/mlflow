"""Validators for Mem0 memory-operation trace cases.

These decide a case's verdict from ids/hashes/counts/revisions alone — they never
read a raw memory body (the fixtures never contain one). They prove that the
trace shape locked by ``tests/mem0/fixtures/memory_operations.jsonl`` carries
enough information to catch the memory failure modes, before any
``mlflow.mem0.autolog()`` wiring exists.

A "case" is one parsed JSONL row: ``{"case_id", "expected", "events": [...]}``.
Events are ordered. Read events (``search``/``get``) return memories, each
carrying a ``revision``; write events (``update``/``delete``) supersede them
(``update`` bumps a memory to a new ``revision``, ``delete`` tombstones it);
``llm.request`` events join back to a memory operation via
``used_memory_operation_ids`` and name the memories they relied on via
``used_memory_ids``.
"""

READ_OPERATIONS = ("search", "get")
WRITE_OPERATIONS = ("update", "delete")


def _max_revision(known, observed):
    """The higher of two revisions, treating a missing revision as no information.

    Revision knowledge is monotonic: a read that reports an older revision than
    one already learned from a write must not move the known revision backward.
    """
    if known is None:
        return observed
    if observed is None:
        return known
    return max(known, observed)


def _memory_ops(case):
    return [e for e in case["events"] if e["kind"] == "memory.operation"]


def _decisions(case):
    return [e for e in case["events"] if e["kind"] == "llm.request"]


def _reads(case):
    return [e for e in _memory_ops(case) if e["operation"] in READ_OPERATIONS]


def _is_read(event):
    return event["kind"] == "memory.operation" and event["operation"] in READ_OPERATIONS


def load_receipt(case):
    """A read op is a valid receipt if it carries scope + consistent result identity."""
    for op in _reads(case):
        results = op.get("results", [])
        if "scope_hash" not in op:
            return "invalid"
        if op.get("result_count") != len(results):
            return "invalid"
    return "valid"


def usefulness_claim(case):
    """`joined` if a decision references an *earlier* read and uses a memory that
    read actually returned; `load_only` otherwise.

    Event order matters: a decision cannot join to a read that comes after it,
    and a decision that names a ``used_memory_id`` absent from the read it joins
    to has not established usefulness.
    """
    reads_by_op = {}
    for event in case["events"]:  # events are ordered
        if _is_read(event):
            reads_by_op[event["memory_operation_id"]] = {
                r["memory_id"] for r in event.get("results", [])
            }
        elif event["kind"] == "llm.request":
            used = set(event.get("used_memory_ids", []))
            for op_id in event.get("used_memory_operation_ids", []):
                if op_id in reads_by_op and used & reads_by_op[op_id]:
                    return "joined"
    return "load_only"


def detect_wrong_scope(case):
    """A returned memory whose scope_hash differs from the operation's scope."""
    for op in _reads(case):
        for result in op.get("results", []):
            if result["scope_hash"] != op["scope_hash"]:
                return True
    return False


def detect_stale_memory(case):
    """A decision relies on a memory version that a later write already superseded.

    ``delete`` tombstones an id permanently. ``update`` bumps a memory to a new
    ``revision``; a decision is stale only when the read it joined to holds an
    older revision than the memory's current one. A *fresh* read after an update
    re-establishes the current revision, so ``update -> fresh read -> use`` is
    not stale. Revision knowledge is monotonic — a later read reporting an older
    revision never overwrites a newer one learned from a write, so
    ``update(rev=2) -> stale read@rev1 -> use`` stays stale.
    """
    current_rev = {}
    deleted = set()
    reads_by_op = {}
    for event in case["events"]:  # events are ordered
        if event["kind"] == "memory.operation":
            operation = event["operation"]
            if operation in READ_OPERATIONS:
                revs = {r["memory_id"]: r.get("revision") for r in event.get("results", [])}
                reads_by_op[event["memory_operation_id"]] = revs
                for mid, rev in revs.items():
                    current_rev[mid] = _max_revision(current_rev.get(mid), rev)
            elif operation == "update":
                for mid in event.get("target_memory_ids", []):
                    current_rev[mid] = event.get("revision")
            elif operation == "delete":
                deleted.update(event.get("target_memory_ids", []))
        elif event["kind"] == "llm.request":
            used = set(event.get("used_memory_ids", []))
            if used & deleted:
                return True
            for op_id in event.get("used_memory_operation_ids", []):
                revs = reads_by_op.get(op_id, {})
                for mid in used & set(revs):
                    if revs[mid] != current_rev.get(mid):
                        return True
    return False


def detect_unreturned_memory(case):
    """A decision names a ``used_memory_id`` absent from the earlier reads it
    joins to — it claims a memory those operations never returned.

    Every claimed id must belong to the union of results across the referenced
    earlier reads; a partial overlap (some claimed ids returned, others foreign)
    still fails, since the foreign ids were never returned.
    """
    reads_by_op = {}
    for event in case["events"]:  # events are ordered
        if _is_read(event):
            reads_by_op[event["memory_operation_id"]] = {
                r["memory_id"] for r in event.get("results", [])
            }
        elif event["kind"] == "llm.request":
            used = set(event.get("used_memory_ids", []))
            if not used:
                continue
            referenced = [
                op_id
                for op_id in event.get("used_memory_operation_ids", [])
                if op_id in reads_by_op
            ]
            if not referenced:
                continue
            returned = set().union(*(reads_by_op[op_id] for op_id in referenced))
            if used - returned:
                return True
    return False


def detect_unjoinable(case):
    """A read that returned results but that no *later* decision references.

    An empty read (zero results) has nothing to join to a decision, so it is not
    unjoinable.
    """
    events = case["events"]
    for i, event in enumerate(events):
        if _is_read(event) and event.get("results"):
            op_id = event["memory_operation_id"]
            referenced_later = any(
                later["kind"] == "llm.request"
                and op_id in later.get("used_memory_operation_ids", [])
                for later in events[i + 1 :]
            )
            if not referenced_later:
                return True
    return False


def requires_raw_payload(case):
    """True when a decision joins a read but metadata cannot identify which memory
    it used: the read has >=2 results tied on all available metadata (scope +
    score) and the decision selects none of the tied results via
    ``used_memory_ids``.

    A tie alone is not enough — ``used_memory_ids`` is plural, so naming any of
    the tied results (one, several, or all of them) is a complete metadata
    selection and needs no raw text. The join is only genuinely ambiguous when
    the decision points at none of the tied members.
    """
    reads_by_op = {}
    for event in case["events"]:  # events are ordered
        if _is_read(event):
            reads_by_op[event["memory_operation_id"]] = event.get("results", [])
        elif event["kind"] == "llm.request":
            used = set(event.get("used_memory_ids", []))
            for op_id in event.get("used_memory_operation_ids", []):
                results = reads_by_op.get(op_id)
                if not results:
                    continue
                buckets = {}
                for result in results:
                    buckets.setdefault((result["scope_hash"], result["score"]), []).append(
                        result["memory_id"]
                    )
                for members in buckets.values():
                    if len(members) >= 2 and not (used & set(members)):
                        return True
    return False


def classify(case):
    """Map a case to its failure class (or 'valid'). Order = severity."""
    if detect_wrong_scope(case):
        return "wrong_scope_retrieval"
    if detect_stale_memory(case):
        return "stale_memory_used"
    if detect_unreturned_memory(case):
        return "unreturned_memory_used"
    if detect_unjoinable(case):
        return "unjoinable_memory"
    if requires_raw_payload(case):
        return "raw_payload_required"
    return "valid"

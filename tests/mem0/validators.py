"""Validators for Mem0 memory-operation trace cases.

These decide a case's verdict from ids/hashes/counts alone — they never read a
raw memory body (the fixtures never contain one). They prove that the trace
shape locked by ``tests/mem0/fixtures/memory_operations.jsonl`` carries enough
information to catch the memory failure modes, before any
``mlflow.mem0.autolog()`` wiring exists.

A "case" is one parsed JSONL row: ``{"case_id", "expected", "events": [...]}``.
Read events (``search``/``get``) return memories; write events
(``update``/``delete``) supersede them; ``llm.request`` events join back to a
memory operation via ``used_memory_operation_ids``.
"""

READ_OPERATIONS = ("search", "get")
WRITE_OPERATIONS = ("update", "delete")


def _memory_ops(case):
    return [e for e in case["events"] if e["kind"] == "memory.operation"]


def _decisions(case):
    return [e for e in case["events"] if e["kind"] == "llm.request"]


def _reads(case):
    return [e for e in _memory_ops(case) if e["operation"] in READ_OPERATIONS]


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
    """`joined` if some memory operation is referenced by a later decision."""
    joined = {mid for d in _decisions(case) for mid in d.get("used_memory_operation_ids", [])}
    for op in _reads(case):
        if op["memory_operation_id"] in joined:
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
    """A decision uses a memory id that an earlier update/delete already superseded."""
    superseded = set()
    for event in case["events"]:  # events are ordered
        if event["kind"] == "memory.operation" and event["operation"] in WRITE_OPERATIONS:
            superseded.update(event.get("target_memory_ids", []))
        elif event["kind"] == "llm.request":
            if superseded.intersection(event.get("used_memory_ids", [])):
                return True
    return False


def detect_unjoinable(case):
    """A retrieved memory operation that no decision references."""
    joined = {mid for d in _decisions(case) for mid in d.get("used_memory_operation_ids", [])}
    return any(op["memory_operation_id"] not in joined for op in _reads(case))


def requires_raw_payload(case):
    """True if a verdict needs raw text: a used result is indistinguishable, by
    metadata alone, from another result in the same operation (same scope + score).
    """
    used = {mid for d in _decisions(case) for mid in d.get("used_memory_ids", [])}
    for op in _reads(case):
        buckets = {}
        for result in op.get("results", []):
            buckets.setdefault((result["scope_hash"], result["score"]), []).append(
                result["memory_id"]
            )
        for members in buckets.values():
            if len(members) >= 2 and used.intersection(members):
                return True
    return False


def classify(case):
    """Map a case to its failure class (or 'valid'). Order = severity."""
    if detect_wrong_scope(case):
        return "wrong_scope_retrieval"
    if detect_stale_memory(case):
        return "stale_memory_used"
    if detect_unjoinable(case):
        return "unjoinable_memory"
    if requires_raw_payload(case):
        return "raw_payload_required"
    return "valid"

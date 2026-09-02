import ast
from collections.abc import Iterator

from clint.rules.base import Rule

_TRACE_CHILD_MODELS = frozenset({"SqlTraceTag", "SqlTraceMetadata", "SqlTraceMetrics"})


class ForbiddenTraceChildMerge(Rule):
    def _message(self) -> str:
        return (
            "Do not merge SqlTraceTag / SqlTraceMetadata / SqlTraceMetrics rows in a loop "
            "directly. Use `_merge_trace_child_rows_in_lock_order` so the rows are merged "
            "in sorted key order and concurrent writers cannot deadlock on the PK index "
            "(see #24338)."
        )

    @staticmethod
    def check(node: ast.For) -> bool:
        """Flag a `for` loop that merges a trace child model with a per-iteration key.

        ``session.merge(SqlTrace{Tag,Metadata,Metrics}(request_id=..., key=<name>, ...))``
        inside a loop is the exact pattern that regresses the #24338 deadlock fix: the loop
        can acquire the PK-index locks in caller-dependent order. Route it through
        ``_merge_trace_child_rows_in_lock_order`` instead.

        Matched only when ``key`` is a bare loop-bound name — the shape every trace-child
        merge site actually uses (``for k, v in d.items(): ... key=k``). This is a
        defense-in-depth backstop, not the primary guarantee (that is the helper plus the
        sorted-order tests), so it deliberately stays narrow to avoid false positives:
        a constant key (``key=TraceTagKey.SPANS_LOCATION``) is a single fixed row that
        cannot self-invert, and the shared helper merges a variable ``model_class`` rather
        than a literal trace-child constructor. A hand-written loop that varied the key via
        an attribute/subscript (``key=item.key``) would slip past — an accepted gap, since
        no such site exists and broadening it would flag safe fixed-attribute merges.

        Only this loop's own body is inspected; a nested loop is judged by its own
        ``visit_For`` so an offending inner loop is reported once, at the inner loop.
        """
        return any(
            ForbiddenTraceChildMerge._is_trace_child_merge(call)
            for call in ForbiddenTraceChildMerge._calls_excluding_nested_loops(node)
        )

    @staticmethod
    def _calls_excluding_nested_loops(node: ast.For) -> Iterator[ast.Call]:
        # Walk the loop body but prune nested loop subtrees at any depth, so a merge that
        # belongs to an inner loop is attributed only to that inner loop (which gets its
        # own visit_For), never doubly reported on an enclosing loop.
        for child in ast.iter_child_nodes(node):
            stack = [child]
            while stack:
                cur = stack.pop()
                if isinstance(cur, (ast.For, ast.AsyncFor)):
                    continue
                if isinstance(cur, ast.Call):
                    yield cur
                stack.extend(ast.iter_child_nodes(cur))

    @staticmethod
    def _is_trace_child_merge(stmt: ast.AST) -> bool:
        match stmt:
            case ast.Call(
                func=ast.Attribute(attr="merge"),
                args=[ast.Call(func=ast.Name(id=model), keywords=keywords), *_],
            ) if model in _TRACE_CHILD_MODELS:
                # Match only a bare loop-bound name key (the shape real sites use). A
                # constant key is a single fixed row and is safe; see check() for the
                # accepted attribute/subscript-key gap.
                return any(kw.arg == "key" and isinstance(kw.value, ast.Name) for kw in keywords)
            case _:
                return False

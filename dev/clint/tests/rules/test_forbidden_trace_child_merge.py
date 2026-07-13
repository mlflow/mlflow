from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Position, Range, lint_file
from clint.rules import ForbiddenTraceChildMerge


def test_flags_looped_trace_metadata_merge(index: SymbolIndex) -> None:
    code = """
for k, v in metadata.items():
    session.merge(SqlTraceMetadata(request_id=trace_id, key=k, value=v))
"""
    config = Config(select={ForbiddenTraceChildMerge.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 1
    assert all(isinstance(r.rule, ForbiddenTraceChildMerge) for r in results)
    assert results[0].range == Range(Position(1, 0))


def test_flags_looped_trace_tag_merge(index: SymbolIndex) -> None:
    code = """
for tag_key, tag_value in tags.items():
    session.merge(SqlTraceTag(request_id=trace_id, key=tag_key, value=tag_value))
"""
    config = Config(select={ForbiddenTraceChildMerge.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 1


def test_flags_looped_trace_metrics_merge(index: SymbolIndex) -> None:
    code = """
for k, v in metrics.items():
    session.merge(SqlTraceMetrics(request_id=trace_id, key=k, value=v))
"""
    config = Config(select={ForbiddenTraceChildMerge.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 1


def test_no_flag_single_row_merge(index: SymbolIndex) -> None:
    # set_trace_tag / archival single fixed-key merges are not in a loop and cannot
    # self-invert their own PK-lock order, so they are safe and must not be flagged.
    code = """
session.merge(SqlTraceTag(request_id=trace_id, key=key, value=value))
"""
    config = Config(select={ForbiddenTraceChildMerge.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 0


def test_no_flag_fixed_key_merge_in_outer_loop(index: SymbolIndex) -> None:
    # A single fixed-key merge (constant/attribute key) inside an outer loop over trace
    # ids writes one row per trace and cannot self-invert the PK-lock order, so it is safe.
    code = """
for trace_id in all_trace_ids:
    session.merge(
        SqlTraceTag(request_id=trace_id, key=TraceTagKey.SPANS_LOCATION, value=loc)
    )
"""
    config = Config(select={ForbiddenTraceChildMerge.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 0


def test_no_flag_helper_variable_merge(index: SymbolIndex) -> None:
    # The shared helper merges a variable model class, not a literal trace-child
    # constructor, so it is not matched even though it merges inside a loop.
    code = """
for key, value in sorted(values.items()):
    session.merge(model_class(request_id=request_id, key=key, value=value))
"""
    config = Config(select={ForbiddenTraceChildMerge.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 0


def test_no_flag_looped_non_trace_merge(index: SymbolIndex) -> None:
    # Looping a merge of an unrelated model is out of scope for this rule.
    code = """
for tag in tags:
    session.merge(SqlTag(key=tag.key, value=tag.value))
"""
    config = Config(select={ForbiddenTraceChildMerge.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 0

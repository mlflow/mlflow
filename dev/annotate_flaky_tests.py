"""Insert @pytest.mark.flaky decorators for classifier-approved flaky tests.

Third stage of the flaky-test pipeline (detect -> classify -> annotate). Reads the
classifier output and, for every verdict with ``action == "annotate"``, adds a
``@pytest.mark.flaky(attempts=N)`` decorator above the corresponding test function so
the conftest retry engine will re-run it. Tests classified ``investigate`` / ``fix`` are
left untouched — a retry would mask a real bug — and are reported for a human instead.

The edit is done with the ``ast`` module (resolving the nodeid's full ``::`` qualifier
chain so a name reused across classes hits the right function) and a line-based insert
that indents from the def's column to match the target. It is idempotent:
a test that already carries a ``flaky`` marker is skipped. A short comment linking back
to the report is added above the decorator so the annotation's origin is traceable.

Usage:
  python dev/annotate_flaky_tests.py --in classified.json [--report annotated.md]
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Annotation:
    nodeid: str
    file: str
    func: str
    attempts: int
    applied: bool
    note: str


def _split_nodeid(nodeid: str) -> tuple[str, list[str]] | None:
    """'tests/f.py::TestX::test_y[case]' -> ('tests/f.py', ['TestX', 'test_y']).

    Returns the file path and the full '::' qualifier chain (enclosing classes plus
    the function), so a test name reused across classes resolves to the right node.
    """
    if "::" not in nodeid:
        return None
    path, _, rest = nodeid.partition("::")
    if not path.endswith(".py"):
        return None
    # Strip any parametrization id ('test_x[case]') from the final segment.
    qualifiers = [seg.split("[")[0] for seg in rest.split("::")]
    return path, qualifiers


def _find_funcdef(
    tree: ast.Module, qualifiers: list[str]
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Descend the qualifier chain (ClassDefs) to the target FunctionDef.

    Matching the full chain (not a bare `ast.walk` by name) avoids annotating the
    wrong function when a test name is reused across classes or module-level helpers.
    """
    *class_path, func = qualifiers
    scope: ast.AST = tree
    for cls in class_path:
        match = next(
            (
                n
                for n in ast.iter_child_nodes(scope)
                if isinstance(n, ast.ClassDef) and n.name == cls
            ),
            None,
        )
        if match is None:
            return None
        scope = match
    for node in ast.iter_child_nodes(scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func:
            return node
    return None


def _already_flaky(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        # matches @pytest.mark.flaky and @pytest.mark.flaky(...)
        call = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(call, ast.Attribute) and call.attr == "flaky":
            return True
    return False


def _imports_pytest(tree: ast.Module) -> bool:
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import) and any(a.name == "pytest" for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "pytest":
            return True
    return False


def _import_insert_line(tree: ast.Module) -> int:
    """1-based line to insert 'import pytest' at: after any module docstring and
    `from __future__` imports (which must stay first), before everything else.
    """
    line = 1
    for node in tree.body:
        is_docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        if is_docstring or is_future:
            line = (node.end_lineno or node.lineno) + 1
        else:
            break
    return line


def annotate_file(path: Path, qualifiers: list[str], attempts: int, report_ref: str) -> Annotation:
    func = qualifiers[-1]
    nodeid = f"{path}::{'::'.join(qualifiers)}"
    source = path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return Annotation(nodeid, str(path), func, attempts, False, "file did not parse")

    node = _find_funcdef(tree, qualifiers)
    if node is None:
        return Annotation(nodeid, str(path), func, attempts, False, "function not found")
    if _already_flaky(node):
        return Annotation(nodeid, str(path), func, attempts, False, "already marked flaky")

    # Insert above the first decorator if present, else above the def line. Indent from
    # the def's column (node.col_offset), NOT the decorator expression's col_offset —
    # a decorator node starts *after* the '@', so its col_offset is one deeper and would
    # over-indent the insert into an IndentationError.
    anchor = node.decorator_list[0] if node.decorator_list else node
    insert_at = anchor.lineno - 1  # 0-based index of the anchor's line
    lines = source.splitlines(keepends=True)
    indent = " " * node.col_offset
    block = (
        f"{indent}# flaky: auto-detected from CI re-runs; see {report_ref}\n"
        f"{indent}@pytest.mark.flaky(attempts={attempts})\n"
    )
    # The decorator references `pytest`; ensure the module imports it, or collection
    # raises NameError. Apply the two inserts bottom-up so the earlier (import) insert
    # doesn't shift the decorator's line index.
    import_at = None if _imports_pytest(tree) else _import_insert_line(tree) - 1
    lines.insert(insert_at, block)
    if import_at is not None:
        lines.insert(import_at, "import pytest\n")
    path.write_text("".join(lines))
    return Annotation(nodeid, str(path), func, attempts, True, "annotated")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="infile", required=True, help="classified.json from the classifier")
    p.add_argument(
        "--report-ref",
        default="the weekly flaky-test report",
        help="Human-readable reference (e.g. issue URL) recorded in the inserted comment.",
    )
    p.add_argument("--report", help="Write a Markdown summary of what was annotated / skipped.")
    args = p.parse_args()

    with open(args.infile) as f:
        classified = json.load(f)

    results: list[Annotation] = []
    for entry in classified:
        verdict = entry.get("verdict", {})
        if verdict.get("action") != "annotate" or not entry.get("test"):
            continue
        split = _split_nodeid(entry["test"])
        if split is None:
            results.append(Annotation(entry["test"], "", "", 0, False, "unparsable nodeid"))
            continue
        rel_path, qualifiers = split
        path = Path(rel_path)
        if not path.exists():
            results.append(
                Annotation(entry["test"], rel_path, qualifiers[-1], 0, False, "file missing")
            )
            continue
        attempts = verdict.get("attempts") or 3
        results.append(annotate_file(path, qualifiers, attempts, args.report_ref))

    applied = [r for r in results if r.applied]
    skipped = [r for r in results if not r.applied]
    for r in results:
        mark = "✓" if r.applied else "–"
        print(f"  {mark} {r.nodeid}  ({r.note})")
    print(f"\nAnnotated {len(applied)}, skipped {len(skipped)}.")

    if args.report:
        lines = ["# Flaky annotations applied", ""]
        if applied:
            lines.append("Added `@pytest.mark.flaky` to:")
            lines += [f"- `{r.nodeid}` — `attempts={r.attempts}`" for r in applied]
        else:
            lines.append("No new annotations applied.")
        if skipped:
            lines += ["", "Skipped:"]
            lines += [f"- `{r.nodeid}` — {r.note}" for r in skipped]
        Path(args.report).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

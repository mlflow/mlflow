"""Insert @pytest.mark.flaky decorators for classifier-approved flaky tests.

Third stage of the flaky-test pipeline (detect -> classify -> annotate). Reads the
classifier output and, for every verdict with ``action == "annotate"``, adds a
``@pytest.mark.flaky(attempts=N)`` decorator above the corresponding test function so
the conftest retry engine will re-run it. Tests classified ``investigate`` / ``fix`` are
left untouched — a retry would mask a real bug — and are reported for a human instead.

The edit is done with the ``ast`` module (find the function by name + line) and a
line-based insert that preserves the function's existing indentation. It is idempotent:
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


def _split_nodeid(nodeid: str) -> tuple[str, str] | None:
    """'tests/foo/test_bar.py::TestX::test_baz' -> ('tests/foo/test_bar.py', 'test_baz')."""
    if "::" not in nodeid:
        return None
    path, _, rest = nodeid.partition("::")
    if not path.endswith(".py"):
        return None
    # Last '::' segment is the function; strip any parametrization id ('test_x[case]').
    func = rest.split("::")[-1].split("[")[0]
    return path, func


def _find_funcdef(tree: ast.Module, func: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
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


def annotate_file(path: Path, func: str, attempts: int, report_ref: str) -> Annotation:
    nodeid = f"{path}::{func}"
    source = path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return Annotation(nodeid, str(path), func, attempts, False, "file did not parse")

    node = _find_funcdef(tree, func)
    if node is None:
        return Annotation(nodeid, str(path), func, attempts, False, "function not found")
    if _already_flaky(node):
        return Annotation(nodeid, str(path), func, attempts, False, "already marked flaky")

    # Insert above the first decorator if present, else above the def line, matching
    # the target's indentation. `lineno` of the earliest decorator == the '@' line.
    anchor = node.decorator_list[0] if node.decorator_list else node
    insert_at = anchor.lineno - 1  # 0-based index of the anchor's line
    lines = source.splitlines(keepends=True)
    indent = " " * anchor.col_offset
    block = (
        f"{indent}# flaky: auto-detected from CI re-runs; see {report_ref}\n"
        f"{indent}@pytest.mark.flaky(attempts={attempts})\n"
    )
    lines.insert(insert_at, block)
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
        rel_path, func = split
        path = Path(rel_path)
        if not path.exists():
            results.append(Annotation(entry["test"], rel_path, func, 0, False, "file missing"))
            continue
        attempts = verdict.get("attempts") or 3
        results.append(annotate_file(path, func, attempts, args.report_ref))

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

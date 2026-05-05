"""Walk high-signal mlflow utility directories and emit helpers_*.md.

For each public function / class / classmethod we find:

- Module path
- Line number (the `def` / `class` line)
- Signature (function args / class bases)
- First sentence of the docstring (single line, trimmed)

Skips:

- Private symbols (leading underscore on the symbol name).
- Test files.
- Auto-generated protobuf modules (`*_pb2.py`).
- `__init__.py` files that only re-export (skipped via SKIP_FILES list).

Output: `<orchestrator>/repo_knowledge/helpers_<slug>.md` plus a
`helpers_index.md` pointer file. Re-run via the
`refresh-helpers.yml` workflow weekly.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import sys
from pathlib import Path

# Targets walked by the script. (display, root_relative_path, recurse, slug).
# `slug` becomes the suffix on the per-section helpers file:
# helpers_exceptions.md, helpers_utils.md, helpers_tracing_utils.md, helpers_types.md.
TARGETS: list[tuple[str, str, bool, str]] = [
    ("`mlflow/exceptions.py`", "mlflow/exceptions.py", False, "exceptions"),
    ("`mlflow/utils/`", "mlflow/utils", True, "utils"),
    ("`mlflow/tracing/utils/`", "mlflow/tracing/utils", True, "tracing_utils"),
    ("`mlflow/types/`", "mlflow/types", False, "types"),
]

APPLIES_TO_BY_SLUG = {
    "exceptions": (
        "any PR that raises MlflowException, RestException, or any subclass; touches "
        "mlflow/exceptions.py; calls error_code / sqlstate / error_class."
    ),
    "utils": (
        "any PR that touches mlflow/utils/; introduces a new utility helper anywhere "
        "in the repo; or could plausibly reuse an existing utility (lazy_load, "
        "annotations, rest_utils, file_utils, etc.)."
    ),
    "tracing_utils": (
        "any PR under mlflow/tracing/, mlflow/<flavor>/autolog.py, "
        "mlflow/<flavor>/chat.py, or any code that emits OTLP spans / sets span "
        "attributes."
    ),
    "types": (
        "any PR that touches mlflow/types/, defines a chat / agent / response "
        "BaseModel, calls validate_compat / model_dump_compat / model_validate, or "
        "threads ChatMessage / ChatTool / ResponsesAgent types."
    ),
}

SKIP_FILES = {
    "mlflow/utils/__init__.py",  # mostly logger setup + re-exports
}


def _first_sentence(docstring: str | None) -> str:
    if not docstring:
        return ""
    text = docstring.strip().replace("\n", " ")
    for sep in (". ", ".\t", ".\n", "?", "!"):
        if sep in text:
            return text.split(sep, 1)[0].strip().rstrip(".") + (
                "." if sep.startswith(".") else sep[-1]
            )
    return text


def _format_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    parts: list[str] = []
    pos_only = list(args.posonlyargs)
    regular = list(args.args)
    if pos_only:
        parts.extend(_arg_to_str(a) for a in pos_only)
        parts.append("/")
    parts.extend(_arg_to_str(a) for a in regular)
    if args.vararg:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")
    parts.extend(_arg_to_str(a) for a in args.kwonlyargs)
    if args.kwarg:
        parts.append("**" + args.kwarg.arg)
    sig = "(" + ", ".join(parts) + ")"
    if node.returns is not None:
        sig += " -> " + ast.unparse(node.returns)
    return sig


def _arg_to_str(arg: ast.arg) -> str:
    s = arg.arg
    if arg.annotation is not None:
        s += ": " + ast.unparse(arg.annotation)
    return s


def _format_class_bases(node: ast.ClassDef) -> str:
    bases = [ast.unparse(b) for b in node.bases]
    return f"({', '.join(bases)})" if bases else ""


def walk_targets(repo_root: Path) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {name: [] for name, _, _, _ in TARGETS}
    for display, rel, recurse, _slug in TARGETS:
        root = repo_root / rel
        if rel.endswith(".py"):
            files = [root]
        elif recurse:
            files = sorted([p for p in root.rglob("*.py") if "__pycache__" not in p.parts])
        else:
            files = sorted(root.glob("*.py"))
        for f in files:
            rel_path = f.relative_to(repo_root).as_posix()
            if rel_path in SKIP_FILES:
                continue
            if rel_path.endswith("_pb2.py"):
                continue
            if f.name.startswith("_") and f.name != "__init__.py":
                continue
            try:
                tree = ast.parse(f.read_text())
            except SyntaxError:
                continue
            for node in tree.body:
                entry = _node_entry(node, rel_path)
                if entry is None:
                    continue
                grouped[display].append(entry)
                if isinstance(node, ast.ClassDef):
                    for sub in node.body:
                        sub_entry = _node_entry(sub, rel_path, parent=node.name)
                        if sub_entry is None:
                            continue
                        if sub_entry["kind"] == "function":
                            sub_entry["kind"] = "method"
                        grouped[display].append(sub_entry)
    return grouped


def _node_entry(
    node: ast.AST, rel_path: str, parent: str | None = None
) -> dict[str, object] | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        name = node.name
        if name.startswith("_") and name != "__init__":
            return None
        if name == "__init__":
            return None
        kind = "function"
        if any(_is_decorator(d, "classmethod") for d in node.decorator_list):
            kind = "classmethod"
        elif any(_is_decorator(d, "staticmethod") for d in node.decorator_list):
            kind = "staticmethod"
        elif any(_is_decorator(d, "property") for d in node.decorator_list):
            return None
        sig = _format_signature(node)
        doc = _first_sentence(ast.get_docstring(node))
        full_name = f"{parent}.{name}" if parent else name
        return {
            "kind": kind,
            "name": full_name,
            "sig": sig,
            "doc": doc,
            "path": rel_path,
            "line": node.lineno,
        }
    if isinstance(node, ast.ClassDef):
        if node.name.startswith("_"):
            return None
        sig = _format_class_bases(node)
        doc = _first_sentence(ast.get_docstring(node))
        return {
            "kind": "class",
            "name": node.name,
            "sig": sig,
            "doc": doc,
            "path": rel_path,
            "line": node.lineno,
        }
    return None


def _is_decorator(dec: ast.AST, name: str) -> bool:
    if isinstance(dec, ast.Name):
        return dec.id == name
    if isinstance(dec, ast.Attribute):
        return dec.attr == name
    if isinstance(dec, ast.Call):
        return _is_decorator(dec.func, name)
    return False


def render_section(display: str, slug: str, entries: list[dict[str, object]], today: str) -> str:
    out: list[str] = []
    applies_to = APPLIES_TO_BY_SLUG.get(slug, "any PR.")
    out.append("---")
    out.append(f"name: helpers_{slug}")
    out.append(
        f"description: Auto-generated public-symbol reference for {display}. "
        "Use this before suggesting a new helper."
    )
    out.append(f"applies_to: {applies_to}")
    out.append(f"last_verified: {today}")
    out.append(
        "citation_policy: each `path:line` is the `def` / `class` line. If the snippet "
        "drifts, search by symbol name."
    )
    out.append(
        "generated_by: .claude/orchestrator/scripts/generate_helpers_md.py "
        "(refreshed weekly by .github/workflows/refresh-helpers.yml)."
    )
    out.append("---")
    out.append("")
    out.append(f"# Helpers: {display}")
    out.append("")
    out.append(
        f"Auto-generated. Walks {display} and lists every public symbol with its "
        "signature and first docstring sentence."
    )
    out.append("")
    out.append("## How to use this file")
    out.append("")
    out.append(
        "- **Before suggesting a new utility function in a review**, grep this file for "
        "the area you're touching. If a helper already exists, point at its `path:line` "
        "instead of asking for a new one."
    )
    out.append(
        "- **Class entries** list public methods in the same row group (`ClassName.method` form)."
    )
    out.append(
        "- **Search by symbol name**, not by line number: line numbers drift after reformats."
    )
    out.append("")
    by_file: dict[str, list[dict[str, object]]] = {}
    for e in entries:
        by_file.setdefault(e["path"], []).append(e)
    single_file = len(by_file) == 1 and f"`{next(iter(by_file))}`" == display
    for path in sorted(by_file):
        file_entries = by_file[path]
        if not single_file:
            out.append(f"## `{path}`")
            out.append("")
        out.append("| Symbol | Kind | Signature | One-line docstring | Line |")
        out.append("|---|---|---|---|---|")
        for e in sorted(file_entries, key=lambda x: x["line"]):
            name = e["name"].replace("|", "\\|")
            sig = e["sig"].replace("|", "\\|").replace("\n", " ")
            doc = e["doc"].replace("|", "\\|").replace("\n", " ")
            if len(doc) > 130:
                doc = doc[:127] + "..."
            if len(sig) > 110:
                sig = sig[:107] + "..."
            out.append(f"| `{name}` | {e['kind']} | `{sig}` | {doc} | {e['line']} |")
        out.append("")
    return "\n".join(out) + "\n"


def render_index(grouped: dict[str, list[dict[str, object]]], today: str) -> str:
    out: list[str] = []
    out.append("---")
    out.append("name: helpers_index")
    out.append(
        "description: Tiny index pointing at the four per-section helpers files. Read "
        "this first to decide which helpers_*.md to load."
    )
    out.append(
        "applies_to: any PR. Read this index, then load the matching helpers_<slug>.md "
        "based on the diff's areas."
    )
    out.append(f"last_verified: {today}")
    out.append("---")
    out.append("")
    out.append("# Helpers index")
    out.append("")
    out.append(
        "`helpers.md` is split into per-section files. Read this index, then open only "
        "the per-section files relevant to the PR's diff."
    )
    out.append("")
    out.append("| File | Section | Symbols | When to read |")
    out.append("|---|---|---:|---|")
    for display, _rel, _recurse, slug in TARGETS:
        n = len(grouped.get(display, []))
        applies = APPLIES_TO_BY_SLUG.get(slug, "")
        out.append(f"| `helpers_{slug}.md` | {display} | {n} | {applies} |")
    out.append("")
    return "\n".join(out) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Path to the mlflow checkout. Defaults to cwd.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=("Output directory. Defaults to <repo-root>/.claude/orchestrator/repo_knowledge."),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root: Path = args.repo_root.resolve()
    out_dir: Path = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else repo_root / ".claude" / "orchestrator" / "repo_knowledge"
    )
    grouped = walk_targets(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    legacy = out_dir / "helpers.md"
    if legacy.exists():
        legacy.unlink()
    total = 0
    for display, _rel, _recurse, slug in TARGETS:
        entries = grouped.get(display, [])
        if not entries:
            continue
        path = out_dir / f"helpers_{slug}.md"
        path.write_text(render_section(display, slug, entries, today))
        total += len(entries)
        sys.stderr.write(f"Wrote {path} ({len(entries)} symbols, {path.stat().st_size:,} bytes)\n")
    index_path = out_dir / "helpers_index.md"
    index_path.write_text(render_index(grouped, today))
    sys.stderr.write(f"Wrote {index_path} ({index_path.stat().st_size:,} bytes)\n")
    sys.stderr.write(f"Total symbols: {total}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

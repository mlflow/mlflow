from __future__ import annotations

import ast
import json
import re
import textwrap
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from clint import rules
from clint.builtin import BUILTIN_MODULES

PARAM_REGEX = re.compile(r"\s+:param\s+\w+:", re.MULTILINE)
RETURN_REGEX = re.compile(r"\s+:returns?:", re.MULTILINE)
DISABLE_COMMENT_REGEX = re.compile(r"clint:\s*disable=([a-z0-9-]+)")


def ignore_map(code: str) -> dict[str, set[int]]:
    """
    Creates a mapping of rule name to line numbers to ignore.

    {
        "<rule_name>": {<line_number>, ...},
        ...
    }
    """
    mapping: dict[str, set[int]] = {}
    readline = iter(code.splitlines(True)).__next__
    for tok in tokenize.generate_tokens(readline):
        if tok.type != tokenize.COMMENT:
            continue
        if m := DISABLE_COMMENT_REGEX.search(tok.string):
            mapping.setdefault(m.group(1), set()).add(tok.start[0])
    return mapping


def _is_log_model(node: ast.AST) -> bool:
    """
    Is this node a call to `log_model`?
    """
    if isinstance(node, ast.Name):
        return "log_model" in node.id

    elif isinstance(node, ast.Attribute):
        return "log_model" in node.attr

    return False


@dataclass
class Violation:
    rule: rules.Rule
    path: Path
    lineno: int
    col_offset: int
    cell: int | None = None

    def __str__(self):
        # Use the same format as ruff
        cell_loc = f"cell {self.cell}:" if self.cell is not None else ""
        return (
            f"{self.path}:{cell_loc}{self.lineno}:{self.col_offset}: "
            f"{self.rule.id}: {self.rule.message}"
        )

    def json(self) -> dict[str, str | int | None]:
        return {
            "type": "error",
            "module": None,
            "obj": None,
            "line": self.lineno,
            "column": self.col_offset,
            "endLine": self.lineno,
            "endColumn": self.col_offset,
            "path": str(self.path),
            "symbol": self.rule.name,
            "message": self.rule.message,
            "message-id": self.rule.id,
        }


@dataclass
class Location:
    lineno: int
    col_offset: int

    @classmethod
    def from_node(cls, node: ast.AST) -> "Location":
        return cls(node.lineno, node.col_offset)


@dataclass
class CodeBlock:
    code: str
    loc: Location


def _get_indent(s: str) -> int:
    return len(s) - len(s.lstrip())


_CODE_BLOCK_HEADER_REGEX = re.compile(r"^\.\.\s+code-block::\s*py(thon)?")
_CODE_BLOCK_OPTION_REGEX = re.compile(r"^:\w+:")


def _iter_code_blocks(docstring: str) -> Iterator[CodeBlock]:
    code_block_loc: Location | None = None
    code_lines: list[str] = []

    for idx, line in enumerate(docstring.split("\n")):
        if code_block_loc:
            indent = _get_indent(line)
            # Are we still in the code block?
            if 0 < indent <= code_block_loc.col_offset:
                code = textwrap.dedent("\n".join(code_lines))
                yield CodeBlock(code=code, loc=code_block_loc)

                code_block_loc = None
                code_lines.clear()
                continue

            # .. code-block:: python
            #     :option:           <- code block may have options
            #     :another-option:   <-
            #
            #     import mlflow      <- code body starts from here
            #     ...
            if not _CODE_BLOCK_OPTION_REGEX.match(line.lstrip()):
                code_lines.append(line)

        else:
            if _CODE_BLOCK_HEADER_REGEX.match(line.lstrip()):
                code_block_loc = Location(idx, _get_indent(line) + 1)

    # The docstring ends with a code block
    if code_lines:
        code = textwrap.dedent("\n".join(code_lines))
        yield CodeBlock(code=code, loc=code_block_loc)


def _parse_docstring_args(docstring: str) -> list[str]:
    args: list[str] = []
    args_header_indent: int | None = None
    first_arg_indent: int | None = None
    arg_name_regex = re.compile(r"(\w+)")
    for line in docstring.split("\n"):
        if args_header_indent is not None:
            indent = _get_indent(line)
            # If we encounter a non-blank line with an indent less than the args header,
            # we are done parsing the args section.
            if 0 < indent <= args_header_indent:
                break

            if not args and first_arg_indent is None:
                first_arg_indent = indent

            if m := arg_name_regex.match(line[first_arg_indent:]):
                args.append(m.group(1))

        elif line.lstrip().startswith("Args:"):
            args_header_indent = _get_indent(line)

    return args


class Linter(ast.NodeVisitor):
    def __init__(self, *, path: Path, ignore: dict[str, set[int]], cell: int | None = None):
        """
        Lints a Python file.

        Args:
            path: Path to the file being linted.
            ignore: Mapping of rule name to line numbers to ignore.
            cell: Index of the cell being linted in a Jupyter notebook.
        """
        self.stack: list[ast.AST] = []
        self.path = path
        self.ignore = ignore
        self.cell = cell
        self.violations: list[Violation] = []

    def _check(self, loc: Location, rule: rules.Rule) -> None:
        if (lines := self.ignore.get(rule.name)) and loc.lineno in lines:
            return
        self.violations.append(
            Violation(
                rule,
                self.path,
                loc.lineno,
                loc.col_offset,
                self.cell,
            )
        )

    def _docstring(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> ast.Constant | None:
        if (
            isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.s, str)
        ):
            return node.body[0].value
        return None

    def _no_rst(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if (nd := self._docstring(node)) and (
            PARAM_REGEX.search(nd.s) or RETURN_REGEX.search(nd.s)
        ):
            self._check(nd, rules.NoRst())

    def _is_in_function(self) -> bool:
        return bool(self.stack)

    def _is_in_class(self) -> bool:
        return self.stack and isinstance(self.stack[-1], ast.ClassDef)

    def _parse_func_args(self, func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        args: list[str] = []
        for arg in func.args.posonlyargs:
            args.append(arg.arg)

        for arg in func.args.args:
            args.append(arg.arg)

        for arg in func.args.kwonlyargs:
            args.append(arg.arg)

        if func.args.vararg:
            args.append(func.args.vararg.arg)

        if func.args.kwarg:
            args.append(func.args.kwarg.arg)

        if self._is_in_class():
            if any(isinstance(d, ast.Name) and d.id == "classmethod" for d in func.decorator_list):
                if "cls":
                    args.remove("cls")
            elif any(
                isinstance(d, ast.Name) and d.id == "staticmethod" for d in func.decorator_list
            ):
                pass
            else:  # Instance method
                if "self":
                    args.remove("self")

        return args

    def _test_name_typo(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if not self.path.name.startswith("test_") or self._is_in_function():
            return

        if node.name.startswith("test") and not node.name.startswith("test_"):
            self._check(Location.from_node(node), rules.TestNameTypo())

    def _mlflow_class_name(self, node: ast.ClassDef) -> None:
        if "MLflow" in node.name or "MLFlow" in node.name:
            self._check(Location.from_node(node), rules.MlflowClassName())

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node)
        self._no_rst(node)
        self._mlflow_class_name(node)
        self.generic_visit(node)
        self.stack.pop()

    def _syntax_error_example(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if docstring_node := self._docstring(node):
            for code_block in _iter_code_blocks(docstring_node.value):
                try:
                    ast.parse(code_block.code)
                except SyntaxError:
                    loc = Location(
                        docstring_node.lineno + code_block.loc.lineno,
                        code_block.loc.col_offset,
                    )
                    self._check(loc, rules.ExampleSyntaxError())

    def _param_mismatch(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if node.name.startswith("_"):
            return
        if docstring_node := self._docstring(node):
            if (doc_args := _parse_docstring_args(docstring_node.value)) and (
                func_args := self._parse_func_args(node)
            ):
                func_args_set = set(func_args)
                doc_args_set = set(doc_args)
                # TODO: Enable this rule
                # if diff := func_args_set - doc_args_set:
                #     self._check(Location.from_node(node), rules.MissingDocstringParam(diff))

                if diff := doc_args_set - func_args_set:
                    self._check(Location.from_node(node), rules.ExtraneousDocstringParam(diff))

                # TODO: Enable this rule
                # if func_args_set == doc_args_set and func_args != doc_args:
                #     self._check(Location.from_node(node), rules.DocstringParamOrder())

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._test_name_typo(node)
        self._syntax_error_example(node)
        self._param_mismatch(node)
        self.stack.append(node)
        self._no_rst(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._test_name_typo(node)
        self._syntax_error_example(node)
        self._param_mismatch(node)
        self.stack.append(node)
        self._no_rst(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        if self._is_in_function():
            for alias in node.names:
                if alias.name.split(".", 1)[0] in BUILTIN_MODULES:
                    self._check(Location.from_node(node), rules.LazyBuiltinImport())
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._is_in_function() and node.module.split(".", 1)[0] in BUILTIN_MODULES:
            self._check(Location.from_node(node), rules.LazyBuiltinImport())
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            self.path.parts[0] in ["tests", "mlflow"]
            and _is_log_model(node.func)
            and any(arg.arg == "artifact_path" for arg in node.keywords)
        ):
            self._check(Location.from_node(node), rules.KeywordArtifactPath())


def _lint_cell(path: Path, cell: dict[str, Any], index: int) -> list[Violation]:
    type_ = cell.get("cell_type")
    if type_ != "code":
        return []

    lines = cell.get("source")
    if not lines:
        return []

    src = "\n".join(lines)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # Ignore non-python cells such as `!pip install ...`
        return []

    linter = Linter(path=path, ignore=ignore_map(src), cell=index)
    linter.visit(tree)
    return linter.violations


def lint_file(path: Path) -> list[Violation]:
    code = path.read_text()
    if path.suffix == ".ipynb":
        if cells := json.loads(code).get("cells"):
            violations = []
            for idx, cell in enumerate(cells, start=1):
                violations.extend(_lint_cell(path, cell, idx))
            return violations
    else:
        linter = Linter(path=path, ignore=ignore_map(code))
        linter.visit(ast.parse(code))
        return linter.violations

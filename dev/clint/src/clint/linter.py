from __future__ import annotations

import ast
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class Rule:
    id: str
    name: str
    message: str

    def format(self, **kwargs) -> str:
        return self.message.format(**kwargs)


@dataclass
class Violation:
    rule: Rule
    path: Path
    lineno: int
    col_offset: int

    def __str__(self):
        return f"{self.path}:{self.lineno}:{self.col_offset}: {self.rule.id}: {self.rule.message}"

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


NO_RST = Rule(
    "MLF0001",
    "no-rst",
    "Do not use RST style. Use Google style instead.",
)
LAZY_BUILTIN_IMPORT = Rule(
    "MLF0002",
    "lazy-builtin-import",
    "Builtin modules must be imported at the top level.",
)
MLFLOW_CLASS_NAME = Rule(
    "MLF0003",
    "mlflow-class-name",
    "Should use `Mlflow` in class name, not `MLflow` or `MLFlow`.",
)
TEST_NAME_TYPO = Rule(
    "MLF0004",
    "test-name-typo",
    "This function looks like a test, but its name does not start with 'test_'.",
)

MISSING_PARAMS = Rule(
    "MLF0006",
    "missing-params",
    "This function has undocumented parameters.",
)
EXTRA_PARAMS = Rule(
    "MLF0007",
    "extra-params",
    "This function has extra parameters in the docstring.",
)


def _get_indent_level(s: str) -> int | None:
    if not s.strip():
        return None
    return len(s) - len(s.lstrip())


import re

_PARAM_NAME_REGEX = re.compile(r"\w+", re.MULTILINE)


def _parse_args_from_docstring(docstring: str) -> list[str]:
    in_args_section = False
    args_header_indent: int | None = None
    args: set[str] = set()
    for line in docstring.splitlines():
        if in_args_section:
            # Are we still in the args section?
            indent = _get_indent_level(line)
            if indent is not None and indent <= args_header_indent:
                break

            if m := _PARAM_NAME_REGEX.match(line.lstrip()):
                args.add(m.group(0))
        else:
            if line.lstrip().startswith("Args:"):
                in_args_section = True
                args_header_indent = _get_indent_level(line)

    return args


def _parse_args_from_node(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = {a.arg for a in node.args.args}.union({a.arg for a in node.args.kwonlyargs})

    if node.args.vararg:
        args.add(node.args.vararg.arg)
    if node.args.kwarg:
        args.add(node.args.kwarg.arg)

    args = args.difference({"self", "cls"})


def _undocumented_params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str] | None:
    if (ds := ast.get_docstring(node)) and (doc_args := _parse_args_from_docstring(ds)):
        if func_args := _parse_args_from_node(node):
            if diff := (func_args.difference(doc_args)):
                return diff
    return None


class Linter(ast.NodeVisitor):
    def __init__(self, path: Path, ignore: dict[str, set[int]]):
        self.stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.path = path
        self.ignore = ignore
        self.violations: list[Violation] = []
        self.parents: list[ast.AST] = []

    def _check(self, node: ast.AST, rule: Rule) -> None:
        if (lines := self.ignore.get(rule.name)) and node.lineno in lines:
            return
        self.violations.append(Violation(rule, self.path, node.lineno, node.col_offset))

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
            self._check(nd, NO_RST)

    def _is_in_function(self) -> bool:
        return bool(self.stack)

    def _test_name_typo(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if not self.path.name.startswith("test_") or self._is_in_function():
            return

        if node.name.startswith("test") and not node.name.startswith("test_"):
            self._check(node, TEST_NAME_TYPO)

    def _mlflow_class_name(self, node: ast.ClassDef) -> None:
        if "MLflow" in node.name or "MLFlow" in node.name:
            self._check(node, MLFLOW_CLASS_NAME)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node)
        self.parents.append(node)
        self._no_rst(node)
        self._mlflow_class_name(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._test_name_typo(node)
        if args := _undocumented_params(node):
            self._check(node, UNDOCUMENTED_PARAMS)

        self.stack.append(node)
        self._no_rst(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._test_name_typo(node)
        self.stack.append(node)
        self._no_rst(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        if self._is_in_function():
            for alias in node.names:
                if alias.name.split(".", 1)[0] in BUILTIN_MODULES:
                    self._check(node, LAZY_BUILTIN_IMPORT)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._is_in_function() and node.module.split(".", 1)[0] in BUILTIN_MODULES:
            self._check(node, LAZY_BUILTIN_IMPORT)
        self.generic_visit(node)


def lint_file(path: Path) -> list[Violation]:
    with open(path) as f:
        code = f.read()
        linter = Linter(path, ignore_map(code))
        linter.visit(ast.parse(code))
        return linter.violations

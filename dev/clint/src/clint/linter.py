from __future__ import annotations

import ast
import re
import tokenize
from dataclasses import dataclass

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


@dataclass
class Violation:
    rule: Rule
    path: str
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
            "path": self.path,
            "symbol": self.rule.name,
            "message": self.rule.message,
            "message-id": self.rule.id,
        }


NO_RST = Rule(
    "Z0001",
    "no-rst",
    "Do not use RST style. Use Google style instead.",
)
LAZY_BUILTIN_IMPORT = Rule(
    "Z0002",
    "lazy-builtin-import",
    "Builtin modules must be imported at the top level.",
)


class Linter(ast.NodeVisitor):
    def __init__(self, path: str, ignore: dict[str, set[int]]):
        self.stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.path = path
        self.ignore = ignore
        self.violations: list[Violation] = []

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

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node)
        self._no_rst(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append(node)
        self._no_rst(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
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


def lint_file(path: str) -> list[Violation]:
    with open(path) as f:
        code = f.read()
        linter = Linter(path, ignore_map(code))
        linter.visit(ast.parse(code))
        return linter.violations

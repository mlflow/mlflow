from __future__ import annotations

import ast
import fnmatch
import functools
import json
import re
import textwrap
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Union

from clint import rules
from clint.builtin import BUILTIN_MODULES
from clint.config import Config

PARAM_REGEX = re.compile(r"\s+:param\s+\w+:", re.MULTILINE)
RETURN_REGEX = re.compile(r"\s+:returns?:", re.MULTILINE)
DISABLE_COMMENT_REGEX = re.compile(r"clint:\s*disable=([a-z0-9-]+)")
MARKDOWN_LINK_RE = re.compile(r"\[.+\]\(.+\)")


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
            mapping.setdefault(m.group(1), set()).add(tok.start[0] - 1)
    return mapping


def _is_set_active_model(node: ast.AST) -> bool:
    """
    Is this node a call to `set_active_model`?
    """
    if isinstance(node, ast.Name):
        return "set_active_model" == node.id

    elif isinstance(node, ast.Attribute):
        return "set_active_model" == node.attr

    return False


@dataclass
class Violation:
    rule: rules.Rule
    path: Path
    loc: Location
    cell: int | None = None

    def __str__(self):
        # Use the same format as ruff
        cell_loc = f"cell {self.cell}:" if self.cell is not None else ""
        return (
            # Since `Location` is 0-indexed, lineno and col_offset are incremented by 1
            f"{self.path}:{cell_loc}{self.loc + Location(1, 1)}: "
            f"{self.rule.id}: {self.rule.message} "
            f"See dev/clint/README.md for instructions on ignoring this rule ({self.rule.name})."
        )

    def json(self) -> dict[str, str | int | None]:
        return {
            "type": "error",
            "module": None,
            "obj": None,
            "line": self.loc.lineno,
            "column": self.loc.col_offset,
            "endLine": self.loc.lineno,
            "endColumn": self.loc.col_offset,
            "path": str(self.path),
            "symbol": self.rule.name,
            "message": self.rule.message,
            "message-id": self.rule.id,
        }


@dataclass
class Location:
    lineno: int
    col_offset: int

    def __str__(self):
        return f"{self.lineno}:{self.col_offset}"

    @classmethod
    def from_node(cls, node: ast.AST) -> "Location":
        return cls(node.lineno - 1, node.col_offset)

    def __add__(self, other: Location) -> Location:
        return Location(self.lineno + other.lineno, self.col_offset + other.col_offset)


@dataclass
class CodeBlock:
    code: str
    loc: Location


def _get_indent(s: str) -> int:
    return len(s) - len(s.lstrip())


_CODE_BLOCK_HEADER_REGEX = re.compile(r"^\.\.\s+code-block::\s*py(thon)?")
_CODE_BLOCK_OPTION_REGEX = re.compile(r"^:\w+:")


def _get_header_indent(s: str) -> int | None:
    if _CODE_BLOCK_HEADER_REGEX.match(s.lstrip()):
        return _get_indent(s)
    return None


def _iter_code_blocks(s: str) -> Iterator[CodeBlock]:
    code_block_loc: Location | None = None
    header_indent: int | None = None
    code_lines: list[str] = []
    line_iter = enumerate(s.splitlines())
    while t := next(line_iter, None):
        idx, line = t
        if code_block_loc:
            indent = _get_indent(line)
            # If we encounter a non-blank line with an indent less than the code block header
            # we are done parsing the code block. Here's an example:
            #
            # .. code-block:: python
            #
            #     print("hello")     # indent > header_indent
            #                        # blank
            # <non-blank>            # non-blank and indent <= header_indent
            if line.strip() and indent <= header_indent:
                code = textwrap.dedent("\n".join(code_lines))
                yield CodeBlock(code=code, loc=code_block_loc)

                code_block_loc = None
                code_lines.clear()
                # It's possible that another code block follows the current one
                header_indent = _get_header_indent(line)
                continue

            code_lines.append(line)

        elif header_indent is not None:
            # Advance the iterator to the code body
            #
            # .. code-block:: python
            #     :option:            # we're here
            #     :another-option:    # skip
            #                         # skip
            #     import mlflow       # stop here
            #     ...
            while True:
                if line.strip() and not _CODE_BLOCK_OPTION_REGEX.match(line.lstrip()):
                    # We are at the first line of the code block
                    code_lines.append(line)
                    break
                idx, line = next(line_iter, (None, None))

            code_block_loc = Location(idx, _get_indent(line))
        else:
            header_indent = _get_header_indent(line)

    # The docstring ends with a code block
    if code_lines:
        code = textwrap.dedent("\n".join(code_lines))
        yield CodeBlock(code=code, loc=code_block_loc)


_MD_OPENING_FENCE_REGEX = re.compile(r"^(`{3,})\s*python\s*$")


def _iter_md_code_blocks(s: str) -> Iterator[CodeBlock]:
    """
    Iterates over code blocks in a Markdown string.
    """
    code_block_loc: Location | None = None
    code_lines: list[str] = []
    closing_fence: str | None = None
    line_iter = enumerate(s.splitlines())
    while t := next(line_iter, None):
        idx, line = t
        if code_block_loc:
            if line.strip() == closing_fence:
                code = textwrap.dedent("\n".join(code_lines))
                yield CodeBlock(code=code, loc=code_block_loc)

                code_block_loc = None
                code_lines.clear()
                closing_fence = None
                continue

            code_lines.append(line)

        elif m := _MD_OPENING_FENCE_REGEX.match(line.lstrip()):
            closing_fence = m.group(1)
            code_block_loc = Location(idx + 1, _get_indent(line))

    # Code block at EOF
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


class _ArtifactPathFinder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.artifact_path_index: int | None = None

    @classmethod
    def parse(cls, file: Path) -> int | None:
        v = cls()
        v.visit(ast.parse(file.read_text()))
        return v.artifact_path_index

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == "log_model":
            artifact_path_index = next(
                (i for i, arg in enumerate(node.args.args) if arg.arg == "artifact_path"), None
            )
            if artifact_path_index is not None:
                self.artifact_path_index = artifact_path_index


@functools.lru_cache(maxsize=32)
def _find_artifact_path_index(call_path: tuple[str, str]) -> int | None:
    """
    Finds the index of the `artifact_path` argument in the function signature of `log_model`.
    """
    for p in Path(*call_path).rglob("*.py"):
        if not p.is_file():
            continue

        if (idx := _ArtifactPathFinder.parse(p)) is not None:
            return idx
    return None


class Linter(ast.NodeVisitor):
    def __init__(
        self,
        *,
        path: Path,
        config: Config,
        ignore: dict[str, set[int]],
        cell: int | None = None,
        offset: Location | None = None,
    ):
        """
        Lints a Python file.

        Args:
            path: Path to the file being linted.
            config: Linter configuration declared within the pyproject.toml file.
            ignore: Mapping of rule name to line numbers to ignore.
            cell: Index of the cell being linted in a Jupyter notebook.
            offset: Offset to apply to the line and column numbers of the violations.
        """
        self.stack: list[ast.AST] = []
        self.path = path
        self.config = config
        self.ignore = ignore
        self.cell = cell
        self.violations: list[Violation] = []
        self.in_type_annotation = False
        self.in_TYPE_CHECKING = False
        self.is_mlflow_init_py = path == Path("mlflow", "__init__.py")
        self.imported_modules: set[str] = set()
        self.lazy_modules: dict[str, Location] = {}
        self.offset = offset or Location(0, 0)

    def _check(self, loc: Location, rule: rules.Rule) -> None:
        if (lines := self.ignore.get(rule.name)) and loc.lineno in lines:
            return
        self.violations.append(
            Violation(
                rule,
                self.path,
                loc + self.offset,
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
        if (n := self._docstring(node)) and (PARAM_REGEX.search(n.s) or RETURN_REGEX.search(n.s)):
            self._check(Location.from_node(n), rules.NoRst())

    def _is_in_function(self) -> bool:
        return self.stack and isinstance(self.stack[-1], (ast.FunctionDef, ast.AsyncFunctionDef))

    def _is_in_class(self) -> bool:
        return self.stack and isinstance(self.stack[-1], ast.ClassDef)

    def _is_at_top_level(self) -> bool:
        return not self.stack

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
                if "cls" in args:
                    args.remove("cls")
            elif any(
                isinstance(d, ast.Name) and d.id == "staticmethod" for d in func.decorator_list
            ):
                pass
            else:  # Instance method
                if "self" in args:
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

    def _is_in_test(self) -> bool:
        return (
            self.path.name.startswith("test_")
            and self.stack
            and self.stack[-1].name.startswith("test_")
        )

    @classmethod
    def visit_example(cls, path: Path, config: Config, example: CodeBlock) -> list[Violation]:
        try:
            tree = ast.parse(example.code)
        except SyntaxError:
            return [Violation(rules.ExampleSyntaxError(), path, example.loc)]

        linter = cls(
            path=path,
            config=config,
            ignore=ignore_map(example.code),
            offset=example.loc,
        )
        linter.visit(tree)
        return [v for v in linter.violations if v.rule.name in config.example_rules]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node)
        self._no_rst(node)
        self._syntax_error_example(node)
        self._mlflow_class_name(node)
        self.generic_visit(node)
        self.stack.pop()

    def _syntax_error_example(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> None:
        if docstring_node := self._docstring(node):
            for code_block in _iter_code_blocks(docstring_node.value):
                code_block.loc.lineno += docstring_node.lineno - 1
                self.violations.extend(Linter.visit_example(self.path, self.config, code_block))

    def _param_mismatch(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # TODO: Remove this guard clause to enforce the docstring param checks for all functions
        if node.name.startswith("_"):
            return
        if docstring_node := self._docstring(node):
            if (doc_args := _parse_docstring_args(docstring_node.value)) and (
                func_args := self._parse_func_args(node)
            ):
                func_args_set = set(func_args)
                doc_args_set = set(doc_args)
                if diff := func_args_set - doc_args_set:
                    self._check(Location.from_node(node), rules.MissingDocstringParam(diff))

                if diff := doc_args_set - func_args_set:
                    self._check(Location.from_node(node), rules.ExtraneousDocstringParam(diff))

                if func_args_set == doc_args_set and func_args != doc_args:
                    params = [a for a, b in zip(func_args, doc_args) if a != b]
                    self._check(Location.from_node(node), rules.DocstringParamOrder(params))

    def _invalid_abstract_method(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if rules.InvalidAbstractMethod.check(node):
            self._check(Location.from_node(node), rules.InvalidAbstractMethod())

    def visit_Name(self, node) -> None:
        if self.in_type_annotation and rules.IncorrectTypeAnnotation.check(node):
            self._check(Location.from_node(node), rules.IncorrectTypeAnnotation(node.id))

        self.generic_visit(node)

    def _markdown_link(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if docstring := self._docstring(node):
            if MARKDOWN_LINK_RE.search(docstring.s):
                self._check(docstring, rules.MarkdownLink())

    def _pytest_mark_repeat(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Only check in test files
        if not self.path.name.startswith("test_"):
            return

        if rules.PytestMarkRepeat.check(node):
            self._check(Location.from_node(node), rules.PytestMarkRepeat())

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._test_name_typo(node)
        self._syntax_error_example(node)
        self._param_mismatch(node)
        self._markdown_link(node)
        self._invalid_abstract_method(node)
        self._pytest_mark_repeat(node)

        for arg in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
            if arg.annotation:
                self.visit_type_annotation(arg.annotation)

        if node.returns:
            self.visit_type_annotation(node.returns)

        self.stack.append(node)
        self._no_rst(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._test_name_typo(node)
        self._syntax_error_example(node)
        self._param_mismatch(node)
        self._markdown_link(node)
        self._invalid_abstract_method(node)
        self._pytest_mark_repeat(node)
        self.stack.append(node)
        self._no_rst(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root_module = alias.name.split(".", 1)[0]
            if self._is_in_function() and root_module in BUILTIN_MODULES:
                self._check(Location.from_node(node), rules.LazyBuiltinImport())

            if (
                alias.name.split(".", 1)[0] == "typing_extensions"
                and alias.name not in self.config.typing_extensions_allowlist
            ):
                self._check(
                    Location.from_node(node),
                    rules.TypingExtensions(
                        full_name=alias.name,
                        allowlist=self.config.typing_extensions_allowlist,
                    ),
                )

            if self._is_at_top_level() and not self.in_TYPE_CHECKING:
                self._check_forbidden_top_level_import(node, root_module)

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root_module = node.module and node.module.split(".", 1)[0]
        if self._is_in_function() and root_module in BUILTIN_MODULES:
            self._check(Location.from_node(node), rules.LazyBuiltinImport())

        if self.in_TYPE_CHECKING and self.is_mlflow_init_py:
            for alias in node.names:
                self.imported_modules.add(f"{node.module}.{alias.name}")

        if root_module == "typing_extensions":
            for alias in node.names:
                full_name = f"{node.module}.{alias.name}"
                if full_name not in self.config.typing_extensions_allowlist:
                    self._check(
                        Location.from_node(node),
                        rules.TypingExtensions(
                            full_name=full_name,
                            allowlist=self.config.typing_extensions_allowlist,
                        ),
                    )

        if self._is_at_top_level() and not self.in_TYPE_CHECKING:
            self._check_forbidden_top_level_import(node, node.module)

        if self.path.parts[0] != "tests" and not self.is_mlflow_init_py:
            for alias in node.names:
                if alias.name.split(".")[-1] == "set_active_model":
                    self._check_forbidden_set_active_model_usage(node)

        self.generic_visit(node)

    def _check_forbidden_top_level_import(
        self, node: Union[ast.Import, ast.ImportFrom], module: str
    ) -> None:
        for file_pat, libs in self.config.forbidden_top_level_imports.items():
            if fnmatch.fnmatch(str(self.path), file_pat) and any(
                module.startswith(lib) for lib in libs
            ):
                self._check(
                    Location.from_node(node),
                    rules.ForbiddenTopLevelImport(module=module),
                )

    def _check_forbidden_set_active_model_usage(
        self,
        node: Union[ast.Import, ast.ImportFrom],
    ) -> None:
        self._check(
            Location.from_node(node),
            rules.ForbiddenSetActiveModelUsage(),
        )

    def _resolve_call(self, node: ast.AST) -> list[str]:
        if isinstance(node, ast.Call):
            return self._resolve_call(node.func)
        elif isinstance(node, ast.Attribute):
            return self._resolve_call(node.value) + [node.attr]
        elif isinstance(node, ast.Name):
            return [node.id]
        return []

    def _log_model_with_artifact_path(self, node: ast.Call) -> bool:
        """
        Returns True if the call looks like `mlflow.<flavor>.log_model(...)` and
        the `artifact_path` argument is specified.
        """
        parts = self._resolve_call(node)
        if len(parts) != 3:
            return False

        first, second, third = parts
        if not (first == "mlflow" and third == "log_model"):
            return False

        artifact_path_idx = _find_artifact_path_index((first, second))
        if artifact_path_idx is None:
            return False

        if len(node.args) > artifact_path_idx:
            return True
        else:
            return any(kw.arg and kw.arg == "artifact_path" for kw in node.keywords)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            self.is_mlflow_init_py
            and isinstance(node.func, ast.Name)
            and node.func.id == "LazyLoader"
        ):
            last_arg = node.args[-1]
            if (
                isinstance(last_arg, ast.Constant)
                and isinstance(last_arg.value, str)
                and last_arg.value.startswith("mlflow.")
            ):
                self.lazy_modules[last_arg.value] = Location.from_node(node)

        if self._log_model_with_artifact_path(node):
            self._check(Location.from_node(node), rules.LogModelArtifactPath())

        if rules.UseSysExecutable.check(node):
            self._check(Location.from_node(node), rules.UseSysExecutable())

        if self.path.parts[0] != "tests" and _is_set_active_model(node.func):
            self._check(Location.from_node(node), rules.ForbiddenSetActiveModelUsage())

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if rules.ImplicitOptional.check(node):
            self._check(Location.from_node(node), rules.ImplicitOptional())

        if node.annotation:
            self.visit_type_annotation(node.annotation)

        self.generic_visit(node)

    @staticmethod
    def _is_os_environ(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr == "environ"
        )

    def visit_Assign(self, node: ast.Assign):
        if self._is_in_test():
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and self._is_os_environ(node.targets[0].value)
            ):
                self._check(Location.from_node(node), rules.OsEnvironSetInTest())

        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete):
        if self._is_in_test():
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and self._is_os_environ(node.targets[0].value)
            ):
                self._check(Location.from_node(node), rules.OsEnvironDeleteInTest())

        self.generic_visit(node)

    def visit_type_annotation(self, node: ast.AST) -> None:
        self.in_type_annotation = True
        self.visit(node)
        self.in_type_annotation = False

    def visit_If(self, node: ast.If) -> None:
        if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            self.in_TYPE_CHECKING = True
        self.generic_visit(node)
        self.in_TYPE_CHECKING = False

    def post_visit(self) -> None:
        if self.is_mlflow_init_py and (diff := self.lazy_modules.keys() - self.imported_modules):
            for mod in diff:
                if loc := self.lazy_modules.get(mod):
                    self._check(loc, rules.LazyModule())


def _has_trace_ui_content(output: dict[str, Any]) -> bool:
    """Check if an output contains MLflow trace UI content."""
    data = output.get("data")
    if not data:
        return False

    # Check only HTML outputs since trace UI content is only added to text/html
    html = data.get("text/html")
    if not html:
        return False

    return any("static-files/lib/notebook-trace-renderer/index.html" in line for line in html)


def _lint_cell(path: Path, config: Config, cell: dict[str, Any], index: int) -> list[Violation]:
    violations: list[Violation] = []
    type_ = cell.get("cell_type")

    # Check for forbidden trace UI iframe in cell outputs
    if outputs := cell.get("outputs"):
        for output in outputs:
            if _has_trace_ui_content(output):
                violations.append(
                    Violation(
                        rules.ForbiddenTraceUIInNotebook(),
                        path,
                        Location(0, 0),
                        cell=index,
                    )
                )
                break

    if type_ != "code":
        return violations

    src = "\n".join(cell.get("source", []))
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # Ignore non-python cells such as `!pip install ...`
        return violations

    linter = Linter(path=path, config=config, ignore=ignore_map(src), cell=index)
    linter.visit(tree)
    violations.extend(linter.violations)

    if not src.strip():
        violations.append(
            Violation(
                rules.EmptyNotebookCell(),
                path,
                Location(0, 0),
                cell=index,
            )
        )
    return violations


def lint_file(path: Path, config: Config) -> list[Violation]:
    code = path.read_text()
    if path.suffix == ".ipynb":
        violations = []
        if cells := json.loads(code).get("cells"):
            for idx, cell in enumerate(cells, start=1):
                violations.extend(_lint_cell(path, config, cell, idx))
        return violations
    elif path.suffix in {".rst", ".md", ".mdx"}:
        violations = []
        code_blocks = (
            _iter_code_blocks(code) if path.suffix == ".rst" else _iter_md_code_blocks(code)
        )
        for code_block in code_blocks:
            violations.extend(Linter.visit_example(path, config, code_block))
        return violations
    else:
        linter = Linter(path=path, config=config, ignore=ignore_map(code))
        linter.visit(ast.parse(code))
        linter.post_visit()
        return linter.violations

import ast
import fnmatch
import json
import re
import textwrap
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, TypeAlias

from typing_extensions import Self

from clint import rules
from clint.builtin import BUILTIN_MODULES
from clint.comments import Noqa, iter_comments
from clint.config import Config
from clint.index import SymbolIndex
from clint.resolver import Resolver
from clint.utils import get_ignored_rules_for_file

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


HasLocation: TypeAlias = (
    ast.expr | ast.stmt | ast.alias | ast.arg | ast.keyword | ast.excepthandler | ast.pattern
)


@dataclass
class Location:
    lineno: int
    col_offset: int

    def __str__(self) -> str:
        return f"{self.lineno}:{self.col_offset}"

    @classmethod
    def from_node(cls, node: HasLocation) -> Self:
        return cls(node.lineno - 1, node.col_offset)

    @classmethod
    def from_noqa(cls, noqa: Noqa) -> Self:
        return cls(noqa.lineno - 1, noqa.col_offset)

    def __add__(self, other: "Location") -> "Location":
        return Location(self.lineno + other.lineno, self.col_offset + other.col_offset)


@dataclass
class Violation:
    rule: rules.Rule
    path: Path
    loc: Location
    cell: int | None = None

    def __str__(self) -> str:
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
            if line.strip() and (header_indent is not None) and indent <= header_indent:
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
                if next_line := next(line_iter, None):
                    idx, line = next_line

            code_block_loc = Location(idx, _get_indent(line))
        else:
            header_indent = _get_header_indent(line)

    # The docstring ends with a code block
    if code_lines and code_block_loc:
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
    if code_lines and code_block_loc:
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


class ExampleVisitor(ast.NodeVisitor):
    def __init__(self, linter: "Linter", index: SymbolIndex) -> None:
        self.linter = linter
        self.index = index
        self.has_log_model = False

    def visit_Call(self, node: ast.Call) -> None:
        if names := self.linter.resolver.resolve(node.func):
            match names:
                case ["mlflow", *_, "log_model"]:
                    self.has_log_model = True
                case ["mlflow", "get_artifact_uri"] if self.has_log_model and len(node.args) == 1:
                    self.linter._check(Location.from_node(node), rules.GetArtifactUri())

        if (
            (resolved := self.linter.resolver.resolve(node.func))
            and resolved[0] == "mlflow"
            # Skip `mlflow.data` because its methods are dynamically created and cannot be checked
            # statically
            and resolved[:2] != ["mlflow", "data"]
            # Skip `mlflow.txtai` because it is provided by the external `mlflow-txtai` package and
            # cannot be checked statically
            and resolved[:2] != ["mlflow", "txtai"]
        ):
            function_name = ".".join(resolved)
            if func_def := self.index.resolve(function_name):
                if not (func_def.has_vararg or func_def.has_kwarg):
                    # Get all argument names from the function signature
                    all_args = func_def.args + func_def.kwonlyargs + func_def.posonlyargs
                    # Skip positional arguments that are already provided
                    remaining_args = all_args[len(node.args) :]
                    sig_args = set(remaining_args)
                    call_args = {kw.arg for kw in node.keywords if kw.arg}
                    if diff := call_args - sig_args:
                        self.linter._check(
                            Location.from_node(node),
                            rules.UnknownMlflowArguments(function_name, diff),
                        )
            else:
                self.linter._check(
                    Location.from_node(node), rules.UnknownMlflowFunction(function_name)
                )
        self.generic_visit(node)


class TypeAnnotationVisitor(ast.NodeVisitor):
    def __init__(self, linter: "Linter") -> None:
        self.linter = linter
        self.stack: list[ast.AST] = []

    def visit(self, node: ast.AST) -> None:
        self.stack.append(node)
        super().visit(node)
        self.stack.pop()

    def visit_Name(self, node: ast.Name) -> None:
        if rules.IncorrectTypeAnnotation.check(node):
            self.linter._check(Location.from_node(node), rules.IncorrectTypeAnnotation(node.id))

        if self._is_bare_generic_type(node):
            self.linter._check(Location.from_node(node), rules.UnparameterizedGenericType(node.id))

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self._is_bare_generic_type(node):
            self.linter._check(
                Location.from_node(node), rules.UnparameterizedGenericType(ast.unparse(node))
            )

        self.generic_visit(node)

    def _is_bare_generic_type(self, node: ast.Name | ast.Attribute) -> bool:
        """Check if this node is a bare generic type (e.g., `dict` or `list` without parameters)."""
        if not rules.UnparameterizedGenericType.is_generic_type(node, self.linter.resolver):
            return False

        # Check if this node is the value of a Subscript (e.g., the 'dict' in 'dict[str, int]').
        # `[:-1]` skips the current node, which is the one being checked.
        for parent in reversed(self.stack[:-1]):
            if isinstance(parent, ast.Subscript) and parent.value is node:
                return False
        return True


class Linter(ast.NodeVisitor):
    def __init__(
        self,
        *,
        path: Path,
        config: Config,
        ignore: dict[str, set[int]],
        index: SymbolIndex,
        cell: int | None = None,
        offset: Location | None = None,
    ) -> None:
        """
        Lints a Python file.

        Args:
            path: Path to the file being linted.
            config: Linter configuration declared within the pyproject.toml file.
            ignore: Mapping of rule name to line numbers to ignore.
            cell: Index of the cell being linted in a Jupyter notebook.
            offset: Offset to apply to the line and column numbers of the violations.
            index: Symbol index for resolving function signatures.
        """
        self.stack: list[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef] = []
        self.path = path
        self.config = config
        self.ignore = ignore
        self.cell = cell
        self.violations: list[Violation] = []
        self.in_TYPE_CHECKING = False
        self.is_mlflow_init_py = path == Path("mlflow", "__init__.py")
        self.imported_modules: set[str] = set()
        self.lazy_modules: dict[str, Location] = {}
        self.offset = offset or Location(0, 0)
        self.resolver = Resolver()
        self.index = index
        self.ignored_rules = get_ignored_rules_for_file(path, config.per_file_ignores)

    def _check(self, loc: Location, rule: rules.Rule) -> None:
        # Skip rules that are not selected in the config
        if rule.name not in self.config.select:
            return
        # Check line-level ignores
        if (lines := self.ignore.get(rule.name)) and loc.lineno in lines:
            return
        # Check per-file ignores
        if rule.name in self.ignored_rules:
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

    def _no_rst(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        if (
            (n := self._docstring(node))
            and isinstance(n.value, str)
            and (PARAM_REGEX.search(n.value) or RETURN_REGEX.search(n.value))
        ):
            self._check(Location.from_node(n), rules.NoRst())

    def _is_in_function(self) -> bool:
        if self.stack:
            return isinstance(self.stack[-1], (ast.FunctionDef, ast.AsyncFunctionDef))
        return False

    def _is_in_class(self) -> bool:
        if self.stack:
            return isinstance(self.stack[-1], ast.ClassDef)
        return False

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

    def _no_class_based_tests(self, node: ast.ClassDef) -> None:
        if rule := rules.NoClassBasedTests.check(node, self.path.name):
            self._check(Location.from_node(node), rule)

    def _redundant_test_docstring(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> None:
        if rule := rules.RedundantTestDocstring.check(node, self.path.name):
            self._check(Location.from_node(node), rule)

    def visit_Module(self, node: ast.Module) -> None:
        if rule := rules.RedundantTestDocstring.check_module(node, self.path.name):
            self._check(Location(0, 0), rule)
        self.generic_visit(node)

    def _is_in_test(self) -> bool:
        if not self.path.name.startswith("test_"):
            return False

        if not self.stack:
            return False

        return self.stack[-1].name.startswith("test_")

    @classmethod
    def visit_example(
        cls, path: Path, config: Config, example: CodeBlock, index: SymbolIndex
    ) -> list[Violation]:
        try:
            tree = ast.parse(example.code)
        except SyntaxError:
            return [Violation(rules.ExampleSyntaxError(), path, example.loc)]

        linter = cls(
            path=path,
            config=config,
            ignore=ignore_map(example.code),
            index=index,
            offset=example.loc,
        )
        linter.visit(tree)
        linter.visit_comments(example.code)
        if index:
            v = ExampleVisitor(linter, index)
            v.visit(tree)
        return [v for v in linter.violations if v.rule.name in config.example_rules]

    def visit_decorators(self, decorator_list: list[ast.expr]) -> None:
        for decorator in decorator_list:
            if rules.InvalidExperimentalDecorator.check(decorator, self.resolver):
                self._check(Location.from_node(decorator), rules.InvalidExperimentalDecorator())

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node)
        self._no_rst(node)
        self._syntax_error_example(node)
        self._mlflow_class_name(node)
        self._no_class_based_tests(node)
        self._redundant_test_docstring(node)
        self.visit_decorators(node.decorator_list)
        self._markdown_link(node)
        with self.resolver.scope():
            self.generic_visit(node)
        self.stack.pop()

    def _syntax_error_example(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> None:
        if node.name.startswith("_"):
            return
        if (docstring_node := self._docstring(node)) and isinstance(docstring_node.value, str):
            for code_block in _iter_code_blocks(docstring_node.value):
                code_block.loc.lineno += docstring_node.lineno - 1
                self.violations.extend(
                    Linter.visit_example(self.path, self.config, code_block, self.index)
                )

    def _param_mismatch(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # TODO: Remove this guard clause to enforce the docstring param checks for all functions
        if node.name.startswith("_"):
            return
        if (docstring_node := self._docstring(node)) and isinstance(docstring_node.value, str):
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
        if rules.InvalidAbstractMethod.check(node, self.resolver):
            self._check(Location.from_node(node), rules.InvalidAbstractMethod())

    def visit_Name(self, node: ast.Name) -> None:
        self.generic_visit(node)

    def _markdown_link(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        if (docstring := self._docstring(node)) and isinstance(docstring.value, str):
            if MARKDOWN_LINK_RE.search(docstring.value):
                self._check(Location.from_node(docstring), rules.MarkdownLink())

    def _pytest_mark_repeat(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Only check in test files
        if not self.path.name.startswith("test_"):
            return

        if deco := rules.PytestMarkRepeat.check(node.decorator_list, self.resolver):
            self._check(Location.from_node(deco), rules.PytestMarkRepeat())

    def _mock_patch_as_decorator(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Only check in test files
        if not self.path.name.startswith("test_"):
            return

        # Check all decorators, not just the first one
        for deco in node.decorator_list:
            if rules.MockPatchAsDecorator.check([deco], self.resolver):
                self._check(Location.from_node(deco), rules.MockPatchAsDecorator())

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._test_name_typo(node)
        self._syntax_error_example(node)
        self._param_mismatch(node)
        self._markdown_link(node)
        self._invalid_abstract_method(node)
        self._pytest_mark_repeat(node)
        self._mock_patch_as_decorator(node)
        self._redundant_test_docstring(node)

        for arg in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
            if arg.annotation:
                self.visit_type_annotation(arg.annotation)

        if node.returns:
            self.visit_type_annotation(node.returns)

        self.stack.append(node)
        self._no_rst(node)
        self.visit_decorators(node.decorator_list)
        with self.resolver.scope():
            self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._test_name_typo(node)
        self._syntax_error_example(node)
        self._param_mismatch(node)
        self._markdown_link(node)
        self._invalid_abstract_method(node)
        self._pytest_mark_repeat(node)
        self._mock_patch_as_decorator(node)
        self._redundant_test_docstring(node)
        self.stack.append(node)
        self._no_rst(node)
        self.visit_decorators(node.decorator_list)
        with self.resolver.scope():
            self.generic_visit(node)
        self.stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        self.resolver.add_import(node)
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
        self.resolver.add_import_from(node)

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

        if node.module and self._is_at_top_level() and not self.in_TYPE_CHECKING:
            self._check_forbidden_top_level_import(node, node.module)

        if not self.is_mlflow_init_py:
            for alias in node.names:
                if alias.name.split(".")[-1] == "set_active_model":
                    self._check_forbidden_set_active_model_usage(node)

        self.generic_visit(node)

    def _check_forbidden_top_level_import(
        self, node: ast.Import | ast.ImportFrom, module: str
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
        node: ast.Import | ast.ImportFrom,
    ) -> None:
        self._check(
            Location.from_node(node),
            rules.ForbiddenSetActiveModelUsage(),
        )

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

        if rules.LogModelArtifactPath.check(node, self.index):
            self._check(Location.from_node(node), rules.LogModelArtifactPath())

        if rules.UseSysExecutable.check(node, self.resolver):
            self._check(Location.from_node(node), rules.UseSysExecutable())

        if rules.ForbiddenSetActiveModelUsage.check(node, self.resolver):
            self._check(Location.from_node(node), rules.ForbiddenSetActiveModelUsage())

        if expr := rules.ForbiddenDeprecationWarning.check(node, self.resolver):
            self._check(Location.from_node(expr), rules.ForbiddenDeprecationWarning())

        if rules.UnnamedThread.check(node, self.resolver):
            self._check(Location.from_node(node), rules.UnnamedThread())

        if rules.ThreadPoolExecutorWithoutThreadNamePrefix.check(node, self.resolver):
            self._check(Location.from_node(node), rules.ThreadPoolExecutorWithoutThreadNamePrefix())

        if rules.IsinstanceUnionSyntax.check(node):
            self._check(Location.from_node(node), rules.IsinstanceUnionSyntax())

        if self._is_in_test() and rules.OsChdirInTest.check(node, self.resolver):
            self._check(Location.from_node(node), rules.OsChdirInTest())

        if self._is_in_test() and rules.TempDirInTest.check(node, self.resolver):
            self._check(Location.from_node(node), rules.TempDirInTest())

        if self._is_in_test() and rules.MockPatchDictEnviron.check(node, self.resolver):
            self._check(Location.from_node(node), rules.MockPatchDictEnviron())

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if rules.ImplicitOptional.check(node):
            self._check(Location.from_node(node.annotation), rules.ImplicitOptional())

        if node.annotation:
            self.visit_type_annotation(node.annotation)

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._is_in_test() and rules.OsEnvironSetInTest.check(node, self.resolver):
            self._check(Location.from_node(node), rules.OsEnvironSetInTest())

        if rules.MultiAssign.check(node):
            self._check(Location.from_node(node), rules.MultiAssign())

        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        if self._is_in_test() and rules.OsEnvironDeleteInTest.check(node, self.resolver):
            self._check(Location.from_node(node), rules.OsEnvironDeleteInTest())
        self.generic_visit(node)

    def visit_type_annotation(self, node: ast.expr) -> None:
        visitor = TypeAnnotationVisitor(self)
        visitor.visit(node)

    def visit_If(self, node: ast.If) -> None:
        if (resolved := self.resolver.resolve(node.test)) and resolved == [
            "typing",
            "TYPE_CHECKING",
        ]:
            self.in_TYPE_CHECKING = True
        self.generic_visit(node)
        self.in_TYPE_CHECKING = False

    def visit_With(self, node: ast.With) -> None:
        # Only check in test files
        if self.path.name.startswith("test_") and rules.NestedMockPatch.check(node, self.resolver):
            self._check(Location.from_node(node), rules.NestedMockPatch())
        self.generic_visit(node)

    def post_visit(self) -> None:
        if self.is_mlflow_init_py and (diff := self.lazy_modules.keys() - self.imported_modules):
            for mod in diff:
                if loc := self.lazy_modules.get(mod):
                    self._check(loc, rules.LazyModule())

    def visit_comments(self, src: str) -> None:
        for comment in iter_comments(src):
            if noqa := Noqa.from_token(comment):
                self.visit_noqa(noqa)

    def visit_noqa(self, noqa: Noqa) -> None:
        if rule := rules.DoNotDisable.check(noqa.rules):
            self._check(Location.from_noqa(noqa), rule)

    def visit_file_content(self, src: str) -> None:
        if rules.NoShebang.check(src):
            self._check(Location(0, 0), rules.NoShebang())


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


def _lint_cell(
    path: Path,
    config: Config,
    cell: dict[str, Any],
    cell_index: int,
    index: SymbolIndex,
) -> list[Violation]:
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
                        cell=cell_index,
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

    linter = Linter(path=path, config=config, ignore=ignore_map(src), index=index, cell=cell_index)
    linter.visit(tree)
    linter.visit_comments(src)
    violations.extend(linter.violations)

    if not src.strip():
        violations.append(
            Violation(
                rules.EmptyNotebookCell(),
                path,
                Location(0, 0),
                cell=cell_index,
            )
        )
    return violations


def _has_h1_header(cells: list[dict[str, Any]]) -> bool:
    return any(
        line.strip().startswith("# ")
        for cell in cells
        if cell.get("cell_type") == "markdown"
        for line in cell.get("source", [])
    )


def lint_file(path: Path, code: str, config: Config, index_path: Path) -> list[Violation]:
    if path.is_absolute():
        raise ValueError(f"Path must be relative: {path}")
    index = SymbolIndex.load(index_path)
    if path.suffix == ".ipynb":
        violations = []
        if cells := json.loads(code).get("cells"):
            for cell_idx, cell in enumerate(cells, start=1):
                violations.extend(
                    _lint_cell(
                        path=path,
                        config=config,
                        index=index,
                        cell=cell,
                        cell_index=cell_idx,
                    )
                )
            if (rules.MissingNotebookH1Header.name in config.select) and not _has_h1_header(cells):
                violations.append(
                    Violation(
                        rules.MissingNotebookH1Header(),
                        path,
                        Location(0, 0),
                    )
                )
        return violations
    elif path.suffix in {".rst", ".md", ".mdx"}:
        violations = []
        code_blocks = (
            _iter_code_blocks(code) if path.suffix == ".rst" else _iter_md_code_blocks(code)
        )
        for code_block in code_blocks:
            violations.extend(Linter.visit_example(path, config, code_block, index))
        return violations
    else:
        linter = Linter(path=path, config=config, ignore=ignore_map(code), index=index)
        module = ast.parse(code)
        linter.visit(module)
        linter.visit_comments(code)
        linter.visit_file_content(code)
        linter.post_visit()
        return linter.violations

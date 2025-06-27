from __future__ import annotations

import ast
import inspect
import itertools
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from packaging.version import InvalidVersion, Version

from clint.resolver import Resolver
from clint.utils import resolve_expr

if TYPE_CHECKING:
    from clint.index import SymbolIndex


class Rule(ABC):
    _CLASS_NAME_TO_RULE_NAME_REGEX = re.compile(r"(?<!^)(?=[A-Z])")
    _id_counter = itertools.count(start=1)
    _generated_id: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only generate ID for concrete classes
        if not inspect.isabstract(cls):
            id_ = next(cls._id_counter)
            cls._generated_id = f"MLF{id_:04d}"

    @property
    def id(self) -> str:
        return self._generated_id

    @abstractmethod
    def _message(self) -> str:
        """
        Return a message that explains this rule.
        """

    @property
    def message(self) -> str:
        return self._message()

    @property
    def name(self) -> str:
        """
        The name of this rule.
        """
        return self._CLASS_NAME_TO_RULE_NAME_REGEX.sub("-", self.__class__.__name__).lower()


class NoRst(Rule):
    def _message(self) -> str:
        return "Do not use RST style. Use Google style instead."


class LazyBuiltinImport(Rule):
    def _message(self) -> str:
        return "Builtin modules must be imported at the top level."


class MlflowClassName(Rule):
    def _message(self) -> str:
        return "Should use `Mlflow` in class name, not `MLflow` or `MLFlow`."


class TestNameTypo(Rule):
    def _message(self) -> str:
        return "This function looks like a test, but its name does not start with 'test_'."


class LogModelArtifactPath(Rule):
    def _message(self) -> str:
        return "`artifact_path` parameter of `log_model` is deprecated. Use `name` instead."

    @staticmethod
    def check(node: ast.Call, index: "SymbolIndex") -> bool:
        """
        Returns True if the call looks like `mlflow.<flavor>.log_model(...)` and
        the `artifact_path` argument is specified.
        """
        parts = resolve_expr(node.func)
        if not parts or len(parts) != 3:
            return False

        first, second, third = parts
        if not (first == "mlflow" and third == "log_model"):
            return False

        # TODO: Remove this once spark flavor supports logging models as logged model artifacts
        if second == "spark":
            return False

        function_name = f"{first}.{second}.log_model"
        artifact_path_idx = LogModelArtifactPath._find_artifact_path_index(index, function_name)
        if artifact_path_idx is None:
            return False

        if len(node.args) > artifact_path_idx:
            return True
        else:
            return any(kw.arg and kw.arg == "artifact_path" for kw in node.keywords)

    @staticmethod
    def _find_artifact_path_index(index: "SymbolIndex", function_name: str) -> int | None:
        """
        Finds the index of the `artifact_path` argument in the function signature of `log_model`
        using the SymbolIndex.
        """
        if f := index.resolve(function_name):
            try:
                return f.all_args.index("artifact_path")
            except ValueError:
                return None
        return None


class ExampleSyntaxError(Rule):
    def _message(self) -> str:
        return "This example has a syntax error."


class MissingDocstringParam(Rule):
    def __init__(self, params: set[str]) -> None:
        self.params = params

    def _message(self) -> str:
        return f"Missing parameters in docstring: {self.params}"


class ExtraneousDocstringParam(Rule):
    def __init__(self, params: set[str]) -> None:
        self.params = params

    def _message(self) -> str:
        return f"Extraneous parameters in docstring: {self.params}"


class DocstringParamOrder(Rule):
    def __init__(self, params: list[str]) -> None:
        self.params = params

    def _message(self) -> str:
        return f"Unordered parameters in docstring: {self.params}"


class ImplicitOptional(Rule):
    def _message(self) -> str:
        return "Use `Optional` if default value is `None`"

    @staticmethod
    def check(node: ast.AnnAssign) -> bool:
        """
        Returns True if the value to assign is `None` but the type annotation is
        not `Optional[...]` or `... | None`. For example: `a: int = None`.
        """
        return ImplicitOptional._is_none(node.value) and not (
            ImplicitOptional._is_optional(node.annotation)
            or ImplicitOptional._is_bitor_none(node.annotation)
        )

    @staticmethod
    def _is_optional(ann: ast.AST) -> bool:
        """
        Returns True if `ann` looks like `Optional[...]`.
        """
        return (
            isinstance(ann, ast.Subscript)
            and isinstance(ann.value, ast.Name)
            and ann.value.id == "Optional"
        )

    @staticmethod
    def _is_bitor_none(ann: ast.AST) -> bool:
        """
        Returns True if `ann` looks like `... | None`.
        """
        return (
            isinstance(ann, ast.BinOp)
            and isinstance(ann.op, ast.BitOr)
            and (isinstance(ann.right, ast.Constant) and ann.right.value is None)
        )

    @staticmethod
    def _is_none(value: ast.AnnAssign) -> bool:
        """
        Returns True if `value` represents `None`.
        """
        return isinstance(value, ast.Constant) and value.value is None


class OsEnvironSetInTest(Rule):
    def _message(self) -> str:
        return "Do not set `os.environ` in test directly. Use `monkeypatch.setenv` (https://docs.pytest.org/en/stable/reference/reference.html#pytest.MonkeyPatch.setenv)."

    @staticmethod
    def check(node: ast.Assign, resolver: Resolver) -> bool:
        """
        Returns True if the assignment is to os.environ[...].
        """
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
            resolved = resolver.resolve(node.targets[0].value)
            return resolved == ["os", "environ"]
        return False


class OsEnvironDeleteInTest(Rule):
    def _message(self) -> str:
        return "Do not delete `os.environ` in test directly. Use `monkeypatch.delenv` (https://docs.pytest.org/en/stable/reference/reference.html#pytest.MonkeyPatch.delenv)."

    @staticmethod
    def check(node: ast.Delete, resolver: Resolver) -> bool:
        """
        Returns True if the deletion is from os.environ[...].
        """
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
            resolved = resolver.resolve(node.targets[0].value)
            return resolved == ["os", "environ"]
        return False


class ForbiddenTopLevelImport(Rule):
    def __init__(self, module: str) -> None:
        self.module = module

    def _message(self) -> str:
        return (
            f"Importing module `{self.module}` at the top level is not allowed "
            "in this file. Use lazy import instead."
        )


class UseSysExecutable(Rule):
    def _message(self) -> str:
        return (
            "Use `[sys.executable, '-m', 'mlflow', ...]` when running mlflow CLI in a subprocess."
        )

    @staticmethod
    def check(node: ast.Call, resolver: Resolver) -> bool:
        """
        Returns True if `node` looks like `subprocess.Popen(["mlflow", ...])`.
        """
        resolved = resolver.resolve(node)
        if (
            resolved
            and len(resolved) == 2
            and resolved[0] == "subprocess"
            and resolved[1] in ["Popen", "run", "check_output", "check_call"]
            and node.args
        ):
            first_arg = node.args[0]
            if isinstance(first_arg, ast.List) and first_arg.elts:
                first_elem = first_arg.elts[0]
                return (
                    isinstance(first_elem, ast.Constant)
                    and isinstance(first_elem.value, str)
                    and first_elem.value == "mlflow"
                )
        return False


class InvalidAbstractMethod(Rule):
    def _message(self) -> str:
        return (
            "Abstract method should only contain a single statement/expression, "
            "and it must be `pass`, `...`, or a docstring."
        )

    @staticmethod
    def _is_abstract_method(
        node: ast.FunctionDef | ast.AsyncFunctionDef, resolver: Resolver
    ) -> bool:
        return any(
            (resolved := resolver.resolve(d)) and resolved == ["abc", "abstractmethod"]
            for d in node.decorator_list
        )

    @staticmethod
    def _has_invalid_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        # Does this abstract method have multiple statements/expressions?
        if len(node.body) > 1:
            return True

        # This abstract method has a single statement/expression.
        # Check if it's `pass`, `...`, or a docstring. If not, it's invalid.
        stmt = node.body[0]

        # Check for `pass`
        if isinstance(stmt, ast.Pass):
            return False

        # Check for `...` or docstring
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            value = stmt.value.value
            # `...` literal or docstring
            return not (value is ... or isinstance(value, str))

        # Any other statement is invalid
        return True

    @staticmethod
    def check(node: ast.FunctionDef | ast.AsyncFunctionDef, resolver: Resolver) -> bool:
        return InvalidAbstractMethod._is_abstract_method(
            node, resolver
        ) and InvalidAbstractMethod._has_invalid_body(node)


class IncorrectTypeAnnotation(Rule):
    MAPPING = {
        "callable": "Callable",
        "any": "Any",
    }

    def __init__(self, type_hint: str) -> None:
        self.type_hint = type_hint

    @staticmethod
    def check(node: ast.Name) -> bool:
        return node.id in IncorrectTypeAnnotation.MAPPING

    def _message(self) -> str:
        if correct_hint := self.MAPPING.get(self.type_hint):
            return f"Did you mean `{correct_hint}` instead of `{self.type_hint}`?"

        raise ValueError(
            f"Unexpected type: {self.type_hint}. It must be one of {list(self.MAPPING)}."
        )


class TypingExtensions(Rule):
    def __init__(self, *, full_name: str, allowlist: list[str]) -> None:
        self.full_name = full_name
        self.allowlist = allowlist

    def _message(self) -> str:
        return (
            f"`{self.full_name}` is not allowed to use. Only {self.allowlist} are allowed. "
            "You can extend `tool.clint.typing-extensions-allowlist` in `pyproject.toml` if needed "
            "but make sure that the version requirement for `typing-extensions` is compatible with "
            "the added types."
        )


class MarkdownLink(Rule):
    def _message(self) -> str:
        return (
            "Markdown link is not supported in docstring. "
            "Use reST link instead (e.g., `Link text <link URL>`_)."
        )


class LazyModule(Rule):
    def _message(self) -> str:
        return "Module loaded by `LazyLoader` must be imported in `TYPE_CHECKING` block."


class EmptyNotebookCell(Rule):
    def _message(self) -> str:
        return "Empty notebook cell. Remove it or add some content."


class ForbiddenSetActiveModelUsage(Rule):
    def _message(self) -> str:
        return (
            "Usage of `set_active_model` is not allowed in mlflow, use `_set_active_model` instead."
        )

    @staticmethod
    def check(node: ast.Call, resolver: Resolver) -> bool:
        """Check if this is a call to set_active_model function."""
        return (
            (resolved := resolver.resolve(node))
            and len(resolved) >= 1
            and resolved[0] == "mlflow"
            and resolved[-1] == "set_active_model"
        )


class ForbiddenTraceUIInNotebook(Rule):
    def _message(self) -> str:
        return (
            "Found the MLflow Trace UI iframe in the notebook. "
            "The trace UI in cell outputs will not render correctly in previews or the website. "
            "Please run `mlflow.tracing.disable_notebook_display()` and rerun the cell "
            "to remove the iframe."
        )


class PytestMarkRepeat(Rule):
    def _message(self) -> str:
        return (
            "@pytest.mark.repeat decorator should not be committed. "
            "This decorator is meant for local testing only to check for flaky tests."
        )

    @staticmethod
    def check(node: ast.FunctionDef | ast.AsyncFunctionDef, resolver: Resolver) -> bool:
        """
        Returns True if the function has @pytest.mark.repeat decorator.
        """
        return any(
            (res := resolver.resolve(deco)) and res == ["pytest", "mark", "repeat"]
            for deco in node.decorator_list
        )


def _is_valid_version(version: str) -> bool:
    try:
        v = Version(version)
        return not (v.is_devrelease or v.is_prerelease or v.is_postrelease)
    except InvalidVersion:
        return False


class UnnamedThread(Rule):
    def _message(self) -> str:
        return (
            "`threading.Thread()` must be called with a `name` argument to improve debugging "
            "and traceability of thread-related issues."
        )

    @staticmethod
    def check(node: ast.Call, resolver: Resolver) -> bool:
        """
        Returns True if the call is threading.Thread() without a name parameter.
        """
        return (
            (resolved := resolver.resolve(node))
            and resolved == ["threading", "Thread"]
            and not any(keyword.arg == "name" for keyword in node.keywords)
        )


class ThreadPoolExecutorWithoutThreadNamePrefix(Rule):
    def _message(self) -> str:
        return (
            "`ThreadPoolExecutor()` must be called with a `thread_name_prefix` argument to improve "
            "debugging and traceability of thread-related issues."
        )

    @staticmethod
    def check(node: ast.Call, resolver: Resolver) -> bool:
        """
        Returns True if the call is ThreadPoolExecutor() without a thread_name_prefix parameter.
        """
        return (
            (resolved := resolver.resolve(node))
            and resolved == ["concurrent", "futures", "ThreadPoolExecutor"]
            and not any(keyword.arg == "thread_name_prefix" for keyword in node.keywords)
        )


class InvalidExperimentalDecorator(Rule):
    def _message(self) -> str:
        return (
            "Invalid usage of `@experimental` decorator. It must be used with a `version` "
            "argument that is a valid semantic version string."
        )

    @staticmethod
    def check(node: ast.expr, resolver: Resolver) -> bool:
        """
        Returns True if the `@experimental` decorator from mlflow.utils.annotations is used
        incorrectly.
        """
        resolved = resolver.resolve(node)
        if not resolved:
            return False

        if resolved != ["mlflow", "utils", "annotations", "experimental"]:
            return False

        if not isinstance(node, ast.Call):
            return True

        version = next((k.value for k in node.keywords if k.arg == "version"), None)
        if version is None:
            # No `version` argument, invalid usage
            return True

        if not isinstance(version, ast.Constant) or not isinstance(version.value, str):
            # `version` is not a string literal, invalid usage
            return True

        if not _is_valid_version(version.value):
            # `version` is not a valid semantic version, # invalid usage
            return True

        return False


class UnparameterizedGenericType(Rule):
    def __init__(self, type_hint: str) -> None:
        self.type_hint = type_hint

    @staticmethod
    def is_generic_type(node: ast.Name | ast.Attribute, resolver: Resolver) -> bool:
        if resolved := resolver.resolve(node):
            return tuple(resolved) in {
                ("typing", "Callable"),
                ("typing", "Sequence"),
            }
        elif isinstance(node, ast.Name):
            return node.id in {
                "dict",
                "list",
                "set",
                "tuple",
                "frozenset",
            }
        return False

    def _message(self) -> str:
        return (
            f"Generic type `{self.type_hint}` must be parameterized "
            "(e.g., `list[str]` rather than `list`)."
        )


class DoNotDisable(Rule):
    DO_NOT_DISABLE = {"B006"}

    def __init__(self, rules: set[str]) -> None:
        self.rules = rules

    @classmethod
    def check(cls, rules: set[str]) -> "DoNotDisable":
        if s := rules.intersection(DoNotDisable.DO_NOT_DISABLE):
            return cls(s)

    def _message(self) -> str:
        return f"DO NOT DISABLE: {self.rules}."


class UnknownMlflowFunction(Rule):
    def __init__(self, function_name: str) -> None:
        self.function_name = function_name

    def _message(self) -> str:
        return (
            f"Unknown MLflow function: `{self.function_name}`. "
            "This function may not exist or could be misspelled."
        )


class UnknownMlflowArguments(Rule):
    def __init__(self, function_name: str, unknown_args: set[str]) -> None:
        self.function_name = function_name
        self.unknown_args = unknown_args

    def _message(self) -> str:
        args_str = ", ".join(f"`{arg}`" for arg in sorted(self.unknown_args))
        return (
            f"Unknown arguments {args_str} passed to `{self.function_name}`. "
            "Check the function signature for valid parameter names."
        )

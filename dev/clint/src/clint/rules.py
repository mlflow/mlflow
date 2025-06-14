from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod

from packaging.version import InvalidVersion, Version


class Rule(ABC):
    _CLASS_NAME_TO_RULE_NAME_REGEX = re.compile(r"(?<!^)(?=[A-Z])")

    @abstractmethod
    def _id(self) -> str:
        """
        Return a unique identifier for this rule.
        """

    @property
    def id(self) -> str:
        return self._id()

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
    def _id(self) -> str:
        return "MLF0001"

    def _message(self) -> str:
        return "Do not use RST style. Use Google style instead."


class LazyBuiltinImport(Rule):
    def _id(self) -> str:
        return "MLF0002"

    def _message(self) -> str:
        return "Builtin modules must be imported at the top level."


class MlflowClassName(Rule):
    def _id(self) -> str:
        return "MLF0003"

    def _message(self) -> str:
        return "Should use `Mlflow` in class name, not `MLflow` or `MLFlow`."


class TestNameTypo(Rule):
    def _id(self) -> str:
        return "MLF0004"

    def _message(self) -> str:
        return "This function looks like a test, but its name does not start with 'test_'."


class LogModelArtifactPath(Rule):
    def _id(self) -> str:
        return "MLF0005"

    def _message(self) -> str:
        return "`artifact_path` parameter of `log_model` is deprecated. Use `name` instead."


class ExampleSyntaxError(Rule):
    def _id(self) -> str:
        return "MLF0006"

    def _message(self) -> str:
        return "This example has a syntax error."


class MissingDocstringParam(Rule):
    def __init__(self, params: set[str]) -> None:
        self.params = params

    def _id(self) -> str:
        return "MLF0007"

    def _message(self) -> str:
        return f"Missing parameters in docstring: {self.params}"


class ExtraneousDocstringParam(Rule):
    def __init__(self, params: set[str]) -> None:
        self.params = params

    def _id(self) -> str:
        return "MLF0008"

    def _message(self) -> str:
        return f"Extraneous parameters in docstring: {self.params}"


class DocstringParamOrder(Rule):
    def __init__(self, params: list[str]) -> None:
        self.params = params

    def _id(self) -> str:
        return "MLF0009"

    def _message(self) -> str:
        return f"Unordered parameters in docstring: {self.params}"


class ImplicitOptional(Rule):
    def _id(self) -> str:
        return "MLF0010"

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
    def _id(self) -> str:
        return "MLF0011"

    def _message(self) -> str:
        return "Do not set `os.environ` in test directly. Use `monkeypatch.setenv` (https://docs.pytest.org/en/stable/reference/reference.html#pytest.MonkeyPatch.setenv)."


class OsEnvironDeleteInTest(Rule):
    def _id(self) -> str:
        return "MLF0012"

    def _message(self) -> str:
        return "Do not delete `os.environ` in test directly. Use `monkeypatch.delenv` (https://docs.pytest.org/en/stable/reference/reference.html#pytest.MonkeyPatch.delenv)."


class ForbiddenTopLevelImport(Rule):
    def __init__(self, module: str) -> None:
        self.module = module

    def _id(self) -> str:
        return "MLF0013"

    def _message(self) -> str:
        return (
            f"Importing module `{self.module}` at the top level is not allowed "
            "in this file. Use lazy import instead."
        )


class UseSysExecutable(Rule):
    def _id(self) -> str:
        return "MLF0014"

    def _message(self) -> str:
        return (
            "Use `[sys.executable, '-m', 'mlflow', ...]` when running mlflow CLI in a subprocess."
        )

    @staticmethod
    def check(node: ast.Call) -> bool:
        """
        Returns True if `node` looks like `subprocess.Popen(["mlflow", ...])`.
        """
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and (node.func.value.id == "subprocess")
            and (node.func.attr in ["Popen", "run", "check_output", "check_call"])
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


def _is_abstract_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        (isinstance(d, ast.Name) and d.id == "abstractmethod")
        or (
            isinstance(d, ast.Attribute)
            and isinstance(d.value, ast.Name)
            and d.value.id == "abc"
            and d.attr == "abstractmethod"
        )
        for d in node.decorator_list
    )


class InvalidAbstractMethod(Rule):
    def _id(self) -> str:
        return "MLF0015"

    def _message(self) -> str:
        return (
            "Abstract method should only contain a single statement/expression, "
            "and it must be `pass`, `...`, or a docstring."
        )

    @staticmethod
    def check(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        return _is_abstract_method(node) and (
            # Does this abstract method have multiple statements/expressions?
            len(node.body) > 1
            # This abstract method has a single statement/expression.
            # Check if it's `pass`, `...`, or a docstring. If not, it's invalid.
            or not (
                # pass
                isinstance(node.body[0], ast.Pass)
                or (
                    isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and (
                        # `...`
                        node.body[0].value.value is ...
                        # docstring
                        or isinstance(node.body[0].value.value, str)
                    )
                )
            )
        )


class IncorrectTypeAnnotation(Rule):
    MAPPING = {
        "callable": "Callable",
        "any": "Any",
    }

    def __init__(self, type_hint: str) -> None:
        self.type_hint = type_hint

    def _id(self) -> str:
        return "MLF0016"

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

    def _id(self) -> str:
        return "MLF0017"

    def _message(self) -> str:
        return (
            f"`{self.full_name}` is not allowed to use. Only {self.allowlist} are allowed. "
            "You can extend `tool.clint.typing-extensions-allowlist` in `pyproject.toml` if needed "
            "but make sure that the version requirement for `typing-extensions` is compatible with "
            "the added types."
        )


class MarkdownLink(Rule):
    def _id(self) -> str:
        return "MLF0018"

    def _message(self) -> str:
        return (
            "Markdown link is not supported in docstring. "
            "Use reST link instead (e.g., `Link text <link URL>`_)."
        )


class LazyModule(Rule):
    def _id(self) -> str:
        return "MLF0019"

    def _message(self) -> str:
        return "Module loaded by `LazyLoader` must be imported in `TYPE_CHECKING` block."


class EmptyNotebookCell(Rule):
    def _id(self) -> str:
        return "MLF0020"

    def _message(self) -> str:
        return "Empty notebook cell. Remove it or add some content."


class ForbiddenSetActiveModelUsage(Rule):
    def _id(self) -> str:
        return "MLF0021"

    def _message(self) -> str:
        return (
            "Usage of `set_active_model` is not allowed in mlflow, use `_set_active_model` instead."
        )


class ForbiddenTraceUIInNotebook(Rule):
    def _id(self) -> str:
        return "MLF0022"

    def _message(self) -> str:
        return (
            "Found the MLflow Trace UI iframe in the notebook. "
            "The trace UI in cell outputs will not render correctly in previews or the website. "
            "Please run `mlflow.tracing.disable_notebook_display()` and rerun the cell "
            "to remove the iframe."
        )


class PytestMarkRepeat(Rule):
    def _id(self) -> str:
        return "MLF0023"

    def _message(self) -> str:
        return (
            "@pytest.mark.repeat decorator should not be committed. "
            "This decorator is meant for local testing only to check for flaky tests."
        )

    @staticmethod
    def check(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """
        Returns True if the function has @pytest.mark.repeat decorator.
        """
        for decorator in node.decorator_list:
            if PytestMarkRepeat._is_pytest_mark_repeat(decorator):
                return True
        return False

    @staticmethod
    def _is_pytest_mark_repeat(decorator: ast.AST) -> bool:
        """
        Check if a decorator is @pytest.mark.repeat in any form:
        - pytest.mark.repeat
        - pytest.mark.repeat(n)
        """
        # Handle direct call like @pytest.mark.repeat(10)
        if isinstance(decorator, ast.Call):
            decorator = decorator.func

        # Check for pytest.mark.repeat attribute access
        if (
            isinstance(decorator, ast.Attribute)
            and decorator.attr == "repeat"
            and isinstance(decorator.value, ast.Attribute)
            and decorator.value.attr == "mark"
            and isinstance(decorator.value.value, ast.Name)
            and decorator.value.value.id == "pytest"
        ):
            return True

        return False


def _is_valid_version(version: str) -> bool:
    try:
        v = Version(version)
        return not (v.is_devrelease or v.is_prerelease or v.is_postrelease)
    except InvalidVersion:
        return False


class UnnamedThread(Rule):
    def _id(self) -> str:
        return "MLF0024"

    def _message(self) -> str:
        return "`threading.Thread()` calls should include a `name` parameter for easier debugging"

    @staticmethod
    def check(node: ast.Call) -> bool:
        """
        Returns True if the call is threading.Thread() without a name parameter.
        """
        # Check if it's a threading.Thread call
        if not UnnamedThread._is_threading_thread_call(node):
            return False

        # Check if name parameter is provided
        return not UnnamedThread._has_name_parameter(node)

    @staticmethod
    def _is_threading_thread_call(node: ast.Call) -> bool:
        """Check if this is a threading.Thread() call."""
        # Check for threading.Thread() pattern
        if isinstance(node.func, ast.Attribute):
            return (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "threading"
                and node.func.attr == "Thread"
            )

        # Check for direct Thread() calls (from threading import Thread)
        if isinstance(node.func, ast.Name) and node.func.id == "Thread":
            return True

        return False

    @staticmethod
    def _has_name_parameter(node: ast.Call) -> bool:
        """Check if the call includes a name parameter."""
        # Check keyword arguments
        return any(keyword.arg == "name" for keyword in node.keywords)


class NonLiteralExperimentalVersion(Rule):
    def _id(self) -> str:
        return "MLF0025"

    def _message(self) -> str:
        return (
            "The `version` argument of `@experimental` must be a string literal that is a valid "
            "semantic version (e.g., '3.0.0')."
        )

    @staticmethod
    def _check(node: ast.expr) -> bool:
        """
        Returns True if the `@experimental` decorator is used incorrectly.
        """
        if isinstance(node, ast.Name) and node.id == "experimental":
            # The code looks like this:
            # ---
            # @experimental
            # def my_function():
            #     ...
            # ---
            # No `version` argument, invalid usage
            return True

        if not isinstance(node, ast.Call):
            # Not a function call, ignore it
            return False

        if not isinstance(node.func, ast.Name):
            # Not a simple function call, ignore it
            return False

        if node.func.id != "experimental":
            # Not the `experimental` decorator, ignore it
            return False

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

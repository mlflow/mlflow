from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod


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


# TODO: Consider dropping this rule once https://github.com/astral-sh/ruff/discussions/13622
#       is supported.
class KeywordArtifactPath(Rule):
    def _id(self) -> str:
        return "MLF0005"

    def _message(self) -> str:
        return (
            "artifact_path must be passed as a positional argument. "
            "See https://github.com/mlflow/mlflow/pull/13268 for why this is necessary."
        )


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

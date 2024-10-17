from __future__ import annotations

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
        return "MLF007"

    def _message(self) -> str:
        return f"Missing parameters in docstring: {self.params}"


class ExtraneousDocstringParam(Rule):
    def __init__(self, params: set[str]) -> None:
        self.params = params

    def _id(self) -> str:
        return "MLF008"

    def _message(self) -> str:
        return f"Extraneous parameters in docstring: {self.params}"


class DocstringParamOrder(Rule):
    def __init__(self, params: list[str]) -> None:
        self.params = params

    def _id(self) -> str:
        return "MLF009"

    def _message(self) -> str:
        return f"Unordered parameters in docstring: {self.params}"

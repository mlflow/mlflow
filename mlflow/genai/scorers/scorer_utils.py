# This file contains utility functions for scorer functionality.

import ast
import inspect
import json
import logging
import re
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable

from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException

if TYPE_CHECKING:
    from mlflow.genai.utils.type import FunctionCall

_logger = logging.getLogger(__name__)

GATEWAY_PROVIDER = "gateway"
INSTRUCTIONS_JUDGE_PYDANTIC_DATA = "instructions_judge_pydantic_data"
BUILTIN_SCORER_PYDANTIC_DATA = "builtin_scorer_pydantic_data"


# FunctionBodyExtractor class is forked from https://github.com/unitycatalog/unitycatalog/blob/20dd3820be332ac04deec4e063099fb863eb3392/ai/core/src/unitycatalog/ai/core/utils/callable_utils.py
class FunctionBodyExtractor(ast.NodeVisitor):
    """
    AST NodeVisitor class to extract the body of a function.
    """

    def __init__(self, func_name: str, source_code: str):
        self.func_name = func_name
        self.source_code = source_code
        self.function_body = ""
        self.indent_unit = 4
        self.found = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not self.found and node.name == self.func_name:
            self.found = True
            self.extract_body(node)

    def extract_body(self, node: ast.FunctionDef):
        body = node.body
        # Skip the docstring
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]

        if not body:
            return

        start_lineno = body[0].lineno
        end_lineno = body[-1].end_lineno

        source_lines = self.source_code.splitlines(keepends=True)
        function_body_lines = source_lines[start_lineno - 1 : end_lineno]

        self.function_body = dedent("".join(function_body_lines)).rstrip("\n")

        if indents := [stmt.col_offset for stmt in body if stmt.col_offset is not None]:
            self.indent_unit = min(indents)


# extract_function_body function is forked from https://github.com/unitycatalog/unitycatalog/blob/20dd3820be332ac04deec4e063099fb863eb3392/ai/core/src/unitycatalog/ai/core/utils/callable_utils.py
def extract_function_body(func: Callable[..., Any]) -> tuple[str, int]:
    """
    Extracts the body of a function as a string without the signature or docstring,
    dedents the code, and returns the indentation unit used in the function (e.g., 2 or 4 spaces).
    """
    source_lines, _ = inspect.getsourcelines(func)
    dedented_source = dedent("".join(source_lines))
    func_name = func.__name__

    extractor = FunctionBodyExtractor(func_name, dedented_source)
    parsed_source = ast.parse(dedented_source)
    extractor.visit(parsed_source)

    return extractor.function_body, extractor.indent_unit


def recreate_function(source: str, signature: str, func_name: str) -> Callable[..., Any]:
    """
    Recreate a function from its source code, signature, and name.

    Args:
        source: The function body source code.
        signature: The function signature string (e.g., "(inputs, outputs)").
        func_name: The name of the function.

    Returns:
        The recreated function.
    """
    import mlflow

    # Parse the signature to build the function definition
    sig_match = re.match(r"\((.*?)\)", signature)
    if not sig_match:
        raise MlflowException(
            f"Invalid signature format: '{signature}'", error_code=INVALID_PARAMETER_VALUE
        )

    params_str = sig_match.group(1).strip()

    # Build the function definition with future annotations to defer type hint evaluation
    func_def = "from __future__ import annotations\n"
    func_def += f"def {func_name}({params_str}):\n"
    # Indent the source code
    indented_source = "\n".join(f"    {line}" for line in source.split("\n"))
    func_def += indented_source

    # Create a namespace with common MLflow imports that scorer functions might use
    # Include mlflow module so type hints like "mlflow.entities.Trace" can be resolved
    import_namespace = {
        "mlflow": mlflow,
    }

    # Import commonly used MLflow classes
    try:
        from mlflow.entities import (
            Assessment,
            AssessmentError,
            AssessmentSource,
            AssessmentSourceType,
            Feedback,
            Trace,
        )
        from mlflow.genai.judges import CategoricalRating

        import_namespace.update(
            {
                "Feedback": Feedback,
                "Assessment": Assessment,
                "AssessmentSource": AssessmentSource,
                "AssessmentError": AssessmentError,
                "AssessmentSourceType": AssessmentSourceType,
                "Trace": Trace,
                "CategoricalRating": CategoricalRating,
            }
        )
    except ImportError:
        pass  # Some imports might not be available in all contexts

    # Local namespace will capture the created function
    local_namespace = {}

    # Execute the function definition with MLflow imports available
    exec(func_def, import_namespace, local_namespace)  # noqa: S102

    # Return the recreated function
    return local_namespace[func_name]


def is_gateway_model(model: str | None) -> bool:
    if model is None:
        return False
    from mlflow.metrics.genai.model_utils import _parse_model_uri

    try:
        provider, _ = _parse_model_uri(model)
        return provider == GATEWAY_PROVIDER
    except MlflowException:
        return False


def extract_endpoint_ref(model: str) -> str:
    from mlflow.metrics.genai.model_utils import _parse_model_uri

    _, endpoint_ref = _parse_model_uri(model)
    return endpoint_ref


def build_gateway_model(endpoint_ref: str) -> str:
    return f"{GATEWAY_PROVIDER}:/{endpoint_ref}"


def extract_model_from_serialized_scorer(serialized_data: dict[str, Any]) -> str | None:
    if ij_data := serialized_data.get(INSTRUCTIONS_JUDGE_PYDANTIC_DATA):
        return ij_data.get("model")
    if bs_data := serialized_data.get(BUILTIN_SCORER_PYDANTIC_DATA):
        return bs_data.get("model")
    return None


def update_model_in_serialized_scorer(
    serialized_data: dict[str, Any], new_model: str | None
) -> dict[str, Any]:
    result = serialized_data.copy()
    if ij_data := result.get(INSTRUCTIONS_JUDGE_PYDANTIC_DATA):
        result[INSTRUCTIONS_JUDGE_PYDANTIC_DATA] = {**ij_data, "model": new_model}
    elif bs_data := result.get(BUILTIN_SCORER_PYDANTIC_DATA):
        result[BUILTIN_SCORER_PYDANTIC_DATA] = {**bs_data, "model": new_model}
    return result


def validate_scorer_name(name: str | None) -> None:
    """
    Validate the scorer name.

    Args:
        name: The scorer name to validate.

    Raises:
        MlflowException: If the name is invalid.
    """
    if name is None:
        raise MlflowException.invalid_parameter_value("Scorer name cannot be None.")
    if not isinstance(name, str):
        raise MlflowException.invalid_parameter_value(
            f"Scorer name must be a string, got {type(name).__name__}."
        )
    if not name.strip():
        raise MlflowException.invalid_parameter_value(
            "Scorer name cannot be empty or contain only whitespace."
        )


def validate_scorer_model(model: str | None) -> None:
    """
    Validate the scorer model string if present.

    Args:
        model: The model string to validate.

    Raises:
        MlflowException: If the model is invalid.
    """
    if model is None:
        return

    if not isinstance(model, str):
        raise MlflowException.invalid_parameter_value(
            f"Scorer model must be a string, got {type(model).__name__}."
        )
    if not model.strip():
        raise MlflowException.invalid_parameter_value(
            "Scorer model cannot be empty or contain only whitespace."
        )


def parse_tool_call_expectations(
    expectations: dict[str, Any] | None,
) -> list["FunctionCall"] | None:
    from mlflow.genai.utils.type import FunctionCall

    if not expectations or "expected_tool_calls" not in expectations:
        return None

    expected_tool_calls = expectations["expected_tool_calls"]
    if not expected_tool_calls:
        return None

    normalized_calls = []
    for call in expected_tool_calls:
        if isinstance(call, FunctionCall):
            normalized_calls.append(call)
        elif isinstance(call, dict):
            name = call.get("name")
            arguments = call.get("arguments")
            if arguments is not None and not isinstance(arguments, dict):
                raise MlflowException(
                    f"Invalid arguments type: {type(arguments)}. Arguments must be a dict."
                )
            normalized_calls.append(FunctionCall(name=name, arguments=arguments))
        else:
            raise MlflowException(
                f"Invalid expected tool call format: {type(call)}. "
                "Expected dict with 'name' and optional 'arguments', or FunctionCall object."
            )

    return normalized_calls


def normalize_tool_call_arguments(args: dict[str, Any] | None) -> dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    raise MlflowException(f"Invalid arguments type: {type(args)}. Arguments must be a dict.")


def get_tool_call_signature(call: "FunctionCall", include_arguments: bool) -> str | None:
    if include_arguments:
        args = json.dumps(normalize_tool_call_arguments(call.arguments), sort_keys=True)
        return f"{call.name}({args})"
    return call.name

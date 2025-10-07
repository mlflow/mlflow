# This file contains utility functions for scorer functionality.

import ast
import inspect
import logging
import re
from textwrap import dedent
from typing import Any, Callable

from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException

_logger = logging.getLogger(__name__)


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

        indents = [stmt.col_offset for stmt in body if stmt.col_offset is not None]
        if indents:
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
    exec(func_def, import_namespace, local_namespace)

    # Return the recreated function
    return local_namespace[func_name]

"""
Databricks Agent Label Schemas Python SDK. For more details see Databricks Agent Evaluation:
<https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html>

The API docs can be found here:
<https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#review-app>
"""

from typing import Literal, Optional, Union

try:
    from databricks.agents import review_app
    from databricks.rag_eval.review_app.label_schemas import (
        EXPECTED_FACTS,
        EXPECTED_RESPONSE,
        GUIDELINES,
        InputCategorical,
        InputCategoricalList,
        InputNumeric,
        InputText,
        InputTextList,
        LabelSchema,
        LabelSchemaType,
    )
except ImportError:
    raise ImportError(
        "The `databricks-agents` package is required to use `mlflow.genai.label_schemas`. "
        "Please install it with `pip install databricks-agents`."
    ) from None


def create_label_schema(
    name: str,
    *,
    type: Literal["feedback", "expectation"],
    title: str,
    input: Union[
        InputCategorical,
        InputCategoricalList,
        InputText,
        InputTextList,
        InputNumeric,
    ],
    instruction: Optional[str] = None,
    enable_comment: bool = False,
    overwrite: bool = False,
) -> LabelSchema:
    """Create a new label schema for the review app.

    A label schema defines the type of input that stakeholders will provide when labeling items
    in the review app.

    Args:
        name: The name of the label schema. Must be unique across the review app.
        type: The type of the label schema. Either "feedback" or "expectation".
        title: The title of the label schema shown to stakeholders.
        input: The input type of the label schema.
        instruction: Optional. The instruction shown to stakeholders.
        enable_comment: Optional. Whether to enable comments for the label schema.
        overwrite: Optional. Whether to overwrite the existing label schema with the same name.

    Returns:
        LabelSchema: The created label schema.
    """
    app = review_app.get_review_app()
    return app.create_label_schema(
        name=name,
        type=type,
        title=title,
        input=input,
        instruction=instruction,
        enable_comment=enable_comment,
        overwrite=overwrite,
    )


def get_label_schema(name: str) -> LabelSchema:
    """Get a label schema from the review app.

    Args:
        name: The name of the label schema to get.

    Returns:
        LabelSchema: The label schema.
    """
    app = review_app.get_review_app()
    label_schema = next(
        (label_schema for label_schema in app.label_schemas if label_schema.name == name),
        None,
    )
    if label_schema is None:
        raise ValueError(f"Label schema with name `{name}` not found")
    return label_schema


def delete_label_schema(name: str) -> review_app.ReviewApp:
    """Delete a label schema from the review app.

    Args:
        name: The name of the label schema to delete.

    Returns:
        ReviewApp: The review app.
    """
    app = review_app.get_review_app()
    return app.delete_label_schema(name)


__all__ = [
    "EXPECTED_FACTS",
    "GUIDELINES",
    "EXPECTED_RESPONSE",
    "LabelSchemaType",
    "LabelSchema",
    "InputCategorical",
    "InputCategoricalList",
    "InputNumeric",
    "InputText",
    "InputTextList",
    "create_label_schema",
    "get_label_schema",
    "delete_label_schema",
]

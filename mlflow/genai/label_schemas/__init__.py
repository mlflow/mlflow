"""
Databricks Agent Label Schemas Python SDK. For more details see Databricks Agent Evaluation:
<https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html>

The API docs can be found here:
<https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#review-app>
"""

from typing import TYPE_CHECKING, Literal

from mlflow.genai.label_schemas.label_schemas import (
    InputCategorical,
    InputCategoricalList,
    InputNumeric,
    InputText,
    InputTextList,
    LabelSchema,
    LabelSchemaType,
)
from mlflow.genai.labeling import ReviewApp

if TYPE_CHECKING:
    from databricks.agents.review_app import ReviewApp

EXPECTED_FACTS = "expected_facts"
GUIDELINES = "guidelines"
EXPECTED_RESPONSE = "expected_response"


def create_label_schema(
    name: str,
    *,
    type: Literal["feedback", "expectation"],
    title: str,
    input: InputCategorical | InputCategoricalList | InputText | InputTextList | InputNumeric,
    instruction: str | None = None,
    enable_comment: bool = False,
    overwrite: bool = False,
) -> LabelSchema:
    """Create a new label schema for the review app.

    A label schema defines the type of input that stakeholders will provide when labeling items
    in the review app.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.

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
    from mlflow.genai.labeling.stores import _get_labeling_store  # Nested to avoid circular import

    store = _get_labeling_store()
    return store.create_label_schema(
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

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.

    Args:
        name: The name of the label schema to get.

    Returns:
        LabelSchema: The label schema.
    """
    from mlflow.genai.labeling.stores import _get_labeling_store  # Nested to avoid circular import

    store = _get_labeling_store()
    return store.get_label_schema(name)


def delete_label_schema(name: str):
    """Delete a label schema from the review app.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.

    Args:
        name: The name of the label schema to delete.
    """
    # Nested to avoid circular import
    from mlflow.genai.labeling.databricks_utils import get_databricks_review_app
    from mlflow.genai.labeling.stores import DatabricksLabelingStore, _get_labeling_store

    store = _get_labeling_store()
    store.delete_label_schema(name)

    # For backwards compatibility, return a ReviewApp instance only if using Databricks store
    if isinstance(store, DatabricksLabelingStore):
        return ReviewApp(get_databricks_review_app())
    else:
        # For non-Databricks stores, we can't return a meaningful ReviewApp
        return None


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

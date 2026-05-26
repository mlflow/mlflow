"""
Databricks Agent Label Schemas Python SDK. For more details see Databricks Agent Evaluation:
<https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html>

The API docs can be found here:
<https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#review-app>
"""

from typing import TYPE_CHECKING, Literal, TypeAlias

from mlflow.genai.label_schemas.label_schemas import (
    InputCategorical,
    InputCategoricalList,
    InputNumeric,
    InputPassFail,
    InputText,
    InputTextList,
    LabelSchema,
    LabelSchemaType,
)
from mlflow.genai.labeling import ReviewApp
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracing.client import TracingClient

if TYPE_CHECKING:
    from databricks.agents.review_app import ReviewApp

EXPECTED_FACTS = "expected_facts"
GUIDELINES = "guidelines"
EXPECTED_RESPONSE = "expected_response"

_OSS_SCHEMA_INPUT: TypeAlias = InputPassFail | InputCategorical | InputNumeric


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
    """Create a new label schema for the review app (Databricks-routed).

    A label schema defines the type of input that stakeholders will provide when labeling items
    in the review app.

    .. note::
        This is the Databricks-routed flow; identity is the schema ``name``
        within the workspace's ReviewApp. For OSS MLflow deployments use
        :func:`create_experiment_label_schema` instead, whose identity is
        ``(experiment_id, name)``. Requires `pip install mlflow[databricks]`.

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


def create_experiment_label_schema(
    experiment_id: str,
    *,
    name: str,
    type: Literal["feedback", "expectation"],
    title: str,
    input: _OSS_SCHEMA_INPUT,
    instruction: str | None = None,
    enable_comment: bool = False,
) -> LabelSchema:
    """Create a new label schema scoped to an OSS experiment.

    Unlike :func:`create_label_schema` (Databricks-routed, identified by
    schema name within a ReviewApp), OSS-native schemas are identified by
    ``(experiment_id, name)``. The server generates a ``schema_id``
    returned on the response. Use :func:`upsert_experiment_label_schema`
    for create-or-replace semantics.

    .. note::
        For Databricks workspaces with a ReviewApp, use
        :func:`create_label_schema` instead (it routes through the
        Databricks agents SDK).
    """
    return TracingClient()._create_label_schema(
        experiment_id=experiment_id,
        name=name,
        type=type,
        title=title,
        input=input,
        instruction=instruction,
        enable_comment=enable_comment,
    )


def get_experiment_label_schema(schema_id: str) -> LabelSchema:
    """Get an OSS-native label schema by its server-generated ``schema_id``."""
    return TracingClient()._get_label_schema(schema_id)


def get_experiment_label_schema_by_name(experiment_id: str, name: str) -> LabelSchema:
    """Get an OSS-native label schema by ``(experiment_id, name)``."""
    return TracingClient()._get_label_schema_by_name(experiment_id, name)


def list_experiment_label_schemas(
    experiment_id: str, max_results: int = 100, page_token: str | None = None
) -> PagedList[LabelSchema]:
    """List OSS-native label schemas for an experiment, paginated."""
    return TracingClient()._list_label_schemas(
        experiment_id, max_results=max_results, page_token=page_token
    )


def update_experiment_label_schema(
    schema_id: str,
    *,
    name: str | None = None,
    title: str | None = None,
    instruction: str | None = None,
    enable_comment: bool | None = None,
    input: _OSS_SCHEMA_INPUT | None = None,
) -> LabelSchema:
    """Sparse-update an OSS-native label schema.

    ``type`` is immutable and not accepted. Fields left as ``None`` are
    unchanged on the server. ``enable_comment=None`` is treated as
    "unchanged"; pass ``True`` or ``False`` to set.

    .. note::
        Empty strings are real values, not "no-op": passing
        ``instruction=""`` replaces the stored value with the empty
        string rather than clearing or preserving it. Pass ``None``
        (the default) to leave the field unchanged.
    """
    return TracingClient()._update_label_schema(
        schema_id,
        name=name,
        title=title,
        instruction=instruction,
        enable_comment=enable_comment,
        input=input,
    )


def upsert_experiment_label_schema(
    experiment_id: str,
    *,
    name: str,
    type: Literal["feedback", "expectation"],
    title: str,
    input: _OSS_SCHEMA_INPUT,
    instruction: str | None = None,
    enable_comment: bool | None = None,
) -> LabelSchema:
    """Atomically create-or-replace an OSS-native label schema.

    Identity is ``(experiment_id, name)``. ``type`` is immutable on
    replace; a type mismatch with the existing row is rejected.
    Omitting ``enable_comment`` preserves the existing value on replace
    (and defaults to ``False`` on create).
    """
    return TracingClient()._upsert_label_schema(
        experiment_id=experiment_id,
        name=name,
        type=type,
        title=title,
        input=input,
        instruction=instruction,
        enable_comment=enable_comment,
    )


def delete_experiment_label_schema(schema_id: str) -> None:
    """Delete an OSS-native label schema. No-op when the schema doesn't exist."""
    TracingClient()._delete_label_schema(schema_id)


__all__ = [
    "EXPECTED_FACTS",
    "GUIDELINES",
    "EXPECTED_RESPONSE",
    "LabelSchemaType",
    "LabelSchema",
    "InputCategorical",
    "InputCategoricalList",
    "InputNumeric",
    "InputPassFail",
    "InputText",
    "InputTextList",
    "create_label_schema",
    "get_label_schema",
    "delete_label_schema",
    # OSS-native CRUD (experiment-scoped, server-generated schema_id)
    "create_experiment_label_schema",
    "get_experiment_label_schema",
    "get_experiment_label_schema_by_name",
    "list_experiment_label_schemas",
    "update_experiment_label_schema",
    "upsert_experiment_label_schema",
    "delete_experiment_label_schema",
]

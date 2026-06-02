"""
Label schemas define how reviewers annotate traces in the review UI.

By default a schema is managed in the MLflow tracking store and scoped to an
experiment (identity ``(experiment_id, name)``, with a server-generated
``schema_id``). On a Databricks tracking URI the same functions route to the
workspace's ReviewApp instead, where a schema is identified by ``name``. The
per-function notes call out the parameters that apply to only one of the two
routing targets.
"""

from typing import TYPE_CHECKING, Literal, TypeAlias

from mlflow.exceptions import MlflowException
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
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracing.client import TracingClient
from mlflow.tracking import get_tracking_uri
from mlflow.utils.annotations import experimental
from mlflow.utils.uri import is_databricks_uri

if TYPE_CHECKING:
    from databricks.agents.review_app import ReviewApp

EXPECTED_FACTS = "expected_facts"
GUIDELINES = "guidelines"
EXPECTED_RESPONSE = "expected_response"

_SCHEMA_INPUT: TypeAlias = (
    InputPassFail
    | InputCategorical
    | InputCategoricalList
    | InputNumeric
    | InputText
    | InputTextList
)


def _reject_databricks_only_params(*, title: str | None, overwrite: bool) -> None:
    # `title` / `overwrite` only apply to the Databricks ReviewApp.
    if title is not None:
        raise MlflowException(
            "`title` is only supported on a Databricks tracking URI (the ReviewApp).",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if overwrite:
        raise MlflowException(
            "`overwrite` is only supported on a Databricks tracking URI (the ReviewApp).",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _reject_tracking_store_only_params(*, experiment_id: str | None, schema_id: str | None) -> None:
    # `experiment_id` / `schema_id` only apply to the tracking store; the
    # ReviewApp identifies schemas by `name`.
    if experiment_id is not None:
        raise MlflowException(
            "`experiment_id` is only supported on a non-Databricks tracking URI.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if schema_id is not None:
        raise MlflowException(
            "`schema_id` is only supported on a non-Databricks tracking URI; use `name`.",
            error_code=INVALID_PARAMETER_VALUE,
        )


@experimental(version="3.13.0")
def create_label_schema(
    name: str,
    *,
    type: Literal["feedback", "expectation"],
    input: _SCHEMA_INPUT,
    instruction: str | None = None,
    enable_comment: bool = False,
    title: str | None = None,
    overwrite: bool = False,
    experiment_id: str | None = None,
) -> LabelSchema:
    """
    Create a label schema.

    By default the schema is created in the MLflow tracking store, scoped to
    ``experiment_id`` (the current experiment when omitted) and identified by
    ``(experiment_id, name)``. On a Databricks tracking URI it is created in
    the workspace ReviewApp instead, identified by ``name``.

    Args:
        name: Schema name. Shown to reviewers as the label prompt and used as
            the assessment key; unique within the experiment.
        type: ``"feedback"`` or ``"expectation"``.
        input: The input widget spec (e.g. :py:class:`InputPassFail`,
            :py:class:`InputCategorical`, :py:class:`InputNumeric`,
            :py:class:`InputText`).
        instruction: Optional supplementary guidance shown to reviewers.
        enable_comment: Whether reviewers can add a free-form rationale.
        title: Databricks ReviewApp only — display title shown to reviewers.
        overwrite: Databricks ReviewApp only — replace an existing schema with
            the same name.
        experiment_id: Tracking store only — parent experiment; defaults to the
            current experiment.

    Returns:
        The created :py:class:`LabelSchema`.
    """
    if is_databricks_uri(get_tracking_uri()):
        _reject_tracking_store_only_params(experiment_id=experiment_id, schema_id=None)
        if title is None:
            raise MlflowException(
                "`title` is required on a Databricks tracking URI (the ReviewApp).",
                error_code=INVALID_PARAMETER_VALUE,
            )
        # Nested to avoid a hard dependency on databricks-agents off Databricks.
        from mlflow.genai.labeling.stores import _get_labeling_store

        return _get_labeling_store().create_label_schema(
            name=name,
            type=type,
            title=title,
            input=input,
            instruction=instruction,
            enable_comment=enable_comment,
            overwrite=overwrite,
        )

    _reject_databricks_only_params(title=title, overwrite=overwrite)
    if experiment_id is None:
        from mlflow.tracking.fluent import _get_experiment_id

        experiment_id = _get_experiment_id()
    return TracingClient()._create_label_schema(
        experiment_id=experiment_id,
        name=name,
        type=type,
        input=input,
        instruction=instruction,
        enable_comment=enable_comment,
    )


@experimental(version="3.13.0")
def get_label_schema(
    name: str | None = None,
    *,
    schema_id: str | None = None,
    experiment_id: str | None = None,
) -> LabelSchema:
    """
    Get a label schema.

    On a Databricks tracking URI, looks up by ``name`` in the ReviewApp.
    Otherwise looks up in the tracking store by ``schema_id``, or by
    ``(experiment_id, name)`` when ``schema_id`` is omitted.
    """
    if is_databricks_uri(get_tracking_uri()):
        _reject_tracking_store_only_params(experiment_id=experiment_id, schema_id=schema_id)
        if name is None:
            raise MlflowException(
                "`name` is required on a Databricks tracking URI.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        from mlflow.genai.labeling.stores import _get_labeling_store

        return _get_labeling_store().get_label_schema(name)

    client = TracingClient()
    if schema_id is not None:
        if name is not None or experiment_id is not None:
            raise MlflowException(
                "Pass either `schema_id` or `(experiment_id, name)`, not both.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return client._get_label_schema(schema_id)
    if experiment_id is None or name is None:
        raise MlflowException(
            "Provide `schema_id`, or both `experiment_id` and `name`.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    return client._get_label_schema_by_name(experiment_id, name)


@experimental(version="3.13.0")
def delete_label_schema(name: str | None = None, *, schema_id: str | None = None):
    """
    Delete a label schema.

    On a Databricks tracking URI, deletes by ``name`` from the ReviewApp (and
    returns a :py:class:`ReviewApp` for backwards compatibility). Otherwise
    deletes by ``schema_id`` from the tracking store (a no-op if it doesn't
    exist) and returns ``None``.
    """
    if is_databricks_uri(get_tracking_uri()):
        if schema_id is not None:
            raise MlflowException(
                "`schema_id` is only supported on a non-Databricks tracking URI; use `name`.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if name is None:
            raise MlflowException(
                "`name` is required on a Databricks tracking URI.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        # Nested to avoid circular import.
        from mlflow.genai.labeling.databricks_utils import get_databricks_review_app
        from mlflow.genai.labeling.stores import DatabricksLabelingStore, _get_labeling_store

        store = _get_labeling_store()
        store.delete_label_schema(name)
        if isinstance(store, DatabricksLabelingStore):
            return ReviewApp(get_databricks_review_app())
        return None

    if name is not None:
        raise MlflowException(
            "`name` is only supported on a Databricks tracking URI; use `schema_id`.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if schema_id is None:
        raise MlflowException("`schema_id` is required.", error_code=INVALID_PARAMETER_VALUE)
    TracingClient()._delete_label_schema(schema_id)
    return None


@experimental(version="3.13.0")
def list_label_schemas(
    experiment_id: str | None = None,
    *,
    max_results: int = 100,
    page_token: str | None = None,
) -> PagedList[LabelSchema]:
    """
    List label schemas for an experiment, paginated.

    Tracking store only; ``experiment_id`` defaults to the current experiment.
    Not supported on a Databricks tracking URI — manage ReviewApp schemas in
    the workspace review UI.
    """
    if is_databricks_uri(get_tracking_uri()):
        raise MlflowException(
            "list_label_schemas is not supported on a Databricks tracking URI; "
            "manage label schemas in the workspace review UI.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if experiment_id is None:
        from mlflow.tracking.fluent import _get_experiment_id

        experiment_id = _get_experiment_id()
    return TracingClient()._list_label_schemas(
        experiment_id, max_results=max_results, page_token=page_token
    )


@experimental(version="3.13.0")
def update_label_schema(
    schema_id: str,
    *,
    name: str | None = None,
    instruction: str | None = None,
    enable_comment: bool | None = None,
    input: _SCHEMA_INPUT | None = None,
) -> LabelSchema:
    """
    Sparse-update a label schema.

    ``type`` is immutable and not accepted. When ``input`` is provided its
    variant (pass/fail, categorical, numeric, text) and a categorical
    schema's ``multi_select`` flag must match the existing schema — only
    within-variant fields (e.g. the option list) may change; switching
    either is rejected. Fields left as ``None`` are unchanged on the server;
    an empty string is a real value that replaces the stored field rather
    than leaving it untouched. Tracking store only — not supported on a
    Databricks tracking URI.
    """
    if is_databricks_uri(get_tracking_uri()):
        raise MlflowException(
            "update_label_schema is not supported on a Databricks tracking URI; "
            "manage label schemas in the workspace review UI.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    return TracingClient()._update_label_schema(
        schema_id,
        name=name,
        instruction=instruction,
        enable_comment=enable_comment,
        input=input,
    )


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
    "list_label_schemas",
    "update_label_schema",
]

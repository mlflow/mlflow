"""
Server-side validation for label schemas.

Rules ported from the Databricks server-side validator
(``ReviewAppRpcValidatorRequestHook``) plus one OSS-specific addition:
type immutability post-create is enforced server-side (Databricks
documents `type` as immutable but does not enforce it).

The validation surface is intentionally split:

- :py:func:`validate_schema_for_create` is called from the store layer's
  create + upsert paths.
- :py:func:`validate_schema_for_update` is called from the store layer's
  patch path and enforces the immutable-field constraints (`type`).

This module is store-layer-internal; callers should not need to import
it directly.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

if TYPE_CHECKING:
    from mlflow.genai.label_schemas.label_schemas import (
        InputCategorical,
        InputNumeric,
        InputPassFail,
        LabelSchema,
        LabelSchemaType,
    )


# Field length and count limits, ported from
# managed-evals/src/reviewapp/ReviewAppRpcValidatorRequestHook.scala.
NAME_MIN_LENGTH = 1
NAME_MAX_LENGTH = 150
NAME_REGEX = re.compile(r"^[a-zA-Z0-9_]+$")

TITLE_MIN_LENGTH = 1
TITLE_MAX_LENGTH = 256

INSTRUCTION_MAX_LENGTH = 1000

PASS_FAIL_LABEL_MIN_LENGTH = 1
PASS_FAIL_LABEL_MAX_LENGTH = 64

CATEGORICAL_OPTIONS_MIN_COUNT = 1
CATEGORICAL_OPTIONS_MAX_COUNT = 100
CATEGORICAL_OPTION_MIN_LENGTH = 1
CATEGORICAL_OPTION_MAX_LENGTH = 64

_SUPPORTED_INPUT_TYPE_NAMES = ("InputPassFail", "InputCategorical", "InputNumeric")


def _invalid(message: str) -> MlflowException:
    return MlflowException(message, error_code=INVALID_PARAMETER_VALUE)


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or len(name) < NAME_MIN_LENGTH:
        raise _invalid(f"Label schema `name` must be a non-empty string; got {name!r}.")
    if len(name) > NAME_MAX_LENGTH:
        raise _invalid(
            f"Label schema `name` must be at most {NAME_MAX_LENGTH} characters; got {len(name)}."
        )
    if not NAME_REGEX.match(name):
        raise _invalid(
            f"Label schema `name` must match {NAME_REGEX.pattern!r} "
            f"(alphanumeric and underscore only); got {name!r}."
        )


def _validate_title(title: str) -> None:
    if not isinstance(title, str) or len(title) < TITLE_MIN_LENGTH:
        raise _invalid(f"Label schema `title` must be a non-empty string; got {title!r}.")
    if len(title) > TITLE_MAX_LENGTH:
        raise _invalid(
            f"Label schema `title` must be at most {TITLE_MAX_LENGTH} characters; got {len(title)}."
        )


def _validate_instruction(instruction: str | None) -> None:
    if instruction is None:
        return
    if not isinstance(instruction, str):
        cls_name = instruction.__class__.__name__
        raise _invalid(f"Label schema `instruction` must be a string or None; got {cls_name}.")
    if len(instruction) > INSTRUCTION_MAX_LENGTH:
        raise _invalid(
            f"Label schema `instruction` must be at most {INSTRUCTION_MAX_LENGTH} characters; "
            f"got {len(instruction)}."
        )


def _validate_pass_fail_input(input_obj: InputPassFail) -> None:
    for field_name in ("positive_label", "negative_label"):
        label = getattr(input_obj, field_name)
        if not isinstance(label, str) or len(label) < PASS_FAIL_LABEL_MIN_LENGTH:
            raise _invalid(
                f"`InputPassFail.{field_name}` must be a non-empty string; got {label!r}."
            )
        if len(label) > PASS_FAIL_LABEL_MAX_LENGTH:
            raise _invalid(
                f"`InputPassFail.{field_name}` must be at most {PASS_FAIL_LABEL_MAX_LENGTH} "
                f"characters; got {len(label)}."
            )
    if input_obj.positive_label == input_obj.negative_label:
        raise _invalid(
            "`InputPassFail.positive_label` and `negative_label` must be distinct; "
            f"got {input_obj.positive_label!r} for both."
        )


def _validate_categorical_options(options) -> None:
    if not isinstance(options, list) or len(options) < CATEGORICAL_OPTIONS_MIN_COUNT:
        raise _invalid(f"`InputCategorical.options` must be a non-empty list; got {options!r}.")
    if len(options) > CATEGORICAL_OPTIONS_MAX_COUNT:
        raise _invalid(
            f"`InputCategorical.options` must have at most {CATEGORICAL_OPTIONS_MAX_COUNT} "
            f"entries; got {len(options)}."
        )
    seen: set[str] = set()
    for opt in options:
        if not isinstance(opt, str) or len(opt) < CATEGORICAL_OPTION_MIN_LENGTH:
            raise _invalid(
                f"`InputCategorical.options` entries must be non-empty strings; got {opt!r}."
            )
        if len(opt) > CATEGORICAL_OPTION_MAX_LENGTH:
            raise _invalid(
                f"`InputCategorical.options` entries must be at most "
                f"{CATEGORICAL_OPTION_MAX_LENGTH} characters; got {len(opt)} for {opt!r}."
            )
        if opt in seen:
            raise _invalid(
                f"`InputCategorical.options` must be deduplicated; {opt!r} appears twice."
            )
        seen.add(opt)


def _validate_categorical_input(
    input_obj: InputCategorical,
    *,
    type: LabelSchemaType,
) -> None:
    from mlflow.genai.label_schemas.label_schemas import LabelSchemaType

    _validate_categorical_options(input_obj.options)

    if type == LabelSchemaType.FEEDBACK:
        if input_obj.semantic_polarity is None:
            raise _invalid(
                "Feedback-type schemas with `InputCategorical` must set "
                "`semantic_polarity` to either 'ascending' or 'descending'."
            )
        if input_obj.semantic_polarity not in ("ascending", "descending"):
            raise _invalid(
                f"`InputCategorical.semantic_polarity` must be 'ascending' or 'descending'; "
                f"got {input_obj.semantic_polarity!r}."
            )


def _validate_numeric_input(
    input_obj: InputNumeric,
    *,
    type: LabelSchemaType,
) -> None:
    from mlflow.genai.label_schemas.label_schemas import LabelSchemaType

    for field_name in ("min_value", "max_value"):
        value = getattr(input_obj, field_name)
        if value is not None and not isinstance(value, (int, float)):
            raise _invalid(
                f"`InputNumeric.{field_name}` must be numeric or None; "
                f"got {value.__class__.__name__}."
            )

    if type == LabelSchemaType.FEEDBACK:
        if input_obj.min_value is None or input_obj.max_value is None:
            raise _invalid(
                "Feedback-type schemas with `InputNumeric` must set both "
                "`min_value` and `max_value` (feedback is always bounded)."
            )
    if input_obj.min_value is not None and input_obj.max_value is not None:
        if input_obj.min_value >= input_obj.max_value:
            raise _invalid(
                f"`InputNumeric.min_value` must be strictly less than `max_value`; "
                f"got min={input_obj.min_value}, max={input_obj.max_value}."
            )


def _validate_input(input_obj, *, type: LabelSchemaType) -> None:
    # Imported here to avoid a circular import; the entity module is the
    # public surface and may import constants from this module later.
    from mlflow.genai.label_schemas.label_schemas import (
        InputCategorical,
        InputNumeric,
        InputPassFail,
    )

    if input_obj is None:
        raise _invalid("Label schema `input` is required.")

    if isinstance(input_obj, InputPassFail):
        _validate_pass_fail_input(input_obj)
        return
    if isinstance(input_obj, InputCategorical):
        _validate_categorical_input(input_obj, type=type)
        return
    if isinstance(input_obj, InputNumeric):
        _validate_numeric_input(input_obj, type=type)
        return

    cls_name = input_obj.__class__.__name__
    raise _invalid(
        f"Label schema `input` of type {cls_name!r} is not supported by the "
        f"OSS server. Supported input types are: {', '.join(_SUPPORTED_INPUT_TYPE_NAMES)}."
    )


def _validate_schema_type(type_value) -> "LabelSchemaType":
    from mlflow.genai.label_schemas.label_schemas import LabelSchemaType

    if isinstance(type_value, LabelSchemaType):
        return type_value
    if isinstance(type_value, str):
        try:
            return LabelSchemaType(type_value)
        except ValueError as exc:
            raise _invalid(
                f"Label schema `type` must be one of "
                f"{[t.value for t in LabelSchemaType]}; got {type_value!r}."
            ) from exc
    raise _invalid(
        f"Label schema `type` must be a LabelSchemaType or string; got {type(type_value).__name__}."
    )


def _validate_enable_comment(enable_comment) -> None:
    # `bool` is a subclass of `int`, so guard against ints sneaking through.
    if not isinstance(enable_comment, bool):
        raise _invalid(
            f"Label schema `enable_comment` must be a bool; "
            f"got {enable_comment.__class__.__name__}."
        )


def validate_schema_for_create(
    *,
    name: str,
    type,
    title: str,
    input,
    instruction: str | None = None,
    enable_comment: bool = False,
) -> None:
    """Validate fields supplied to ``create_label_schema`` / ``upsert_label_schema``.

    Raises:
        MlflowException(INVALID_PARAMETER_VALUE): if any field violates the rules.
    """
    _validate_name(name)
    schema_type = _validate_schema_type(type)
    _validate_title(title)
    _validate_instruction(instruction)
    _validate_enable_comment(enable_comment)
    _validate_input(input, type=schema_type)


def validate_schema_for_update(
    *,
    existing: LabelSchema,
    name: str | None,
    title: str | None,
    instruction: str | None,
    enable_comment: bool | None,
    input,
) -> None:
    """Validate fields supplied to ``update_label_schema``.

    Enforces `type` immutability implicitly (no `type` parameter is accepted).
    Validates whichever fields are non-None.

    Raises:
        MlflowException(INVALID_PARAMETER_VALUE): if any field violates the rules.
    """
    if name is not None:
        _validate_name(name)
    if title is not None:
        _validate_title(title)
    if instruction is not None:
        _validate_instruction(instruction)
    if enable_comment is not None:
        _validate_enable_comment(enable_comment)
    if input is not None:
        _validate_input(input, type=existing.type)

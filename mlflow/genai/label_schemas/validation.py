"""
Server-side validation for label schemas.

Type immutability post-create is enforced server-side (the field is
documented as immutable but the entity does not enforce it on its own).

The validation surface is intentionally split:

- :py:func:`validate_schema_for_create` is called from the store layer's
  create path.
- :py:func:`validate_schema_for_update` is called from the store layer's
  patch path and enforces the immutable-field constraints (`type`).

This module is store-layer-internal; callers should not need to import
it directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlflow.exceptions import MlflowException

if TYPE_CHECKING:
    from mlflow.genai.label_schemas.label_schemas import (
        InputNumeric,
        InputPassFail,
        InputText,
        LabelSchema,
        LabelSchemaType,
    )


# Field length and count limits. `name` is the reviewer-facing label and
# the assessment key, so it can read as a prompt, e.g. "Is the answer
# correct?". Its bound matches the assessment key/name length used
# elsewhere in the tracking store (250) for consistency.
NAME_MIN_LENGTH = 1
NAME_MAX_LENGTH = 250

INSTRUCTION_MAX_LENGTH = 1000

PASS_FAIL_LABEL_MIN_LENGTH = 1
PASS_FAIL_LABEL_MAX_LENGTH = 64

CATEGORICAL_OPTIONS_MIN_COUNT = 1
CATEGORICAL_OPTIONS_MAX_COUNT = 10
CATEGORICAL_OPTION_MIN_LENGTH = 1
CATEGORICAL_OPTION_MAX_LENGTH = 64

TEXT_MAX_LENGTH_MIN = 1

_SUPPORTED_INPUT_TYPE_NAMES = ("InputPassFail", "InputCategorical", "InputNumeric", "InputText")


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or len(name) < NAME_MIN_LENGTH:
        raise MlflowException.invalid_parameter_value(
            f"Label schema `name` must be a non-empty string; got {name!r}."
        )
    if len(name) > NAME_MAX_LENGTH:
        raise MlflowException.invalid_parameter_value(
            f"Label schema `name` must be at most {NAME_MAX_LENGTH} characters; got {len(name)}."
        )


def _validate_instruction(instruction: str | None) -> None:
    if instruction is None:
        return
    if not isinstance(instruction, str):
        cls_name = instruction.__class__.__name__
        raise MlflowException.invalid_parameter_value(
            f"Label schema `instruction` must be a string or None; got {cls_name}."
        )
    if len(instruction) > INSTRUCTION_MAX_LENGTH:
        raise MlflowException.invalid_parameter_value(
            f"Label schema `instruction` must be at most {INSTRUCTION_MAX_LENGTH} characters; "
            f"got {len(instruction)}."
        )


def _validate_pass_fail_input(input_obj: InputPassFail) -> None:
    for field_name in ("positive_label", "negative_label"):
        label = getattr(input_obj, field_name)
        if not isinstance(label, str) or len(label) < PASS_FAIL_LABEL_MIN_LENGTH:
            raise MlflowException.invalid_parameter_value(
                f"`InputPassFail.{field_name}` must be a non-empty string; got {label!r}."
            )
        if len(label) > PASS_FAIL_LABEL_MAX_LENGTH:
            raise MlflowException.invalid_parameter_value(
                f"`InputPassFail.{field_name}` must be at most {PASS_FAIL_LABEL_MAX_LENGTH} "
                f"characters; got {len(label)}."
            )
    if input_obj.positive_label == input_obj.negative_label:
        raise MlflowException.invalid_parameter_value(
            "`InputPassFail.positive_label` and `negative_label` must be distinct; "
            f"got {input_obj.positive_label!r} for both."
        )


def _validate_categorical_options(options) -> None:
    if not isinstance(options, list) or len(options) < CATEGORICAL_OPTIONS_MIN_COUNT:
        raise MlflowException.invalid_parameter_value(
            f"`InputCategorical.options` must be a non-empty list; got {options!r}."
        )
    if len(options) > CATEGORICAL_OPTIONS_MAX_COUNT:
        raise MlflowException.invalid_parameter_value(
            f"`InputCategorical.options` must have at most {CATEGORICAL_OPTIONS_MAX_COUNT} "
            f"entries; got {len(options)}."
        )
    seen: set[str] = set()
    for opt in options:
        if not isinstance(opt, str) or len(opt) < CATEGORICAL_OPTION_MIN_LENGTH:
            raise MlflowException.invalid_parameter_value(
                f"`InputCategorical.options` entries must be non-empty strings; got {opt!r}."
            )
        if len(opt) > CATEGORICAL_OPTION_MAX_LENGTH:
            raise MlflowException.invalid_parameter_value(
                f"`InputCategorical.options` entries must be at most "
                f"{CATEGORICAL_OPTION_MAX_LENGTH} characters; got {len(opt)} for {opt!r}."
            )
        if opt in seen:
            raise MlflowException.invalid_parameter_value(
                f"`InputCategorical.options` must be deduplicated; {opt!r} appears twice."
            )
        seen.add(opt)


def _validate_text_input(input_obj: InputText) -> None:
    max_length = input_obj.max_length
    if max_length is None:
        return
    if not isinstance(max_length, int) or isinstance(max_length, bool):
        raise MlflowException.invalid_parameter_value(
            f"`InputText.max_length` must be an int or None; got {max_length.__class__.__name__}."
        )
    if max_length < TEXT_MAX_LENGTH_MIN:
        raise MlflowException.invalid_parameter_value(
            f"`InputText.max_length` must be at least {TEXT_MAX_LENGTH_MIN}; got {max_length}."
        )


def _validate_numeric_input(input_obj: InputNumeric) -> None:
    for field_name in ("min_value", "max_value"):
        value = getattr(input_obj, field_name)
        # `bool` is a subclass of `int`; reject it so True/False can't slip
        # through as 1/0 bounds (mirrors `InputText.max_length`).
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise MlflowException.invalid_parameter_value(
                f"`InputNumeric.{field_name}` must be numeric or None; "
                f"got {value.__class__.__name__}."
            )

    if input_obj.min_value is not None and input_obj.max_value is not None:
        if input_obj.min_value >= input_obj.max_value:
            raise MlflowException.invalid_parameter_value(
                f"`InputNumeric.min_value` must be strictly less than `max_value`; "
                f"got min={input_obj.min_value}, max={input_obj.max_value}."
            )


def _validate_input(input_obj) -> None:
    # Imported here to avoid a circular import; the entity module is the
    # public surface and may import constants from this module later.
    from mlflow.genai.label_schemas.label_schemas import (
        InputCategorical,
        InputNumeric,
        InputPassFail,
        InputText,
    )

    if input_obj is None:
        raise MlflowException.invalid_parameter_value("Label schema `input` is required.")

    if isinstance(input_obj, InputPassFail):
        _validate_pass_fail_input(input_obj)
        return
    if isinstance(input_obj, InputCategorical):
        _validate_categorical_options(input_obj.options)
        if not isinstance(input_obj.multi_select, bool):
            raise MlflowException.invalid_parameter_value(
                f"`InputCategorical.multi_select` must be a bool; "
                f"got {input_obj.multi_select.__class__.__name__}."
            )
        return
    if isinstance(input_obj, InputNumeric):
        _validate_numeric_input(input_obj)
        return
    if isinstance(input_obj, InputText):
        _validate_text_input(input_obj)
        return

    cls_name = input_obj.__class__.__name__
    raise MlflowException.invalid_parameter_value(
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
            raise MlflowException.invalid_parameter_value(
                f"Label schema `type` must be one of "
                f"{[t.value for t in LabelSchemaType]}; got {type_value!r}."
            ) from exc
    raise MlflowException.invalid_parameter_value(
        f"Label schema `type` must be a LabelSchemaType or string; got {type(type_value).__name__}."
    )


def _validate_enable_comment(enable_comment) -> None:
    # `bool` is a subclass of `int`, so guard against ints sneaking through.
    if not isinstance(enable_comment, bool):
        raise MlflowException.invalid_parameter_value(
            f"Label schema `enable_comment` must be a bool; "
            f"got {enable_comment.__class__.__name__}."
        )


def validate_schema_for_create(
    *,
    name: str,
    type,
    input,
    instruction: str | None = None,
    enable_comment: bool = False,
) -> None:
    """Validate fields supplied to ``create_label_schema``.

    Raises:
        MlflowException(INVALID_PARAMETER_VALUE): if any field violates the rules.
    """
    _validate_name(name)
    _validate_schema_type(type)
    _validate_instruction(instruction)
    _validate_enable_comment(enable_comment)
    _validate_input(input)


def validate_schema_for_update(
    *,
    existing: LabelSchema,
    name: str | None,
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
    if instruction is not None:
        _validate_instruction(instruction)
    if enable_comment is not None:
        _validate_enable_comment(enable_comment)
    if input is not None:
        _validate_input(input)

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from mlflow.exceptions import MlflowException
from mlflow.genai.utils.enum_utils import StrEnum
from mlflow.protos import label_schemas_pb2 as _ls_pb
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

if TYPE_CHECKING:
    from databricks.agents.review_app import label_schemas as _label_schemas

    _InputCategorical = _label_schemas.InputCategorical
    _InputCategoricalList = _label_schemas.InputCategoricalList
    _InputNumeric = _label_schemas.InputNumeric
    _InputText = _label_schemas.InputText
    _InputTextList = _label_schemas.InputTextList
    _LabelSchema = _label_schemas.LabelSchema

DatabricksInputType = TypeVar("DatabricksInputType")
_InputType = TypeVar("_InputType", bound="InputType")


class InputType(ABC):
    """Base class for all input types."""

    @abstractmethod
    def _to_databricks_input(self) -> DatabricksInputType:
        """Convert to the internal Databricks input type."""

    @classmethod
    @abstractmethod
    def _from_databricks_input(cls, input_obj: DatabricksInputType) -> _InputType:
        """Create from the internal Databricks input type."""


@dataclass
class InputCategorical(InputType):
    """A categorical input for collecting assessments from stakeholders.

    Renders as a single-select dropdown by default; set ``multi_select=True``
    to render as a multi-select. The author controls option ordering
    directly, so there is no separate polarity hint.
    """

    options: list[str]
    """List of available options for the categorical selection."""

    multi_select: bool = False
    """When ``True``, the widget allows multiple options to be selected and
    the assessment value becomes a list of strings. Defaults to ``False``
    (single-select)."""

    def _to_databricks_input(self) -> "_InputCategorical":
        """Convert to the internal Databricks input type.

        The ``multi_select`` field has no Databricks equivalent and is
        dropped when routing to Databricks. A warning is emitted when a
        non-default value is silently discarded so callers can detect
        intent loss.
        """
        from databricks.agents.review_app import label_schemas as _label_schemas

        if self.multi_select:
            warnings.warn(
                "InputCategorical field `multi_select` has no Databricks "
                "equivalent and is being dropped when routing this schema to "
                "Databricks. Set `multi_select=False` (the default) for "
                "Databricks-routed schemas, or use the MLflow tracking store.",
                UserWarning,
                stacklevel=2,
            )
        return _label_schemas.InputCategorical(options=self.options)

    @classmethod
    def _from_databricks_input(cls, input_obj: "_InputCategorical") -> "InputCategorical":
        """Create from the internal Databricks input type."""
        return cls(options=input_obj.options)

    def to_proto(self) -> _ls_pb.InputCategorical:
        return _ls_pb.InputCategorical(options=list(self.options), multi_select=self.multi_select)

    @classmethod
    def from_proto(cls, proto: _ls_pb.InputCategorical) -> "InputCategorical":
        return cls(options=list(proto.options), multi_select=proto.multi_select)


@dataclass
class InputCategoricalList(InputType):
    """A multi-select dropdown for collecting assessments from stakeholders.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    options: list[str]
    """List of available options for the multi-select categorical (dropdown)."""

    def _to_databricks_input(self) -> "_InputCategoricalList":
        """Convert to the internal Databricks input type."""
        from databricks.agents.review_app import label_schemas as _label_schemas

        return _label_schemas.InputCategoricalList(options=self.options)

    @classmethod
    def _from_databricks_input(cls, input_obj: "_InputCategoricalList") -> "InputCategoricalList":
        """Create from the internal Databricks input type."""
        return cls(options=input_obj.options)


@dataclass
class InputTextList(InputType):
    """Like `Text`, but allows multiple entries.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    max_length_each: int | None = None
    """Maximum character length for each individual text entry. None means no limit."""

    max_count: int | None = None
    """Maximum number of text entries allowed. None means no limit."""

    def _to_databricks_input(self) -> "_InputTextList":
        """Convert to the internal Databricks input type."""
        from databricks.agents.review_app import label_schemas as _label_schemas

        return _label_schemas.InputTextList(
            max_length_each=self.max_length_each, max_count=self.max_count
        )

    @classmethod
    def _from_databricks_input(cls, input_obj: "_InputTextList") -> "InputTextList":
        """Create from the internal Databricks input type."""
        return cls(max_length_each=input_obj.max_length_each, max_count=input_obj.max_count)


@dataclass
class InputText(InputType):
    """A free-form text box for collecting assessments from stakeholders.

    Supported by both feedback and expectation schemas; use it to capture
    free-form rationale or ground-truth text from reviewers.
    """

    max_length: int | None = None
    """Maximum character length for the text input. None means no limit."""

    def _to_databricks_input(self) -> "_InputText":
        """Convert to the internal Databricks input type."""
        from databricks.agents.review_app import label_schemas as _label_schemas

        return _label_schemas.InputText(max_length=self.max_length)

    @classmethod
    def _from_databricks_input(cls, input_obj: "_InputText") -> "InputText":
        """Create from the internal Databricks input type."""
        return cls(max_length=input_obj.max_length)

    def to_proto(self) -> _ls_pb.InputText:
        proto = _ls_pb.InputText()
        if self.max_length is not None:
            proto.max_length = self.max_length
        return proto

    @classmethod
    def from_proto(cls, proto: _ls_pb.InputText) -> "InputText":
        return cls(max_length=proto.max_length if proto.HasField("max_length") else None)


@dataclass
class InputNumeric(InputType):
    """A numeric input for collecting assessments from stakeholders.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    min_value: float | None = None
    """Minimum allowed numeric value. None means no minimum limit."""

    max_value: float | None = None
    """Maximum allowed numeric value. None means no maximum limit."""

    def _to_databricks_input(self) -> "_InputNumeric":
        """Convert to the internal Databricks input type."""
        from databricks.agents.review_app import label_schemas as _label_schemas

        return _label_schemas.InputNumeric(min_value=self.min_value, max_value=self.max_value)

    @classmethod
    def _from_databricks_input(cls, input_obj: "_InputNumeric") -> "InputNumeric":
        """Create from the internal Databricks input type."""
        return cls(min_value=input_obj.min_value, max_value=input_obj.max_value)

    def to_proto(self) -> _ls_pb.InputNumeric:
        proto = _ls_pb.InputNumeric()
        if self.min_value is not None:
            proto.min_value = self.min_value
        if self.max_value is not None:
            proto.max_value = self.max_value
        return proto

    @classmethod
    def from_proto(cls, proto: _ls_pb.InputNumeric) -> "InputNumeric":
        return cls(
            min_value=proto.min_value if proto.HasField("min_value") else None,
            max_value=proto.max_value if proto.HasField("max_value") else None,
        )


@dataclass
class InputPassFail(InputType):
    """A Pass/Fail input for collecting feedback from stakeholders.

    Renders as a thumbs-up / thumbs-down toggle in the review UI. The
    ``positive_label`` and ``negative_label`` fields carry the
    positive/negative meaning directly so the UI never has to infer
    "good" vs "bad" from option strings.

    For example, a correctness schema sets
    ``InputPassFail(positive_label="Correct", negative_label="Incorrect")``;
    a hallucination schema sets
    ``InputPassFail(positive_label="Hallucinated", negative_label="Grounded")``.
    The stored assessment value is a ``bool``; ``True`` corresponds to the
    ``positive_label``.

    This input type has no Databricks ReviewApp equivalent. Databricks-routed
    schemas should use :py:class:`InputCategorical` instead; the Databricks
    conversion methods raise ``NotImplementedError``.
    """

    positive_label: str
    """Label shown next to the thumbs-up button (e.g., "Correct", "Pass")."""

    negative_label: str
    """Label shown next to the thumbs-down button (e.g., "Incorrect", "Fail")."""

    def _to_databricks_input(self) -> DatabricksInputType:
        raise NotImplementedError(
            "InputPassFail has no Databricks counterpart; Databricks-routed "
            "schemas should use InputCategorical with explicit positive/negative options."
        )

    @classmethod
    def _from_databricks_input(cls, input_obj: DatabricksInputType) -> "InputPassFail":
        raise NotImplementedError(
            "InputPassFail has no Databricks counterpart; Databricks-routed schemas "
            "use InputCategorical."
        )

    def to_proto(self) -> _ls_pb.InputPassFail:
        return _ls_pb.InputPassFail(
            positive_label=self.positive_label, negative_label=self.negative_label
        )

    @classmethod
    def from_proto(cls, proto: _ls_pb.InputPassFail) -> "InputPassFail":
        return cls(positive_label=proto.positive_label, negative_label=proto.negative_label)


class LabelSchemaType(StrEnum):
    """Type of label schema."""

    FEEDBACK = "feedback"
    EXPECTATION = "expectation"

    def to_proto(self) -> int:
        if self is LabelSchemaType.FEEDBACK:
            return _ls_pb.FEEDBACK
        return _ls_pb.EXPECTATION

    @classmethod
    def from_proto(cls, proto: int) -> "LabelSchemaType":
        if proto == _ls_pb.FEEDBACK:
            return cls.FEEDBACK
        if proto == _ls_pb.EXPECTATION:
            return cls.EXPECTATION
        raise MlflowException(
            f"Label schema `type` must be one of FEEDBACK or EXPECTATION; "
            f"got proto enum value {proto}.",
            error_code=INVALID_PARAMETER_VALUE,
        )


@dataclass(frozen=True)
class LabelSchema:
    """A label schema for collecting input from stakeholders.

    Identity is ``(experiment_id, name)``. The tracking-store identification
    fields (``schema_id``, ``experiment_id``, audit timestamps) are
    populated by the backend when the schema is created via the SDK; they
    are ``None`` for Databricks-routed schemas where identity lives on the
    parent ReviewApp.
    """

    name: str
    """Unique name identifier for the label schema within an experiment."""

    type: LabelSchemaType
    """Type of the label schema, either 'feedback' or 'expectation'."""

    input: (
        InputPassFail
        | InputCategorical
        | InputCategoricalList
        | InputText
        | InputTextList
        | InputNumeric
    )
    """
    Input type specification that defines how stakeholders will provide their assessment
    (e.g., Pass/Fail toggle, categorical dropdown, text box, numeric input).
    """

    instruction: str | None = None
    """Optional detailed instructions shown to stakeholders for guidance."""

    enable_comment: bool = False
    """Whether to enable additional comment functionality for reviewers."""

    schema_id: str | None = None
    """Server-generated identifier, set when the schema is created through
    the MLflow tracking store. ``None`` for Databricks-routed schemas
    (identity there is `(review_app_id, name)`)."""

    experiment_id: str | None = None
    """Parent experiment. ``None`` for Databricks-routed schemas."""

    created_by: str | None = None
    """User who created the schema. ``None`` for Databricks-routed schemas."""

    created_at: int | None = None
    """Creation timestamp in milliseconds. ``None`` for Databricks-routed schemas."""

    updated_at: int | None = None
    """Last update timestamp in milliseconds. ``None`` for Databricks-routed schemas."""

    @classmethod
    def _convert_databricks_input(cls, input_obj):
        """Convert a Databricks input type to the corresponding MLflow input type."""
        from databricks.agents.review_app import label_schemas as _label_schemas

        input_type_mapping = {
            _label_schemas.InputCategorical: InputCategorical,
            _label_schemas.InputCategoricalList: InputCategoricalList,
            _label_schemas.InputText: InputText,
            _label_schemas.InputTextList: InputTextList,
            _label_schemas.InputNumeric: InputNumeric,
        }

        input_class = input_type_mapping.get(type(input_obj))
        if input_class is None:
            raise ValueError(f"Unknown input type: {type(input_obj)}")

        return input_class._from_databricks_input(input_obj)

    @classmethod
    def _from_databricks_label_schema(cls, schema: "_LabelSchema") -> "LabelSchema":
        """Convert from the internal Databricks label schema type."""

        return cls(
            name=schema.name,
            type=schema.type,
            input=cls._convert_databricks_input(schema.input),
            instruction=schema.instruction,
            enable_comment=schema.enable_comment,
        )

    def to_proto(self) -> _ls_pb.LabelSchema:
        proto = _ls_pb.LabelSchema(
            name=self.name,
            type=self.type.to_proto(),
            enable_comment=self.enable_comment,
            input=_input_to_proto(self.input),
        )
        if self.schema_id is not None:
            proto.schema_id = self.schema_id
        if self.experiment_id is not None:
            proto.experiment_id = self.experiment_id
        if self.instruction is not None:
            proto.instruction = self.instruction
        if self.created_by is not None:
            proto.created_by = self.created_by
        if self.created_at is not None:
            proto.created_at = self.created_at
        if self.updated_at is not None:
            # Entity field is `updated_at`; proto field is `last_updated_at`
            # (matches Databricks RPC convention); SQL column is
            # `last_update_time` (matches existing SqlExperiment / SqlRun).
            proto.last_updated_at = self.updated_at
        return proto

    @classmethod
    def from_proto(cls, proto: _ls_pb.LabelSchema) -> "LabelSchema":
        return cls(
            name=proto.name,
            type=LabelSchemaType.from_proto(proto.type),
            input=_input_from_proto(proto.input),
            instruction=proto.instruction if proto.HasField("instruction") else None,
            enable_comment=proto.enable_comment,
            schema_id=proto.schema_id if proto.HasField("schema_id") else None,
            experiment_id=proto.experiment_id if proto.HasField("experiment_id") else None,
            created_by=proto.created_by if proto.HasField("created_by") else None,
            created_at=proto.created_at if proto.HasField("created_at") else None,
            updated_at=proto.last_updated_at if proto.HasField("last_updated_at") else None,
        )


def _input_to_proto(input_obj) -> _ls_pb.LabelSchemaInput:
    """Wrap a tracking-store input dataclass in a LabelSchemaInput oneof.

    Raises:
        MlflowException: if `input_obj` is a Databricks-only type
            (InputTextList / InputCategoricalList). These have no wire
            representation and are rejected during validation.
    """
    if isinstance(input_obj, InputPassFail):
        return _ls_pb.LabelSchemaInput(pass_fail=input_obj.to_proto())
    if isinstance(input_obj, InputCategorical):
        return _ls_pb.LabelSchemaInput(categorical=input_obj.to_proto())
    if isinstance(input_obj, InputNumeric):
        return _ls_pb.LabelSchemaInput(numeric=input_obj.to_proto())
    if isinstance(input_obj, InputText):
        return _ls_pb.LabelSchemaInput(text=input_obj.to_proto())
    raise MlflowException(
        f"Label schema input of type {input_obj.__class__.__name__!r} cannot be "
        "serialized to proto. Supported types are InputPassFail, InputCategorical, "
        "InputNumeric, InputText.",
        error_code=INVALID_PARAMETER_VALUE,
    )


def _input_from_proto(proto: _ls_pb.LabelSchemaInput):
    """Unwrap a LabelSchemaInput oneof to the matching input dataclass.

    Raises:
        MlflowException: if no oneof variant is set.
    """
    variant = proto.WhichOneof("input")
    if variant == "pass_fail":
        return InputPassFail.from_proto(proto.pass_fail)
    if variant == "categorical":
        return InputCategorical.from_proto(proto.categorical)
    if variant == "numeric":
        return InputNumeric.from_proto(proto.numeric)
    if variant == "text":
        return InputText.from_proto(proto.text)
    raise MlflowException(
        "Label schema `input` must have exactly one of `pass_fail`, `categorical`, "
        "`numeric`, or `text` set; got an empty oneof.",
        error_code=INVALID_PARAMETER_VALUE,
    )

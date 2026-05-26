from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar

from mlflow.genai.utils.enum_utils import StrEnum

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
    to render as a multi-select. For feedback-type schemas the
    ``semantic_polarity`` field is required so the UI knows which option
    direction is positive (e.g., "good" vs "bad" sorting and coloring).
    """

    options: list[str]
    """List of available options for the categorical selection."""

    semantic_polarity: Literal["ascending", "descending"] | None = None
    """Polarity hint for feedback-type schemas. ``ascending`` means the first
    option is most positive; ``descending`` means the first is most negative.
    Required for OSS feedback-type schemas; ignored on Databricks-routed
    schemas."""

    multi_select: bool = False
    """When ``True``, the widget allows multiple options to be selected and
    the assessment value becomes a list of strings. Defaults to ``False``
    (single-select)."""

    def _to_databricks_input(self) -> "_InputCategorical":
        """Convert to the internal Databricks input type.

        The OSS-only ``semantic_polarity`` and ``multi_select`` fields are
        dropped since Databricks doesn't model them.
        """
        from databricks.agents.review_app import label_schemas as _label_schemas

        return _label_schemas.InputCategorical(options=self.options)

    @classmethod
    def _from_databricks_input(cls, input_obj: "_InputCategorical") -> "InputCategorical":
        """Create from the internal Databricks input type."""
        return cls(options=input_obj.options)


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

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
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


@dataclass
class InputPassFail:
    """A Pass/Fail input for collecting feedback from stakeholders.

    Renders as a thumbs-up / thumbs-down toggle in the SME review UI. The
    ``positive_label`` and ``negative_label`` fields carry the semantic
    polarity directly so the UI never has to infer "good" vs "bad" from
    option strings.

    For example, a correctness schema sets
    ``InputPassFail(positive_label="Correct", negative_label="Incorrect")``;
    a hallucination schema sets
    ``InputPassFail(positive_label="Hallucinated", negative_label="Grounded")``.
    The stored assessment value is a ``bool``; ``True`` corresponds to the
    ``positive_label``.

    This input type is OSS-only. Databricks-routed schemas should use
    :py:class:`InputCategorical` instead.
    """

    positive_label: str
    """Label shown next to the thumbs-up button (e.g., "Correct", "Pass")."""

    negative_label: str
    """Label shown next to the thumbs-down button (e.g., "Incorrect", "Fail")."""


class LabelSchemaType(StrEnum):
    """Type of label schema."""

    FEEDBACK = "feedback"
    EXPECTATION = "expectation"


@dataclass(frozen=True)
class LabelSchema:
    """A label schema for collecting input from stakeholders.

    Identity is ``(experiment_id, name)``. The OSS-side identification
    fields (``schema_id``, ``experiment_id``, audit timestamps) are
    populated by the backend when the schema is created via the SDK; they
    are ``None`` for Databricks-routed schemas where identity lives on the
    parent ReviewApp.
    """

    name: str
    """Unique name identifier for the label schema within an experiment."""

    type: LabelSchemaType
    """Type of the label schema, either 'feedback' or 'expectation'."""

    title: str
    """Display title shown to stakeholders in the labeling review UI."""

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
    """Server-generated identifier for OSS-native schemas. ``None`` for
    Databricks-routed schemas (identity there is `(review_app_id, name)`)."""

    experiment_id: str | None = None
    """Parent experiment for OSS-native schemas. ``None`` for Databricks-routed."""

    created_by: str | None = None
    """User who created the schema (OSS-native)."""

    created_at: int | None = None
    """Creation timestamp in milliseconds (OSS-native)."""

    updated_at: int | None = None
    """Last update timestamp in milliseconds (OSS-native)."""

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
            title=schema.title,
            input=cls._convert_databricks_input(schema.input),
            instruction=schema.instruction,
            enable_comment=schema.enable_comment,
        )

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, TypeVar, Union

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
    """A single-select dropdown for collecting assessments from stakeholders.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    options: list[str]
    """List of available options for the categorical selection."""

    def _to_databricks_input(self) -> "_InputCategorical":
        """Convert to the internal Databricks input type."""
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

    max_length_each: Optional[int] = None
    """Maximum character length for each individual text entry. None means no limit."""

    max_count: Optional[int] = None
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

    max_length: Optional[int] = None
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

    min_value: Optional[float] = None
    """Minimum allowed numeric value. None means no minimum limit."""

    max_value: Optional[float] = None
    """Maximum allowed numeric value. None means no maximum limit."""

    def _to_databricks_input(self) -> "_InputNumeric":
        """Convert to the internal Databricks input type."""
        from databricks.agents.review_app import label_schemas as _label_schemas

        return _label_schemas.InputNumeric(min_value=self.min_value, max_value=self.max_value)

    @classmethod
    def _from_databricks_input(cls, input_obj: "_InputNumeric") -> "InputNumeric":
        """Create from the internal Databricks input type."""
        return cls(min_value=input_obj.min_value, max_value=input_obj.max_value)


class LabelSchemaType(StrEnum):
    """Type of label schema."""

    FEEDBACK = "feedback"
    EXPECTATION = "expectation"


@dataclass(frozen=True)
class LabelSchema:
    """A label schema for collecting input from stakeholders.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    name: str
    """Unique name identifier for the label schema."""

    type: LabelSchemaType
    """Type of the label schema, either 'feedback' or 'expectation'."""

    title: str
    """Display title shown to stakeholders in the labeling review UI."""

    input: Union[InputCategorical, InputCategoricalList, InputText, InputTextList, InputNumeric]
    """
    Input type specification that defines how stakeholders will provide their assessment
    (e.g., dropdown, text box, numeric input)
    """
    instruction: Optional[str] = None
    """Optional detailed instructions shown to stakeholders for guidance."""

    enable_comment: bool = False
    """Whether to enable additional comment functionality for reviewers."""

    @classmethod
    def _from_databricks_label_schema(cls, schema: "_LabelSchema") -> "LabelSchema":
        """Convert from the internal Databricks label schema type."""

        return cls(
            name=schema.name,
            type=schema.type,
            title=schema.title,
            input=schema.input._from_databricks_input(),
            instruction=schema.instruction,
            enable_comment=schema.enable_comment,
        )

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, TypeVar, Union

from mlflow.genai.utils.enum_utils import StrEnum

if TYPE_CHECKING:
    from typing import TypeAlias

    from databricks.agents.review_app import label_schemas as _label_schemas

    _InputCategorical: TypeAlias = _label_schemas.InputCategorical
    _InputCategoricalList: TypeAlias = _label_schemas.InputCategoricalList
    _InputNumeric: TypeAlias = _label_schemas.InputNumeric
    _InputText: TypeAlias = _label_schemas.InputText
    _InputTextList: TypeAlias = _label_schemas.InputTextList
    _LabelSchema: TypeAlias = _label_schemas.LabelSchema

# TypeVar for generic InputType subclass return types
DatabricksInputType = TypeVar("DatabricksInputType", bound="InputType")

if TYPE_CHECKING:
    # Type alias for all possible Databricks input types
    DatabricksInput: TypeAlias = Union[
        "_InputCategorical",
        "_InputCategoricalList",
        "_InputText",
        "_InputTextList",
        "_InputNumeric",
    ]


class InputType(ABC):
    """Base class for all input types."""

    @abstractmethod
    def _to_databricks_input(self) -> "DatabricksInput":
        """Convert to the internal Databricks input type."""

    @classmethod
    @abstractmethod
    def _from_databricks_input(
        cls: type[DatabricksInputType],
        input_obj: "DatabricksInput",
    ) -> DatabricksInputType:
        """Create from the internal Databricks input type."""


@dataclass
class InputCategorical(InputType):
    """A single-select dropdown for collecting assessments from stakeholders.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    options: list[str]

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
    max_count: Optional[int] = None

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
    max_value: Optional[float] = None

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

    name: str  # Must be unique across the review app.
    type: LabelSchemaType

    # Title shown in the Review UI as the title of the task.
    # e.g., "Does the response contain sensitive information?"
    title: str

    input: Union[InputCategorical, InputCategoricalList, InputText, InputTextList, InputNumeric]

    instruction: Optional[str] = None
    enable_comment: bool = False

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

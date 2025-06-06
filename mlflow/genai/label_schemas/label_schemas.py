from typing import TYPE_CHECKING, Literal, Optional, Union

from mlflow.genai.utils.enum_utils import StrEnum

if TYPE_CHECKING:
    from databricks.agents.review_app.label_schemas import (
        InputCategorical as _InputCategorical,
    )
    from databricks.agents.review_app.label_schemas import (
        InputCategoricalList as _InputCategoricalList,
    )
    from databricks.agents.review_app.label_schemas import (
        InputNumeric as _InputNumeric,
    )
    from databricks.agents.review_app.label_schemas import (
        InputText as _InputText,
    )
    from databricks.agents.review_app.label_schemas import (
        InputTextList as _InputTextList,
    )
    from databricks.agents.review_app.label_schemas import (
        LabelSchema as _LabelSchema,
    )


class InputCategorical:
    """A single-select dropdown for collecting assessments from stakeholders.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    def __init__(self, label_schema_input: "_InputCategorical"):
        self._input = label_schema_input

    @property
    def options(self) -> list[str]:
        return self._input.options


class InputCategoricalList:
    """A multi-select dropdown for collecting assessments from stakeholders.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    def __init__(self, label_schema_input: "_InputCategoricalList"):
        self._input = label_schema_input

    @property
    def options(self) -> list[str]:
        return self._input.options


class InputTextList:
    """Like `Text`, but allows multiple entries.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    def __init__(self, label_schema_input: "_InputTextList"):
        self._input = label_schema_input

    @property
    def max_length_each(self) -> Optional[int]:
        return self._input.max_length_each

    @property
    def max_count(self) -> Optional[int]:
        return self._input.max_count


class InputText:
    """A free-form text box for collecting assessments from stakeholders.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    def __init__(self, label_schema_input: "_InputText"):
        self._input = label_schema_input

    @property
    def max_length(self) -> Optional[int]:
        return self._input.max_length


class InputNumeric:
    """A numeric input for collecting assessments from stakeholders.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    def __init__(self, label_schema_input: "_InputNumeric"):
        self._input = label_schema_input

    @property
    def min_value(self) -> Optional[float]:
        return self._input.min_value

    @property
    def max_value(self) -> Optional[float]:
        return self._input.max_value


class LabelSchemaType(StrEnum):
    """Type of label schema."""

    FEEDBACK = "feedback"
    EXPECTATION = "expectation"


class LabelSchema:
    """A label schema for collecting input from stakeholders.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    def __init__(
        self,
        label_schema: "_LabelSchema",
    ):
        self._schema = label_schema

    @property
    def name(self) -> str:
        return self._schema.name

    @property
    def type(self) -> Literal["feedback", "expectation"]:
        return self._schema.type

    @property
    def title(self) -> str:
        return self._schema.title

    @property
    def input(
        self,
    ) -> Union[
        InputCategorical,
        InputCategoricalList,
        InputText,
        InputTextList,
        InputNumeric,
    ]:
        return self._schema.input

    @property
    def instruction(self) -> Optional[str]:
        return self._schema.instruction

    @property
    def enable_comment(self) -> bool:
        return self._schema.enable_comment

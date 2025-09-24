"""Entity models for dimension discovery and correlation analysis."""

from dataclasses import dataclass
from typing import Any

from mlflow.entities._mlflow_object import _MlflowObject


class DimensionType:
    """Types of dimensions available for analysis."""

    STATUS = "status"
    SPAN_TYPE = "span.type"
    LATENCY = "latency"
    TOOL = "tool"
    TAG = "tag"
    ASSESSMENT = "assessment"


class ParameterType:
    """Types of parameters a dimension can have."""

    VALUE = "value"  # Single value equality
    THRESHOLD = "threshold"  # Numeric threshold with operator
    OPERATOR = "operator"  # Comparison operator (>, <, >=, <=)


class NPMIStrength:
    """NPMI correlation strength classifications."""

    STRONG = "strong"  # > 0.7
    MODERATE = "moderate"  # 0.4 - 0.7
    WEAK = "weak"  # 0.1 - 0.4
    NEGLIGIBLE = "negligible"  # < 0.1


@dataclass
class DimensionParameter(_MlflowObject):
    """
    Parameter definition for a dimension.

    Attributes:
        name: Parameter name (e.g., "value", "threshold")
        type: Type of the parameter
        required: Whether this parameter is required
        description: Human-readable description
        allowed_values: List of allowed values (for enumerated types)
        default_value: Default value if not specified
    """

    name: str
    type: ParameterType
    required: bool = True
    description: str | None = None
    allowed_values: list[str] | None = None
    default_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "description": self.description,
            "allowed_values": self.allowed_values,
            "default_value": self.default_value,
        }


@dataclass
class DimensionDefinition(_MlflowObject):
    """
    Definition of a dimension available for analysis.

    Attributes:
        name: Dimension name (e.g., "status", "tools.gpt-4", "tags.environment")
        type: Type of dimension
        display_name: Human-readable display name
        description: Description of what this dimension represents
        parameters: List of parameters this dimension accepts
        available_values: Available values discovered from data
        count: Number of occurrences in the dataset
    """

    name: str
    type: DimensionType
    display_name: str
    description: str | None = None
    parameters: list[DimensionParameter] | None = None
    available_values: list[str] | None = None
    count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "display_name": self.display_name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters] if self.parameters else [],
            "available_values": self.available_values,
            "count": self.count,
        }


@dataclass
class DimensionValue(_MlflowObject):
    """
    A specific dimension with its parameter values set.

    Attributes:
        dimension_name: Name of the dimension
        parameters: Dictionary of parameter name to value
    """

    dimension_name: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension_name": self.dimension_name,
            "parameters": self.parameters,
        }


@dataclass
class DimensionsDiscoveryResponse(_MlflowObject):
    """
    Response containing all discovered dimensions.

    Attributes:
        dimensions: List of available dimensions
        total_traces: Total number of traces analyzed
        time_range_ms: Time range of data (start, end) in milliseconds
    """

    dimensions: list[DimensionDefinition]
    total_traces: int
    time_range_ms: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": [d.to_dict() for d in self.dimensions],
            "total_traces": self.total_traces,
            "time_range_ms": self.time_range_ms,
        }


@dataclass
class NPMICalculationResponse(_MlflowObject):
    """
    Response from NPMI calculation between two dimensions.

    Attributes:
        dimension1: First dimension
        dimension2: Second dimension
        npmi: NPMI score (-1 to 1)
        npmi_smoothed: Smoothed NPMI score
        strength: Classification of correlation strength
        dimension1_count: Number of traces matching dimension1
        dimension2_count: Number of traces matching dimension2
        joint_count: Number of traces matching both dimensions
        total_count: Total number of traces analyzed
    """

    dimension1: DimensionValue
    dimension2: DimensionValue
    npmi: float
    npmi_smoothed: float | None
    strength: NPMIStrength
    dimension1_count: int
    dimension2_count: int
    joint_count: int
    total_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension1": self.dimension1.to_dict(),
            "dimension2": self.dimension2.to_dict(),
            "npmi": self.npmi,
            "npmi_smoothed": self.npmi_smoothed,
            "strength": self.strength,
            "dimension1_count": self.dimension1_count,
            "dimension2_count": self.dimension2_count,
            "joint_count": self.joint_count,
            "total_count": self.total_count,
        }


DIMENSION_PARAMETER_DEFINITIONS = {
    DimensionType.STATUS: [
        DimensionParameter(
            name="value",
            type=ParameterType.VALUE,
            required=True,
            description="Status value to filter for",
            allowed_values=["OK", "ERROR", "UNSET"],
        )
    ],
    DimensionType.SPAN_TYPE: [
        DimensionParameter(
            name="value",
            type=ParameterType.VALUE,
            required=True,
            description="Span type to filter for",
            allowed_values=["CHAIN", "TOOL", "LLM", "RETRIEVER", "EMBEDDING", "AGENT", "UNKNOWN"],
        )
    ],
    DimensionType.LATENCY: [
        DimensionParameter(
            name="threshold",
            type=ParameterType.THRESHOLD,
            required=True,
            description="Latency threshold in milliseconds",
        ),
        DimensionParameter(
            name="operator",
            type=ParameterType.OPERATOR,
            required=True,
            description="Comparison operator",
            allowed_values=[">", "<", ">=", "<="],
            default_value=">",
        ),
    ],
    DimensionType.TOOL: [
        DimensionParameter(
            name="name",
            type=ParameterType.VALUE,
            required=True,
            description="Tool name to filter for",
        )
    ],
    DimensionType.TAG: [
        DimensionParameter(
            name="key",
            type=ParameterType.VALUE,
            required=True,
            description="Tag key to filter for",
        ),
        DimensionParameter(
            name="value",
            type=ParameterType.VALUE,
            required=False,
            description="Tag value to filter for (optional)",
        ),
    ],
    DimensionType.ASSESSMENT: [
        DimensionParameter(
            name="name",
            type=ParameterType.VALUE,
            required=True,
            description="Assessment name to filter for",
        ),
        DimensionParameter(
            name="value",
            type=ParameterType.VALUE,
            required=False,
            description="Assessment feedback value to filter for",
        ),
    ],
}
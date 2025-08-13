import warnings
from dataclasses import asdict, dataclass
from typing import Any

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.exceptions import MlflowException
from mlflow.protos.assessments_pb2 import AssessmentSource as ProtoAssessmentSource
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental


@experimental(version="2.21.0")
@dataclass
class AssessmentSource(_MlflowObject):
    """
    Source of an assessment (human, LLM as a judge with GPT-4, etc).

    When recording an assessment, MLflow mandates providing a source information
    to keep track of how the assessment is conducted.

    Args:
        source_type: The type of the assessment source. Must be one of the values in
            the AssessmentSourceType enum or an instance of the enumerator value.
        source_id: An identifier for the source, e.g. user ID or LLM judge ID. If not
            provided, the default value "default" is used.

    Note:

    The legacy AssessmentSourceType "AI_JUDGE" is deprecated and will be resolved as
    "LLM_JUDGE". You will receive a warning if using this deprecated value. This legacy
    term will be removed in a future version of MLflow.

    Example:

    Human annotation can be represented with a source type of "HUMAN":

    .. code-block:: python

        import mlflow
        from mlflow.entities.assessment import AssessmentSource, AssessmentSourceType

        source = AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,  # or "HUMAN"
            source_id="bob@example.com",
        )

    LLM-as-a-judge can be represented with a source type of "LLM_JUDGE":

    .. code-block:: python

        import mlflow
        from mlflow.entities.assessment import AssessmentSource, AssessmentSourceType

        source = AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,  # or "LLM_JUDGE"
            source_id="gpt-4o-mini",
        )

    Heuristic evaluation can be represented with a source type of "CODE":

    .. code-block:: python

        import mlflow
        from mlflow.entities.assessment import AssessmentSource, AssessmentSourceType

        source = AssessmentSource(
            source_type=AssessmentSourceType.CODE,  # or "CODE"
            source_id="repo/evaluation_script.py",
        )

    To record more context about the assessment, you can use the `metadata` field of
    the assessment logging APIs as well.
    """

    source_type: str
    source_id: str = "default"

    def __post_init__(self):
        # Perform the standardization on source_type after initialization
        self.source_type = AssessmentSourceType._standardize(self.source_type)

    def to_dictionary(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dictionary(cls, source_dict: dict[str, Any]) -> "AssessmentSource":
        return cls(**source_dict)

    def to_proto(self):
        source = ProtoAssessmentSource()
        source.source_type = ProtoAssessmentSource.SourceType.Value(self.source_type)
        if self.source_id is not None:
            source.source_id = self.source_id
        return source

    @classmethod
    def from_proto(cls, proto):
        return AssessmentSource(
            source_type=AssessmentSourceType.from_proto(proto.source_type),
            source_id=proto.source_id if proto.source_id else None,
        )


@experimental(version="2.21.0")
class AssessmentSourceType:
    """
    Enumeration and validator for assessment source types.

    This class provides constants for valid assessment source types and handles validation
    and standardization of source type values. It supports both direct constant access and
    instance creation with string validation.

    The class automatically handles:
    - Case-insensitive string inputs (converts to uppercase)
    - Deprecation warnings for legacy values (AI_JUDGE → LLM_JUDGE)
    - Validation of source type values

    Available source types:
        - HUMAN: Assessment performed by a human evaluator
        - LLM_JUDGE: Assessment performed by an LLM-as-a-judge (e.g., GPT-4)
        - CODE: Assessment performed by deterministic code/heuristics
        - SOURCE_TYPE_UNSPECIFIED: Default when source type is not specified

    Note:
        The legacy "AI_JUDGE" type is deprecated and automatically converted to "LLM_JUDGE"
        with a deprecation warning. This ensures backward compatibility while encouraging
        migration to the new terminology.

    Example:
        Using class constants directly:

        .. code-block:: python

            from mlflow.entities.assessment import AssessmentSource, AssessmentSourceType

            # Direct constant usage
            source = AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4")

        String validation through instance creation:

        .. code-block:: python

            # String input - case insensitive
            source = AssessmentSource(
                source_type="llm_judge",  # Will be standardized to "LLM_JUDGE"
                source_id="gpt-4",
            )

            # Deprecated value - triggers warning
            source = AssessmentSource(
                source_type="AI_JUDGE",  # Warning: converts to "LLM_JUDGE"
                source_id="gpt-4",
            )
    """

    SOURCE_TYPE_UNSPECIFIED = "SOURCE_TYPE_UNSPECIFIED"
    LLM_JUDGE = "LLM_JUDGE"
    AI_JUDGE = "AI_JUDGE"  # Deprecated, use LLM_JUDGE instead
    HUMAN = "HUMAN"
    CODE = "CODE"
    _SOURCE_TYPES = [SOURCE_TYPE_UNSPECIFIED, LLM_JUDGE, HUMAN, CODE]

    def __init__(self, source_type: str):
        self._source_type = AssessmentSourceType._parse(source_type)

    @staticmethod
    def _parse(source_type: str) -> str:
        source_type = source_type.upper()

        # Backwards compatibility shim for mlflow.evaluations.AssessmentSourceType
        if source_type == AssessmentSourceType.AI_JUDGE:
            warnings.warn(
                "AI_JUDGE is deprecated. Use LLM_JUDGE instead.",
                DeprecationWarning,
            )
            source_type = AssessmentSourceType.LLM_JUDGE

        if source_type not in AssessmentSourceType._SOURCE_TYPES:
            raise MlflowException(
                message=(
                    f"Invalid assessment source type: {source_type}. "
                    f"Valid source types: {AssessmentSourceType._SOURCE_TYPES}"
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )
        return source_type

    def __str__(self):
        return self._source_type

    @staticmethod
    def _standardize(source_type: str) -> str:
        return str(AssessmentSourceType(source_type))

    @classmethod
    def from_proto(cls, proto_source_type) -> str:
        return ProtoAssessmentSource.SourceType.Name(proto_source_type)

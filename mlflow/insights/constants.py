from abc import ABC

from mlflow.exceptions import MlflowException

INSIGHTS_ANALYSIS_FILE_NAME: str = "analysis.yaml"
INSIGHTS_RUN_TAG_NAME_KEY: str = "mlflow.insights.name"


class _Constant(ABC):
    @classmethod
    def _valid_members(cls) -> set[str]:
        return {
            value
            for attr, value in ((a, getattr(cls, a)) for a in dir(cls))
            if not attr.startswith("_") and attr.isupper() and isinstance(value, str)
        }

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value in cls._valid_members

    @classmethod
    def validate(cls, value: str) -> str:
        if value not in cls._valid_members():
            raise MlflowException.invalid_parameter_value(
                f"Invalid configuration supplied for {cls.__name__}. "
                f"Valid entries are: {cls._valid_members()}"
            )

    @classmethod
    def values(cls) -> set[str]:
        return cls._valid_members()


class AnalysisStatus(_Constant):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    ARCHIVED = "ARCHIVED"
    ERROR = "ERROR"


class HypothesisStatus(_Constant):
    TESTING = "TESTING"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


class IssueSeverity(_Constant):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IssueStatus(_Constant):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"

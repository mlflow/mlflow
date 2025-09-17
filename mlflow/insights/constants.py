from enum import Enum

INSIGHTS_ANALYSIS_FILE_NAME = "analysis.yaml"
INSIGHTS_RUN_TAG_NAME_KEY = "mlflow.insights.name"


class AnalysisStatus(str, Enum):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    ARCHIVED = "ARCHIVED"
    ERROR = "ERROR"


class HypothesisStatus(str, Enum):
    TESTING = "TESTING"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


class IssueSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IssueStatus(str, Enum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"

"""Constants for MLflow Insights Store SQL operations."""


class TimeGranularity:
    """Time bucket sizes for aggregations."""

    HOUR_1 = "1h"
    HOUR_6 = "6h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_7 = "7d"
    DAY_30 = "30d"

    @staticmethod
    def get_hours(granularity: str) -> int:
        """Get the number of hours for a granularity string."""
        mapping = {
            TimeGranularity.HOUR_1: 1,
            TimeGranularity.HOUR_6: 6,
            TimeGranularity.HOUR_12: 12,
            TimeGranularity.DAY_1: 24,
            TimeGranularity.DAY_7: 168,
            TimeGranularity.DAY_30: 720,
        }
        return mapping.get(granularity, 1)

    @staticmethod
    def get_milliseconds(granularity: str) -> int:
        """Get the number of milliseconds for a granularity string."""
        return TimeGranularity.get_hours(granularity) * 3600 * 1000


class DatabaseDialect:
    """Database dialect types."""

    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MSSQL = "mssql"
    SQLITE = "sqlite"


class TraceStatus:
    """Trace status values."""

    OK = "OK"
    ERROR = "ERROR"
    UNSET = "UNSET"


class SpanType:
    """Span type values."""

    CHAIN = "CHAIN"
    TOOL = "TOOL"
    LLM = "LLM"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
    AGENT = "AGENT"
    UNKNOWN = "UNKNOWN"


class SpanStatusCode:
    """Span status code values."""

    OK = "OK"
    ERROR = "ERROR"
    UNSET = "UNSET"


class ComparisonOperator:
    """Comparison operators for filtering."""

    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    EQ = "="
    NEQ = "!="


# Query aliases - Used in SQL queries as column aliases
class QueryAliases:
    """Common query aliases."""

    BUCKET_TIME = "bucket_time"
    REQUEST_COUNT = "request_count"
    TRACE_COUNT = "trace_count"
    USAGE_COUNT = "usage_count"
    ERROR_COUNT = "error_count"
    MEAN_LATENCY = "mean_latency"
    MIN_TIME = "min_time"
    MAX_TIME = "max_time"
    TOOL_NAME = "tool_name"
    SPAN_COUNT = "span_count"
    AVG_LATENCY_MS = "avg_latency_ms"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"
    COUNT = "count"


# Filter string templates for dimension to filter conversion
class FilterTemplates:
    """Templates for converting dimensions to filter strings."""

    STATUS = "attributes.status = '{value}'"
    LATENCY = "attributes.execution_time_ms {operator} {threshold}"
    SPAN_TYPE = "span.attributes.span_type = '{value}'"
    TOOL = "span.name = '{name}' AND span.attributes.span_type = 'TOOL'"
    TAG_WITH_VALUE = "tags.{key} = '{value}'"
    TAG_EXISTS = "tags.{key} IS NOT NULL"
    ASSESSMENT_WITH_VALUE = "assessment.name = '{name}' AND assessment.feedback.value = '{value}'"
    ASSESSMENT_EXISTS = "assessment.name = '{name}'"


# Query limits to prevent excessive memory usage
class QueryLimits:
    """Query result limits."""

    MAX_TRACE_SAMPLE = 10000  # Max traces to sample for discovery
    MAX_TOOLS = 50  # Max tools to discover
    MAX_TAG_KEYS = 20  # Max tag keys to discover
    MAX_TAG_VALUES = 10  # Max unique values per tag
    MAX_ASSESSMENTS = 20  # Max assessments to discover
    MAX_ERROR_SPANS = 10  # Max error spans to return
    MAX_SLOW_TOOLS = 10  # Max slow tools to return
    DEFAULT_SAMPLE_SIZE = 100  # Default sample size for quality metrics


# Quality metric thresholds for analysis
class QualityThresholds:
    """Thresholds for quality metric analysis."""

    MINIMAL_RESPONSE_LENGTH = 20  # Characters
    RUSHED_PROCESSING_MS = 100  # Milliseconds
    VERBOSITY_LENGTH = 1000  # Characters
    QUALITY_ISSUE_PATTERN = r"error|fail|issue|problem"  # Regex pattern


# NPMI correlation strength thresholds
class NPMIThresholds:
    """NPMI correlation strength thresholds."""

    STRONG = 0.7  # > 0.7 is strong correlation
    MODERATE = 0.4  # 0.4 - 0.7 is moderate
    WEAK = 0.1  # 0.1 - 0.4 is weak
    # < 0.1 is negligible


# Percentiles for latency calculations
DEFAULT_PERCENTILES = [50, 90, 95, 99]
SUMMARY_PERCENTILES = [50, 90, 99]  # For summaries

# Unit conversion constants
NANOSECONDS_TO_MILLISECONDS = 1e6
SECONDS_TO_MILLISECONDS = 1000
MINUTES_TO_SECONDS = 60
HOURS_TO_MINUTES = 60

# Dimension type prefixes
class DimensionPrefix:
    """Prefixes used for dimension names."""

    TOOL = "tool."
    TAG = "tag."
    ASSESSMENT = "assessment."


# Default values
DEFAULT_TIME_BUCKET = TimeGranularity.HOUR_1
DEFAULT_COMPARISON_OPERATOR = ComparisonOperator.GT
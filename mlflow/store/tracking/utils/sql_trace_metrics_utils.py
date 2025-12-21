from dataclasses import dataclass
from datetime import datetime, timezone

import sqlalchemy
from sqlalchemy import Column, and_, case, exists, func, literal_column
from sqlalchemy.orm.query import Query

from mlflow.entities.trace_metrics import (
    AggregationType,
    MetricAggregation,
    MetricDataPoint,
    MetricViewType,
)
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import (
    SqlAssessments,
    SqlSpan,
    SqlTraceInfo,
    SqlTraceMetadata,
    SqlTraceMetrics,
    SqlTraceTag,
)
from mlflow.tracing.constant import (
    AssessmentMetricDimensionKey,
    AssessmentMetricKey,
    AssessmentMetricSearchKey,
    SpanMetricDimensionKey,
    SpanMetricKey,
    SpanMetricSearchKey,
    TraceMetricDimensionKey,
    TraceMetricKey,
    TraceMetricSearchKey,
    TraceTagKey,
)
from mlflow.utils.search_utils import SearchTraceMetricsUtils


@dataclass
class TraceMetricsConfig:
    """
    Configuration for traces metrics.

    Args:
        aggregation_types: Supported aggregation types to apply to the metrics.
        dimensions: Supported dimensions to group metrics by.
    """

    aggregation_types: set[AggregationType]
    dimensions: set[str]


# TraceMetricKey -> TraceMetricsConfig mapping for traces
TRACES_METRICS_CONFIGS: dict[TraceMetricKey, TraceMetricsConfig] = {
    TraceMetricKey.TRACE_COUNT: TraceMetricsConfig(
        aggregation_types={AggregationType.COUNT},
        dimensions={TraceMetricDimensionKey.TRACE_NAME, TraceMetricDimensionKey.TRACE_STATUS},
    ),
    TraceMetricKey.LATENCY: TraceMetricsConfig(
        aggregation_types={AggregationType.AVG, AggregationType.PERCENTILE},
        dimensions={TraceMetricDimensionKey.TRACE_NAME},
    ),
    TraceMetricKey.INPUT_TOKENS: TraceMetricsConfig(
        aggregation_types={AggregationType.SUM, AggregationType.AVG, AggregationType.PERCENTILE},
        dimensions={TraceMetricDimensionKey.TRACE_NAME},
    ),
    TraceMetricKey.OUTPUT_TOKENS: TraceMetricsConfig(
        aggregation_types={AggregationType.SUM, AggregationType.AVG, AggregationType.PERCENTILE},
        dimensions={TraceMetricDimensionKey.TRACE_NAME},
    ),
    TraceMetricKey.TOTAL_TOKENS: TraceMetricsConfig(
        aggregation_types={AggregationType.SUM, AggregationType.AVG, AggregationType.PERCENTILE},
        dimensions={TraceMetricDimensionKey.TRACE_NAME},
    ),
}

# SpanMetricKey -> TraceMetricsConfig mapping for spans
SPANS_METRICS_CONFIGS: dict[SpanMetricKey, TraceMetricsConfig] = {
    SpanMetricKey.SPAN_COUNT: TraceMetricsConfig(
        aggregation_types={AggregationType.COUNT},
        dimensions={SpanMetricDimensionKey.SPAN_NAME, SpanMetricDimensionKey.SPAN_TYPE},
    ),
    SpanMetricKey.LATENCY: TraceMetricsConfig(
        aggregation_types={AggregationType.AVG, AggregationType.PERCENTILE},
        dimensions={SpanMetricDimensionKey.SPAN_NAME, SpanMetricDimensionKey.SPAN_STATUS},
    ),
}

ASSESSMENTS_METRICS_CONFIGS: dict[str, TraceMetricsConfig] = {
    AssessmentMetricKey.ASSESSMENT_COUNT: TraceMetricsConfig(
        aggregation_types={AggregationType.COUNT},
        dimensions={
            AssessmentMetricDimensionKey.ASSESSMENT_NAME,
            AssessmentMetricDimensionKey.ASSESSMENT_VALUE,
        },
    ),
    AssessmentMetricKey.ASSESSMENT_VALUE: TraceMetricsConfig(
        aggregation_types={AggregationType.AVG, AggregationType.PERCENTILE},
        dimensions={AssessmentMetricDimensionKey.ASSESSMENT_NAME},
    ),
}

VIEW_TYPE_CONFIGS: dict[MetricViewType, dict[str, TraceMetricsConfig]] = {
    MetricViewType.TRACES: TRACES_METRICS_CONFIGS,
    MetricViewType.SPANS: SPANS_METRICS_CONFIGS,
    MetricViewType.ASSESSMENTS: ASSESSMENTS_METRICS_CONFIGS,
}

TIME_BUCKET_LABEL = "time_bucket"


def get_percentile_aggregation(db_type: str, percentile_value: float, column):
    """
    Get percentile aggregation function based on database type.

    Args:
        db_type: Database type (e.g., "postgresql", "mssql", "mysql", "sqlite")
        percentile_value: Percentile value between 0 and 100 (e.g., 50 for median)
        column: SQLAlchemy column to compute percentile on

    Returns:
        SQLAlchemy aggregation function for percentile

    Note:
        - PostgreSQL and MSSQL use native PERCENTILE_CONT functions
        - MySQL/SQLite uses an approximation with MAX/MIN weighted calculation
    """
    percentile_value = percentile_value / 100
    match db_type:
        case "postgresql":
            return func.percentile_cont(percentile_value).within_group(column)
        # MSSQL contains PERCENTILE_CONT function, but it's easier to use the
        # approximation formula since the function requires a window function.
        case "mssql" | "mysql" | "sqlite":
            # MySQL and SQLite don't have PERCENTILE_CONT, so we use an approximation
            # For grouped data, we use a weighted average of min and max
            # This is a rough approximation: P = min + percentile * (max - min)
            return func.min(column) + percentile_value * (func.max(column) - func.min(column))


def get_time_bucket_expression(
    view_type: MetricViewType, time_interval_seconds: int, db_type: str
) -> Column:
    """Get time bucket expression for grouping timestamps.

    Args:
        view_type: Type of metrics view (e.g., TRACES, SPANS)
        time_interval_seconds: Time interval in seconds for bucketing
        db_type: Database type (e.g., "postgresql", "mssql", "mysql", "sqlite")

    Returns:
        SQLAlchemy column expression for time bucket
    """
    # Convert time_interval_seconds to milliseconds
    bucket_size_ms = time_interval_seconds * 1000

    if db_type == "mssql":
        # MSSQL requires the exact same SQL text in SELECT, GROUP BY, and ORDER BY clauses.
        # We use literal_column to generate identical SQL text across all clauses.
        match view_type:
            case MetricViewType.TRACES:
                column_name = "timestamp_ms"
            case MetricViewType.SPANS:
                # For spans, timestamp is an expression (start_time_unix_nano / 1000000)
                # rather than a simple column. Build the complete expression inline.
                column_name = "start_time_unix_nano / 1000000"
            case MetricViewType.ASSESSMENTS:
                column_name = "created_timestamp"
        expr_str = f"floor({column_name} / {bucket_size_ms}) * {bucket_size_ms}"
        return literal_column(expr_str)
    else:
        # For non-MSSQL databases, use SQLAlchemy expressions
        match view_type:
            case MetricViewType.TRACES:
                timestamp_column = SqlTraceInfo.timestamp_ms
            case MetricViewType.SPANS:
                # Convert nanoseconds to milliseconds
                timestamp_column = SqlSpan.start_time_unix_nano / 1000000
            case MetricViewType.ASSESSMENTS:
                timestamp_column = SqlAssessments.created_timestamp
        # This floors the timestamp to the nearest bucket boundary
        return func.floor(timestamp_column / bucket_size_ms) * bucket_size_ms


def _get_aggregation_expression(aggregation: MetricAggregation, db_type: str, column) -> Column:
    """
    Get the SQL aggregation expression for the given aggregation type and column.

    Args:
        aggregation: The aggregation of the metric
        db_type: Database type (for percentile calculations)
        column: The column to aggregate

    Returns:
        SQLAlchemy column expression for the aggregation
    """
    match aggregation.aggregation_type:
        case AggregationType.COUNT:
            return func.count(column)
        case AggregationType.SUM:
            return func.sum(column)
        case AggregationType.AVG:
            return func.avg(column)
        case AggregationType.PERCENTILE:
            return get_percentile_aggregation(db_type, aggregation.percentile_value, column)
        case _:
            raise MlflowException.invalid_parameter_value(
                f"Unsupported aggregation type: {aggregation.aggregation_type}",
            )


def _get_assessment_numeric_value_column(json_column: Column) -> Column:
    """
    Extract numeric value from JSON-encoded assessment value.

    Handles conversion of JSON primitives to numeric values:
    - JSON true/false -> 1/0
    - JSON numbers -> numeric value
    - other JSON-encoded values -> NULL

    Args:
        json_column: Column containing JSON-encoded value

    Returns:
        Column expression that extracts numeric value or NULL for non-numeric values
    """
    return case(
        (json_column == "true", 1.0),
        (json_column == "false", 0.0),
        # Skip null, strings, lists, and dicts (JSON null/objects/arrays)
        (json_column == "null", None),
        (func.substring(json_column, 1, 1).in_(['"', "[", "{"]), None),
        # For numbers, cast to float
        else_=func.cast(json_column, sqlalchemy.Float),
    )


def _get_column_to_aggregate(view_type: MetricViewType, metric_name: str) -> Column:
    """
    Get the SQL column for the given metric name and view type.

    Args:
        metric_name: Name of the metric to query
        view_type: Type of metrics view (e.g., TRACES, SPANS, ASSESSMENTS)

    Returns:
        SQLAlchemy column to aggregate
    """
    match view_type:
        case MetricViewType.TRACES:
            match metric_name:
                case TraceMetricKey.TRACE_COUNT:
                    return SqlTraceInfo.request_id
                case TraceMetricKey.LATENCY:
                    return SqlTraceInfo.execution_time_ms
                case metric_name if metric_name in TraceMetricKey.token_usage_keys():
                    return SqlTraceMetrics.value
        case MetricViewType.SPANS:
            match metric_name:
                case SpanMetricKey.SPAN_COUNT:
                    return SqlSpan.span_id
                case SpanMetricKey.LATENCY:
                    # Span latency in milliseconds (nanoseconds converted to ms)
                    return (SqlSpan.end_time_unix_nano - SqlSpan.start_time_unix_nano) // 1000000
        case MetricViewType.ASSESSMENTS:
            match metric_name:
                case AssessmentMetricKey.ASSESSMENT_COUNT:
                    return SqlAssessments.assessment_id
                case "assessment_value":
                    return _get_assessment_numeric_value_column(SqlAssessments.value)

    raise MlflowException.invalid_parameter_value(
        f"Unsupported metric name: {metric_name} for view type {view_type}",
    )


def _apply_dimension_to_query(
    query: Query, dimension: str, view_type: MetricViewType
) -> tuple[Query, Column]:
    """
    Apply dimension-specific logic to query and return the dimension column.

    Args:
        query: SQLAlchemy query to modify
        dimension: Dimension name to apply
        view_type: Type of metrics view (e.g., TRACES, SPANS, ASSESSMENTS)

    Returns:
        Tuple of (modified query, labeled dimension column)
    """
    match view_type:
        case MetricViewType.TRACES:
            match dimension:
                case TraceMetricDimensionKey.TRACE_NAME:
                    # Join with SqlTraceTag to get trace name
                    query = query.join(
                        SqlTraceTag,
                        and_(
                            SqlTraceInfo.request_id == SqlTraceTag.request_id,
                            SqlTraceTag.key == TraceTagKey.TRACE_NAME,
                        ),
                    )
                    return query, SqlTraceTag.value.label(TraceMetricDimensionKey.TRACE_NAME)
                case TraceMetricDimensionKey.TRACE_STATUS:
                    return query, SqlTraceInfo.status.label(TraceMetricDimensionKey.TRACE_STATUS)
        case MetricViewType.SPANS:
            match dimension:
                case SpanMetricDimensionKey.SPAN_NAME:
                    return query, SqlSpan.name.label(SpanMetricDimensionKey.SPAN_NAME)
                case SpanMetricDimensionKey.SPAN_TYPE:
                    return query, SqlSpan.type.label(SpanMetricDimensionKey.SPAN_TYPE)
                case SpanMetricDimensionKey.SPAN_STATUS:
                    return query, SqlSpan.status.label(SpanMetricDimensionKey.SPAN_STATUS)
        case MetricViewType.ASSESSMENTS:
            match dimension:
                case AssessmentMetricDimensionKey.ASSESSMENT_NAME:
                    return query, SqlAssessments.name.label(
                        AssessmentMetricDimensionKey.ASSESSMENT_NAME
                    )
                case AssessmentMetricDimensionKey.ASSESSMENT_VALUE:
                    return query, SqlAssessments.value.label(
                        AssessmentMetricDimensionKey.ASSESSMENT_VALUE
                    )
    raise MlflowException.invalid_parameter_value(
        f"Unsupported dimension `{dimension}` with view type {view_type}"
    )


def _apply_view_initial_join(query: Query, view_type: MetricViewType) -> Query:
    """
    Apply initial join required for the view type.

    Args:
        query: SQLAlchemy query (starting from SqlTraceInfo)
        view_type: Type of metrics view (e.g., TRACES, SPANS, ASSESSMENTS)

    Returns:
        Modified query with view-specific joins
    """
    match view_type:
        case MetricViewType.SPANS:
            query = query.join(SqlSpan, SqlSpan.trace_id == SqlTraceInfo.request_id)
        case MetricViewType.ASSESSMENTS:
            query = query.join(SqlAssessments, SqlAssessments.trace_id == SqlTraceInfo.request_id)
    return query


def _apply_metric_specific_joins(
    query: Query, metric_name: str, view_type: MetricViewType
) -> Query:
    """
    Apply metric-specific joins to the query.

    Args:
        query: SQLAlchemy query to modify
        metric_name: Name of the metric being queried
        view_type: Type of metrics view (e.g., TRACES, SPANS)

    Returns:
        Modified query with necessary joins
    """
    match view_type:
        case MetricViewType.TRACES:
            # Join with SqlTraceMetrics for token usage metrics
            if metric_name in TraceMetricKey.token_usage_keys():
                query = query.join(
                    SqlTraceMetrics,
                    and_(
                        SqlTraceInfo.request_id == SqlTraceMetrics.request_id,
                        SqlTraceMetrics.key == metric_name,
                    ),
                )
    return query


def _apply_filters(query: Query, filters: list[str], view_type: MetricViewType) -> Query:
    """
    Apply filters to the query.

    Args:
        query: SQLAlchemy query to filter
        filters: List of filter strings
        view_type: Type of metrics view

    Returns:
        Filtered query
    """
    if not filters:
        return query

    for filter_string in filters:
        parsed_filter = SearchTraceMetricsUtils.parse_search_filter(filter_string)
        match parsed_filter.view_type:
            case TraceMetricSearchKey.VIEW_TYPE:
                match parsed_filter.entity:
                    case TraceMetricSearchKey.STATUS:
                        query = query.filter(SqlTraceInfo.status == parsed_filter.value)
                    case TraceMetricSearchKey.METADATA:
                        metadata_filter = exists().where(
                            and_(
                                SqlTraceMetadata.request_id == SqlTraceInfo.request_id,
                                SqlTraceMetadata.key == parsed_filter.key,
                                SqlTraceMetadata.value == parsed_filter.value,
                            )
                        )
                        query = query.filter(metadata_filter)
                    case TraceMetricSearchKey.TAG:
                        tag_filter = exists().where(
                            and_(
                                SqlTraceTag.request_id == SqlTraceInfo.request_id,
                                SqlTraceTag.key == parsed_filter.key,
                                SqlTraceTag.value == parsed_filter.value,
                            )
                        )
                        query = query.filter(tag_filter)
            case SpanMetricSearchKey.VIEW_TYPE:
                if view_type != MetricViewType.SPANS:
                    raise MlflowException.invalid_parameter_value(
                        f"Filtering by span is only supported for {MetricViewType.SPANS} view "
                        f"type, got {view_type}",
                    )
                match parsed_filter.entity:
                    case SpanMetricSearchKey.NAME:
                        query = query.filter(SqlSpan.name == parsed_filter.value)
                    case SpanMetricSearchKey.STATUS:
                        query = query.filter(SqlSpan.status == parsed_filter.value)
                    case SpanMetricSearchKey.TYPE:
                        query = query.filter(SqlSpan.type == parsed_filter.value)
            case AssessmentMetricSearchKey.VIEW_TYPE:
                if view_type != MetricViewType.ASSESSMENTS:
                    raise MlflowException.invalid_parameter_value(
                        "Filtering by assessment is only supported for "
                        f"{MetricViewType.ASSESSMENTS} view type, got {view_type}",
                    )
                match parsed_filter.entity:
                    case AssessmentMetricSearchKey.NAME:
                        query = query.filter(SqlAssessments.name == parsed_filter.value)
                    case AssessmentMetricSearchKey.TYPE:
                        query = query.filter(SqlAssessments.assessment_type == parsed_filter.value)

    return query


def query_metrics(
    view_type: MetricViewType,
    db_type: str,
    query: Query,
    metric_name: str,
    aggregations: list[MetricAggregation],
    dimensions: list[str] | None,
    filters: list[str] | None,
    time_interval_seconds: int | None,
    max_results: int,
) -> list[MetricDataPoint]:
    """Unified query metrics function for all view types.

    Args:
        view_type: Type of metrics view (e.g., TRACES, SPANS)
        db_type: Database type (e.g., "postgresql", "mssql", "mysql")
        query: Base SQLAlchemy query
        metric_name: Name of the metric to query
        aggregations: List of aggregations to compute
        dimensions: List of dimensions to group by
        filters: List of filter strings (each parsed by SearchTraceUtils), combined with AND
        time_interval_seconds: Time interval in seconds for time bucketing
        max_results: Maximum number of results to return

    Returns:
        List of MetricDataPoint objects
    """
    # Apply view-specific initial join
    query = _apply_view_initial_join(query, view_type)

    query = _apply_filters(query, filters, view_type)

    # Group by dimension columns, labeled for SELECT
    dimension_columns = []

    if time_interval_seconds:
        time_bucket_expr = get_time_bucket_expression(view_type, time_interval_seconds, db_type)
        dimension_columns.append(time_bucket_expr.label(TIME_BUCKET_LABEL))

    for dimension in dimensions or []:
        query, dimension_column = _apply_dimension_to_query(query, dimension, view_type)
        dimension_columns.append(dimension_column)

    # Apply metric-specific joins
    query = _apply_metric_specific_joins(query, metric_name, view_type)

    # Build aggregation expressions
    agg_column = _get_column_to_aggregate(view_type, metric_name)
    aggregation_results = {
        str(aggregation): _get_aggregation_expression(aggregation, db_type, agg_column)
        for aggregation in aggregations
    }

    # select columns: dimensions first, then aggregations
    select_columns = dimension_columns.copy()
    for agg_label, agg_func in aggregation_results.items():
        select_columns.append(agg_func.label(agg_label))
    query = query.with_entities(*select_columns)

    # Extract underlying column expressions from labeled columns for GROUP BY/ORDER BY
    if dimension_columns:
        group_by_columns = [col.element for col in dimension_columns]
        query = query.group_by(*group_by_columns)
        # order by time bucket first, then by other dimensions
        query = query.order_by(*group_by_columns)

    results = query.limit(max_results).all()

    return convert_results_to_metric_data_points(
        results, select_columns, len(dimension_columns), metric_name
    )


def validate_query_trace_metrics_params(
    view_type: MetricViewType,
    metric_name: str,
    aggregations: list[MetricAggregation],
    dimensions: list[str] | None,
):
    """Validate parameters for query_trace_metrics.

    Args:
        view_type: Type of metrics view (e.g., TRACES, SPANS, ASSESSMENTS)
        metric_name: Name of the metric to query
        aggregations: List of aggregations to compute
        dimensions: List of dimensions to group by

    Raises:
        MlflowException: If any parameter is invalid
    """
    if view_type not in VIEW_TYPE_CONFIGS:
        supported_view_types = [vt.value for vt in VIEW_TYPE_CONFIGS.keys()]
        raise MlflowException.invalid_parameter_value(
            f"view_type must be one of {supported_view_types}, got '{view_type.value}'",
        )

    view_type_config = VIEW_TYPE_CONFIGS[view_type]
    if metric_name not in view_type_config:
        raise MlflowException.invalid_parameter_value(
            f"metric_name must be one of {list(view_type_config.keys())}, got '{metric_name}'",
        )

    metrics_config = view_type_config[metric_name]
    aggregation_types = [agg.aggregation_type for agg in aggregations]
    if invalid_agg_types := (set(aggregation_types) - metrics_config.aggregation_types):
        supported_aggs = sorted([a.value for a in metrics_config.aggregation_types])
        invalid_aggs = sorted([a.value for a in invalid_agg_types])
        raise MlflowException.invalid_parameter_value(
            f"Found invalid aggregation_type(s): {invalid_aggs}. "
            f"Supported aggregation types: {supported_aggs}",
        )

    dimensions_list = dimensions or []
    if invalid_dimensions := (set(dimensions_list) - metrics_config.dimensions):
        supported_dims = sorted([d for d in metrics_config.dimensions if d is not None])
        raise MlflowException.invalid_parameter_value(
            f"Found invalid dimension(s): {sorted(invalid_dimensions)}. "
            f"Supported dimensions: {supported_dims}",
        )


def convert_results_to_metric_data_points(
    results: list[tuple[...]],
    select_columns: list[Column],
    num_dimensions: int,
    metric_name: str,
) -> list[MetricDataPoint]:
    """
    Convert query results to MetricDataPoint objects.

    Args:
        results: List of tuples containing query results
        select_columns: List of labeled column objects (dimensions + aggregations)
        num_dimensions: Number of dimension columns
        metric_name: Name of the metric being queried

    Returns:
        List of MetricDataPoint objects
    """
    data_points = []
    for row in results:
        # Split row values into dimensions and aggregations based on select_columns
        dims = {col.name: row[i] for i, col in enumerate(select_columns[:num_dimensions])}

        # Convert time_bucket from milliseconds to ISO 8601 datetime string
        if TIME_BUCKET_LABEL in dims:
            timestamp_ms = float(dims[TIME_BUCKET_LABEL])
            timestamp_sec = timestamp_ms / 1000.0
            dt = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
            dims[TIME_BUCKET_LABEL] = dt.isoformat()

        values = {
            col.name: row[i + num_dimensions]
            for i, col in enumerate(select_columns[num_dimensions:])
        }

        data_points.append(
            MetricDataPoint(
                dimensions=dims,
                metric_name=metric_name,
                values=values,
            )
        )
    return data_points

from datetime import datetime, timezone
from typing import TypedDict

from sqlalchemy import Column, and_, func
from sqlalchemy.orm.query import Query

from mlflow.entities.trace_metrics import (
    AggregationType,
    MetricDataPoint,
    MetricsViewType,
    TimeGranularity,
)
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import SqlTraceInfo, SqlTraceTag
from mlflow.tracing.constant import TraceTagKey


class TraceMetricsConfig(TypedDict):
    """
    Configuration for traces metrics.

    Args:
        aggregation_types: Supported aggregation types to apply to the metrics.
        dimensions: Supported dimensions to group metrics by.
        filter_fields: Supported fields to filter on.
    """

    aggregation_types: set[AggregationType]
    dimensions: set[str]
    # TODO: support this
    filter_fields: set[str] | None


# metric_name -> TraceMetricsConfig mapping for traces
TRACES_METRICS_CONFIGS: dict[str, TraceMetricsConfig] = {
    "trace": {
        "aggregation_types": {AggregationType.COUNT},
        "dimensions": {"name", "status"},
        "filter_fields": None,
    },
    "latency": {
        "aggregation_types": {
            AggregationType.AVG,
            AggregationType.P50,
            AggregationType.P75,
            AggregationType.P90,
            AggregationType.P95,
            AggregationType.P99,
        },
        "dimensions": {"name"},
        "filter_fields": None,
    },
}

# TODO: add spans and assessments metrics configs
VIEW_TYPE_CONFIGS: dict[MetricsViewType, dict[str, TraceMetricsConfig]] = {
    MetricsViewType.TRACES: TRACES_METRICS_CONFIGS,
}

TIME_BUCKET_LABEL = "time_bucket"


def get_percentile_aggregation(db_type: str, percentile: float, column):
    """
    Get percentile aggregation function based on database type.

    Args:
        db_type: Database type (e.g., "postgresql", "mssql", "mysql", "sqlite")
        percentile: Percentile value between 0 and 1
        column: SQLAlchemy column to compute percentile on

    Returns:
        SQLAlchemy aggregation function for percentile

    Note:
        - PostgreSQL and MSSQL use native PERCENTILE_CONT functions
        - MySQL/SQLite uses an approximation with MAX/MIN weighted calculation
    """
    match db_type:
        case "postgresql":
            return func.percentile_cont(percentile).within_group(column)
        case "mssql":
            return func.percentile_cont(percentile).within_group(column.asc())
        case "mysql" | "sqlite":
            # MySQL and SQLite don't have PERCENTILE_CONT, so we use an approximation
            # For grouped data, we use a weighted average of min and max
            # This is a rough approximation: P = min + percentile * (max - min)
            return func.min(column) + percentile * (func.max(column) - func.min(column))


def get_time_bucket_expression(
    timestamp_column: Column, time_granularity: TimeGranularity
) -> Column:
    """Get time bucket expression for grouping timestamps.

    Args:
        timestamp_column: SQLAlchemy column containing timestamps in milliseconds
        time_granularity: Time granularity for bucketing

    Returns:
        SQLAlchemy column expression for time bucket
    """
    # Convert time_granularity to milliseconds
    granularity_ms_map = {
        TimeGranularity.MINUTE: 60 * 1000,
        TimeGranularity.HOUR: 60 * 60 * 1000,
        TimeGranularity.DAY: 24 * 60 * 60 * 1000,
        TimeGranularity.WEEK: 7 * 24 * 60 * 60 * 1000,
        TimeGranularity.MONTH: 30 * 24 * 60 * 60 * 1000,
    }

    bucket_size_ms = granularity_ms_map[time_granularity]

    # This floors the timestamp to the nearest bucket boundary
    return func.floor(timestamp_column / bucket_size_ms) * bucket_size_ms


def query_metrics_for_traces_view(
    db_type: str,
    query: Query,
    metric_name: str,
    aggregation_types: list[AggregationType],
    dimensions: list[str] | None,
    filters: list[str],
    time_granularity: TimeGranularity | None,
    max_results: int,
) -> list[MetricDataPoint]:
    """Query metrics for traces view.

    Args:
        db_type: Database type (e.g., "postgresql", "mssql", "mysql")
        query: Base SQLAlchemy query
        metric_name: Name of the metric to query
        aggregation_types: List of aggregation types to compute
        dimensions: List of dimensions to group by
        filters: List of filter strings
        time_granularity: Time granularity for time bucketing
        max_results: Maximum number of results to return

    Returns:
        List of MetricDataPoint objects
    """
    # TODO: Apply additional filters from filter_string parameter

    # Group by dimension columns, labeled for SELECT
    dimension_columns = []

    if time_granularity:
        time_bucket_expr = get_time_bucket_expression(SqlTraceInfo.timestamp_ms, time_granularity)
        dimension_columns.append(time_bucket_expr.label(TIME_BUCKET_LABEL))

    for dimension in dimensions or []:
        if dimension == "name":
            # Join with SqlTraceTag to get trace name
            query = query.join(
                SqlTraceTag,
                and_(
                    SqlTraceInfo.request_id == SqlTraceTag.request_id,
                    SqlTraceTag.key == TraceTagKey.TRACE_NAME,
                ),
            )
            dimension_columns.append(SqlTraceTag.value.label("name"))
        elif dimension == "status":
            dimension_columns.append(SqlTraceInfo.status.label("status"))
        else:
            raise NotImplementedError(
                f"dimension {dimension} is not supported for view type {MetricsViewType.TRACES}"
            )

    aggregation_results = {}

    if metric_name == "trace":
        aggregation_results[AggregationType.COUNT] = func.count(SqlTraceInfo.request_id)
    elif metric_name == "latency":
        for agg_type in aggregation_types:
            if agg_type == AggregationType.AVG:
                aggregation_results[agg_type] = func.avg(SqlTraceInfo.execution_time_ms)
            elif percentile := agg_type.map_to_percentile():
                aggregation_results[agg_type] = get_percentile_aggregation(
                    db_type, percentile, SqlTraceInfo.execution_time_ms
                )

    # select columns: dimensions first, then aggregations
    select_columns = dimension_columns.copy()
    for agg_type, agg_func in aggregation_results.items():
        select_columns.append(agg_func.label(agg_type.value))

    # Extract underlying column expressions from labeled columns for GROUP BY/ORDER BY
    if dimension_columns:
        group_by_columns = [col.element for col in dimension_columns]
        query = query.group_by(*group_by_columns)
        # order by time bucket first, then by other dimensions
        query = query.order_by(*group_by_columns)

    results = query.with_entities(*select_columns).limit(max_results).all()

    return convert_results_to_metric_data_points(
        results, select_columns, len(dimension_columns), metric_name
    )


def validate_query_trace_metrics_params(
    view_type: MetricsViewType,
    metric_name: str,
    aggregation_types: list[AggregationType],
    dimensions: list[str] | None,
):
    """Validate parameters for query_trace_metrics.

    Args:
        view_type: Type of metrics view (e.g., TRACES, SPANS, ASSESSMENTS)
        metric_name: Name of the metric to query
        aggregation_types: List of aggregation types to compute
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
    if invalid_agg_types := (set(aggregation_types) - metrics_config["aggregation_types"]):
        supported_aggs = sorted([a.value for a in metrics_config["aggregation_types"]])
        invalid_aggs = sorted([a.value for a in invalid_agg_types])
        raise MlflowException.invalid_parameter_value(
            f"Found invalid aggregation_type(s): {invalid_aggs}. "
            f"Supported aggregation types: {supported_aggs}",
        )

    dimensions_list = dimensions or []
    if invalid_dimensions := (set(dimensions_list) - metrics_config["dimensions"]):
        supported_dims = sorted([d for d in metrics_config["dimensions"] if d is not None])
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

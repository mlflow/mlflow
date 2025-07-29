"""
Trace archival functionality for MLflow that enables archiving traces to Delta tables.
"""

import importlib.util
import logging

from mlflow.exceptions import MlflowException
from mlflow.genai.experimental.databricks_trace_exporter_utils import DatabricksTraceServerClient
from mlflow.tracking import MlflowClient
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils.annotations import experimental
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATABRICKS_TRACE_ROLLING_DELETION_ENABLED,
    MLFLOW_DATABRICKS_TRACE_STORAGE_TABLE,
)

_logger = logging.getLogger(__name__)

# Supported schema version for trace archival
SUPPORTED_SCHEMA_VERSION = "v1"

# OTEL event names for trace snapshot, assessments snapshot and tags snapshot
TRACE_SNAPSHOT_OTEL_EVENT_NAME = "genai.trace.snapshot"
ASSESSMENTS_SNAPSHOT_OTEL_EVENT_NAME = "genai.assessments.snapshot"
TAGS_SNAPSHOT_OTEL_EVENT_NAME = "genai.tags.snapshot"


def _validate_schema_versions(spans_version: str, events_version: str) -> None:
    """
    Validate that both spans and events tables use supported schema versions.

    Args:
        spans_version: Schema version of the spans table
        events_version: Schema version of the events table

    Raises:
        MlflowException: If either table uses an unsupported schema version
    """
    if spans_version != SUPPORTED_SCHEMA_VERSION:
        raise MlflowException(
            f"Unsupported spans table schema version: {spans_version}. "
            f"Only {SUPPORTED_SCHEMA_VERSION} is supported for GenAI trace views."
        )

    if events_version != SUPPORTED_SCHEMA_VERSION:
        raise MlflowException(
            f"Unsupported events table schema version: {events_version}. "
            f"Only {SUPPORTED_SCHEMA_VERSION} is supported for GenAI trace views."
        )

    _logger.debug(
        f"Schema version validation passed: spans={spans_version}, events={events_version}"
    )


def _create_genai_trace_view(view_name: str, spans_table: str, events_table: str) -> None:
    """
    Create a logical view for GenAI trace data that combines spans and events tables.

    Args:
        view_name: The name of the final view to create (e.g., 'catalog.schema.trace_logs_12345')
        spans_table: The name of the table containing raw spans data
        events_table: The name of the table containing raw events data

    Raises:
        MlflowException: If view creation fails
    """
    try:
        spark = _get_active_spark_session()
        if spark is None:
            from pyspark.sql import SparkSession

            spark = SparkSession.builder.getOrCreate()

        query = f"""
            CREATE OR REPLACE VIEW {view_name} AS
            WITH
            -- 1. Extract trace metadata from trace snapshot events
            trace_snapshots AS (
              SELECT
                trace_id,
                PARSE_JSON(body) AS trace_data -- Parse the JSON body as a VARIANT
              FROM
                {events_table}
              WHERE
                event_name = '{TRACE_SNAPSHOT_OTEL_EVENT_NAME}'
            ),
            -- 2. Aggregate spans grouped by trace_id
            spans_agg AS (
              SELECT
                trace_id,
                COLLECT_LIST(STRUCT(*)) AS spans -- keep the original span record in tact
              FROM
                {spans_table}
              GROUP BY
                trace_id
            ),
            -- 3. Aggregated valid assessments grouped by trace_id
            assessments_agg AS (
              SELECT
                trace_id,
                -- Collect and parse the JSON body as a VARIANT
                COLLECT_LIST(parse_json(body)) AS assessments
              FROM
                (
                -- Select the latest assessment snapshot for each trace
                SELECT
                  trace_id,
                  body,
                  attributes['valid'] AS is_valid,
                  attributes['assessment_id'] AS assessment_id,
                  ROW_NUMBER() OVER (
                    PARTITION BY attributes['assessment_id']
                    ORDER BY time_unix_nano DESC
                  ) AS rn
                FROM
                  {events_table}
                WHERE
                  event_name = '{ASSESSMENTS_SNAPSHOT_OTEL_EVENT_NAME}'
                )
              WHERE
                -- only keep the latest assessment snapshots that are still valid
                rn = 1 AND is_valid = 'true'
              GROUP BY
                trace_id
            ),
            -- 4. Latest tags grouped by trace_id
            latest_tags AS (
              SELECT
                trace_id,
                FIRST_VALUE(body) OVER (
                    PARTITION BY trace_id
                    ORDER BY time_unix_nano DESC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                  ) AS tag_json
              FROM
                {events_table}
              WHERE
                event_name = '{TAGS_SNAPSHOT_OTEL_EVENT_NAME}'
              QUALIFY
                ROW_NUMBER() OVER (PARTITION BY trace_id ORDER BY time_unix_nano DESC) = 1
            ),
            -- 5. Extract the root span which contains the full request and response
            -- and not previews from the trace snapshot event
            root_span as (
            SELECT
                trace_id,
                attributes['mlflow.spanInputs'] as request,
                attributes['mlflow.spanOutputs'] as response
            FROM
                {spans_table}
            where
                parent_span_id = "" or parent_span_id is null
            )
            -- 6. Main query - join the trace metadata with associated tags, assessments and spans.
            -- All transformation are moved here to keep the joins performant
            SELECT
              ts.trace_data:trace_id::STRING AS trace_id,
              ts.trace_data:client_request_id::STRING AS client_request_id,
              TIMESTAMP_MILLIS(CAST(ts.trace_data:request_time::DOUBLE AS BIGINT)) AS request_time,
              ts.trace_data:state::STRING AS state,
              ts.trace_data:execution_duration::DOUBLE AS execution_duration_ms,
              ts.trace_data:request_preview::STRING AS request_preview,
              ts.trace_data:response_preview::STRING AS response_preview,
              rs.request AS request,
              rs.response AS response,
              ts.trace_data:trace_metadata AS trace_metadata,
              COALESCE(FROM_JSON(lt.tag_json, 'MAP<STRING, STRING>'), MAP()) AS tags,
              ts.trace_data:trace_location AS trace_location,
              sa.spans,
              COALESCE(
                TRANSFORM(
                  aa.assessments,
                  body -> STRUCT(
                    body:assessment_id::STRING AS assessment_id,
                    body:trace_id::STRING AS trace_id,
                    body:assessment_name::STRING AS assessment_name,
                    FROM_JSON(
                        body:source::STRING,
                        'STRUCT<source_id: STRING, source_type: STRING>'
                    ) AS source,
                    TIMESTAMP_MILLIS(CAST(body:create_time::DOUBLE AS BIGINT)) AS create_time,
                    TIMESTAMP_MILLIS(
                        CAST(body:last_update_time::DOUBLE AS BIGINT)
                    ) AS last_update_time,
                    FROM_JSON(body:expectation::STRING,
                        'STRUCT<
                            value: STRING,
                            serialized_value: STRUCT<serialization_format: STRING, value: STRING>
                            >
                        ') AS expectation,
                    FROM_JSON(
                        body:feedback::STRING,
                        'STRUCT<
                            value: STRING,
                            error: STRUCT<
                                error_code: STRING,
                                error_message: STRING,
                                stack_trace: STRING
                            >
                         >'
                    ) AS feedback,
                    body:rationale::STRING AS rationale,
                    FROM_JSON(body:metadata::STRING, 'MAP<STRING, STRING>') AS metadata,
                    body:span_id::STRING AS span_id,
                    body:valid::BOOLEAN AS valid
                  )
                ),
                ARRAY()
              ) AS assessments
            FROM
              trace_snapshots ts
              LEFT JOIN latest_tags lt ON ts.trace_id = lt.trace_id
              LEFT JOIN assessments_agg aa ON ts.trace_id = aa.trace_id
              LEFT JOIN spans_agg sa ON ts.trace_id = sa.trace_id
              LEFT JOIN root_span rs ON ts.trace_id = rs.trace_id;
            """

        spark.sql(query)
        _logger.info(f"Successfully created trace archival view: {view_name}")

    except Exception as e:
        raise MlflowException(f"Failed to create trace archival view {view_name}") from e


def _enable_trace_rolling_deletion(experiment_id: str) -> None:
    """
    Enable rolling deletion for traces in the specified experiment.

    This function sets an experiment tag to enable automatic rolling deletion
    of traces based on the configured retention policy.

    Args:
        experiment_id: The MLflow experiment ID to enable rolling deletion for

    Raises:
        MlflowException: If setting the experiment tag fails
    """
    try:
        _logger.debug(f"Enabling trace rolling deletion for experiment {experiment_id}")
        MlflowClient().set_experiment_tag(
            experiment_id, MLFLOW_DATABRICKS_TRACE_ROLLING_DELETION_ENABLED, "true"
        )
        _logger.debug(f"Successfully enabled trace rolling deletion for experiment {experiment_id}")
    except Exception as e:
        error_msg = f"Failed to enable trace rolling deletion for experiment {experiment_id}: {e!s}"
        raise MlflowException(error_msg) from e


def _do_enable_databricks_archival(
    experiment_id: str, catalog: str, schema: str, table_prefix: str = "trace_logs"
) -> str:
    """
    Enable trace archival by orchestrating the full archival enablement process.
    Note that this operation is idempotent such that if the archival is already enabled,
    it will return the existing view name without making any changes.

    Args:
        experiment_id: The MLflow experiment ID to enable archival for
        catalog: The Unity Catalog catalog name where tables will be created
        schema: The Unity Catalog schema name where tables will be created
        table_prefix: Prefix for the archival view name

    Returns:
        The name of the created trace archival view

    Raises:
        MlflowException: If any step of the archival process fails
    """
    trace_archival_location = f"{catalog}.{schema}.{table_prefix}_{experiment_id}"

    try:
        # 1. Create trace destination using client
        # The backend API is idempotent and will return existing configuration if it already exists
        # It will only throw ALREADY_EXISTS error if the table schema versions have changed
        _logger.info(
            f"Creating archival configuration for experiment {experiment_id} in {catalog}.{schema}"
        )

        trace_archive_config = DatabricksTraceServerClient().create_trace_destination(
            experiment_id=experiment_id,
            catalog=catalog,
            schema=schema,
            table_prefix=table_prefix,
        )
        _logger.debug(
            f"Trace archival enabled with Spans table: {trace_archive_config.spans_table_name}, "
            f"Events table: {trace_archive_config.events_table_name}, "
            f"Spans schema version: {trace_archive_config.spans_schema_version}, "
            f"Events schema version: {trace_archive_config.events_schema_version}"
        )

        # 3. Validate schema versions before proceeding
        _validate_schema_versions(
            trace_archive_config.spans_schema_version, trace_archive_config.events_schema_version
        )

        # 4 Create the logical view
        _logger.info(f"Creating trace archival at: {trace_archival_location}")
        _create_genai_trace_view(
            trace_archival_location,
            trace_archive_config.spans_table_name,
            trace_archive_config.events_table_name,
        )

        # 5. Set experiment tag to track the archival location
        MlflowClient().set_experiment_tag(
            experiment_id, MLFLOW_DATABRICKS_TRACE_STORAGE_TABLE, trace_archival_location
        )

        # 6. Enable rolling deletion for the experiment
        _enable_trace_rolling_deletion(experiment_id)

        _logger.info(
            f"Trace archival to Databricks enabled successfully for experiment {experiment_id} "
            f"with target archival available at: {trace_archival_location}"
        )

        return trace_archival_location

    except Exception as e:
        _logger.error(f"Failed to enable trace archival for experiment {experiment_id}: {e!s}")
        raise MlflowException(
            f"Failed to enable trace archival for experiment {experiment_id}: {e!s}"
        ) from e


@experimental(version="3.2.0")
def enable_databricks_trace_archival(
    experiment_id: str, catalog: str, schema: str, table_prefix: str = "trace_logs"
) -> str:
    """
    Enable trace archival for an MLflow experiment by creating Delta tables and views.

    This function sets up the infrastructure needed to archive traces from an MLflow experiment
    to Unity Catalog Delta tables. It:
    1. Calls the Databricks trace server to create trace destination metadata
    2. Creates a logical view that combines the raw otel spans and events tables
       created by trace server
    3. Sets an experiment tag indicating where the archival data is stored

    This function is idempotent - if archival is already enabled for the experiment,
    it returns the existing view name without making any changes.

    TODO: move this orchestration to the mlflow backend once this feature
    graduates from private preview

    Args:
        experiment_id: The MLflow experiment ID to enable archival for.
        catalog: The Unity Catalog catalog name where tables will be created.
        schema: The Unity Catalog schema name where tables will be created.
        table_prefix: The prefix for the archival table and view names. Defaults to "trace_logs".

    Returns:
        The name of the created trace archival view in the format:
        "{catalog}.{schema}.{table_prefix}_{experiment_id}"

    Raises:
        MlflowException: If the trace destination creation fails, table creation fails,
            or experiment tag setting fails.

    Example:
        >>> import mlflow.tracing
        >>> view_name = mlflow.tracing.enable_databricks_archival(
        ...     "12345", "my_catalog", "my_schema", "my_prefix"
        ... )
        >>> print(view_name)
        my_catalog.my_schema.my_prefix_12345
    """

    if importlib.util.find_spec("databricks.agents") is None:
        raise ImportError(
            "The `mlflow[databricks]` package is required to use databricks trace archival."
            "Please install it with `pip install mlflow[databricks]`."
        )

    return _do_enable_databricks_archival(experiment_id, catalog, schema, table_prefix)

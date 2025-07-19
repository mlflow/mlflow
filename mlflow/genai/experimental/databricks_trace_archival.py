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
from mlflow.utils.mlflow_tags import MLFLOW_DATABRICKS_TRACE_STORAGE_TABLE

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
              WITH trace_snapshots AS (
                  -- 1. Extract trace metadata from trace snapshot events
                  SELECT
                    trace_id,
                    -- Parse the JSON body to extract all trace metadata
                    FROM_JSON(
                      body,
                      'STRUCT<
                        trace_id: STRING,
                        client_request_id: STRING,
                        trace_location: STRUCT<
                          type: STRING,
                          mlflow_experiment: STRUCT<experiment_id: STRING>,
                          inference_table: STRUCT<full_table_name: STRING>
                        >,
                        request_preview: STRING,
                        response_preview: STRING,
                        request_time: BIGINT,
                        execution_duration: DOUBLE,
                        state: STRING,
                        trace_metadata: MAP<STRING, STRING>
                      >'
                    ) AS trace_data
                  FROM
                    {events_table}
                  WHERE
                    event_name = '{TRACE_SNAPSHOT_OTEL_EVENT_NAME}'
                ),
                -- 2. Aggregate all spans for a given trace
                spans_agg AS (
                  SELECT
                    trace_id,
                    COLLECT_LIST(
                      NAMED_STRUCT(
                        'span_id', span_id,
                        'trace_id', trace_id,
                        'parent_id', parent_span_id,
                        'start_time', TIMESTAMP_MILLIS(
                            CAST(start_time_unix_nano / 1000000 AS BIGINT)
                        ),
                        'end_time', TIMESTAMP_MILLIS(CAST(end_time_unix_nano / 1000000 AS BIGINT)),
                        'status_code', GET_JSON_OBJECT(status, '$.code'),
                        'status_message', GET_JSON_OBJECT(status, '$.message'),
                        'name', name,
                        'attributes', attributes,
                        'events',
                        CASE
                          WHEN events IS NOT NULL AND size(events) > 0
                          THEN TRANSFORM(
                            events,
                            e -> NAMED_STRUCT(
                              'name', GET_JSON_OBJECT(e, '$.name'),
                              'timestamp', TIMESTAMP_MILLIS(
                                  CAST(
                                      CAST(GET_JSON_OBJECT(e, '$.time_unix_nano') AS BIGINT)
                                      / 1000000
                                      AS BIGINT
                                  )
                              ),
                              'attributes', GET_JSON_OBJECT(e, '$.attributes')
                            )
                          )
                          ELSE ARRAY()
                        END
                      )
                    ) AS spans
                  FROM {spans_table}
                  GROUP BY trace_id
                ),
                -- 3. Aggregated the valid assessments grouped by trace_id
                assessments_agg AS (
                  SELECT
                      trace_id,
                      COLLECT_LIST(
                          FROM_JSON(
                              body,
                              'STRUCT<
                                  assessment_id: STRING,
                                  trace_id: STRING,
                                  assessment_name: STRING,
                                  source: STRUCT<source_id: STRING, source_type: STRING>,
                                  create_time: TIMESTAMP,
                                  last_update_time: TIMESTAMP,
                                  expectation: STRUCT<value: STRING>,
                                  feedback: STRUCT<
                                      value: STRING,
                                      error: STRUCT<error_code: STRING, error_message: STRING>
                                  >,
                                  rationale: STRING,
                                  metadata: MAP<STRING, STRING>,
                                  span_id: STRING
                              >'
                          )
                      ) AS assessments
                  FROM (
                      SELECT
                          trace_id,
                          body
                      FROM {events_table}
                      WHERE event_name = '{ASSESSMENTS_SNAPSHOT_OTEL_EVENT_NAME}'
                          AND attributes['valid'] = 'true'
                      QUALIFY ROW_NUMBER() OVER (
                          PARTITION BY trace_id, attributes['assessment_id']
                          ORDER BY time_unix_nano DESC
                      ) = 1
                  )
                  GROUP BY trace_id
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
                  FROM {events_table}
                  WHERE event_name = '{TAGS_SNAPSHOT_OTEL_EVENT_NAME}'
                  QUALIFY ROW_NUMBER() OVER (PARTITION BY trace_id ORDER BY time_unix_nano DESC) = 1
                )
                -- 5. Main query - join the trace metadata with associated tags,
                -- assessments and spans
                SELECT
                  ts.trace_data.trace_id,
                  ts.trace_data.client_request_id,
                  TIMESTAMP_MILLIS(ts.trace_data.request_time) AS request_time,
                  ts.trace_data.state,
                  ts.trace_data.execution_duration AS execution_duration_ms,
                  ts.trace_data.request_preview AS request,
                  ts.trace_data.response_preview AS response,
                  ts.trace_data.trace_metadata,
                  COALESCE(FROM_JSON(lt.tag_json, 'MAP<STRING, STRING>'), MAP()) AS tags,
                  ts.trace_data.trace_location,
                  COALESCE(aa.assessments, ARRAY()) AS assessments,
                  COALESCE(sa.spans, ARRAY()) AS spans
                FROM trace_snapshots ts
                  LEFT JOIN latest_tags lt ON ts.trace_id = lt.trace_id
                  LEFT JOIN assessments_agg aa ON ts.trace_id = aa.trace_id
                  LEFT JOIN spans_agg sa ON ts.trace_id = sa.trace_id;
            """

        spark.sql(query)
        _logger.info(f"Successfully created trace archival view: {view_name}")

    except Exception as e:
        raise MlflowException(f"Failed to create trace archival view {view_name}") from e


def _do_enable_databricks_archival(
    experiment_id: str, catalog: str, schema: str, table_prefix: str = "trace_logs"
) -> str:
    """
    Enable trace archival by orchestrating the full archival enablement process.

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
    # Check if archival is already enabled by looking for the experiment tag
    mlflow_client = MlflowClient()
    try:
        experiment = mlflow_client.get_experiment(experiment_id)
        if experiment and experiment.tags:
            existing_view_name = experiment.tags.get(MLFLOW_DATABRICKS_TRACE_STORAGE_TABLE)
            if existing_view_name:
                _logger.info(
                    f"Trace archival already enabled for experiment {experiment_id}. "
                    f"Using existing view: {existing_view_name}"
                )
                return existing_view_name

    except Exception as e:
        _logger.debug(f"Could not check experiment tags for {experiment_id}: {e}")
        # Continue with normal flow if we can't check tags

    trace_archival_location = f"{catalog}.{schema}.{table_prefix}_{experiment_id}"

    try:
        # 1. Create trace destination using client
        _logger.info(
            f"Creating archival configuration for experiment {experiment_id} in {catalog}.{schema}"
        )

        try:
            trace_archive_config = DatabricksTraceServerClient().create_trace_destination(
                experiment_id=experiment_id,
                catalog=catalog,
                schema=schema,
                table_prefix=table_prefix,
            )
        except MlflowException as e:
            # The backend API is not idempotent, so we need to handle the ALREADY_EXISTS error
            if "ALREADY_EXISTS" in str(e):
                _logger.info(
                    f"Trace archival already exists for experiment {experiment_id}. "
                    f"Returning expected view: {trace_archival_location}"
                )
                return trace_archival_location
            else:
                # Re-raise other MlflowExceptions
                raise
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
        mlflow_client.set_experiment_tag(
            experiment_id, MLFLOW_DATABRICKS_TRACE_STORAGE_TABLE, trace_archival_location
        )

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

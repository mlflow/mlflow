"""
Trace archival functionality for MLflow that enables archiving traces to Delta tables.
"""

import importlib.util
import logging
from typing import Optional

from mlflow.exceptions import MlflowException
from mlflow.genai.experimental.databricks_trace_storage_config import (
    DatabricksTraceDeltaStorageConfig,
)
from mlflow.protos.databricks_trace_server_pb2 import (
    CreateTraceDestinationRequest,
    GetTraceDestinationRequest,
)
from mlflow.protos.databricks_trace_server_pb2 import (
    TraceDestination as ProtoTraceDestination,
)
from mlflow.protos.databricks_trace_server_pb2 import (
    TraceLocation as ProtoTraceLocation,
)
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.mlflow_tags import MLFLOW_DATABRICKS_TRACE_STORAGE_TABLE
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import call_endpoint

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


@experimental(version="3.2.0")
def get_databricks_trace_storage_config(
    experiment_id: str,
) -> Optional[DatabricksTraceDeltaStorageConfig]:
    """
    Get the trace storage configuration for an experiment if archival is enabled.

    This function checks if trace archival is configured for the given experiment by calling
    the GetTraceDestination API. If archival is enabled, it returns the configuration including
    table names and schema versions. If not configured, it returns None.

    Args:
        experiment_id: The MLflow experiment ID to check for trace archival configuration.

    Returns:
        DatabricksTraceDeltaStorageConfig if archival is enabled for the experiment,
        None if archival is not configured.

    Raises:
        MlflowException: If there's an error calling the API (other than 404/not found).

    Example:
        >>> config = get_databricks_trace_storage_config("12345")
        >>> if config:
        ...     print(f"Spans table: {config.spans_table_name}")
        ...     print(f"Events table: {config.events_table_name}")
        ... else:
        ...     print("Trace archival not configured for this experiment")
    """
    if importlib.util.find_spec("databricks.agents") is None:
        return None

    try:
        # Create proto request with experiment ID
        proto_trace_location = ProtoTraceLocation()
        proto_trace_location.type = ProtoTraceLocation.TraceLocationType.MLFLOW_EXPERIMENT
        proto_trace_location.mlflow_experiment.experiment_id = experiment_id

        proto_request = GetTraceDestinationRequest(
            trace_location=proto_trace_location,
        )

        # Call the GetTraceDestination API
        request_body = message_to_json(proto_request)

        _logger.debug(f"Checking trace archival configuration for experiment {experiment_id}")

        trace_destination_proto = call_endpoint(
            host_creds=get_databricks_host_creds(),
            endpoint=f"/api/2.0/tracing/trace-destinations/mlflow-experiments/{experiment_id}",
            method="GET",
            json_body=request_body,
            response_proto=ProtoTraceDestination(),
        )

        # Convert to our config object
        config = DatabricksTraceDeltaStorageConfig.from_proto(trace_destination_proto)

        _logger.debug(
            f"Found trace archival configuration for experiment {experiment_id}: "
            f"spans={config.spans_table_name}, events={config.events_table_name}"
        )

        return config

    except MlflowException as e:
        # Check if this is a 404 (not configured) vs other error
        if "404" in str(e) or "not found" in str(e).lower():
            _logger.debug(f"No trace archival configuration found for experiment {experiment_id}")
            return None
        else:
            _logger.error(
                f"Error checking trace archival configuration for experiment {experiment_id}: {e}"
            )
            raise
    except Exception as e:
        _logger.error(
            f"Unexpected error checking trace archival configuration for experiment "
            f"{experiment_id}: {e}"
        )
        raise MlflowException(
            f"Failed to check trace archival configuration for experiment {experiment_id}: {e}"
        ) from e


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
    trace_archival_location = f"{catalog}.{schema}.{table_prefix}_{experiment_id}"

    try:
        # 1. Create proto request directly (internal implementation detail)
        proto_trace_location = ProtoTraceLocation()
        proto_trace_location.type = ProtoTraceLocation.TraceLocationType.MLFLOW_EXPERIMENT
        proto_trace_location.mlflow_experiment.experiment_id = experiment_id

        proto_request = CreateTraceDestinationRequest(
            trace_location=proto_trace_location,
            uc_catalog=catalog,
            uc_schema=schema,
            uc_table_prefix=table_prefix,
        )

        # 2. Call the trace server CreateTraceDestination API
        request_body = message_to_json(proto_request)

        _logger.info(
            f"Creating archival configuration for experiment {experiment_id} in {catalog}.{schema}"
        )
        trace_archive_config_proto = call_endpoint(
            host_creds=get_databricks_host_creds(),
            endpoint="/api/2.0/tracing/trace-destinations",
            method="POST",
            json_body=request_body,
            response_proto=ProtoTraceDestination(),
        )
        trace_archive_config = DatabricksTraceDeltaStorageConfig.from_proto(
            trace_archive_config_proto
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
        from mlflow.tracking import MlflowClient

        MlflowClient().set_experiment_tag(
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

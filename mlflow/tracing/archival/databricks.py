"""
Trace archival functionality for MLflow that enables archiving traces to Delta tables.
"""

import logging

import mlflow
from mlflow.entities.trace_archive_configuration import TraceArchiveConfiguration
from mlflow.environment_variables import MLFLOW_HTTP_REQUEST_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey
from mlflow.protos.databricks_trace_server_pb2 import (
    CreateTraceDestinationRequest,
    TraceLocation as ProtoTraceLocation,
    TraceDestination as ProtoTraceDestination,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.mlflow_tags import MLFLOW_EXPERIMENT_TRACE_ARCHIVAL_TABLE
from mlflow.utils.rest_utils import call_endpoint
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils.proto_json_utils import message_to_json

_logger = logging.getLogger(__name__)

# Supported schema version for trace archival
SUPPORTED_SCHEMA_VERSION = "v1"


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
    
    _logger.debug(f"Schema version validation passed: spans={spans_version}, events={events_version}")


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
            WITH root_spans AS ( -- 1. Filter and extract attributes efficiently
              SELECT
                trace_id,
                -- Extract all needed attributes in single pass
                NULL AS client_request_id, -- client_request_id is populated via separate path, not available in span attributes
                attributes['{SpanAttributeKey.INPUTS}'] AS request,
                attributes['{SpanAttributeKey.OUTPUTS}'] AS response,
                attributes['{SpanAttributeKey.EXPERIMENT_ID}'] AS experiment_id,
                -- Build proper trace_metadata with only relevant keys in the root span attributes
                MAP_FILTER(
                  attributes,
                  (k, v) -> k IN (
                    '{TraceMetadataKey.SOURCE_RUN}',
                    '{TraceMetadataKey.MODEL_ID}', 
                    '{TraceMetadataKey.INPUTS}',
                    '{TraceMetadataKey.OUTPUTS}',
                    '{TraceMetadataKey.TOKEN_USAGE}',
                    '{TraceMetadataKey.SIZE_STATS}',
                    '{TraceMetadataKey.TRACE_USER}',
                    '{TraceMetadataKey.TRACE_SESSION}'
                  )
                ) AS trace_metadata,
                TIMESTAMP_MILLIS(CAST(start_time_unix_nano / 1000000 AS BIGINT)) AS request_time,
                GET_JSON_OBJECT(status, '$.code') AS state,
                (end_time_unix_nano - start_time_unix_nano) / 1000000 AS execution_duration_ms
              FROM
                {spans_table}
              WHERE
                (
                  parent_span_id = ''
                  OR parent_span_id IS NULL
                )
                AND start_time_unix_nano IS NOT NULL
                AND attributes['{SpanAttributeKey.EXPERIMENT_ID}'] IS NOT NULL
            ),
            -- 2. Group the spans by trace_id
            spans_agg AS (
              SELECT
                trace_id,
                COLLECT_LIST(
                  NAMED_STRUCT(
                    'span_id',span_id,
                    'trace_id', trace_id,
                    'parent_id', parent_span_id,
                    'start_time', TIMESTAMP_MILLIS(CAST(start_time_unix_nano / 1000000 AS BIGINT)),
                    'end_time', TIMESTAMP_MILLIS(CAST(end_time_unix_nano / 1000000 AS BIGINT)),
                    'status_code', GET_JSON_OBJECT(status, '$.code'),
                    'status_message', GET_JSON_OBJECT(status, '$.message'),
                    'name', name,
                    'attributes', attributes,
                    'events',
                    CASE
                      WHEN
                        events IS NOT NULL
                        AND size(events) > 0
                      THEN
                        TRANSFORM(
                          events,
                          e -> NAMED_STRUCT(
                            'name', GET_JSON_OBJECT(e, '$.name'),
                            'timestamp', TIMESTAMP_MILLIS(CAST(CAST(GET_JSON_OBJECT(e, '$.time_unix_nano') AS BIGINT) / 1000000 AS BIGINT)),
                            'attributes', GET_JSON_OBJECT(e, '$.attributes')
                          )
                        )
                      ELSE ARRAY()
                    END
                  )
                ) AS spans
              FROM
                {spans_table}
              GROUP BY
                trace_id
            ),
            -- 3. Batch process the events associated with assessments
            assessment_events_parsed AS (
              SELECT
                trace_id,
                -- Use single FROM_JSON instead of multiple GET_JSON_OBJECT calls since it is more efficient
                FROM_JSON(
                  body,
                  'STRUCT<
                        assessment_id: STRING,
                        trace_id: STRING,
                        assessment_name: STRING,
                        source: STRUCT<source_id: STRING, source_type: STRING>,
                        create_time: STRING,
                        last_update_time: STRING,
                        expectation: STRUCT<value: STRING>,
                        feedback: STRUCT<
                          value: STRING,
                          error: STRUCT<error_code: STRING, error_message: STRING>
                        >,
                        rationale: STRING,
                        metadata: STRING,
                        span_id: STRING
                      >'
                ) AS parsed_body
              FROM
                {events_table}
              WHERE
                event_name = 'genai.assessments.update'
            ),
            -- 4. Aggregate assessments by trace_id
            assessments_agg AS (
              SELECT
                trace_id,
                COLLECT_LIST(
                  NAMED_STRUCT(
                    'assessment_id', parsed_body.assessment_id,
                    'trace_id', parsed_body.trace_id,
                    'name', parsed_body.assessment_name,
                    'source', parsed_body.source,
                    'create_time', CAST(parsed_body.create_time AS TIMESTAMP),
                    'last_update_time', CAST(parsed_body.last_update_time AS TIMESTAMP),
                    'expectation', parsed_body.expectation,
                    'feedback', parsed_body.feedback,
                    'rationale', parsed_body.rationale,
                    'metadata', FROM_JSON(parsed_body.metadata, 'MAP<STRING, STRING>'),
                    'span_id', parsed_body.span_id
                  )
                ) AS assessments
              FROM
                assessment_events_parsed
              GROUP BY
                trace_id
            ),
            -- 5. Tag events are snapshots so apply a window function to resolve the latest tags for each trace_id
            latest_tags AS (
              SELECT
                trace_id,
                -- Use FIRST_VALUE with proper window frame
                FIRST_VALUE(body) OVER (
                    PARTITION BY trace_id
                    ORDER BY time_unix_nano DESC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                  ) AS tag_json
              FROM
                {events_table}
              WHERE
                -- tags update events provide a snapshot of the tags at the time of the event
                event_name = 'genai.tags.update'
              QUALIFY
                ROW_NUMBER() OVER (PARTITION BY trace_id ORDER BY time_unix_nano DESC) = 1
            )
            -- 6. Main query with optimized structure joining the root spans with the assessments and tags
            SELECT
              rs.trace_id,
              rs.client_request_id,
              rs.request_time,
              rs.state,
              rs.execution_duration_ms,
              rs.request,
              rs.response,
              rs.trace_metadata,
              -- Optimized tags processing
              COALESCE(FROM_JSON(lt.tag_json, 'MAP<STRING, STRING>'), MAP()) AS tags,
              -- MLflow experiment-only trace_location (filtered for valid experiment_id)
              NAMED_STRUCT(
                'type', 'MLFLOW_EXPERIMENT',
                'mlflow_experiment', NAMED_STRUCT('experiment_id', rs.experiment_id)
              ) AS trace_location,
              COALESCE(aa.assessments, ARRAY()) AS assessments,
              COALESCE(sa.spans, ARRAY()) AS spans
            FROM
              root_spans rs
                LEFT JOIN latest_tags lt
                  ON rs.trace_id = lt.trace_id
                LEFT JOIN assessments_agg aa
                  ON rs.trace_id = aa.trace_id
                LEFT JOIN spans_agg sa
                  ON rs.trace_id = sa.trace_id
            """
        
        spark.sql(query)
        _logger.info(f"Successfully configured Databricks trace archival to: {view_name}")
        
    except Exception as e:
        raise MlflowException(f"Failed to configure Databricks trace archival to: {view_name}") from e


def _do_enable_databricks_archival(experiment_id: str, catalog: str, schema: str, table_prefix: str = "trace_logs") -> str:
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
    trace_archival_view_name = f"{catalog}.{schema}.{table_prefix}_{experiment_id}"
    
    try:
        # 1. Create proto request
        proto_trace_location = ProtoTraceLocation()
        proto_trace_location.type = ProtoTraceLocation.TraceLocationType.MLFLOW_EXPERIMENT
        proto_trace_location.mlflow_experiment.experiment_id = experiment_id
        
        proto_request = CreateTraceDestinationRequest(
            trace_location=proto_trace_location,
            uc_catalog=catalog,
            uc_schema=schema,
            uc_table_prefix=table_prefix
        )
        
        # 2. Call the backend to create the trace archival metadata and the raw otel spans/events tables
        request_body = message_to_json(proto_request)
        
        _logger.info(f"Creating Databricks archival configuration for experiment {experiment_id} in {catalog}.{schema}")
        trace_archive_config_proto = call_endpoint(
            host_creds=get_databricks_host_creds(),
            endpoint="/api/2.0/tracing/trace-destinations",
            method="POST",
            json_body=request_body,
            response_proto=ProtoTraceDestination(),
        )
        trace_archive_config = TraceArchiveConfiguration.from_proto(trace_archive_config_proto)
        
        _logger.debug(f"Databricks trace archival enabled with Spans table: {trace_archive_config.spans_table_name}, "
                    f"Events table: {trace_archive_config.events_table_name}, "
                    f"Spans schema version: {trace_archive_config.spans_schema_version}, "
                    f"Events schema version: {trace_archive_config.events_schema_version}")
        
        # 3. Validate schema versions before proceeding
        _validate_schema_versions(trace_archive_config.spans_schema_version, trace_archive_config.events_schema_version)
        
        # 4 Create the logical view
        _logger.info(f"Creating Databricks trace archival to: {trace_archival_view_name}")
        _create_genai_trace_view(trace_archival_view_name, trace_archive_config.spans_table_name, trace_archive_config.events_table_name)
        
        # 5. Set experiment tag to track the archival location
        from mlflow.tracking import MlflowClient
        MlflowClient().set_experiment_tag(experiment_id, MLFLOW_EXPERIMENT_TRACE_ARCHIVAL_TABLE, trace_archival_view_name)
        
        _logger.info(f"Databricks trace archival enabled successfully for experiment {experiment_id} with target archival available at: {trace_archival_view_name}")
        
        return trace_archival_view_name
        
    except Exception as e:
        raise MlflowException(
            f"Failed to enable Databricks archival for experiment {experiment_id}: {str(e)}"
        ) from e


@experimental
def enable_databricks_archival(experiment_id: str, catalog: str, schema: str, table_prefix: str = "trace_logs") -> str:
    """
    Enable trace archival for an MLflow experiment by creating Delta tables and views.
    
    This function sets up the infrastructure needed to archive traces from an MLflow experiment
    to Unity Catalog Delta tables. It:
    1. Calls the Databricks trace server to create trace destination metadata
    2. Creates a logical view that combines the raw otel spans and events tables created by trace server
    3. Sets an experiment tag indicating where the archival data is stored
    
    TODO: move this orchestration to the mlflow backend once this feature graduates from private preview

    Args:
        experiment_id: The MLflow experiment ID to enable archival for.
        catalog: The Unity Catalog catalog name where tables will be created.
        schema: The Unity Catalog schema name where tables will be created.
        table_prefix: Prefix for the archival view name (defaults to "trace_logs")
        
    Returns:
        The name of the created trace archival view in the format:
        "{catalog}.{schema}.{table_prefix}_{experiment_id}"
        
    Raises:
        MlflowException: If the trace destination creation fails, table creation fails,
            or experiment tag setting fails.
            
    Example:
        >>> import mlflow.tracing
        >>> view_name = mlflow.tracing.enable_databricks_archival("12345", "my_catalog", "my_schema", "my_prefix")
        >>> print(view_name)
        my_catalog.my_schema.my_prefix_12345
    """
    return _do_enable_databricks_archival(experiment_id, catalog, schema, table_prefix)
"""
Trace archival functionality for MLflow that enables archiving traces to Delta tables.
"""

import logging

import mlflow
from mlflow.environment_variables import MLFLOW_HTTP_REQUEST_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_trace_server_pb2 import (
    CreateTraceDestinationRequest,
    TraceLocation as ProtoTraceLocation,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import http_request
from mlflow.utils._spark_utils import _get_active_spark_session
from google.protobuf.json_format import MessageToDict

_logger = logging.getLogger(__name__)

@experimental(version="3.2.0")
def enable_trace_archival(experiment_id: str, catalog: str, schema: str, table_prefix: str = "trace_logs") -> str:
    """
    Enable trace archival for an MLflow experiment by creating Delta tables and views.
    
    This function sets up the infrastructure needed to archive traces from an MLflow experiment
    to Unity Catalog Delta tables. It:
    1. Calls the Databricks trace server to create trace destination metadata
    2. Creates a logical view that combines the raw otel spans and events tables created by trace server
    3. Sets an experiment tag indicating where the archival data is stored
    
    Args:
        experiment_id: The MLflow experiment ID to enable archival for.
        catalog: The Unity Catalog catalog name where tables will be created.
        schema: The Unity Catalog schema name where tables will be created.
        
    Returns:
        The name of the created trace archival view in the format:
        "{catalog}.{schema}.trace_logs_{experiment_id}"
        P
    Raises:
        MlflowException: If the trace destination creation fails, table creation fails,
            or experiment tag setting fails.
            
    Example:
        >>> import mlflow.tracing
        >>> view_name = mlflow.tracing.enable_trace_archival("12345", "my_catalog", "my_schema")
        >>> print(view_name)
        my_catalog.my_schema.trace_logs_12345
    """
    try:
        # 1. Create proto request directly (internal implementation detail)
        proto_trace_location = ProtoTraceLocation()
        proto_trace_location.type = ProtoTraceLocation.TraceLocationType.MLFLOW_EXPERIMENT
        proto_trace_location.mlflow_experiment.experiment_id = experiment_id
        
        proto_request = CreateTraceDestinationRequest(
            trace_location=proto_trace_location,
            uc_catalog=catalog,
            uc_schema=schema,
            uc_table_prefix=f"{table_prefix}_{experiment_id}"
        )
        
        # 2. Call the trace server CreateTraceDestination API
        request_body = MessageToDict(proto_request, preserving_proto_field_name=True)
        
        # Debug logging
        host_creds = get_databricks_host_creds()
        timeout_value = MLFLOW_HTTP_REQUEST_TIMEOUT.get()
        
        # Increase timeout for debugging - can be overridden with env var
        timeout_value = max(timeout_value, 10) 
        
        _logger.info(f"Creating trace destination for experiment {experiment_id} in {catalog}.{schema}")
        
        import time
        start_time = time.time()
        
        try:
            res = http_request(
                host_creds=host_creds,
                endpoint="/api/2.0/tracing/trace-destinations",
                method="POST",
                timeout=timeout_value,
                json=request_body,
            )
            
        except Exception as e:
            _logger.error(f"Failed to create trace destination for experiment {experiment_id}: {str(e)}")
            raise
        
        if res.status_code != 200:
            raise MlflowException(
                f"Failed to create trace destination for experiment {experiment_id}. "
                f"Status: {res.status_code}, Response: {res.text}"
            )
        
        # 3. Parse response to get table names
        response_data = res.json()
        spans_table_name = response_data["spans_table_name"]
        events_table_name = response_data["events_table_name"]
        _logger.debug(f"Trace destination created. Spans table: {spans_table_name}, "
                    f"Events table: {events_table_name}")
        
        
        # 4. Create the logical view
        # TODO: validate the table version before creating the view 
        trace_archival_location = f"{catalog}.{schema}.trace_logs_{experiment_id}"
        _logger.info(f"Creating trace archival at: {trace_archival_location}")
        _create_genai_trace_view(trace_archival_location, spans_table_name, events_table_name)
        
        # 5. Set experiment tag to track the archival location
        mlflow.set_experiment_tag("trace_archival_table", trace_archival_location)
        
        _logger.info(f"Trace archival enabled successfully for experiment {experiment_id}. "
                    f"View created: {trace_archival_location}")
        
        return trace_archival_location
        
    except Exception as e:
        _logger.error(f"Failed to enable trace archival for experiment {experiment_id}: {str(e)}")
        raise MlflowException(
            f"Failed to enable trace archival for experiment {experiment_id}: {str(e)}"
        ) from e

def _create_genai_trace_view(final_view: str, raw_spans_table: str, raw_events_table: str) -> None:
    """
    Create a logical view for GenAI trace data that combines spans and events tables.
    
    Args:
        final_view: The name of the final view to create (e.g., 'catalog.schema.trace_logs_12345')
        raw_spans_table: The name of the table containing raw spans data
        raw_events_table: The name of the table containing raw events data
    """
    try:
        spark = _get_active_spark_session()
        if spark is None:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
        
        query = f"""
        CREATE OR REPLACE VIEW {final_view} AS
        WITH root_spans AS ( -- 1. Filter and extract attributes efficiently
          SELECT
            trace_id,
            -- Extract all needed attributes in single pass
            attributes['mlflow.traceRequestId'] AS client_request_id,
            attributes['mlflow.spanInputs'] AS request,
            attributes['mlflow.spanOutputs'] AS response,
            attributes['mlflow.experimentId'] AS experiment_id,
            attributes['inference.table_name'] AS inference_table_name,
            attributes AS trace_metadata,
            TIMESTAMP(start_time_unix_nano / 1000000000) AS request_time,
            GET_JSON_OBJECT(status, '$.code') AS state,
            (end_time_unix_nano - start_time_unix_nano) / 1000000 AS execution_duration_ms
          FROM
            {raw_spans_table}
          WHERE
            (
              parent_span_id = ''
              OR parent_span_id IS NULL
            )
            AND start_time_unix_nano IS NOT NULL
        ),
        -- 2. Optimize spans aggregation
        spans_agg AS (
          SELECT
            trace_id,
            COLLECT_LIST(
              NAMED_STRUCT(
                'span_id',
                span_id,
                'trace_id',
                trace_id,
                'parent_id',
                parent_span_id,
                'start_time',
                TIMESTAMP(start_time_unix_nano / 1000000000),
                'end_time',
                TIMESTAMP(end_time_unix_nano / 1000000000),
                'status_code',
                GET_JSON_OBJECT(status, '$.code'),
                'status_message',
                GET_JSON_OBJECT(status, '$.message'),
                'name',
                name,
                'attributes',
                attributes,
                -- Simplified event processing
                'events',
                CASE
                  WHEN
                    events IS NOT NULL
                    AND size(events) > 0
                  THEN
                    TRANSFORM(
                      events,
                      e -> NAMED_STRUCT(
                        'name',
                        GET_JSON_OBJECT(e, '$.name'),
                        'timestamp',
                        TIMESTAMP(CAST(GET_JSON_OBJECT(e, '$.time_unix_nano') AS BIGINT) / 1000000000),
                        'attributes',
                        GET_JSON_OBJECT(e, '$.attributes')
                      )
                    )
                  ELSE ARRAY()
                END
              )
            ) AS spans
          FROM
            {raw_spans_table}
          GROUP BY
            trace_id
        ),
        -- 3. Batch JSON parsing for assessments
        assessment_events_parsed AS (
          SELECT
            trace_id,
            -- Use single FROM_JSON instead of multiple GET_JSON_OBJECT calls
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
            {raw_events_table}
          WHERE
            event_name = 'genai.assessments.insert'
        ),
        -- 4. Aggregate assessments
        assessments_agg AS (
          SELECT
            trace_id,
            COLLECT_LIST(
              NAMED_STRUCT(
                'assessment_id',
                parsed_body.assessment_id,
                'trace_id',
                parsed_body.trace_id,
                'name',
                parsed_body.assessment_name,
                'source',
                parsed_body.source,
                'create_time',
                CAST(parsed_body.create_time AS TIMESTAMP),
                'last_update_time',
                CAST(parsed_body.last_update_time AS TIMESTAMP),
                'expectation',
                parsed_body.expectation,
                'feedback',
                parsed_body.feedback,
                'rationale',
                parsed_body.rationale,
                'metadata',
                FROM_JSON(parsed_body.metadata, 'MAP<STRING, STRING>'),
                'span_id',
                parsed_body.span_id
              )
            ) AS assessments
          FROM
            assessment_events_parsed
          GROUP BY
            trace_id
        ),
        -- 5. Optimize tags with window function
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
            {raw_events_table}
          WHERE
            event_name = 'genai.tags.insert'
          QUALIFY
            ROW_NUMBER() OVER (PARTITION BY trace_id ORDER BY time_unix_nano DESC) = 1
        )
        -- 6. Main query with optimized structure
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
          -- Simplified trace_location construction
          NAMED_STRUCT(
            'type',
            CASE
              WHEN rs.experiment_id IS NOT NULL THEN 'mlflow_experiment'
              WHEN rs.inference_table_name IS NOT NULL THEN 'inference_table'
              ELSE NULL
            END,
            'mlflow_experiment',
            CASE
              WHEN rs.experiment_id IS NOT NULL THEN NAMED_STRUCT('experiment_id', rs.experiment_id)
              ELSE CAST(NULL AS STRUCT<experiment_id: STRING>)
            END,
            'inference_table',
            CASE
              WHEN
                rs.inference_table_name IS NOT NULL
              THEN
                NAMED_STRUCT('full_table_name', rs.inference_table_name)
              ELSE CAST(NULL AS STRUCT<full_table_name: STRING>)
            END
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
        _logger.info(f"Successfully created trace archival view: {final_view}")
        
    except Exception as e:
        raise MlflowException(f"Failed to create trace archival view {final_view}: {str(e)}") from e
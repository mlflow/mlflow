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
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import http_request
from mlflow.utils._spark_utils import _get_active_spark_session
from google.protobuf.json_format import MessageToDict

_logger = logging.getLogger(__name__)


def enable_trace_archival(experiment_id: str, catalog: str, schema: str, table_prefix: str = "trace_logs") -> str:
    """
    Enable trace archival for an MLflow experiment by creating Delta tables and views.
    
    This function sets up the infrastructure needed to archive traces from an MLflow experiment
    to Unity Catalog Delta tables. It:
    1. Calls the Databricks trace server to create trace destination metadata
    2. Creates spans and events tables in the specified Unity Catalog location
    3. Creates a logical view that combines the spans and events tables
    4. Sets an experiment tag indicating where the archival data is stored
    
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
        
        # 4. Create the spans and events tables
        _logger.debug("Creating spans and events tables. Spans table: {spans_table_name}, "
                    f"Events table: {events_table_name}")
        _create_spans_table(spans_table_name)
        _create_events_table(events_table_name)
        
        # 5. Create the logical view
        trace_archival_location = f"{catalog}.{schema}.trace_logs_{experiment_id}"
        _logger.info(f"Creating trace archival at: {trace_archival_location}")
        _create_genai_trace_view(trace_archival_location, spans_table_name, events_table_name)
        
        # 6. Set experiment tag to track the archival location
        mlflow.set_experiment_tag("trace_archival_table", trace_archival_location)
        
        _logger.info(f"Trace archival enabled successfully for experiment {experiment_id}. "
                    f"View created: {trace_archival_location}")
        
        return trace_archival_location
        
    except Exception as e:
        _logger.error(f"Failed to enable trace archival for experiment {experiment_id}: {str(e)}")
        raise MlflowException(
            f"Failed to enable trace archival for experiment {experiment_id}: {str(e)}"
        ) from e


def _create_spans_table(spans_table_name: str) -> None:
    """
    Create a Delta table for storing OpenTelemetry spans.
    
    Args:
        spans_table_name: The full qualified name of the spans table to create.
    """
    try:
        spark = _get_active_spark_session()
        if spark is None:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
        
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {spans_table_name}
            (
              trace_id STRING,
              span_id STRING,
              trace_state STRING,
              parent_span_id STRING,
              flags INT,
              name STRING,
              kind STRING,
              start_time_unix_nano LONG,
              end_time_unix_nano LONG,
              attributes MAP<STRING, STRING>,
              dropped_attributes_count INT,
              events ARRAY<STRING>,
              dropped_events_count INT,
              links ARRAY<STRING>,
              dropped_links_count INT,
              status_code STRING,
              status_message STRING
            )
            USING DELTA;
        """)
        
        _logger.debug(f"Successfully created spans table: {spans_table_name}")
        
    except Exception as e:
        raise MlflowException(f"Failed to create spans table {spans_table_name}: {str(e)}") from e


def _create_events_table(events_table_name: str) -> None:
    """
    Create a Delta table for storing OpenTelemetry events.
    
    Args:
        events_table_name: The full qualified name of the events table to create.
    """
    try:
        spark = _get_active_spark_session()
        if spark is None:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
        
        spark.sql(f"""  
            CREATE TABLE IF NOT EXISTS {events_table_name} (
              event_name STRING,
              trace_id STRING,
              span_id STRING,
              time_unix_nano LONG,
              observed_time_unix_nano LONG,
              severity_number STRING,
              severity_text STRING,
              body STRING,
              attributes MAP<STRING, STRING>,
              dropped_attributes_count INT,
              flags INT
            )
            USING DELTA;
        """)
        
        _logger.debug(f"Successfully created events table: {events_table_name}")
        
    except Exception as e:
        raise MlflowException(f"Failed to create events table {events_table_name}: {str(e)}") from e


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
        -- Identify root spans (spans without parent or with parent outside trace)
        WITH root_spans AS (
          SELECT *
          FROM {raw_spans_table}
          WHERE parent_span_id = '' OR parent_span_id IS NULL
        ),

        -- Parse attributes MAP for root spans and extract trace metadata
        trace_metadata AS (
          SELECT
            trace_id,
            attributes['mlflow.traceRequestId'] AS client_request_id,
            TIMESTAMP(start_time_unix_nano / 1000000000) AS request_time,
            status_code AS state,
            (end_time_unix_nano - start_time_unix_nano) / 1000000 AS execution_duration_ms,
            attributes['mlflow.spanInputs'] AS request,
            attributes['mlflow.spanOutputs'] AS response,
            -- Store the full attributes MAP as trace_metadata
            attributes AS trace_metadata,
            -- Create trace_location structure based on attributes
            CASE 
              WHEN attributes['mlflow.experimentId'] IS NOT NULL THEN 
                NAMED_STRUCT(
                  'type', 'mlflow_experiment',
                  'mlflow_experiment', NAMED_STRUCT(
                    'experiment_id', attributes['mlflow.experimentId']
                  ),
                  'inference_table', CAST(NULL AS STRUCT<full_table_name: STRING>)
                )
              WHEN attributes['inference.table_name'] IS NOT NULL THEN 
                NAMED_STRUCT(
                  'type', 'inference_table',
                  'mlflow_experiment', CAST(NULL AS STRUCT<experiment_id: STRING>),
                  'inference_table', NAMED_STRUCT(
                    'full_table_name', attributes['inference.table_name']
                  )
                )
              ELSE 
                NAMED_STRUCT(
                  'type', CAST(NULL AS STRING),
                  'mlflow_experiment', CAST(NULL AS STRUCT<experiment_id: STRING>),
                  'inference_table', CAST(NULL AS STRUCT<full_table_name: STRING>)
                )
            END AS trace_location
          FROM root_spans
        ),

        -- Process tags from events (taking the latest event)
        latest_tags AS (
          SELECT 
            trace_id,
            body AS tag_json
          FROM (
            SELECT 
              trace_id,
              body,
              ROW_NUMBER() OVER (PARTITION BY trace_id ORDER BY time_unix_nano DESC) AS rn
            FROM {raw_events_table}
            WHERE event_name = 'genai.tags.insert'
          ) ranked_tags
          WHERE rn = 1
        ),

        -- Collect and parse assessment events
        assessment_events AS (
          SELECT
            trace_id,
            -- Parse the JSON body string into individual fields
            GET_JSON_OBJECT(body, '$.assessment_id') AS assessment_id,
            GET_JSON_OBJECT(body, '$.trace_id') AS assessment_trace_id,
            GET_JSON_OBJECT(body, '$.name') AS assessment_name,
            GET_JSON_OBJECT(body, '$.source.source_id') AS source_id,
            GET_JSON_OBJECT(body, '$.source.source_type') AS source_type,
            CAST(GET_JSON_OBJECT(body, '$.create_time_ms') AS LONG) AS create_time_ms,
            CAST(GET_JSON_OBJECT(body, '$.last_update_time_ms') AS LONG) AS last_update_time_ms,
            GET_JSON_OBJECT(body, '$.expectation.value') AS expectation_value,
            GET_JSON_OBJECT(body, '$.feedback.value') AS feedback_value,
            GET_JSON_OBJECT(body, '$.feedback.error.error_code') AS feedback_error_code,
            GET_JSON_OBJECT(body, '$.feedback.error.error_message') AS feedback_error_message,
            GET_JSON_OBJECT(body, '$.rationale') AS rationale,
            GET_JSON_OBJECT(body, '$.metadata') AS metadata_json,
            GET_JSON_OBJECT(body, '$.span_id') AS assessment_span_id
          FROM {raw_events_table}
          WHERE event_name = 'genai.assessments.insert'
        ),

        assessments_transformed AS (
          SELECT
            trace_id,
            NAMED_STRUCT(
              'assessment_id', assessment_id,
              'trace_id', assessment_trace_id,
              'name', assessment_name,
              'source', NAMED_STRUCT(
                'source_id', source_id,
                'source_type', source_type
              ),
              'create_time', TIMESTAMP(create_time_ms / 1000),
              'last_update_time', TIMESTAMP(last_update_time_ms / 1000),
              'expectation', NAMED_STRUCT('value', expectation_value),
              'feedback', NAMED_STRUCT(
                'value', feedback_value,
                'error', NAMED_STRUCT(
                  'error_code', feedback_error_code,
                  'error_message', feedback_error_message
                )
              ),
              'rationale', rationale,
              'metadata', FROM_JSON(metadata_json, 'MAP<STRING, STRING>'),
              'span_id', assessment_span_id
            ) AS assessment
          FROM assessment_events
        ),

        -- Group assessments by trace_id
        assessments_grouped AS (
          SELECT
            trace_id,
            COLLECT_LIST(assessment) AS assessments
          FROM assessments_transformed
          GROUP BY trace_id
        ),

        -- Transform all spans - handle events as ARRAY<STRING>
        transformed_spans AS (
          SELECT
            span_id,
            trace_id,
            parent_span_id AS parent_id,
            TIMESTAMP(start_time_unix_nano / 1000000000) AS start_time,
            TIMESTAMP(end_time_unix_nano / 1000000000) AS end_time,
            status_code,
            status_message,
            name,
            attributes,
            -- Handle events as ARRAY<STRING> - parse each JSON string in the array
            CASE 
              WHEN events IS NOT NULL AND size(events) > 0 THEN
                TRANSFORM(
                  events,
                  event_json -> FROM_JSON(event_json, 'STRUCT<name: STRING, time_unix_nano: LONG, attributes: STRING>')
                )
              ELSE 
                ARRAY()
            END AS parsed_events
          FROM {raw_spans_table}
        ),

        -- Transform parsed events to final format
        spans_with_events AS (
          SELECT
            span_id,
            trace_id,
            parent_id,
            start_time,
            end_time,
            status_code,
            status_message,
            name,
            attributes,
            -- Transform events to match expected structure
            TRANSFORM(
              parsed_events,
              e -> NAMED_STRUCT(
                'name', e.name,
                'timestamp', TIMESTAMP(e.time_unix_nano / 1000000000),
                'attributes', e.attributes
              )
            ) AS events
          FROM transformed_spans
        ),

        -- Group spans by trace_id
        spans_grouped AS (
          SELECT 
            trace_id,
            COLLECT_LIST(
              NAMED_STRUCT(
                'span_id', span_id,
                'trace_id', trace_id,
                'parent_id', parent_id,
                'start_time', start_time,
                'end_time', end_time,
                'status_code', status_code,
                'status_message', status_message,
                'name', name,
                'attributes', attributes,
                'events', events
              )
            ) AS spans
          FROM spans_with_events
          GROUP BY trace_id
        )

        -- Main query joining all components
        SELECT
          tm.trace_id,
          tm.client_request_id,
          tm.request_time,
          tm.state,
          tm.execution_duration_ms,
          tm.request,
          tm.response,
          tm.trace_metadata,
          -- Parse tags JSON into map
          CASE WHEN lt.tag_json IS NOT NULL 
            THEN FROM_JSON(lt.tag_json, 'MAP<STRING, STRING>') 
            ELSE MAP()
          END AS tags,
          tm.trace_location,
          -- Add assessments
          COALESCE(ag.assessments, ARRAY()) AS assessments,
          -- Add spans
          COALESCE(sg.spans, ARRAY()) AS spans
        FROM trace_metadata tm
        LEFT JOIN latest_tags lt ON tm.trace_id = lt.trace_id
        LEFT JOIN assessments_grouped ag ON tm.trace_id = ag.trace_id
        LEFT JOIN spans_grouped sg ON tm.trace_id = sg.trace_id
        """
        
        spark.sql(query)
        _logger.info(f"Successfully created trace archival view: {final_view}")
        
    except Exception as e:
        raise MlflowException(f"Failed to create trace archival view {final_view}: {str(e)}") from e
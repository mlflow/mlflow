"""
Trace archival functionality for MLflow that enables archiving traces to Delta tables.
"""

import logging

import mlflow
from mlflow.entities.trace_archive_configuration import TraceArchiveConfiguration
from mlflow.environment_variables import MLFLOW_HTTP_REQUEST_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_trace_server_pb2 import (
    CreateTraceDestinationRequest,
    TraceLocation as ProtoTraceLocation,
    TraceDestination as ProtoTraceDestination,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.mlflow_tags import MLFLOW_EXPERIMENT_TRACE_ARCHIVAL_TABLE
from mlflow.utils.rest_utils import http_request
from mlflow.utils._spark_utils import _get_active_spark_session
from google.protobuf.json_format import MessageToDict

_logger = logging.getLogger(__name__)

# Supported schema version for trace archival
SUPPORTED_SCHEMA_VERSION = "v1"


class DatabricksArchivalManager:
    """
    Manages the creation and validation of Databricks trace archival infrastructure.
    
    This class handles the complex workflow of setting up trace archival including:
    - Schema version validation
    - GenAI trace view creation
    - Persistence of metadata
    """
    
    def __init__(self, experiment_id: str, catalog: str, schema: str, table_prefix: str = "trace_logs"):
        """
        Initialize the DatabricksArchivalManager.
        
        Args:
            experiment_id: The MLflow experiment ID to enable archival for
            catalog: The Unity Catalog catalog name where tables will be created
            schema: The Unity Catalog schema name where tables will be created
            table_prefix: Prefix for the archival view name
        """
        self.experiment_id = experiment_id
        self.catalog = catalog
        self.schema = schema
        self.table_prefix = table_prefix
        self.trace_archival_location = f"{catalog}.{schema}.trace_logs_{experiment_id}"
    
    def validate_schema_versions(self, spans_version: str, events_version: str) -> None:
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
    
    def create_genai_trace_view(self, view_name: str, spans_table: str, events_table: str) -> None:
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
                {spans_table}
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
                {spans_table}
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
                {events_table}
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
                {events_table}
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
            _logger.info(f"Successfully created trace archival view: {view_name}")
            
        except Exception as e:
            raise MlflowException(f"Failed to create trace archival view {view_name}: {str(e)}") from e
    
    def enable_archival(self) -> str:
        """
        Enable trace archival by orchestrating the full archival process.
        
        Returns:
            The name of the created trace archival view
            
        Raises:
            MlflowException: If any step of the archival process fails
        """
        try:
            # 1. Create proto request directly (internal implementation detail)
            proto_trace_location = ProtoTraceLocation()
            proto_trace_location.type = ProtoTraceLocation.TraceLocationType.MLFLOW_EXPERIMENT
            proto_trace_location.mlflow_experiment.experiment_id = self.experiment_id
            
            proto_request = CreateTraceDestinationRequest(
                trace_location=proto_trace_location,
                uc_catalog=self.catalog,
                uc_schema=self.schema,
                uc_table_prefix=f"{self.table_prefix}_{self.experiment_id}"
            )
            
            # 2. Call the trace server CreateTraceDestination API
            request_body = MessageToDict(proto_request, preserving_proto_field_name=True)
            
            _logger.info(f"Creating trace destination for experiment {self.experiment_id} in {self.catalog}.{self.schema}")
            try:
                res = http_request(
                    host_creds=get_databricks_host_creds(),
                    endpoint="/api/2.0/tracing/trace-destinations",
                    method="POST",
                    timeout=MLFLOW_HTTP_REQUEST_TIMEOUT.get(),
                    json=request_body,
                )
                
            except Exception as e:
                _logger.error(f"Failed to create trace destination for experiment {self.experiment_id}: {str(e)}")
                raise
            
            if res.status_code != 200:
                raise MlflowException(
                    f"Failed to create trace destination for experiment {self.experiment_id}. "
                    f"Status: {res.status_code}, Response: {res.text}"
                )
            
            # 3. Parse response into TraceArchiveConfiguration entity
            response_data = res.json()
            
            # Convert JSON response to protobuf and then to entity
            proto_response = ProtoTraceDestination()
            proto_response.trace_location.CopyFrom(proto_trace_location)
            proto_response.spans_table_name = response_data["spans_table_name"]
            proto_response.events_table_name = response_data["events_table_name"]
            proto_response.spans_schema_version = response_data["spans_schema_version"]
            proto_response.events_schema_version = response_data["events_schema_version"]
            
            trace_config = TraceArchiveConfiguration.from_proto(proto_response)
            
            _logger.debug(f"Trace destination created with Spans table: {trace_config.spans_table_name}, "
                        f"Events table: {trace_config.events_table_name}, "
                        f"Spans schema version: {trace_config.spans_schema_version}, "
                        f"Events schema version: {trace_config.events_schema_version}")
            
            # 4. Validate schema versions before proceeding
            self.validate_schema_versions(trace_config.spans_schema_version, trace_config.events_schema_version)
            
            # 5. Create the logical view
            _logger.info(f"Creating trace archival at: {self.trace_archival_location}")
            self.create_genai_trace_view(self.trace_archival_location, trace_config.spans_table_name, trace_config.events_table_name)
            
            # 6. Set experiment tag to track the archival location
            mlflow.set_experiment_tag(MLFLOW_EXPERIMENT_TRACE_ARCHIVAL_TABLE, self.trace_archival_location)
            
            _logger.info(f"Trace archival enabled successfully for experiment {self.experiment_id}. "
                        f"View created: {self.trace_archival_location}")
            
            return self.trace_archival_location
            
        except Exception as e:
            _logger.error(f"Failed to enable trace archival for experiment {self.experiment_id}: {str(e)}")
            raise MlflowException(
                f"Failed to enable trace archival for experiment {self.experiment_id}: {str(e)}"
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
    
    Args:
        experiment_id: The MLflow experiment ID to enable archival for.
        catalog: The Unity Catalog catalog name where tables will be created.
        schema: The Unity Catalog schema name where tables will be created.
        
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
    manager = DatabricksArchivalManager(experiment_id, catalog, schema, table_prefix)
    return manager.enable_archival()
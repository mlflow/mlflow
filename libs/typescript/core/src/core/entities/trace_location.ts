const UC_SCHEMA_DEFAULT_SPANS_TABLE_NAME = 'mlflow_experiment_trace_otel_spans';
const UC_SCHEMA_DEFAULT_LOGS_TABLE_NAME = 'mlflow_experiment_trace_otel_logs';

/**
 * Types of trace locations
 */
export enum TraceLocationType {
  /**
   * Unspecified trace location type
   */
  TRACE_LOCATION_TYPE_UNSPECIFIED = 'TRACE_LOCATION_TYPE_UNSPECIFIED',

  /**
   * Trace is stored in an MLflow experiment
   */
  MLFLOW_EXPERIMENT = 'MLFLOW_EXPERIMENT',

  /**
   * Trace is stored in a Databricks inference table (deprecated)
   */
  INFERENCE_TABLE = 'INFERENCE_TABLE',

  /**
   * Trace is stored in a Databricks Unity Catalog
   */
  UC_SCHEMA = 'UC_SCHEMA'
}

/**
 * Interface representing an MLflow experiment location
 */
export interface MlflowExperimentLocation {
  /**
   * The ID of the MLflow experiment where the trace is stored
   */
  experimentId: string;
}

/**
 * Interface representing a Databricks inference table location (deprecated)
 */
export interface InferenceTableLocation {
  /**
   * The fully qualified name of the inference table where the trace is stored
   */
  fullTableName: string;
}

/**
 * Interface representing a Databricks Unity Catalog Table location
 */
export interface UCSchemaLocation{
  /**
   * The Unity Catalog schema where the trace is stored
   */
  catalog_name: string;
  schema_name: string;
  _otel_spans_table_name?: string;
  _otel_logs_table_name?: string;
}

/**
 * Interface representing the location where the trace is stored
 */
export interface TraceLocation {
  /**
   * The type of the trace location
   */
  type: TraceLocationType;

  /**
   * The MLflow experiment location
   * Set this when the location type is MLflow experiment
   */
  mlflowExperiment?: MlflowExperimentLocation;

  /**
   * The inference table location
   * Set this when the location type is Databricks Inference table
   */
  inferenceTable?: InferenceTableLocation;

  /**
   * The UC schema location
   * Set this when the location type is Databricks Unity Catalog schema
   */
  ucSchema?: UCSchemaLocation;
}

/**
 * Create a TraceLocation from an experiment ID
 * @param experimentId The ID of the MLflow experiment
 */
export function createTraceLocationFromExperimentId(experimentId: string): TraceLocation {
  if (typeof experimentId !== 'string') {
    throw new Error('experimentId must be a string');
  }

  return {
    type: TraceLocationType.MLFLOW_EXPERIMENT,
    mlflowExperiment: {
      experimentId: experimentId
    }
  };
}

/**
 * Create a TraceLocation from a UC schema
 * @param ucSchema The catalog and schema name in Databricks Unity Catalog
 */
export function createTraceLocationFromUCSchema(catalog: string, schema: string): TraceLocation {
  if (typeof catalog !== 'string' || typeof schema !== 'string') {
    throw new Error('catalog and schema names must be a string');
  }

  return {
    type: TraceLocationType.UC_SCHEMA,
    ucSchema: {
      catalog_name: catalog,
      schema_name: schema,
    }
  }
}


export function getFullTableName(ucSchema: UCSchemaLocation): string {
  if (ucSchema._otel_spans_table_name) {
    return `${ucSchema.catalog_name}.${ucSchema.schema_name}.${ucSchema._otel_spans_table_name}`;
  }
  return `${ucSchema.catalog_name}.${ucSchema.schema_name}.${UC_SCHEMA_DEFAULT_SPANS_TABLE_NAME}`;
}

export function getLocationType(location: MlflowExperimentLocation | InferenceTableLocation |     UCSchemaLocation | undefined): TraceLocationType {
  if (location == null) {
    return TraceLocationType.TRACE_LOCATION_TYPE_UNSPECIFIED;
  }
  if ("experimentId" in location) {
    return TraceLocationType.MLFLOW_EXPERIMENT;
  }
  if ("catalog_name" in location && "schema_name" in location) {
    return TraceLocationType.UC_SCHEMA;
  }
  if ("fullTableName" in location) {
    return TraceLocationType.INFERENCE_TABLE;
  }
  return TraceLocationType.TRACE_LOCATION_TYPE_UNSPECIFIED;
}
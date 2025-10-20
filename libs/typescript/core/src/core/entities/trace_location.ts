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
export function createTraceLocationFromUCSchema(ucSchema: string): TraceLocation {
  const [catalog_name, schema_name] = ucSchema.split(".");
  return {
    type: TraceLocationType.UC_SCHEMA,
    ucSchema: {
      catalog_name: catalog_name,
      schema_name: schema_name,
    }
  }
}

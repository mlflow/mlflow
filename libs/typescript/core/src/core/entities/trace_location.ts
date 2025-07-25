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
   * Trace is stored in a Databricks inference table
   */
  INFERENCE_TABLE = 'INFERENCE_TABLE'
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
 * Interface representing a Databricks inference table location
 */
export interface InferenceTableLocation {
  /**
   * The fully qualified name of the inference table where the trace is stored
   */
  fullTableName: string;
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

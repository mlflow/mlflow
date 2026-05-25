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
  INFERENCE_TABLE = 'INFERENCE_TABLE',

  /**
   * Trace is stored under a Databricks Unity Catalog schema (no fixed
   * table prefix). The backend selects default span/log table names.
   */
  UC_SCHEMA = 'UC_SCHEMA',

  /**
   * Trace is stored under a Databricks Unity Catalog table prefix.
   * The user-supplied prefix determines the span/log table names.
   */
  UC_TABLE_PREFIX = 'UC_TABLE_PREFIX',
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
 * Interface representing a Databricks Unity Catalog schema location.
 * Mirrors Python's `UCSchemaLocation`.
 */
export interface UCSchemaLocation {
  catalogName: string;
  schemaName: string;
  /**
   * Backend-populated bare spans table name (no catalog/schema prefix).
   * The full name is `${catalogName}.${schemaName}.${otelSpansTableName}`,
   * computed by `getOtelSpansTableName`.
   */
  otelSpansTableName?: string;
  /**
   * Backend-populated bare logs table name (no catalog/schema prefix).
   * The full name is `${catalogName}.${schemaName}.${otelLogsTableName}`.
   */
  otelLogsTableName?: string;
}

/**
 * Interface representing a Databricks Unity Catalog table-prefix location.
 * Mirrors Python's `UnityCatalog` location.
 */
export interface UnityCatalogLocation {
  catalogName: string;
  schemaName: string;
  /** Customer-supplied table prefix; required for trace ID location string. */
  tablePrefix?: string;
  /** Backend-populated fully qualified spans table name. */
  otelSpansTableName?: string;
  /** Backend-populated fully qualified logs table name. */
  otelLogsTableName?: string;
  /** Backend-populated fully qualified annotations table name. */
  annotationsTableName?: string;
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
   * The Databricks UC schema location. Set when type is UC_SCHEMA.
   */
  ucSchema?: UCSchemaLocation;

  /**
   * The Databricks UC table-prefix location. Set when type is UC_TABLE_PREFIX.
   */
  ucTablePrefix?: UnityCatalogLocation;
}

const UC_SCHEMA_DEFAULT_SPANS_TABLE_NAME = 'mlflow_experiment_trace_otel_spans';

/**
 * Returns "catalog.schema" for a UC schema location.
 */
export function ucSchemaLocationString(location: UCSchemaLocation): string {
  return `${location.catalogName}.${location.schemaName}`;
}

/**
 * Returns "catalog.schema.table_prefix" for a UC table-prefix location.
 * Throws if the prefix is not set; the prefix is required for trace IDs.
 */
export function ucTablePrefixLocationString(location: UnityCatalogLocation): string {
  if (!location.tablePrefix) {
    throw new Error(
      'Unity Catalog table_prefix is required to build a trace location string. ' +
        'Provide a tablePrefix when constructing the UnityCatalog destination.',
    );
  }
  return `${location.catalogName}.${location.schemaName}.${location.tablePrefix}`;
}

/**
 * Get the location string used for V4 trace IDs and UC OTLP routing
 * (i.e. the `X-Databricks-UC-Table-Name` header value).
 */
export function getUcLocationString(traceLocation: TraceLocation): string | null {
  if (traceLocation.type === TraceLocationType.UC_TABLE_PREFIX && traceLocation.ucTablePrefix) {
    return ucTablePrefixLocationString(traceLocation.ucTablePrefix);
  }
  if (traceLocation.type === TraceLocationType.UC_SCHEMA && traceLocation.ucSchema) {
    return ucSchemaLocationString(traceLocation.ucSchema);
  }
  return null;
}

/**
 * Get the fully qualified OTel spans table name to use as the
 * `X-Databricks-UC-Table-Name` header when exporting spans for this
 * trace via OTLP.
 */
export function getOtelSpansTableName(traceLocation: TraceLocation): string | null {
  if (traceLocation.type === TraceLocationType.UC_TABLE_PREFIX && traceLocation.ucTablePrefix) {
    return traceLocation.ucTablePrefix.otelSpansTableName ?? null;
  }
  if (traceLocation.type === TraceLocationType.UC_SCHEMA && traceLocation.ucSchema) {
    const loc = traceLocation.ucSchema;
    const table = loc.otelSpansTableName ?? UC_SCHEMA_DEFAULT_SPANS_TABLE_NAME;
    return `${loc.catalogName}.${loc.schemaName}.${table}`;
  }
  return null;
}

/**
 * Create a TraceLocation from an experiment ID
 * @param experimentId The ID of the MLflow experiment
 */
export function createTraceLocationFromExperimentId(experimentId: string): TraceLocation {
  return {
    type: TraceLocationType.MLFLOW_EXPERIMENT,
    mlflowExperiment: {
      experimentId: experimentId,
    },
  };
}

/**
 * Create a TraceLocation from a Databricks UC schema (no fixed table prefix).
 */
export function createTraceLocationFromUcSchema(
  catalogName: string,
  schemaName: string,
): TraceLocation {
  return {
    type: TraceLocationType.UC_SCHEMA,
    ucSchema: { catalogName, schemaName },
  };
}

/**
 * Create a TraceLocation from a Databricks UC table-prefix location.
 */
export function createTraceLocationFromUcTablePrefix(
  catalogName: string,
  schemaName: string,
  tablePrefix: string,
): TraceLocation {
  return {
    type: TraceLocationType.UC_TABLE_PREFIX,
    ucTablePrefix: { catalogName, schemaName, tablePrefix },
  };
}

/**
 * True iff the trace is stored in a Databricks Unity Catalog location.
 */
export function isUcTraceLocation(traceLocation: TraceLocation): boolean {
  return (
    traceLocation.type === TraceLocationType.UC_SCHEMA ||
    traceLocation.type === TraceLocationType.UC_TABLE_PREFIX
  );
}

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
   * The Databricks UC table-prefix location. Set when type is UC_TABLE_PREFIX.
   */
  ucTablePrefix?: UnityCatalogLocation;
}

export interface SerializedTraceLocation {
  type: TraceLocationType;
  mlflow_experiment?: { experiment_id: string };
  inference_table?: { full_table_name: string };
  uc_table_prefix?: {
    catalog_name: string;
    schema_name: string;
    table_prefix?: string;
    otel_spans_table_name?: string;
    otel_logs_table_name?: string;
    annotations_table_name?: string;
  };
}

export function serializeTraceLocation(loc: TraceLocation): SerializedTraceLocation {
  const out: SerializedTraceLocation = { type: loc.type };
  if (loc.mlflowExperiment) {
    out.mlflow_experiment = { experiment_id: loc.mlflowExperiment.experimentId };
  }
  if (loc.inferenceTable) {
    out.inference_table = { full_table_name: loc.inferenceTable.fullTableName };
  }
  if (loc.ucTablePrefix) {
    const uc = loc.ucTablePrefix;
    out.uc_table_prefix = {
      catalog_name: uc.catalogName,
      schema_name: uc.schemaName,
      ...(uc.tablePrefix ? { table_prefix: uc.tablePrefix } : {}),
      ...(uc.otelSpansTableName ? { otel_spans_table_name: uc.otelSpansTableName } : {}),
      ...(uc.otelLogsTableName ? { otel_logs_table_name: uc.otelLogsTableName } : {}),
      ...(uc.annotationsTableName ? { annotations_table_name: uc.annotationsTableName } : {}),
    };
  }
  return out;
}

export function deserializeTraceLocation(json: SerializedTraceLocation | undefined): TraceLocation {
  if (!json?.type) {
    throw new Error('Invalid trace location: missing type.');
  }

  return {
    type: json.type,
    mlflowExperiment: json?.mlflow_experiment
      ? { experimentId: json.mlflow_experiment.experiment_id }
      : undefined,
    inferenceTable: json?.inference_table
      ? { fullTableName: json.inference_table.full_table_name }
      : undefined,
    ucTablePrefix: json?.uc_table_prefix
      ? {
          catalogName: json.uc_table_prefix.catalog_name,
          schemaName: json.uc_table_prefix.schema_name,
          tablePrefix: json.uc_table_prefix.table_prefix,
          otelSpansTableName: json.uc_table_prefix.otel_spans_table_name,
          otelLogsTableName: json.uc_table_prefix.otel_logs_table_name,
          annotationsTableName: json.uc_table_prefix.annotations_table_name,
        }
      : undefined,
  };
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
  return null;
}

/**
 * Get the fully qualified OTel spans table name to use as the
 * `X-Databricks-UC-Table-Name` header when exporting spans for this
 * trace via OTLP.
 *
 * Falls back to `<catalog>.<schema>.<table_prefix>_otel_spans`, which is the
 * default spans table name Databricks creates when a UC trace location is
 * provisioned. Customers with a custom backend-provisioned spans table can
 * override by setting `ucTablePrefix.otelSpansTableName`.
 */
export function getOtelSpansTableName(traceLocation: TraceLocation): string | null {
  if (traceLocation.type === TraceLocationType.UC_TABLE_PREFIX && traceLocation.ucTablePrefix) {
    const loc = traceLocation.ucTablePrefix;
    if (loc.otelSpansTableName) {
      return loc.otelSpansTableName;
    }
    if (loc.tablePrefix) {
      return `${loc.catalogName}.${loc.schemaName}.${loc.tablePrefix}_otel_spans`;
    }
    return null;
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
  return traceLocation.type === TraceLocationType.UC_TABLE_PREFIX;
}

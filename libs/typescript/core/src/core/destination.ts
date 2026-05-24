import type { MlflowClient } from '../clients/client';
import type { UCSchemaLocation, UnityCatalogLocation } from './entities/trace_location';

/**
 * Databricks experiment tag keys carrying the linked UC trace location.
 * Matches Python's `MLFLOW_EXPERIMENT_DATABRICKS_TRACE_*` constants in
 * mlflow/utils/mlflow_tags.py.
 */
export const DATABRICKS_TRACE_DESTINATION_PATH_TAG =
  'mlflow.experiment.databricksTraceDestinationPath';
export const DATABRICKS_TRACE_SPAN_STORAGE_TABLE_TAG =
  'mlflow.experiment.databricksTraceSpanStorageTable';
export const DATABRICKS_TRACE_LOG_STORAGE_TABLE_TAG =
  'mlflow.experiment.databricksTraceLogStorageTable';
export const DATABRICKS_TRACE_ANNOTATIONS_TABLE_TAG =
  'mlflow.experiment.databricksTraceAnnotationsTable';

/**
 * A user-configured destination for traces, mirroring Python's
 * `mlflow.tracing.set_destination` API. Currently the supported destinations
 * are MLflow experiments (the default, configured via `init`), Databricks
 * Unity Catalog schemas, and Databricks Unity Catalog table prefixes.
 *
 * Setting a `UnityCatalog*` destination causes the SDK to:
 *   - generate V4 trace IDs (`trace:/<location>/<hex>`)
 *   - create trace info via the V4 CreateTraceInfo endpoint, so trace tags
 *     and metadata set via `updateCurrentTrace` persist on UC-backed traces
 *   - export spans via OTLP to `/api/2.0/otel/v1/traces`
 */
export type TraceDestination = UnityCatalogDestination | UcSchemaDestination;

export interface UnityCatalogDestination {
  kind: 'uc_table_prefix';
  location: UnityCatalogLocation;
}

export interface UcSchemaDestination {
  kind: 'uc_schema';
  location: UCSchemaLocation;
}

/**
 * Construct a Databricks UC table-prefix destination.
 * Equivalent to Python's `UnityCatalog(catalog_name, schema_name, table_prefix)`.
 */
export function unityCatalogDestination(args: {
  catalogName: string;
  schemaName: string;
  tablePrefix: string;
}): UnityCatalogDestination {
  if (!args.catalogName || !args.schemaName || !args.tablePrefix) {
    throw new Error('unityCatalogDestination requires catalogName, schemaName, and tablePrefix.');
  }
  return {
    kind: 'uc_table_prefix',
    location: {
      catalogName: args.catalogName,
      schemaName: args.schemaName,
      tablePrefix: args.tablePrefix,
    },
  };
}

/**
 * Construct a Databricks UC schema destination (no fixed table prefix).
 * Equivalent to Python's `UCSchemaLocation(catalog_name, schema_name)`.
 */
export function ucSchemaDestination(args: {
  catalogName: string;
  schemaName: string;
}): UcSchemaDestination {
  if (!args.catalogName || !args.schemaName) {
    throw new Error('ucSchemaDestination requires catalogName and schemaName.');
  }
  return {
    kind: 'uc_schema',
    location: { catalogName: args.catalogName, schemaName: args.schemaName },
  };
}

let currentDestination: TraceDestination | null = null;

/**
 * Set the active trace destination. Must be called before traces are
 * created. Pass `null` to clear.
 *
 * Mirrors Python's `mlflow.tracing.set_destination`. The change takes
 * effect on the next call to `init()`, which rebuilds the span processor
 * stack. If called after `init()`, callers must re-invoke `init(...)` for
 * the new destination to be wired up.
 */
export function setDestination(destination: TraceDestination | null): void {
  currentDestination = destination;
}

/**
 * Get the active trace destination, or null if none is set.
 */
export function getDestination(): TraceDestination | null {
  return currentDestination;
}

/**
 * Reset the destination. For testing purposes only.
 * @internal
 */
export function resetDestination(): void {
  currentDestination = null;
}

/**
 * Parse a UC table-prefix destination from the Databricks experiment tags
 * (`mlflow.experiment.databricksTrace*`). Returns null if the experiment is
 * not linked to a UC trace destination.
 *
 * Mirrors Python's `Experiment._resolve_trace_location_from_tags`.
 */
export function destinationFromExperimentTags(
  tags: Record<string, string>,
): UnityCatalogDestination | null {
  const path = tags[DATABRICKS_TRACE_DESTINATION_PATH_TAG];
  if (!path) {
    return null;
  }
  const parts = path.split('.');
  if (parts.length !== 3 || parts.some((p) => !p)) {
    return null;
  }
  const [catalogName, schemaName, tablePrefix] = parts;
  return {
    kind: 'uc_table_prefix',
    location: {
      catalogName,
      schemaName,
      tablePrefix,
      otelSpansTableName: tags[DATABRICKS_TRACE_SPAN_STORAGE_TABLE_TAG],
      otelLogsTableName: tags[DATABRICKS_TRACE_LOG_STORAGE_TABLE_TAG],
      annotationsTableName: tags[DATABRICKS_TRACE_ANNOTATIONS_TABLE_TAG],
    },
  };
}

/**
 * Fetch the experiment and, if it's linked to a UC trace location via the
 * Databricks experiment tags, return a `UnityCatalogDestination` with the
 * backend-provided spans / logs / annotations table names populated.
 *
 * Returns null when the experiment has no UC trace destination tags or the
 * GetExperiment call returns 404.
 *
 * Mirrors Python's `_resolve_experiment_uc_location` in
 * `mlflow/tracing/provider.py`.
 */
export async function resolveDestinationFromExperiment(
  client: MlflowClient,
  experimentId: string,
): Promise<UnityCatalogDestination | null> {
  const experiment = await client.getExperiment(experimentId);
  if (!experiment) {
    return null;
  }
  return destinationFromExperimentTags(experiment.tags);
}

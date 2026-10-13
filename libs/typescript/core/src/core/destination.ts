import type { UnityCatalogLocation } from './entities/trace_location';

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
 * Parse a UC table-prefix location from the Databricks experiment tags
 * (`mlflow.experiment.databricksTrace*`). Returns null when the experiment is
 * not linked to a UC trace destination.
 *
 * Mirrors Python's `Experiment._resolve_trace_location_from_tags`.
 */
export function ucLocationFromExperimentTags(
  tags: Record<string, string>,
): UnityCatalogLocation | null {
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
    catalogName,
    schemaName,
    tablePrefix,
    otelSpansTableName: tags[DATABRICKS_TRACE_SPAN_STORAGE_TABLE_TAG],
    otelLogsTableName: tags[DATABRICKS_TRACE_LOG_STORAGE_TABLE_TAG],
    annotationsTableName: tags[DATABRICKS_TRACE_ANNOTATIONS_TABLE_TAG],
  };
}

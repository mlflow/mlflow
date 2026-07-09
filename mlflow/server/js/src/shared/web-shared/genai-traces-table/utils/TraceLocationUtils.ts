import type {
  ModelTraceLocationMlflowExperiment,
  ModelTraceLocationUcSchema,
  ModelTraceLocationUcTablePrefix,
  ModelTrace,
  ModelTraceInfoV3,
} from '../../model-trace-explorer/ModelTrace.types';
import { isV3ModelTraceInfo } from '../../model-trace-explorer/ModelTraceExplorer.utils';

/**
 * Helper utility: creates experiment trace location descriptor based on provided experiment ID.
 */
export const createTraceLocationForExperiment = (experimentId: string): ModelTraceLocationMlflowExperiment => ({
  type: 'MLFLOW_EXPERIMENT',
  mlflow_experiment: {
    experiment_id: experimentId,
  },
});

export const createTraceLocationForUCSchema = (ucSchemaFullPath: string): ModelTraceLocationUcSchema => {
  const [catalog_name, schema_name] = ucSchemaFullPath.split('.');
  return {
    type: 'UC_SCHEMA',
    uc_schema: {
      catalog_name,
      schema_name,
    },
  };
};

export const createTraceLocationForUCTablePrefix = (fullPath: string): ModelTraceLocationUcTablePrefix => {
  const [catalog_name, schema_name, table_prefix] = fullPath.split('.');
  return {
    type: 'UC_TABLE_PREFIX',
    uc_table_prefix: {
      catalog_name,
      schema_name,
      table_prefix,
    },
  };
};

/**
 * Returns true if the given trace location uses a V4-compatible type (UC_SCHEMA or UC_TABLE_PREFIX).
 */
export const isV4TraceLocation = (location: { type: string }): boolean =>
  location.type === 'UC_SCHEMA' || location.type === 'UC_TABLE_PREFIX';

/**
 * Returns true if the destination path has 3 dot-separated parts (catalog.schema.prefix),
 * indicating a UC table prefix location rather than a UC schema location.
 */
export const isTablePrefixDestinationPath = (path: string): boolean => {
  return path.split('.').length === 3;
};

/**
 * Auto-detects the location type from a destination path string:
 * - 2 parts (catalog.schema) -> UC_SCHEMA
 * - 3 parts (catalog.schema.prefix) -> UC_TABLE_PREFIX
 */
export const createTraceLocationForDestinationPath = (
  destinationPath: string,
): ModelTraceLocationUcSchema | ModelTraceLocationUcTablePrefix => {
  if (isTablePrefixDestinationPath(destinationPath)) {
    return createTraceLocationForUCTablePrefix(destinationPath);
  }
  return createTraceLocationForUCSchema(destinationPath);
};

/**
 * Determines if a trace (by provided info object) supports being queried using V4 API.
 * UC_SCHEMA and UC_TABLE_PREFIX located traces are supported.
 */
export const doesTraceSupportV4API = (
  traceInfo?: ModelTrace['info'] | Pick<ModelTraceInfoV3, 'trace_location' | 'trace_id'>,
) => {
  return Boolean(
    traceInfo &&
    isV3ModelTraceInfo(traceInfo) &&
    traceInfo.trace_location &&
    isV4TraceLocation(traceInfo.trace_location),
  );
};

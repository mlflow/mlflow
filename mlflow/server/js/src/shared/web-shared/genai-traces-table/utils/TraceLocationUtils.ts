import {
  type ModelTraceLocationMlflowExperiment,
  type ModelTraceLocationUcSchema,
  isV3ModelTraceInfo,
  type ModelTrace,
} from '../../model-trace-explorer';

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

/**
 * Determines if a trace (by provided info object) supports being queried using V4 API.
 * For now, only UC_SCHEMA-located traces are supported.
 */
export const doesTraceSupportV4API = (traceInfo?: ModelTrace['info']) => {
  return Boolean(traceInfo && isV3ModelTraceInfo(traceInfo) && traceInfo.trace_location?.type === 'UC_SCHEMA');
};

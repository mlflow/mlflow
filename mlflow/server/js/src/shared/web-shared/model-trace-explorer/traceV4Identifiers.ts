import type { ModelTraceInfoV3, ModelTraceLocation } from './ModelTrace.types';

export const isV4TraceId = (traceId: string): boolean => {
  return traceId.startsWith('trace:/');
};

export const createTraceV4SerializedLocation = (location: ModelTraceLocation): string | undefined => {
  if (location.type === 'MLFLOW_EXPERIMENT') {
    return location.mlflow_experiment?.experiment_id;
  }
  if (location.type === 'INFERENCE_TABLE') {
    return location.inference_table?.full_table_name;
  }
  if (location.type === 'UC_SCHEMA') {
    return `${location.uc_schema?.catalog_name}.${location.uc_schema?.schema_name}`;
  }
  if (location.type === 'UC_TABLE_PREFIX') {
    return `${location.uc_table_prefix?.catalog_name}.${location.uc_table_prefix?.schema_name}.${location.uc_table_prefix?.table_prefix}`;
  }
  return undefined;
};

export const parseTraceV4SerializedLocation = (locationString: string): ModelTraceLocation => {
  const parts = locationString.split('.');
  if (parts.length >= 3 && parts[0] && parts[1] && parts[2]) {
    return {
      type: 'UC_TABLE_PREFIX',
      uc_table_prefix: { catalog_name: parts[0], schema_name: parts[1], table_prefix: parts[2] },
    };
  }
  if (parts.length >= 2 && parts[0] && parts[1]) {
    return { type: 'UC_SCHEMA', uc_schema: { catalog_name: parts[0], schema_name: parts[1] } };
  }
  return { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: locationString } };
};

export const createTraceV4LongIdentifier = (modelTraceInfo: ModelTraceInfoV3): string => {
  if (!modelTraceInfo.trace_location) {
    return modelTraceInfo.trace_id;
  }
  const serializedLocation = createTraceV4SerializedLocation(modelTraceInfo.trace_location);
  if (!serializedLocation) {
    return modelTraceInfo.trace_id;
  }

  return `trace:/${serializedLocation}/${modelTraceInfo.trace_id}`;
};

type TraceIdWithLocation = {
  trace_id: string;
  trace_location: string;
};

export const parseV4TraceId = (traceId: string): TraceIdWithLocation | undefined => {
  if (!isV4TraceId(traceId)) {
    return undefined;
  }
  const [, trace_location, trace_id] = traceId.split('/');

  // TODO: Support other trace locations
  return {
    trace_id,
    trace_location,
  };
};

/**
 * Parses a V4 trace ID back into a stub of model trace info with trace ID and location
 */
export const parseV4TraceIdToObject = (traceId: string): Partial<ModelTraceInfoV3> | undefined => {
  const parsedTraceId = parseV4TraceId(traceId);
  if (!parsedTraceId) {
    return undefined;
  }
  return {
    trace_id: parsedTraceId.trace_id,
    trace_location: parseTraceV4SerializedLocation(parsedTraceId.trace_location),
  };
};

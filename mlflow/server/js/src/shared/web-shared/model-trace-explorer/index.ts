export { ModelTraceExplorerFrameRenderer } from './ModelTraceExplorerFrameRenderer';

export const ModelSpanType = {
  LLM: 'LLM',
  CHAIN: 'CHAIN',
  AGENT: 'AGENT',
  TOOL: 'TOOL',
  FUNCTION: 'FUNCTION',
  CHAT_MODEL: 'CHAT_MODEL',
  RETRIEVER: 'RETRIEVER',
  PARSER: 'PARSER',
  EMBEDDING: 'EMBEDDING',
  RERANKER: 'RERANKER',
  UNKNOWN: 'UNKNOWN',
} as const;

export const ModelIconType = {
  MODELS: 'models',
  DOCUMENT: 'document',
  CONNECT: 'connect',
} as const;

/**
 * Represents a single model trace span.
 * Based on https://github.com/mlflow/mlflow/blob/tracing/mlflow/entities/span.py
 */
export type ModelTraceSpan = {
  context: {
    span_id: string;
    trace_id: string;
  };
  name: string;
  /* deprecated, renamed to `parent_id` */
  parent_span_id?: string | null;
  parent_id?: string | null;
  /* deprecated, contained in attributes['mlflow.spanType'] */
  span_type?: typeof ModelSpanType[keyof typeof ModelSpanType] | string;
  /* deprecated, migrated to `status_code` and `status_message` */
  status?: ModelTraceStatus;
  status_code?: string;
  status_message?: string | null;
  start_time: number;
  end_time: number;
  /* deprecated, contained in attributes['mlflow.spanInputs'] */
  inputs?: any;
  /* deprecated, contained in attributes['mlflow.spanOutputs'] */
  outputs?: any;
  attributes?: Record<string, any>;
  /* metadata for ui usage logging */
  type: typeof ModelSpanType[keyof typeof ModelSpanType];
};

export type ModelTraceEvent = {
  name: string;
  timestamp?: number;
  attributes?: Record<string, any>;
};

export type ModelTraceData = {
  spans: ModelTraceSpan[];
};

/**
 * Represents a single model trace object.
 * Based on https://github.com/mlflow/mlflow/blob/8e44d102e9568d09d9dc376136d13a5a5d1ab46f/mlflow/tracing/types/model.py#L11
 */
export type ModelTrace = {
  /* deprecated, renamed to `data` */
  trace_data?: ModelTraceData;
  /* deprecated, renamed to `info` */
  trace_info?: ModelTraceInfo;
  data: ModelTraceData;
  info: ModelTraceInfo;
};

export type ModelTraceInfo = {
  request_id?: string;
  experiment_id?: string;
  timestamp_ms?: number;
  execution_time_ms?: number;
  status?: ModelTraceStatus['description'];
  attributes?: Record<string, any>;
  request_metadata?: { key: string; value: string }[];
  tags?: { key: string; value: string }[];
};

export type ModelTraceStatusUnset = {
  description: 'UNSET';
  status_code: 0;
};

export type ModelTraceStatusOk = {
  description: 'OK';
  status_code: 1;
};

export type ModelTraceStatusError = {
  description: 'ERROR';
  status_code: 2;
};

export type ModelTraceStatusInProgress = {
  description: 'IN_PROGRESS';
  status_code: 3;
};

export const ModelTraceSpanType = {
  LLM: 'LLM',
  CHAIN: 'CHAIN',
  AGENT: 'AGENT',
  TOOL: 'TOOL',
  CHAT_MODEL: 'CHAT_MODEL',
  RETRIEVER: 'RETRIEVER',
  PARSER: 'PARSER',
  EMBEDDING: 'EMBEDDING',
  RERANKER: 'RERANKER',
  UNKNOWN: 'UNKNOWN',
} as const;

export type ModelTraceStatus =
  | ModelTraceStatusUnset
  | ModelTraceStatusOk
  | ModelTraceStatusError
  | ModelTraceStatusInProgress;

export const ModelTraceChildToParentFrameMessage = {
  Ready: 'READY',
} as const;

type ModelTraceFrameReadyMessage = {
  type: typeof ModelTraceChildToParentFrameMessage.Ready;
};

export const ModelTraceParentToChildFrameMessage = {
  UpdateTrace: 'UPDATE_TRACE',
} as const;

type ModelTraceFrameUpdateTraceMessage = {
  type: typeof ModelTraceParentToChildFrameMessage.UpdateTrace;
  traceData: ModelTrace;
};

export type ModelTraceChildToParentFrameMessageType = ModelTraceFrameReadyMessage;
export type ModelTraceParentToChildFrameMessageType = ModelTraceFrameUpdateTraceMessage;

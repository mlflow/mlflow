/**
 * Constants for MLflow Tracing
 */

/**
 * Enum for span types that can be used with MLflow Tracing
 */
export enum SpanType {
  LLM = 'LLM',
  CHAIN = 'CHAIN',
  AGENT = 'AGENT',
  TOOL = 'TOOL',
  CHAT_MODEL = 'CHAT_MODEL',
  RETRIEVER = 'RETRIEVER',
  PARSER = 'PARSER',
  EMBEDDING = 'EMBEDDING',
  RERANKER = 'RERANKER',
  MEMORY = 'MEMORY',
  UNKNOWN = 'UNKNOWN'
}

/**
 * Constants for MLflow span attribute keys
 */
export const SpanAttributeKey = {
  EXPERIMENT_ID: 'mlflow.experimentId',
  TRACE_ID: 'mlflow.traceRequestId',
  INPUTS: 'mlflow.spanInputs',
  OUTPUTS: 'mlflow.spanOutputs',
  SPAN_TYPE: 'mlflow.spanType',
  // This attribute is used to store token usage information from LLM responses.
  // Stored in {"input_tokens": int, "output_tokens": int, "total_tokens": int} format.
  TOKEN_USAGE: 'mlflow.chat.tokenUsage',
  // This attribute indicates which flavor/format generated the LLM span. This is
  // used by downstream (e.g., UI) to determine the message format for parsing.
  MESSAGE_FORMAT: 'mlflow.message.format'
};

/**
 * Constants for MLflow trace metadata keys
 */
export const TraceMetadataKey = {
  SOURCE_RUN: 'mlflow.sourceRun',
  MODEL_ID: 'mlflow.modelId',
  SIZE_BYTES: 'mlflow.trace.sizeBytes',
  SCHEMA_VERSION: 'mlflow.trace_schema.version',
  TOKEN_USAGE: 'mlflow.trace.tokenUsage',
  // Deprecated, do not use. These fields are used for storing trace request and response
  // in MLflow 2.x. In MLflow 3.x, these are replaced in favor of the request_preview and
  // response_preview fields in the trace info.
  // TODO: Remove this once the new trace table UI is available that is based on MLflow V3 trace.
  INPUTS: 'mlflow.traceInputs',
  OUTPUTS: 'mlflow.traceOutputs'
};

/**
 * Constants for MLflow trace tag keys
 */
export const TraceTagKey = {
  MLFLOW_ARTIFACT_LOCATION: 'mlflow.artifactLocation'
};

/**
 * Current version of the MLflow trace schema
 */
export const TRACE_SCHEMA_VERSION = '3';

/**
 * The prefix for MLflow trace IDs
 */
export const TRACE_ID_PREFIX = 'tr-';

/**
 * The default name for spans if the name is not provided when starting a span
 */
export const DEFAULT_SPAN_NAME = 'span';

/**
 * Trace ID for no-op spans
 */
export const NO_OP_SPAN_TRACE_ID = 'no-op-span-trace-id';

/**
 * Constants for token usage keys (matching Python TokenUsageKey)
 */
export const TokenUsageKey = {
  INPUT_TOKENS: 'input_tokens',
  OUTPUT_TOKENS: 'output_tokens',
  TOTAL_TOKENS: 'total_tokens'
};

/**
 * Max length of the request/response preview in the trace info.
 */
export const REQUEST_RESPONSE_PREVIEW_MAX_LENGTH = 1000;

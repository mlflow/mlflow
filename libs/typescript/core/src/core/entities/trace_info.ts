import type { TraceLocation, TraceLocationType } from './trace_location';
import type { TraceState } from './trace_state';
import { TraceMetadataKey } from '../constants';

interface SerializedTraceLocation {
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

function serializeTraceLocation(loc: TraceLocation): SerializedTraceLocation {
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

function deserializeTraceLocation(json: SerializedTraceLocation | undefined): TraceLocation {
  return {
    type: json?.type as TraceLocationType,
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
 * Interface for token usage information
 */
export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
}

/**
 * Metadata about a trace, such as its ID, location, timestamp, etc.
 */
export class TraceInfo {
  /**
   * The primary identifier for the trace
   */
  traceId: string;

  /**
   * The location where the trace is stored
   */
  traceLocation: TraceLocation;

  /**
   * Start time of the trace, in milliseconds
   */
  requestTime: number;

  /**
   * State of the trace
   */
  state: TraceState;

  /**
   * Request to the model/agent (JSON-encoded, may be truncated)
   */
  requestPreview?: string;

  /**
   * Response from the model/agent (JSON-encoded, may be truncated)
   */
  responsePreview?: string;

  /**
   * Client supplied request ID associated with the trace
   */
  clientRequestId?: string;

  /**
   * Duration of the trace, in milliseconds
   */
  executionDuration?: number;

  /**
   * Key-value pairs associated with the trace (immutable)
   */
  traceMetadata: Record<string, string>;

  /**
   * Tags associated with the trace (mutable)
   */
  tags: Record<string, string>;

  /**
   * List of assessments associated with the trace.
   * TODO: Assessments are not yet supported in the TypeScript SDK.
   */
  assessments: any[];

  /**
   * Create a new TraceInfo instance
   * @param params TraceInfo parameters
   */
  constructor(params: {
    traceId: string;
    traceLocation: TraceLocation;
    requestTime: number;
    state: TraceState;
    requestPreview?: string;
    responsePreview?: string;
    clientRequestId?: string;
    executionDuration?: number;
    traceMetadata?: Record<string, string>;
    tags?: Record<string, string>;
    assessments?: any[];
  }) {
    this.traceId = params.traceId;
    this.traceLocation = params.traceLocation;
    this.requestTime = params.requestTime;
    this.state = params.state;
    this.requestPreview = params.requestPreview;
    this.responsePreview = params.responsePreview;
    this.clientRequestId = params.clientRequestId;
    this.executionDuration = params.executionDuration;
    this.traceMetadata = params.traceMetadata || {};
    this.tags = params.tags || {};
    // TODO: Assessments are not yet supported in the TypeScript SDK.
    this.assessments = [];
  }

  /**
   * Convert this TraceInfo instance to JSON format
   * @returns JSON object representation of the TraceInfo
   */
  toJson(): SerializedTraceInfo {
    return {
      trace_id: this.traceId,
      client_request_id: this.clientRequestId,
      trace_location: serializeTraceLocation(this.traceLocation),
      request_preview: this.requestPreview,
      response_preview: this.responsePreview,
      request_time: new Date(this.requestTime).toISOString(),
      execution_duration:
        this.executionDuration != null ? `${this.executionDuration / 1000}s` : undefined,
      state: this.state,
      trace_metadata: this.traceMetadata,
      tags: this.tags,
      assessments: this.assessments,
    };
  }

  /**
   * Get aggregated token usage information for this trace.
   * Returns null if no token usage data is available.
   * @returns Token usage object or null
   */
  get tokenUsage(): TokenUsage | null {
    const tokenUsageJson = this.traceMetadata[TraceMetadataKey.TOKEN_USAGE];

    if (!tokenUsageJson) {
      return null;
    }

    const usage = JSON.parse(tokenUsageJson) as TokenUsage;
    const result: TokenUsage = {
      input_tokens: usage.input_tokens,
      output_tokens: usage.output_tokens,
      total_tokens: usage.total_tokens,
    };
    if (usage.cache_read_input_tokens != null) {
      result.cache_read_input_tokens = usage.cache_read_input_tokens;
    }
    if (usage.cache_creation_input_tokens != null) {
      result.cache_creation_input_tokens = usage.cache_creation_input_tokens;
    }
    return result;
  }

  /**
   * Create a TraceInfo instance from JSON data
   * @param json JSON object containing trace info data
   * @returns TraceInfo instance
   */
  static fromJson(json: SerializedTraceInfo): TraceInfo {
    return new TraceInfo({
      traceId: json.trace_id,
      clientRequestId: json.client_request_id,
      traceLocation: deserializeTraceLocation(json.trace_location),
      requestPreview: json.request_preview,
      responsePreview: json.response_preview,
      requestTime: json.request_time != null ? new Date(json.request_time).getTime() : Date.now(),
      executionDuration:
        json.execution_duration != null
          ? parseFloat(json.execution_duration.replace('s', '')) * 1000
          : undefined,
      state: json.state,
      traceMetadata: json.trace_metadata || {},
      tags: json.tags || {},
      assessments: json.assessments || [],
    });
  }
}

export interface SerializedTraceInfo {
  trace_id: string;
  client_request_id?: string;
  trace_location: SerializedTraceLocation;
  request_preview?: string;
  response_preview?: string;
  // "request_time": "2025-06-15T14:07:41.282Z"
  request_time: string;
  execution_duration?: string;
  state: TraceState;
  trace_metadata: Record<string, string>;
  tags: Record<string, string>;
  // TODO: Define proper type for assessments once supported
  assessments: any[];
}

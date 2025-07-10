import type { TraceLocation, TraceLocationType } from './trace_location';
import type { TraceState } from './trace_state';
import { TraceMetadataKey } from '../constants';

/**
 * Interface for token usage information
 */
export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
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
      trace_location: {
        type: this.traceLocation.type,
        mlflow_experiment: this.traceLocation.mlflowExperiment
          ? {
              experiment_id: this.traceLocation.mlflowExperiment.experimentId
            }
          : undefined,
        inference_table: this.traceLocation.inferenceTable
          ? {
              full_table_name: this.traceLocation.inferenceTable.fullTableName
            }
          : undefined
      },
      request_preview: this.requestPreview,
      response_preview: this.responsePreview,
      request_time: new Date(this.requestTime).toISOString(),
      execution_duration:
        this.executionDuration != null ? `${this.executionDuration / 1000}s` : undefined,
      state: this.state,
      trace_metadata: this.traceMetadata,
      tags: this.tags,
      assessments: this.assessments
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
    return {
      input_tokens: usage.input_tokens,
      output_tokens: usage.output_tokens,
      total_tokens: usage.total_tokens
    };
  }

  /**
   * Create a TraceInfo instance from JSON data
   * @param json JSON object containing trace info data
   * @returns TraceInfo instance
   */
  static fromJson(json: SerializedTraceInfo): TraceInfo {
    /* eslint-disable @typescript-eslint/no-unsafe-member-access */
    return new TraceInfo({
      traceId: json.trace_id,
      clientRequestId: json.client_request_id,
      traceLocation: {
        type: json.trace_location?.type as TraceLocationType,
        mlflowExperiment: json.trace_location?.mlflow_experiment
          ? { experimentId: json.trace_location.mlflow_experiment.experiment_id }
          : undefined,
        inferenceTable: json.trace_location?.inference_table
          ? { fullTableName: json.trace_location.inference_table.full_table_name }
          : undefined
      },
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
      assessments: json.assessments || []
    });
    /* eslint-enable @typescript-eslint/no-unsafe-member-access */
  }
}

export interface SerializedTraceInfo {
  trace_id: string;
  client_request_id?: string;
  trace_location: {
    type: TraceLocationType;
    mlflow_experiment?: {
      experiment_id: string;
    };
    inference_table?: {
      full_table_name: string;
    };
  };
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

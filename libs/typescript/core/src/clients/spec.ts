/**
 * MLflow API Request/Response Specifications
 *
 * This module defines the types and interfaces for MLflow API communication,
 * including request payloads and response structures for all trace-related endpoints.
 */

import type { TraceInfo } from '../core/entities/trace_info';
import { ArtifactCredentialType } from './artifacts/databricks';

/**
 * Create a new TraceInfo entity in the backend.
 */
export namespace StartTraceV3 {
  export const getEndpoint = (host: string) => `${host}/api/3.0/mlflow/traces`;

  export interface Request {
    trace: {
      trace_info: Parameters<typeof TraceInfo.fromJson>[0];
    };
  }

  export interface Response {
    trace: {
      trace_info: Parameters<typeof TraceInfo.fromJson>[0];
    };
  }
}

/**
 * Get the TraceInfo entity for a given trace ID.
 */
export namespace GetTraceInfoV3 {
  export const getEndpoint = (host: string, traceId: string) =>
    `${host}/api/3.0/mlflow/traces/${traceId}`;

  export interface Response {
    trace: {
      trace_info: Parameters<typeof TraceInfo.fromJson>[0];
    };
  }
}

/** Create Experiment (used for testing) */
export namespace CreateExperiment {
  export const getEndpoint = (host: string) => `${host}/api/2.0/mlflow/experiments/create`;

  export interface Request {
    name?: string;
    artifact_location?: string;
    tags?: Record<string, string>;
  }

  export interface Response {
    experiment_id: string;
  }
}

/** Delete Experiment (used for testing) */
export namespace DeleteExperiment {
  export const getEndpoint = (host: string) => `${host}/api/2.0/mlflow/experiments/delete`;

  export interface Request {
    experiment_id: string;
  }
}

/**
 * Get credentials for uploading trace data to the artifact store. Only used for Databricks.
 */
export namespace GetCredentialsForTraceDataUpload {
  export const getEndpoint = (host: string, traceId: string) =>
    `${host}/api/2.0/mlflow/traces/${traceId}/credentials-for-data-upload`;

  export interface Response {
    credential_info: {
      type: ArtifactCredentialType;
      signed_uri: string;
    };
  }
}

/**
 * Get credentials for downloading trace data from the artifact store. Only used for Databricks.
 */
export namespace GetCredentialsForTraceDataDownload {
  export const getEndpoint = (host: string, traceId: string) =>
    `${host}/api/2.0/mlflow/traces/${traceId}/credentials-for-data-download`;

  export interface Response {
    credential_info: {
      type: ArtifactCredentialType;
      signed_uri: string;
    };
  }
}

/**
 * Search for traces
 */
export namespace SearchTracesV3 {
  export const getEndpoint = (host: string) => `${host}/api/3.0/mlflow/traces/search`;

  /**
   * TraceLocation for search request - specifies where to search for traces
   */
  export interface TraceLocation {
    type: 'MLFLOW_EXPERIMENT' | 'INFERENCE_TABLE' | 'TRACE_LOCATION_TYPE_UNSPECIFIED';
    mlflow_experiment?: {
      experiment_id: string;
    };
    inference_table?: {
      full_table_name: string;
    };
  }

  export interface Request {
    /** List of locations to search over */
    locations?: TraceLocation[];
    /** Filter expression (e.g. "trace.status = 'OK'") */
    filter?: string;
    /** Maximum number of traces to return (max 500, default 100) */
    max_results?: number;
    /** List of columns for ordering results (e.g. ["timestamp_ms DESC"]) */
    order_by?: string[];
    /** Token for pagination */
    page_token?: string;
  }

  export interface Response {
    /** List of traces matching the search criteria */
    traces: Parameters<typeof TraceInfo.fromJson>[0][];
    /** Token for fetching the next page of results */
    next_page_token?: string;
  }
}

/**
 * MLflow API Request/Response Specifications
 *
 * This module defines the types and interfaces for MLflow API communication,
 * including request payloads and response structures for all trace-related endpoints.
 */

import { TraceInfo } from '../core/entities/trace_info';
import { Trace } from '../core/entities/trace';
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
 * Create a new TraceInfo entity using the V4 API (UC-backed locations).
 */
export namespace CreateTraceV4 {
  export const getEndpoint = (host: string, locationId: string, traceId: string) =>
    `${host}/api/4.0/mlflow/traces/${locationId}/${traceId}/info`;

  export type Request = Parameters<typeof TraceInfo.fromJson>[0];

  export type Response = Parameters<typeof TraceInfo.fromJson>[0];
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

/**
 * Log spans to a UC-backed trace location using the Databricks OTLP proxy.
 * The payload must be an OTLP ExportTraceServiceRequest encoded as protobuf.
 */
export namespace LogSpans {
  export const DATABRICKS_UC_TABLE_HEADER = 'X-Databricks-UC-Table-Name';
  export const CONTENT_TYPE = 'application/x-protobuf';
  export const DEFAULT_SPAN_TABLE_NAME = 'mlflow_experiment_trace_otel_spans';

  export const getEndpoint = (host: string) => `${host}/api/2.0/otel/v1/traces`;

  /**
   * Returns the HTTP headers required for logging spans to the Databricks OTLP proxy.
   *
   * @param ucSchema - The Unity Catalog schema name (required for UC logging)
   * @param databricksToken - Optional Databricks personal access token. If provided, sets Authorization.
   * @returns A headers object suitable for fetch/AJAX requests.
   */
  export const getHeaders = (
    ucSchema: string,
    databricksToken?: string
  ): Record<string, string> => {
    const headers: Record<string, string> = {
      [LogSpans.DATABRICKS_UC_TABLE_HEADER]: `${ucSchema}.${LogSpans.DEFAULT_SPAN_TABLE_NAME}`,
      'Content-Type': LogSpans.CONTENT_TYPE,
    };
    if (databricksToken) {
      headers['Authorization'] = `Bearer ${databricksToken}`;
    }
    return headers;
  };
  // Request/response interfaces are defined by OTLP
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

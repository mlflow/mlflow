/**
 * MLflow API Request/Response Specifications
 *
 * This module defines the types and interfaces for MLflow API communication,
 * including request payloads and response structures for all trace-related endpoints.
 */

import type { TraceInfo } from '../core/entities/trace_info';
import type { SerializedAssessment } from '../core/entities/assessment';
import type { SerializedTraceLocation } from '../core/entities/trace_location';
import { ArtifactCredentialType } from './artifacts/databricks';
import type { SerializedIssue } from '../core/entities/issue';

const getIssueEndpoint = (host: string, issueId?: string) =>
  `${host}/api/3.0/mlflow/issues${issueId === undefined ? '' : `/${encodeURIComponent(issueId)}`}`;

export namespace CreateIssue {
  export const getEndpoint = (host: string) => getIssueEndpoint(host);
  export interface Request {
    experiment_id: string;
    name: string;
    description: string;
    severity: SerializedIssue['severity'];
    status?: SerializedIssue['status'];
    categories?: string[];
    metadata?: Record<string, string>;
    issue_id?: string;
    last_seen_timestamp?: number;
    source_run_id?: string;
  }
  export interface Response {
    issue: SerializedIssue;
  }
}

export namespace GetIssue {
  export const getEndpoint = getIssueEndpoint;
  export type Response = CreateIssue.Response;
}

export namespace UpdateIssue {
  export const getEndpoint = getIssueEndpoint;
  export interface Request {
    issue_id: string;
    name?: string;
    description?: string;
    status?: SerializedIssue['status'];
    severity?: SerializedIssue['severity'];
    categories?: string[];
    metadata?: Record<string, string>;
    delete_metadata_keys?: string[];
    last_seen_timestamp?: number;
  }
  export type Response = CreateIssue.Response;
}

export namespace SearchIssues {
  export const getEndpoint = (host: string) => `${getIssueEndpoint(host)}/search`;
  export interface Request {
    experiment_id: string;
    filter_string?: string;
    max_results?: number;
    order_by?: string[];
    page_token?: string;
    view?: 'basic' | 'full';
  }
  export interface Response {
    issues?: SerializedIssue[];
    next_page_token?: string;
    total_count?: number | string;
  }
}

export namespace DatabricksIssues {
  export const getEndpoint = (host: string, experimentId: string, issueId?: string) =>
    `${host}/api/2.0/mlflow/experiments/${encodeURIComponent(experimentId)}/issues${
      issueId === undefined ? '' : `/${encodeURIComponent(issueId)}`
    }`;

  export interface Resource {
    issue_id: string;
    title: string;
    description?: string;
    status: 'PENDING' | 'REJECTED' | 'RESOLVED';
    severity?: 'LOW' | 'MEDIUM' | 'HIGH' | 'ISSUE_SEVERITY_UNSPECIFIED';
    create_time: string;
    update_time: string;
    last_seen_time?: string;
    source_type?: string;
    source_id?: string;
    categories?: string[];
    metadata?: Record<string, string>;
    created_by?: string;
    last_updated_by?: string;
  }

  export interface CreateRequest {
    title: string;
    description: string;
    severity: string;
    status: string;
    source_type: 'MANUAL' | 'ADHOC_RUN';
    source_id?: string;
    last_seen_time?: string;
    categories?: string[];
    metadata?: Record<string, string>;
  }

  export interface UpdateRequest {
    name: string;
    title?: string;
    description?: string;
    severity?: string;
    status?: string;
    last_seen_time?: string;
    categories?: string[];
    metadata?: Record<string, string>;
  }

  export interface SearchRequest {
    parent: string;
    statuses: string[];
    page_size?: number;
    page_token?: string;
    sort_key?: string;
    sort_order?: 'ASCENDING' | 'DESCENDING';
    view: string;
  }

  export interface SearchResponse {
    issues?: Resource[];
    next_page_token?: string;
    total_size?: number | string;
  }
}

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

/**
 * Search trace metadata using the V3 traces API.
 */
export namespace SearchTracesV3 {
  export const getEndpoint = (host: string) => `${host}/api/3.0/mlflow/traces/search`;

  export interface Request {
    locations: SerializedTraceLocation[];
    filter?: string;
    max_results?: number;
    order_by?: string[];
    page_token?: string;
  }

  export interface Response {
    traces?: Parameters<typeof TraceInfo.fromJson>[0][];
    next_page_token?: string;
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

/** Get Experiment (used for UC destination auto-resolution) */
export namespace GetExperiment {
  export const getEndpoint = (host: string, experimentId: string) =>
    `${host}/api/2.0/mlflow/experiments/get?experiment_id=${encodeURIComponent(experimentId)}`;

  export interface Response {
    experiment?: {
      experiment_id: string;
      name: string;
      artifact_location?: string;
      lifecycle_stage?: string;
      tags?: { key: string; value: string }[];
    };
  }
}

/** Get Experiment By Name */
export namespace GetExperimentByName {
  export const getEndpoint = (host: string, experimentName: string) =>
    `${host}/api/2.0/mlflow/experiments/get-by-name?experiment_name=${encodeURIComponent(experimentName)}`;

  export interface Response {
    experiment?: {
      experiment_id: string;
      name: string;
    };
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
 * Create trace info using the V4 API path. Used for Databricks Unity Catalog
 * backed traces (UC schema or UC table prefix locations).
 *
 * Endpoint: POST /api/4.0/mlflow/traces/{location}/{otel_trace_id}/info
 * where `{location}` is "catalog.schema" or "catalog.schema.table_prefix",
 * and `{otel_trace_id}` is the hex OTel trace ID (no `trace:/<location>/` prefix).
 */
export namespace CreateTraceInfoV4 {
  export const getEndpoint = (host: string, location: string, otelTraceId: string) =>
    `${host}/api/4.0/mlflow/traces/${encodeURIComponent(location)}/${otelTraceId}/info`;

  // The Databricks RPC convention extracts `location_id` and `trace_info.trace_id`
  // from the URL path; the HTTP body is the serialized TraceInfo proto directly
  // (not wrapped in `{ trace_info: ... }`). Mirrors Python's
  // `message_to_json(trace_info.to_proto())`.
  export type Request = Parameters<typeof TraceInfo.fromJson>[0];

  // Backend returns a TraceInfo proto serialized as JSON.
  export type Response = Parameters<typeof TraceInfo.fromJson>[0];
}

/**
 * OTLP span upload endpoint for Databricks Unity Catalog backed traces.
 *
 * Endpoint: POST /api/2.0/otel/v1/traces
 * Required header: `X-Databricks-UC-Table-Name: <fully_qualified_spans_table>`
 * Content-Type: application/x-protobuf (OTLP/HTTP+protobuf); the Databricks
 * endpoint does not accept the OTLP/HTTP+JSON form.
 */
export namespace ExportOtlpTraces {
  export const getEndpoint = (host: string) => `${host}/api/2.0/otel/v1/traces`;
}

/**
 * Create an assessment on a trace using the V3 traces API.
 * Endpoint: POST /api/3.0/mlflow/traces/{trace_id}/assessments
 */
export namespace CreateAssessment {
  export const getEndpoint = (host: string, traceId: string) =>
    `${host}/api/3.0/mlflow/traces/${encodeURIComponent(traceId)}/assessments`;

  export interface Request {
    assessment: SerializedAssessment;
  }

  export interface Response {
    assessment: SerializedAssessment;
  }
}

/**
 * OTLP span upload endpoint for OSS (non-Databricks) tracking servers.
 *
 * Endpoint: POST /v1/traces
 * Required header: `x-mlflow-experiment-id: <experiment_id>`
 * Content-Type: application/x-protobuf (OTLP/HTTP+protobuf).
 */
export namespace ExportOtlpTracesOss {
  export const getEndpoint = (host: string) => `${host}/v1/traces`;
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

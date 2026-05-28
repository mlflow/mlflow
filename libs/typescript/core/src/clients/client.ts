import { TraceInfo } from '../core/entities/trace_info';
import { Trace } from '../core/entities/trace';
import {
  CreateExperiment,
  CreateTraceInfoV4,
  DeleteExperiment,
  ExportOtlpTraces,
  GetExperiment,
  GetExperimentByName,
  GetTraceInfoV3,
  SearchTracesV3,
  StartTraceV3,
} from './spec';
import { makeRequest, MlflowHttpError } from './utils';
import { TraceData } from '../core/entities/trace_data';
import { ArtifactsClient, getArtifactsClient } from './artifacts';
import { AuthProvider, HeadersProvider } from '../auth';
import { DATABRICKS_UC_TABLE_HEADER } from '../core/constants';
import {
  createTraceLocationFromExperimentId,
  serializeTraceLocation,
  type TraceLocation,
} from '../core/entities/trace_location';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import type { ReadableSpan as OTelReadableSpan } from '@opentelemetry/sdk-trace-base';
import { ExportResultCode, type ExportResult } from '@opentelemetry/core';

export interface SearchTracesOptions {
  experimentIds?: string[];
  locations?: TraceLocation[];
  filter?: string;
  maxResults?: number;
  orderBy?: string[];
  pageToken?: string;
}

export interface SearchTracesResult {
  traces: TraceInfo[];
  nextPageToken?: string;
}

/**
 * Client for MLflow tracing operations.
 *
 * Supports multiple authentication methods:
 * - Databricks: PAT tokens, OAuth M2M, Azure CLI/MSI, Google Cloud
 * - OSS MLflow: Basic auth, Bearer tokens, or no auth
 */
export class MlflowClient {
  /** Client implementation to upload/download trace data artifacts */
  private artifactsClient: ArtifactsClient;

  /** Headers provider for authenticated requests */
  private headersProvider: HeadersProvider;

  /** Host URL for API requests */
  private hostUrl: string;

  /**
   * Creates a new MlflowClient.
   *
   * @param trackingUri - The tracking URI (e.g., "databricks", "http://localhost:5000")
   * @param authProvider - The authentication provider to get tokens for authenticated requests
   */
  constructor(options: { trackingUri: string; authProvider: AuthProvider }) {
    this.headersProvider = options.authProvider.getHeadersProvider();
    this.hostUrl = options.authProvider.getHost();

    this.artifactsClient = getArtifactsClient({
      trackingUri: options.trackingUri,
      host: this.hostUrl,
      authProvider: options.authProvider,
    });
  }

  /**
   * Get the host URL for this client.
   */
  getHost(): string {
    return this.hostUrl;
  }

  // === TRACE LOGGING METHODS ===
  /**
   * Create a new TraceInfo record in the backend store.
   * Corresponding to the Python SDK's start_trace_v3() method.
   *
   * Note: the backend API is named as "Start" due to unfortunate miscommunication.
   * The API is indeed called at the "end" of a trace, not the "start".
   */
  async createTrace(traceInfo: TraceInfo): Promise<TraceInfo> {
    const url = StartTraceV3.getEndpoint(this.hostUrl);
    const payload: StartTraceV3.Request = { trace: { trace_info: traceInfo.toJson() } };
    const response = await makeRequest<StartTraceV3.Response>(
      'POST',
      url,
      this.headersProvider,
      payload,
    );
    return TraceInfo.fromJson(response.trace.trace_info);
  }

  /**
   * Create a new TraceInfo record in the Databricks V4 (Unity Catalog)
   * trace-info endpoint. Used for UC-backed traces so that trace-level
   * tags and metadata persist alongside spans uploaded via OTLP.
   *
   * Corresponds to Python's `DatabricksRestStore._start_trace_v4`.
   *
   * @param location - The UC location string ("catalog.schema" or
   *                   "catalog.schema.table_prefix") that comes from
   *                   the trace ID `trace:/<location>/<hex>`.
   * @param otelTraceId - The hex OTel trace ID portion.
   * @param traceInfo - The full TraceInfo to persist; the backend reads
   *                    `tags`, `trace_metadata`, `state`, durations, etc.
   */
  async createTraceInfoV4(
    location: string,
    otelTraceId: string,
    traceInfo: TraceInfo,
  ): Promise<TraceInfo> {
    const url = CreateTraceInfoV4.getEndpoint(this.hostUrl, location, otelTraceId);
    const payload: CreateTraceInfoV4.Request = traceInfo.toJson();
    const response = await makeRequest<CreateTraceInfoV4.Response>(
      'POST',
      url,
      this.headersProvider,
      payload,
    );
    return TraceInfo.fromJson(response);
  }

  /**
   * Upload OTel spans to a Databricks Unity Catalog location via the OTLP
   * HTTP+protobuf endpoint. The `spansTableName` is the fully qualified
   * spans table (catalog.schema.table) and is forwarded as the
   * `X-Databricks-UC-Table-Name` header used by Databricks to route the
   * payload to the correct UC location.
   *
   * Uses `@opentelemetry/exporter-trace-otlp-proto` which serializes the
   * spans to the OTLP `ExportTraceServiceRequest` protobuf wire format —
   * the Databricks endpoint only accepts `application/x-protobuf`, not
   * the OTLP/HTTP+JSON form.
   */
  async exportOtlpSpansToUc(spans: OTelReadableSpan[], spansTableName: string): Promise<void> {
    if (spans.length === 0) {
      return;
    }
    // Fetch fresh auth headers for every export so OAuth-rotated tokens are
    // honored. The exporter takes a static headers map at construction time,
    // so we rebuild it per call rather than caching an instance.
    const authHeaders = await this.headersProvider();
    const exporter = new OTLPTraceExporter({
      url: ExportOtlpTraces.getEndpoint(this.hostUrl),
      headers: {
        ...authHeaders,
        [DATABRICKS_UC_TABLE_HEADER]: spansTableName,
      },
    });
    try {
      await new Promise<void>((resolve, reject) => {
        exporter.export(spans, (result: ExportResult) => {
          if (result.code === ExportResultCode.SUCCESS) {
            resolve();
          } else {
            reject(result.error ?? new Error(`OTLP span export failed with code ${result.code}`));
          }
        });
      });
    } finally {
      // Drain any pending state so the exporter doesn't keep handles open.
      await exporter.shutdown().catch(() => undefined);
    }
  }

  // === TRACE RETRIEVAL METHODS ===
  /**
   * Get a single trace by ID
   * Fetches both trace info and trace data from backend
   * Corresponds to Python: client.get_trace()
   */
  async getTrace(traceId: string): Promise<Trace> {
    const traceInfo = await this.getTraceInfo(traceId);
    const traceData = await this.artifactsClient.downloadTraceData(traceInfo);
    return new Trace(traceInfo, traceData);
  }

  /**
   * Get trace info using V3 API
   * Endpoint: GET /api/3.0/mlflow/traces/{trace_id}
   */
  async getTraceInfo(traceId: string): Promise<TraceInfo> {
    const url = GetTraceInfoV3.getEndpoint(this.hostUrl, traceId);
    const response = await makeRequest<GetTraceInfoV3.Response>('GET', url, this.headersProvider);

    // The V3 API returns a Trace object with trace_info field
    if (response.trace?.trace_info) {
      return TraceInfo.fromJson(response.trace.trace_info);
    }

    throw new Error(`Invalid response format: missing trace_info: ${JSON.stringify(response)}`);
  }

  /**
   * Search trace metadata using the V3 traces API.
   */
  async searchTraces(options: SearchTracesOptions): Promise<SearchTracesResult> {
    const locations = [
      ...(options.locations ?? []),
      ...(options.experimentIds ?? []).map(createTraceLocationFromExperimentId),
    ];
    if (locations.length === 0) {
      throw new Error('searchTraces requires at least one experiment ID or trace location.');
    }

    const url = SearchTracesV3.getEndpoint(this.hostUrl);
    const payload: SearchTracesV3.Request = {
      locations: locations.map(serializeTraceLocation),
      filter: options.filter,
      max_results: options.maxResults,
      order_by: options.orderBy,
      page_token: options.pageToken,
    };
    const response = await makeRequest<SearchTracesV3.Response>(
      'POST',
      url,
      this.headersProvider,
      payload,
    );

    return {
      traces: response.traces.map((trace) => TraceInfo.fromJson(trace)),
      nextPageToken: response.next_page_token || undefined,
    };
  }

  /**
   * Upload trace data to the artifact store.
   */
  async uploadTraceData(traceInfo: TraceInfo, traceData: TraceData): Promise<void> {
    await this.artifactsClient.uploadTraceData(traceInfo, traceData);
  }

  // === EXPERIMENT METHODS  ===
  /**
   * Create a new experiment
   */
  async createExperiment(
    name: string,
    artifactLocation?: string,
    tags?: Record<string, string>,
  ): Promise<string> {
    const url = CreateExperiment.getEndpoint(this.hostUrl);
    const payload: CreateExperiment.Request = { name, artifact_location: artifactLocation, tags };
    const response = await makeRequest<CreateExperiment.Response>(
      'POST',
      url,
      this.headersProvider,
      payload,
    );
    return response.experiment_id;
  }

  /**
   * Get an experiment by ID, including its tags. Used to auto-resolve UC
   * trace destinations from a Databricks-linked experiment.
   */
  async getExperiment(
    experimentId: string,
  ): Promise<{ experimentId: string; name: string; tags: Record<string, string> } | null> {
    const url = GetExperiment.getEndpoint(this.hostUrl, experimentId);
    try {
      const response = await makeRequest<GetExperiment.Response>('GET', url, this.headersProvider);
      const exp = response.experiment;
      if (!exp?.experiment_id) {
        return null;
      }
      const tags: Record<string, string> = {};
      for (const tag of exp.tags ?? []) {
        tags[tag.key] = tag.value;
      }
      return { experimentId: exp.experiment_id, name: exp.name, tags };
    } catch (error) {
      if (
        error instanceof MlflowHttpError &&
        (error.status === 404 || error.errorCode === 'RESOURCE_DOES_NOT_EXIST')
      ) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Get an experiment by name.
   */
  async getExperimentByName(name: string): Promise<{ experimentId: string; name: string } | null> {
    const url = GetExperimentByName.getEndpoint(this.hostUrl, name);
    try {
      const response = await makeRequest<GetExperimentByName.Response>(
        'GET',
        url,
        this.headersProvider,
      );
      if (!response.experiment?.experiment_id || !response.experiment?.name) {
        return null;
      }
      return {
        experimentId: response.experiment.experiment_id,
        name: response.experiment.name,
      };
    } catch (error) {
      if (
        error instanceof MlflowHttpError &&
        (error.status === 404 || error.errorCode === 'RESOURCE_DOES_NOT_EXIST')
      ) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Delete an experiment
   */
  async deleteExperiment(experimentId: string): Promise<void> {
    const url = DeleteExperiment.getEndpoint(this.hostUrl);
    const payload: DeleteExperiment.Request = { experiment_id: experimentId };
    await makeRequest<void>('POST', url, this.headersProvider, payload);
  }
}

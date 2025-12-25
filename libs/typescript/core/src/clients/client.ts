import { TraceInfo } from '../core/entities/trace_info';
import { Trace } from '../core/entities/trace';
import {
  CreateExperiment,
  DeleteExperiment,
  GetTraceInfoV3,
  GetTrace,
  StartTraceV3,
  SearchTracesV3
} from './spec';
import { getRequestHeaders, makeRequest } from './utils';
import { TraceData } from '../core/entities/trace_data';
import { ArtifactsClient, getArtifactsClient } from './artifacts';
import { getConfig } from '../core/config';
import { Span } from '../core/entities/span';
import { TraceTagKey, SpansLocation } from '../core/constants';

/**
 * Client for MLflow tracing operations
 */
export class MlflowClient {
  /** MLflow tracking server host or Databricks workspace URL */
  private host: string;
  /** Databricks personal access token */
  private databricksToken?: string;
  /** Client implementation to upload/download trace data artifacts */
  private artifactsClient: ArtifactsClient;
  /** The tracking server username for basic auth */
  private trackingServerUsername?: string;
  /** The tracking server password for basic auth */
  private trackingServerPassword?: string;

  constructor(options: {
    trackingUri: string;
    host: string;
    databricksToken?: string;
    trackingServerUsername?: string;
    trackingServerPassword?: string;
  }) {
    this.host = options.host;
    this.databricksToken = options.databricksToken;
    this.trackingServerUsername = options.trackingServerUsername;
    this.trackingServerPassword = options.trackingServerPassword;
    this.artifactsClient = getArtifactsClient({
      trackingUri: options.trackingUri,
      host: options.host,
      databricksToken: options.databricksToken
    });
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
    const url = StartTraceV3.getEndpoint(this.host);
    const payload: StartTraceV3.Request = { trace: { trace_info: traceInfo.toJson() } };
    const response = await makeRequest<StartTraceV3.Response>(
      'POST',
      url,
      getRequestHeaders(
        this.databricksToken,
        this.trackingServerUsername,
        this.trackingServerPassword
      ),
      payload
    );
    return TraceInfo.fromJson(response.trace.trace_info);
  }

  // === TRACE RETRIEVAL METHODS ===
  /**
   * Get a single trace by ID.
   *
   * Fetches both trace info and trace data from backend. The location of span data
   * is determined by the `mlflow.trace.spansLocation` tag:
   * - TRACKING_STORE: Spans are fetched via the GetTrace API (proto format)
   * - ARTIFACT_REPO: Spans are downloaded from the artifact store
   *
   */
  async getTrace(traceId: string): Promise<Trace> {
    // First get trace info to determine where spans are stored
    const traceInfo = await this.getTraceInfo(traceId);
    const spansLocation = traceInfo.tags[TraceTagKey.SPANS_LOCATION];

    // If spans are in tracking store, use the GetTrace API
    if (spansLocation === SpansLocation.TRACKING_STORE) {
      return this.getTraceFromTrackingStore(traceId, traceInfo);
    }

    // Otherwise, download from artifact store
    return this.getTraceFromArtifactStore(traceInfo);
  }

  /**
   * Fetch trace with spans from tracking store using GetTrace API.
   * Used when spans are stored in the tracking store (SPANS_LOCATION = TRACKING_STORE).
   */
  private async getTraceFromTrackingStore(traceId: string, traceInfo: TraceInfo): Promise<Trace> {
    const url = GetTrace.getEndpoint(this.host, traceId);
    const response = await makeRequest<GetTrace.Response>(
      'GET',
      url,
      getRequestHeaders(
        this.databricksToken,
        this.trackingServerUsername,
        this.trackingServerPassword
      )
    );
    const spans = (response.trace.spans || []).map((s) => Span.fromOtelProto(s));
    return new Trace(traceInfo, new TraceData(spans));
  }

  /**
   * Fetch trace with spans from artifact store.
   * Used when spans are stored in artifacts (SPANS_LOCATION = ARTIFACT_REPO or not set).
   */
  private async getTraceFromArtifactStore(traceInfo: TraceInfo): Promise<Trace> {
    const traceData = await this.artifactsClient.downloadTraceData(traceInfo);
    return new Trace(traceInfo, traceData);
  }

  /**
   * Get trace info using V3 API
   * Endpoint: GET /api/3.0/mlflow/traces/{trace_id}
   */
  async getTraceInfo(traceId: string): Promise<TraceInfo> {
    const url = GetTraceInfoV3.getEndpoint(this.host, traceId);
    const response = await makeRequest<GetTraceInfoV3.Response>(
      'GET',
      url,
      getRequestHeaders(
        this.databricksToken,
        this.trackingServerUsername,
        this.trackingServerPassword
      )
    );

    // The V3 API returns a Trace object with trace_info field
    if (response.trace?.trace_info) {
      return TraceInfo.fromJson(response.trace.trace_info);
    }

    throw new Error(`Invalid response format: missing trace_info: ${JSON.stringify(response)}`);
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
    tags?: Record<string, string>
  ): Promise<string> {
    const url = CreateExperiment.getEndpoint(this.host);
    const payload: CreateExperiment.Request = { name, artifact_location: artifactLocation, tags };
    const response = await makeRequest<CreateExperiment.Response>(
      'POST',
      url,
      getRequestHeaders(
        this.databricksToken,
        this.trackingServerUsername,
        this.trackingServerPassword
      ),
      payload
    );
    return response.experiment_id;
  }

  /**
   * Delete an experiment
   */
  async deleteExperiment(experimentId: string): Promise<void> {
    const url = DeleteExperiment.getEndpoint(this.host);
    const payload: DeleteExperiment.Request = { experiment_id: experimentId };
    await makeRequest<void>(
      'POST',
      url,
      getRequestHeaders(
        this.databricksToken,
        this.trackingServerUsername,
        this.trackingServerPassword
      ),
      payload
    );
  }

  /**
   * Search for traces that match the given filter criteria.
   *
   * @param options - Search options
   * @param options.experimentIds - List of experiment IDs to search over. If not provided,
   *   defaults to the experiment ID set via `mlflow.init()`.
   * @param options.filterString - Filter expression (e.g. "trace.status = 'OK'")
   * @param options.maxResults - Maximum number of traces to return (default 100, max 500)
   * @param options.orderBy - List of columns for ordering results (e.g. ["timestamp_ms DESC"])
   * @param options.pageToken - Token for pagination from a previous search
   * @param options.includeSpans - If true, fetch full trace data including spans. Default is true.
   * @returns Object containing traces and optional next page token
   */
  async searchTraces(
    options: {
      experimentIds?: string[];
      filterString?: string;
      maxResults?: number;
      orderBy?: string[];
      pageToken?: string;
      includeSpans?: boolean;
    } = {}
  ): Promise<{ traces: Trace[]; nextPageToken?: string }> {
    const url = SearchTracesV3.getEndpoint(this.host);
    const includeSpans = options.includeSpans ?? true;

    // Use provided experimentIds, or default to the experiment ID from init()
    const experimentIds =
      options.experimentIds && options.experimentIds.length > 0
        ? options.experimentIds
        : [getConfig().experimentId];

    const locations: SearchTracesV3.TraceLocation[] = experimentIds.map(
      (experimentId) => ({
        type: 'MLFLOW_EXPERIMENT' as const,
        mlflow_experiment: { experiment_id: experimentId }
      })
    );

    const payload: SearchTracesV3.Request = {
      locations,
      filter: options.filterString,
      max_results: options.maxResults,
      order_by: options.orderBy,
      page_token: options.pageToken
    };

    const response = await makeRequest<SearchTracesV3.Response>(
      'POST',
      url,
      getRequestHeaders(
        this.databricksToken,
        this.trackingServerUsername,
        this.trackingServerPassword
      ),
      payload
    );

    const traceInfos = (response.traces || []).map((t) => TraceInfo.fromJson(t));

    // If includeSpans is false, return traces with empty span data
    if (!includeSpans) {
      return {
        traces: traceInfos.map((info) => new Trace(info, new TraceData([]))),
        nextPageToken: response.next_page_token
      };
    }

    const traces = await Promise.all(
      traceInfos.map((info) =>
        this.getTrace(info.traceId).catch(() => new Trace(info, new TraceData([])))
      )
    );

    return {
      traces,
      nextPageToken: response.next_page_token
    };
  }
}

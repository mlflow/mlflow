import { TraceInfo } from '../core/entities/trace_info';
import { Trace } from '../core/entities/trace';
import {
  CreateExperiment,
  DeleteExperiment,
  GetTraceInfoV3,
  SearchTracesV3,
  StartTraceV3,
} from './spec';
import { makeRequest } from './utils';
import { TraceData } from '../core/entities/trace_data';
import { ArtifactsClient, getArtifactsClient } from './artifacts';
import { AuthProvider, HeadersProvider } from '../auth';

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
   * Upload trace data to the artifact store.
   */
  async uploadTraceData(traceInfo: TraceInfo, traceData: TraceData): Promise<void> {
    await this.artifactsClient.uploadTraceData(traceInfo, traceData);
  }

  // === TRACE SEARCH METHODS ===
  /**
   * Search for traces matching given criteria.
   * Corresponds to Python: client.search_traces()
   *
   * Endpoint: POST /api/3.0/mlflow/traces/search
   *
   * @param options - Search parameters
   * @param options.experimentIds - Experiment IDs to search within
   * @param options.filter - SQL-like filter string, e.g. "trace.status = 'OK'"
   * @param options.maxResults - Maximum results to return (default 100, max 500)
   * @param options.orderBy - Sort columns, e.g. ["timestamp_ms DESC"]
   * @param options.pageToken - Token for fetching the next page
   * @returns A paged list of TraceInfo objects
   */
  async searchTraces(options: {
    experimentIds?: string[];
    filter?: string;
    maxResults?: number;
    orderBy?: string[];
    pageToken?: string;
  } = {}): Promise<{ items: TraceInfo[]; nextPageToken?: string }> {
    const url = SearchTracesV3.getEndpoint(this.hostUrl);

    const locations = options.experimentIds?.map(
      (id): SearchTracesV3.ExperimentLocation => ({
        type: 'MLFLOW_EXPERIMENT',
        mlflow_experiment: { experiment_id: id },
      }),
    );

    const payload: SearchTracesV3.Request = {
      locations,
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

    const traces = (response.traces ?? []).map((t) => TraceInfo.fromJson(t));
    return {
      items: traces,
      nextPageToken: response.next_page_token,
    };
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
   * Delete an experiment
   */
  async deleteExperiment(experimentId: string): Promise<void> {
    const url = DeleteExperiment.getEndpoint(this.hostUrl);
    const payload: DeleteExperiment.Request = { experiment_id: experimentId };
    await makeRequest<void>('POST', url, this.headersProvider, payload);
  }
}

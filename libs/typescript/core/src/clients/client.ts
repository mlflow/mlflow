import { TraceInfo } from '../core/entities/trace_info';
import { Trace } from '../core/entities/trace';
import { CreateExperiment, DeleteExperiment, GetTraceInfoV3, StartTraceV3 } from './spec';
import { getRequestHeaders, makeRequest } from './utils';
import { TraceData } from '../core/entities/trace_data';
import { ArtifactsClient, getArtifactsClient } from './artifacts';
import { AuthProvider, HeadersProvider } from '../auth';

/**
 * Options for creating an MlflowClient.
 *
 * Supports two modes:
 * 1. New mode: Use an AuthProvider for authentication (recommended)
 * 2. Legacy mode: Use individual host/token/username/password options (backwards compatible)
 */
export interface MlflowClientOptions {
  /**
   * The tracking URI (e.g., "databricks", "http://localhost:5000")
   */
  trackingUri: string;

  /**
   * Authentication provider for the client.
   * When provided, this takes precedence over legacy auth options.
   */
  authProvider?: AuthProvider;

  /**
   * MLflow tracking server host or Databricks workspace URL.
   * @deprecated Use authProvider instead
   */
  host?: string;

  /**
   * Databricks personal access token.
   * @deprecated Use authProvider instead
   */
  databricksToken?: string;

  /**
   * The tracking server username for basic auth.
   * @deprecated Use authProvider instead
   */
  trackingServerUsername?: string;

  /**
   * The tracking server password for basic auth.
   * @deprecated Use authProvider instead
   */
  trackingServerPassword?: string;
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
   * @param options - Client configuration options
   */
  constructor(options: MlflowClientOptions) {
    if (options.authProvider) {
      // New mode: Use AuthProvider
      this.headersProvider = options.authProvider.getHeadersProvider();
      this.hostUrl = options.authProvider.getHost();
    } else if (options.host) {
      // Legacy mode: Use individual options (backwards compatibility)
      this.hostUrl = options.host;
      this.headersProvider = this.createLegacyHeadersProvider(
        options.databricksToken,
        options.trackingServerUsername,
        options.trackingServerPassword
      );
    } else {
      throw new Error('MlflowClient requires either an authProvider or a host option');
    }

    this.artifactsClient = getArtifactsClient({
      trackingUri: options.trackingUri,
      host: this.hostUrl,
      authProvider: options.authProvider,
      // Legacy options for backwards compatibility
      databricksToken: options.databricksToken
    });
  }

  /**
   * Create a legacy headers provider from individual auth options.
   * This maintains backwards compatibility with existing code.
   */
  private createLegacyHeadersProvider(
    databricksToken?: string,
    username?: string,
    password?: string
  ): HeadersProvider {
    return async () => {
      return getRequestHeaders(databricksToken, username, password);
    };
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
    const headers = await this.headersProvider();
    const response = await makeRequest<StartTraceV3.Response>('POST', url, headers, payload);
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
    const headers = await this.headersProvider();
    const response = await makeRequest<GetTraceInfoV3.Response>('GET', url, headers);

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
    const url = CreateExperiment.getEndpoint(this.hostUrl);
    const payload: CreateExperiment.Request = { name, artifact_location: artifactLocation, tags };
    const headers = await this.headersProvider();
    const response = await makeRequest<CreateExperiment.Response>('POST', url, headers, payload);
    return response.experiment_id;
  }

  /**
   * Delete an experiment
   */
  async deleteExperiment(experimentId: string): Promise<void> {
    const url = DeleteExperiment.getEndpoint(this.hostUrl);
    const payload: DeleteExperiment.Request = { experiment_id: experimentId };
    const headers = await this.headersProvider();
    await makeRequest<void>('POST', url, headers, payload);
  }
}

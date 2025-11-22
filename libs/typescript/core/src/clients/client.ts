import { TraceInfo } from '../core/entities/trace_info';
import { Trace } from '../core/entities/trace';
import { CreateExperiment, DeleteExperiment, GetTraceInfoV3, StartTraceV3 } from './spec';
import { getRequestHeaders, makeRequest } from './utils';
import { TraceData } from '../core/entities/trace_data';
import { ArtifactsClient, getArtifactsClient } from './artifacts';
import { DatabricksAuthProvider } from '../core/databricks_auth';

/**
 * Client for MLflow tracing operations
 */
export class MlflowClient {
  /** MLflow tracking server host or Databricks workspace URL */
  private host: string;
  /** Databricks personal access token */
  private databricksToken?: string;
  /** Databricks auth provider (handles PAT or OAuth client credentials) */
  private databricksAuthProvider?: DatabricksAuthProvider;
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
    databricksAuthProvider?: DatabricksAuthProvider;
    trackingServerUsername?: string;
    trackingServerPassword?: string;
  }) {
    this.host = options.host;
    this.databricksToken = options.databricksToken;
    this.databricksAuthProvider = options.databricksAuthProvider;
    this.trackingServerUsername = options.trackingServerUsername;
    this.trackingServerPassword = options.trackingServerPassword;
    this.artifactsClient = getArtifactsClient({
      trackingUri: options.trackingUri,
      host: options.host,
      databricksToken: options.databricksToken,
      databricksAuthProvider: options.databricksAuthProvider
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
      await this.buildRequestHeaders(),
      payload
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
    const url = GetTraceInfoV3.getEndpoint(this.host, traceId);
    const response = await makeRequest<GetTraceInfoV3.Response>(
      'GET',
      url,
      await this.buildRequestHeaders()
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
      await this.buildRequestHeaders(),
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
    await makeRequest<void>('POST', url, await this.buildRequestHeaders(), payload);
  }

  private async buildRequestHeaders(): Promise<Record<string, string>> {
    let token: string | undefined;

    if (this.databricksAuthProvider) {
      token = await this.databricksAuthProvider.getAccessToken();

      if (!token) {
        throw new Error(
          'Databricks access token could not be acquired. Verify client credentials or token configuration.'
        );
      }
    } else {
      token = this.databricksToken;
    }

    return getRequestHeaders(token, this.trackingServerUsername, this.trackingServerPassword);
  }
}

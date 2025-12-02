import { AuthProvider } from '../auth/types';
import {
  NoAuthProvider,
  PersonalAccessTokenProvider,
  BasicAuthProvider
} from '../auth/providers';
import { TraceInfo } from '../core/entities/trace_info';
import { Trace } from '../core/entities/trace';
import { CreateExperiment, DeleteExperiment, GetTraceInfoV3, StartTraceV3 } from './spec';
import { makeAuthenticatedRequest } from './utils';
import { TraceData } from '../core/entities/trace_data';
import { ArtifactsClient, getArtifactsClient } from './artifacts';

/**
 * Client for MLflow tracing operations
 */
export class MlflowClient {
  /** MLflow tracking server host or Databricks workspace URL */
  private host: string;
  /** Authentication provider for API requests */
  private authProvider: AuthProvider;
  /** Client implementation to upload/download trace data artifacts */
  private artifactsClient: ArtifactsClient;

  constructor(options: {
    trackingUri: string;
    host: string;
    authProvider?: AuthProvider;
    /** @deprecated Use authProvider instead */
    databricksToken?: string;
    /** @deprecated Use authProvider instead */
    trackingServerUsername?: string;
    /** @deprecated Use authProvider instead */
    trackingServerPassword?: string;
  }) {
    this.host = options.host;

    // Initialize auth provider
    if (options.authProvider) {
      this.authProvider = options.authProvider;
    } else if (options.databricksToken) {
      // Legacy: create PAT provider from token
      this.authProvider = new PersonalAccessTokenProvider(options.databricksToken);
    } else if (options.trackingServerUsername && options.trackingServerPassword) {
      // Legacy: create Basic Auth provider
      this.authProvider = new BasicAuthProvider(
        options.trackingServerUsername,
        options.trackingServerPassword
      );
    } else {
      // No auth
      this.authProvider = new NoAuthProvider();
    }

    this.artifactsClient = getArtifactsClient({
      trackingUri: options.trackingUri,
      host: options.host,
      authProvider: this.authProvider,
      // Pass legacy credentials for backwards compatibility
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
    const response = await makeAuthenticatedRequest<StartTraceV3.Response>(
      'POST',
      url,
      this.authProvider,
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
    const response = await makeAuthenticatedRequest<GetTraceInfoV3.Response>(
      'GET',
      url,
      this.authProvider
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
    const response = await makeAuthenticatedRequest<CreateExperiment.Response>(
      'POST',
      url,
      this.authProvider,
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
    await makeAuthenticatedRequest<void>('POST', url, this.authProvider, payload);
  }
}

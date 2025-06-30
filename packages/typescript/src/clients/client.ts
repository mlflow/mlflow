import { TraceInfo } from '../core/entities/trace_info';
import { Trace } from '../core/entities/trace';
import { CreateExperiment, DeleteExperiment, GetTraceInfoV3, StartTraceV3 } from './spec';
import { getRequestHeaders, makeRequest } from './utils';
import { TraceData } from '../core/entities/trace_data';

/**
 * Client for MLflow tracing operations
 */
export class MlflowClient {
  /** MLflow tracking server host or Databricks workspace URL */
  private host: string;
  /** Personal access token */
  private token?: string;

  constructor(options: { host: string; token?: string }) {
    // The host is guaranteed to be set by the init() function
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    this.host = options.host!;
    this.token = options.token;
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
      getRequestHeaders(this.token),
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
    // TODO: Implement trace data download
    return new Trace(traceInfo, new TraceData([]));
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
      getRequestHeaders(this.token)
    );

    // The V3 API returns a Trace object with trace_info field
    if (response.trace?.trace_info) {
      return TraceInfo.fromJson(response.trace.trace_info);
    }

    throw new Error(`Invalid response format: missing trace_info: ${JSON.stringify(response)}`);
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
      getRequestHeaders(this.token),
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
    await makeRequest<void>('POST', url, getRequestHeaders(this.token), payload);
  }
}

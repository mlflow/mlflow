import { TraceTagKey } from '../../core/constants';
import { SerializedTraceData, TraceData } from '../../core/entities/trace_data';
import { TraceInfo } from '../../core/entities/trace_info';
import { getRequestHeaders, makeRequest } from '../utils';
import { ArtifactsClient } from './base';

/**
 * Trace data file name constant - matches Python SDK
 */
const TRACE_DATA_FILE_NAME = 'traces.json';

/**
 * MLflow OSS Artifacts Client
 *
 * Implements artifact upload/download for OSS MLflow Tracking Server using the standard
 * HTTP artifact repository endpoints. Based on Python HttpArtifactRepository.
 */
export class MlflowArtifactsClient implements ArtifactsClient {
  private readonly host: string;

  constructor(options: { host: string }) {
    this.host = options.host;
  }
  /**
   * Upload trace data to MLflow artifact storage.
   *
   * Equivalent to Python's upload_trace_data() method which uses log_artifact()
   * under the hood to upload the trace data as a JSON file.
   *
   * @param traceInfo The trace information containing artifact URI
   * @param traceData The trace data to upload
   */
  async uploadTraceData(traceInfo: TraceInfo, traceData: TraceData): Promise<void> {
    // Serialize trace data to JSON string (equivalent to Python's json.dumps)
    const traceDataJson = traceData.toJson();

    // Upload trace data to the artifact store
    const artifactUrl = this.getArtifactUrlForTrace(traceInfo);
    const headers = getRequestHeaders();
    await makeRequest<void>('PUT', artifactUrl, headers, traceDataJson);
  }

  /**
   * Download trace data from MLflow artifact storage.
   *
   * Equivalent to Python's download_trace_data() method which downloads
   * the traces.json file and parses it back to TraceData.
   *
   * @param traceInfo The trace information containing artifact URI
   * @returns The downloaded and parsed trace data
   */
  async downloadTraceData(traceInfo: TraceInfo): Promise<TraceData> {
    // Download the trace data file
    const artifactUrl = this.getArtifactUrlForTrace(traceInfo);
    const headers = getRequestHeaders();

    const traceDataJson = await makeRequest<SerializedTraceData>('GET', artifactUrl, headers);

    // Parse JSON back to TraceData (equivalent to Python's try_read_trace_data)
    try {
      return TraceData.fromJson(traceDataJson);
    } catch (error) {
      throw new Error(
        `Failed to parse trace data JSON: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  /**
   * Construct the artifact URL from the trace info. The artifact location is set as a tag
   * on the trace info by the backend.
   *
   * This implements the same URI resolution logic as Python's MlflowArtifactsRepository.resolve_uri()
   * converting mlflow-artifacts:// URIs to HTTP endpoints.
   */
  private getArtifactUrlForTrace(traceInfo: TraceInfo): string {
    const artifactUri = traceInfo.tags[TraceTagKey.MLFLOW_ARTIFACT_LOCATION];

    if (!artifactUri) {
      throw new Error('Artifact location not found in trace tags');
    }

    // Resolve mlflow-artifacts:// URI to HTTP endpoint
    return this.resolveArtifactUri(artifactUri, TRACE_DATA_FILE_NAME);
  }

  /**
   * Resolve mlflow-artifacts:// URI to HTTP endpoint.
   *
   * Equivalent to Python's MlflowArtifactsRepository.resolve_uri() method.
   * Transforms URIs like "mlflow-artifacts:/0/traces/tr-abc123/artifacts"
   * to "http://localhost:5000/api/2.0/mlflow-artifacts/artifacts/0/traces/tr-abc123/artifacts/traces.json"
   *
   * @param artifactUri The mlflow-artifacts:// URI from trace tags
   * @param fileName The file name to append (e.g., "traces.json")
   * @returns The resolved HTTP endpoint URL
   */
  private resolveArtifactUri(artifactUri: string, fileName: string): string {
    const baseApiPath = '/api/2.0/mlflow-artifacts/artifacts';
    const url = new URL(artifactUri);

    if (url.protocol !== 'mlflow-artifacts:') {
      throw new Error(`Expected mlflow-artifacts:// URI, got ${url.protocol}`);
    }

    // Construct the final HTTP URL
    const cleanHost = this.host.replace(/\/$/, ''); // Remove trailing slash
    return `${cleanHost}${baseApiPath}${url.pathname}/${fileName}`;
  }
}

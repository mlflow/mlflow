import { SerializedTraceData, TraceData } from '../../core/entities/trace_data';
import { TraceInfo } from '../../core/entities/trace_info';
import { JSONBig } from '../../core/utils/json';
import { GetCredentialsForTraceDataDownload, GetCredentialsForTraceDataUpload } from '../spec';
import { getRequestHeaders, makeRequest } from '../utils';
import { ArtifactsClient } from './base';

export class DatabricksArtifactsClient implements ArtifactsClient {
  private host: string;
  private token?: string;

  constructor(options: { host: string; token?: string }) {
    this.host = options.host;
    this.token = options.token;
  }

  /**
   * Private wrapper for fetch to enable easier testing
   */
  private async httpFetch(url: string, options?: RequestInit): Promise<Response> {
    return await fetch(url, options);
  }

  /**
   * Upload trace data (spans) to the backend using artifact repository pattern.
   *
   * 1. Get credentials for upload
   * 2. Serialize trace data to JSON
   * 3. Upload to cloud storage using the credentials
   */
  async uploadTraceData(traceInfo: TraceInfo, traceData: TraceData): Promise<void> {
    try {
      const credentials = await this.getCredentialsForTraceDataUpload(traceInfo.traceId);
      const traceDataJson = JSONBig.stringify(traceData.toJson());
      await this.uploadToCloudStorage(credentials, traceDataJson);
    } catch (error) {
      console.warn(`Trace data upload failed for ${traceInfo.traceId}:`, error);
      throw error;
    }
  }

  /**
   * Download trace data (spans) from cloud storage
   * Uses artifact repository pattern with signed URLs
   */
  async downloadTraceData(traceInfo: TraceInfo): Promise<TraceData> {
    try {
      const credentials = await this.getCredentialsForTraceDataDownload(traceInfo.traceId);
      const traceDataJson = await this.downloadFromSignedUrl(credentials);
      return TraceData.fromJson(traceDataJson);
    } catch (error) {
      console.warn(`Failed to download trace data for ${traceInfo.traceId}:`, error);

      // Return empty trace data if download fails
      // This allows getting trace info even if data is missing
      return new TraceData([]);
    }
  }

  /**
   * Get credentials for uploading trace data
   * Endpoint: GET /api/2.0/mlflow/traces/{request_id}/credentials-for-data-upload
   */
  private async getCredentialsForTraceDataUpload(traceId: string): Promise<ArtifactCredentialInfo> {
    const url = GetCredentialsForTraceDataUpload.getEndpoint(this.host, traceId);
    const response = await makeRequest<GetCredentialsForTraceDataUpload.Response>(
      'GET',
      url,
      getRequestHeaders(this.token)
    );
    return response.credential_info;
  }

  /**
   * Get credentials for downloading trace data
   * Endpoint: GET /mlflow/traces/{trace_id}/credentials-for-data-download
   */
  private async getCredentialsForTraceDataDownload(
    traceId: string
  ): Promise<ArtifactCredentialInfo> {
    const url = GetCredentialsForTraceDataDownload.getEndpoint(this.host, traceId);
    const response = await makeRequest<GetCredentialsForTraceDataDownload.Response>(
      'GET',
      url,
      getRequestHeaders(this.token)
    );

    if (response.credential_info) {
      return response.credential_info;
    } else {
      throw new Error('Invalid response format: missing credential_info');
    }
  }

  /**
   * Upload data to cloud storage using the provided credentials
   */
  private async uploadToCloudStorage(
    credentials: ArtifactCredentialInfo,
    data: string
  ): Promise<void> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };

    // Add headers from credentials (if they exist)
    if (credentials.headers && Array.isArray(credentials.headers)) {
      credentials.headers.forEach((header) => {
        headers[header.name] = header.value;
      });
    }

    switch (credentials.type) {
      case 'AWS_PRESIGNED_URL':
      case 'GCP_SIGNED_URL':
        await this.uploadToSignedUrl(credentials.signed_uri, data, headers, credentials.type);
        break;
      // TODO: Implement Azure upload
      case 'AZURE_SAS_URI':
      case 'AZURE_ADLS_GEN2_SAS_URI':
        throw new Error(
          `Azure upload not yet implemented for credential type: ${credentials.type}`
        );
      default:
        throw new Error(`Unsupported credential type: ${credentials.type as string}`);
    }
  }

  /**
   * Upload data to cloud storage using signed URL (AWS S3 or GCP Storage)
   */
  private async uploadToSignedUrl(
    signedUrl: string,
    data: string,
    headers: Record<string, string>,
    credentialType: string
  ): Promise<void> {
    try {
      const response = await this.httpFetch(signedUrl, {
        method: 'PUT',
        headers,
        body: data
      });

      if (!response.ok) {
        throw new Error(
          `${credentialType} upload failed: ${response.status} ${response.statusText}`
        );
      }
    } catch (error) {
      throw new Error(`Failed to upload to ${credentialType}: ${(error as Error).message}`);
    }
  }

  /**
   * Download data from cloud storage using signed URL
   */
  private async downloadFromSignedUrl(
    credentials: ArtifactCredentialInfo
  ): Promise<SerializedTraceData> {
    const headers: Record<string, string> = {};

    // Add headers from credentials (if they exist)
    if (credentials.headers && Array.isArray(credentials.headers)) {
      credentials.headers.forEach((header) => {
        headers[header.name] = header.value;
      });
    }

    try {
      const response = await this.httpFetch(credentials.signed_uri, {
        method: 'GET',
        headers
      });

      if (!response.ok) {
        if (response.status === 404) {
          throw new Error(`Trace data not found (404)`);
        }
        throw new Error(`Download failed: ${response.status} ${response.statusText}`);
      }

      const textData = await response.text();
      try {
        return JSONBig.parse(textData) as SerializedTraceData;
      } catch (parseError) {
        throw new Error(`Trace data corrupted: invalid JSON - ${(parseError as Error).message}`);
      }
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error(`Failed to download trace data: ${String(error)}`);
    }
  }
}

/**
 * HTTP header for artifact upload/download
 */
export interface HttpHeader {
  name: string;
  value: string;
}

/**
 * Artifact credential information for upload/download
 */
export interface ArtifactCredentialInfo {
  /** ID of the MLflow Run containing the artifact */
  run_id?: string;

  /** Relative path to the artifact */
  path?: string;

  /** Signed URI credential for artifact access */
  signed_uri: string;

  /** HTTP headers for upload/download (optional, may not be present) */
  headers?: HttpHeader[];

  /** Type of signed credential URI */
  type: ArtifactCredentialType;
}

/**
 * Enum for artifact credential types.
 * This ensures type safety when handling different cloud storage providers.
 */
export type ArtifactCredentialType =
  | 'AWS_PRESIGNED_URL'
  | 'GCP_SIGNED_URL'
  | 'AZURE_SAS_URI'
  | 'AZURE_ADLS_GEN2_SAS_URI';

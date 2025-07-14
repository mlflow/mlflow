import { SerializedTraceData, TraceData } from '../../core/entities/trace_data';
import { TraceInfo } from '../../core/entities/trace_info';
import { JSONBig } from '../../core/utils/json';
import { GetCredentialsForTraceDataDownload, GetCredentialsForTraceDataUpload } from '../spec';
import { getRequestHeaders, makeRequest } from '../utils';
import { ArtifactsClient } from './base';

export class DatabricksArtifactsClient implements ArtifactsClient {
  private host: string;
  private databricksToken?: string;

  constructor(options: { host: string; databricksToken?: string }) {
    this.host = options.host;
    this.databricksToken = options.databricksToken;
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
      console.error(`Trace data upload failed for ${traceInfo.traceId}:`, error);
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
      console.error(`Failed to download trace data for ${traceInfo.traceId}:`, error);

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
      getRequestHeaders(this.databricksToken)
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
      getRequestHeaders(this.databricksToken)
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
      case 'AZURE_SAS_URI':
        await this.uploadToAzureBlob(credentials.signed_uri, data, headers);
        break;
      case 'AZURE_ADLS_GEN2_SAS_URI':
        await this.uploadToAzureAdlsGen2(credentials.signed_uri, data, headers);
        break;
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
      const response = await fetch(signedUrl, {
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
   * Upload data to Azure Blob Storage using SAS URI
   * Uses simple PUT for all uploads since traces rarely exceed 100MB
   * https://learn.microsoft.com/en-us/azure/storage/common/storage-sas-overview
   */
  private async uploadToAzureBlob(
    sasUri: string,
    data: string,
    headers: Record<string, string>
  ): Promise<void> {
    try {
      const response = await fetch(sasUri, {
        method: 'PUT',
        headers: {
          ...headers,
          'x-ms-blob-type': 'BlockBlob',
          'Content-Type': 'application/json'
        },
        body: data
      });

      if (!response.ok) {
        throw new Error(`Azure Blob upload failed: ${response.status} ${response.statusText}`);
      }
    } catch (error) {
      throw new Error(`Failed to upload to Azure Blob Storage: ${(error as Error).message}`);
    }
  }

  /**
   * Upload data to Azure Data Lake Storage Gen2 using SAS URI
   * https://learn.microsoft.com/en-us/rest/api/storageservices/data-lake-storage-gen2
   */
  private async uploadToAzureAdlsGen2(
    sasUri: string,
    data: string,
    headers: Record<string, string>
  ): Promise<void> {
    try {
      const dataBuffer = new TextEncoder().encode(data);

      // ADLS Gen2 uses a different API pattern - create file then append data
      // Create the file
      const createUrl = `${sasUri}&resource=file`;
      const createResponse = await fetch(createUrl, {
        method: 'PUT',
        headers: {
          ...headers,
          'Content-Length': '0'
        }
      });

      if (!createResponse.ok) {
        throw new Error(
          `Azure ADLS Gen2 file creation failed: ${createResponse.status} ${createResponse.statusText}`
        );
      }

      // Append data to the file
      const appendUrl = `${sasUri}&action=append&position=0`;
      const appendResponse = await fetch(appendUrl, {
        method: 'PATCH',
        headers: {
          ...headers,
          'Content-Type': 'application/octet-stream'
        },
        body: dataBuffer
      });

      if (!appendResponse.ok) {
        throw new Error(
          `Azure ADLS Gen2 data append failed: ${appendResponse.status} ${appendResponse.statusText}`
        );
      }

      // Flush the data to complete the upload
      const flushUrl = `${sasUri}&action=flush&position=${dataBuffer.length}`;
      const flushResponse = await fetch(flushUrl, {
        method: 'PATCH',
        headers: {
          ...headers,
          'Content-Length': '0'
        }
      });

      if (!flushResponse.ok) {
        throw new Error(
          `Azure ADLS Gen2 flush failed: ${flushResponse.status} ${flushResponse.statusText}`
        );
      }
    } catch (error) {
      throw new Error(`Failed to upload to Azure ADLS Gen2: ${(error as Error).message}`);
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
      const response = await fetch(credentials.signed_uri, {
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

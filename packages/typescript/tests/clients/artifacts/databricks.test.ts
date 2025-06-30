import { TraceInfo } from '../../../src/core/entities/trace_info';
import { TraceData } from '../../../src/core/entities/trace_data';
import { TraceLocationType } from '../../../src/core/entities/trace_location';
import { TraceState } from '../../../src/core/entities/trace_state';
import { makeRequest } from '../../../src/clients/utils';
import { DatabricksArtifactsClient } from '../../../src/clients/artifacts/databricks';

// Mock the makeRequest function
// eslint-disable-next-line @typescript-eslint/no-unsafe-return
jest.mock('../../../src/clients/utils', () => ({
  ...jest.requireActual('../../../src/clients/utils'),
  makeRequest: jest.fn()
}));

describe('DatabricksArtifactsClient', () => {
  let client: DatabricksArtifactsClient;
  let mockHttpFetch: jest.SpyInstance;
  const mockMakeRequest = makeRequest as jest.MockedFunction<typeof makeRequest>;
  const testHost = 'https://dbc-12345.cloud.databricks.com';
  const testToken = 'test-token';

  beforeEach(() => {
    client = new DatabricksArtifactsClient({ host: testHost, token: testToken });
    mockMakeRequest.mockClear();

    // Spy on the private httpFetch method
    mockHttpFetch = jest.spyOn(client as any, 'httpFetch');
    mockHttpFetch.mockClear();
  });

  afterEach(() => {
    mockHttpFetch.mockRestore();
  });

  describe('uploadTraceData', () => {
    it('should get upload credentials and upload to AWS S3 signed URL', async () => {
      const traceInfo = new TraceInfo({
        traceId: 'tr-databricks-123',
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: { experimentId: '0' }
        },
        state: TraceState.OK,
        requestTime: 1000
      });

      const traceData = new TraceData([]);

      // Mock credentials response
      mockMakeRequest.mockResolvedValueOnce({
        credential_info: {
          type: 'AWS_PRESIGNED_URL',
          signed_uri: 'https://s3.amazonaws.com/bucket/traces/tr-databricks-123?signature=xyz',
          headers: [{ name: 'x-amz-server-side-encryption', value: 'AES256' }]
        }
      });

      // Mock successful S3 upload
      mockHttpFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        statusText: 'OK'
      } as Response);

      await client.uploadTraceData(traceInfo, traceData);

      // Verify credentials request
      expect(mockMakeRequest).toHaveBeenCalledWith(
        'GET',
        'https://dbc-12345.cloud.databricks.com/api/2.0/mlflow/traces/tr-databricks-123/credentials-for-data-upload',
        {
          'Content-Type': 'application/json',
          Authorization: 'Bearer test-token'
        }
      );

      // Verify S3 upload
      expect(mockHttpFetch).toHaveBeenCalledWith(
        'https://s3.amazonaws.com/bucket/traces/tr-databricks-123?signature=xyz',
        {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
            'x-amz-server-side-encryption': 'AES256'
          },
          body: expect.stringContaining('"spans"')
        }
      );
    });

    it('should get upload credentials and upload to GCP signed URL', async () => {
      const traceInfo = new TraceInfo({
        traceId: 'tr-gcp-456',
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: { experimentId: '1' }
        },
        state: TraceState.OK,
        requestTime: 2000
      });

      const traceData = new TraceData([]);

      // Mock GCP credentials response
      mockMakeRequest.mockResolvedValueOnce({
        credential_info: {
          type: 'GCP_SIGNED_URL',
          signed_uri: 'https://storage.googleapis.com/bucket/traces/tr-gcp-456?signature=abc'
        }
      });

      // Mock successful GCP upload
      mockHttpFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        statusText: 'OK'
      } as Response);

      await client.uploadTraceData(traceInfo, traceData);

      // Verify GCP upload
      expect(mockHttpFetch).toHaveBeenCalledWith(
        'https://storage.googleapis.com/bucket/traces/tr-gcp-456?signature=abc',
        {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json'
          },
          body: expect.stringContaining('"spans"')
        }
      );
    });

    it('should throw error for unsupported Azure credential types', async () => {
      const traceInfo = new TraceInfo({
        traceId: 'tr-azure-789',
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: { experimentId: '2' }
        },
        state: TraceState.OK,
        requestTime: 3000
      });

      const traceData = new TraceData([]);

      // Mock Azure credentials response
      mockMakeRequest.mockResolvedValueOnce({
        credential_info: {
          type: 'AZURE_SAS_URI',
          signed_uri: 'https://storage.azure.com/traces/tr-azure-789?sas=token'
        }
      });

      await expect(client.uploadTraceData(traceInfo, traceData)).rejects.toThrow(
        'Azure upload not yet implemented for credential type: AZURE_SAS_URI'
      );

      expect(mockHttpFetch).not.toHaveBeenCalled();
    });

    it('should handle upload failures', async () => {
      const traceInfo = new TraceInfo({
        traceId: 'tr-fail-upload',
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: { experimentId: '0' }
        },
        state: TraceState.ERROR,
        requestTime: 4000
      });

      const traceData = new TraceData([]);

      // Mock credentials response
      mockMakeRequest.mockResolvedValueOnce({
        credential_info: {
          type: 'AWS_PRESIGNED_URL',
          signed_uri: 'https://s3.amazonaws.com/bucket/traces/tr-fail-upload?signature=xyz'
        }
      });

      // Mock failed S3 upload
      mockHttpFetch.mockResolvedValueOnce({
        ok: false,
        status: 403,
        statusText: 'Forbidden'
      } as Response);

      await expect(client.uploadTraceData(traceInfo, traceData)).rejects.toThrow(
        'AWS_PRESIGNED_URL upload failed: 403 Forbidden'
      );
    });
  });

  describe('downloadTraceData', () => {
    it('should get download credentials and download from AWS S3 signed URL', async () => {
      const traceInfo = new TraceInfo({
        traceId: 'tr-download-123',
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: { experimentId: '5' }
        },
        state: TraceState.OK,
        requestTime: 5000
      });

      // Mock credentials response
      mockMakeRequest.mockResolvedValueOnce({
        credential_info: {
          type: 'AWS_PRESIGNED_URL',
          signed_uri: 'https://s3.amazonaws.com/bucket/traces/tr-download-123?signature=xyz',
          headers: [{ name: 'x-amz-server-side-encryption', value: 'AES256' }]
        }
      });

      const mockTraceData = {
        spans: [
          {
            span_id: 'c3Bhbi1kb3dubG9hZGVk', // base64 encoded
            trace_id: 'dHItZG93bmxvYWQtMTIz', // base64 encoded
            name: 'downloaded-span',
            start_time: '5000000000',
            end_time: '5100000000',
            status: { code: 'OK' },
            attributes: {}
          }
        ]
      };

      // Mock successful S3 download
      mockHttpFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        statusText: 'OK',
        text: () => Promise.resolve(JSON.stringify(mockTraceData))
      } as Response);

      const result = await client.downloadTraceData(traceInfo);

      // Verify credentials request
      expect(mockMakeRequest).toHaveBeenCalledWith(
        'GET',
        'https://dbc-12345.cloud.databricks.com/api/2.0/mlflow/traces/tr-download-123/credentials-for-data-download',
        {
          'Content-Type': 'application/json',
          Authorization: 'Bearer test-token'
        }
      );

      // Verify S3 download
      expect(mockHttpFetch).toHaveBeenCalledWith(
        'https://s3.amazonaws.com/bucket/traces/tr-download-123?signature=xyz',
        {
          method: 'GET',
          headers: {
            'x-amz-server-side-encryption': 'AES256'
          }
        }
      );

      expect(result).toBeInstanceOf(TraceData);
      expect(result.spans).toHaveLength(1);
      expect(result.spans[0].name).toBe('downloaded-span');
    });
  });

  describe('Error Handling', () => {
    it('should log and re-throw upload errors', async () => {
      const traceInfo = new TraceInfo({
        traceId: 'tr-error-log',
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: { experimentId: '0' }
        },
        state: TraceState.ERROR,
        requestTime: 9000
      });

      const traceData = new TraceData([]);

      // Mock network error
      mockMakeRequest.mockRejectedValueOnce(new Error('Network error'));

      // Spy on console.error
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();

      await expect(client.uploadTraceData(traceInfo, traceData)).rejects.toThrow('Network error');

      expect(consoleSpy).toHaveBeenCalledWith(
        'Trace data upload failed for tr-error-log:',
        expect.any(Error)
      );

      consoleSpy.mockRestore();
    });

    it('should log download errors but return empty data', async () => {
      const traceInfo = new TraceInfo({
        traceId: 'tr-download-error',
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: { experimentId: '0' }
        },
        state: TraceState.OK,
        requestTime: 10000
      });

      // Mock network error
      mockMakeRequest.mockRejectedValueOnce(new Error('Connection timeout'));

      // Spy on console.warn
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();

      const result = await client.downloadTraceData(traceInfo);

      expect(consoleSpy).toHaveBeenCalledWith(
        'Failed to download trace data for tr-download-error:',
        expect.any(Error)
      );

      expect(result).toBeInstanceOf(TraceData);
      expect(result.spans).toHaveLength(0);

      consoleSpy.mockRestore();
    });
  });

  describe('Credential Types', () => {
    const credentialTypes = [
      { type: 'AWS_PRESIGNED_URL', url: 'https://s3.amazonaws.com/bucket/file' },
      { type: 'GCP_SIGNED_URL', url: 'https://storage.googleapis.com/bucket/file' }
    ];

    credentialTypes.forEach(({ type, url }) => {
      it(`should handle ${type} credential type`, async () => {
        const traceInfo = new TraceInfo({
          traceId: `tr-${type}`,
          traceLocation: {
            type: TraceLocationType.MLFLOW_EXPERIMENT,
            mlflowExperiment: { experimentId: '0' }
          },
          state: TraceState.OK,
          requestTime: 11000
        });

        const traceData = new TraceData([]);

        mockMakeRequest.mockResolvedValueOnce({
          credential_info: {
            type,
            signed_uri: `${url}?signature=test`
          }
        });

        mockHttpFetch.mockResolvedValueOnce({
          ok: true,
          status: 200,
          statusText: 'OK'
        } as Response);

        await client.uploadTraceData(traceInfo, traceData);

        expect(mockHttpFetch).toHaveBeenCalledWith(
          `${url}?signature=test`,
          expect.objectContaining({
            method: 'PUT'
          })
        );
      });
    });

    const azureTypes = ['AZURE_SAS_URI', 'AZURE_ADLS_GEN2_SAS_URI'];

    azureTypes.forEach((type) => {
      it(`should throw not implemented error for ${type}`, async () => {
        const traceInfo = new TraceInfo({
          traceId: `tr-${type}`,
          traceLocation: {
            type: TraceLocationType.MLFLOW_EXPERIMENT,
            mlflowExperiment: { experimentId: '0' }
          },
          state: TraceState.OK,
          requestTime: 12000
        });

        const traceData = new TraceData([]);

        mockMakeRequest.mockResolvedValueOnce({
          credential_info: {
            type: type as any,
            signed_uri: 'https://storage.azure.com/file?sas=token'
          }
        });

        await expect(client.uploadTraceData(traceInfo, traceData)).rejects.toThrow(
          `Azure upload not yet implemented for credential type: ${type}`
        );
      });
    });
  });
});

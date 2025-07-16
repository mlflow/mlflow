import { TraceInfo } from '../../../src/core/entities/trace_info';
import { TraceData } from '../../../src/core/entities/trace_data';
import { TraceLocationType } from '../../../src/core/entities/trace_location';
import { TraceState } from '../../../src/core/entities/trace_state';
import { DatabricksArtifactsClient } from '../../../src/clients/artifacts/databricks';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';

describe('DatabricksArtifactsClient', () => {
  let client: DatabricksArtifactsClient;
  const testHost = 'https://dbc-12345.cloud.databricks.com';
  const testToken = 'test-token';

  let server: ReturnType<typeof setupServer>;

  beforeAll(() => {
    server = setupServer();
    server.listen();
  });

  afterAll(() => {
    server.close();
  });

  beforeEach(() => {
    client = new DatabricksArtifactsClient({ host: testHost, databricksToken: testToken });
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
      server.use(
        http.get(
          `${testHost}/api/2.0/mlflow/traces/tr-databricks-123/credentials-for-data-upload`,
          () => {
            return HttpResponse.json({
              credential_info: {
                type: 'AWS_PRESIGNED_URL',
                signed_uri:
                  'https://s3.amazonaws.com/bucket/traces/tr-databricks-123?signature=xyz',
                headers: [{ name: 'x-amz-server-side-encryption', value: 'AES256' }]
              }
            });
          }
        )
      );

      // Mock successful S3 upload
      server.use(
        http.put('https://s3.amazonaws.com/bucket/traces/tr-databricks-123', () => {
          return HttpResponse.json({}, { status: 200 });
        })
      );

      await expect(client.uploadTraceData(traceInfo, traceData)).resolves.toBeUndefined();
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
      server.use(
        http.get(`${testHost}/api/2.0/mlflow/traces/tr-gcp-456/credentials-for-data-upload`, () => {
          return HttpResponse.json({
            credential_info: {
              type: 'GCP_SIGNED_URL',
              signed_uri: 'https://storage.googleapis.com/bucket/traces/tr-gcp-456?signature=abc'
            }
          });
        })
      );

      // Mock successful GCP upload
      server.use(
        http.put('https://storage.googleapis.com/bucket/traces/tr-gcp-456', () => {
          return HttpResponse.json({}, { status: 200 });
        })
      );

      await expect(client.uploadTraceData(traceInfo, traceData)).resolves.toBeUndefined();
    });

    it('should get upload credentials and upload to Azure Blob Storage', async () => {
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
      server.use(
        http.get(
          `${testHost}/api/2.0/mlflow/traces/tr-azure-789/credentials-for-data-upload`,
          () => {
            return HttpResponse.json({
              credential_info: {
                type: 'AZURE_SAS_URI',
                signed_uri: 'https://storage.azure.com/traces/tr-azure-789?sas=token'
              }
            });
          }
        )
      );

      // Mock Azure Blob Storage upload
      server.use(
        http.put('https://storage.azure.com/traces/tr-azure-789', () => {
          return new HttpResponse(null, { status: 201 });
        })
      );

      await expect(client.uploadTraceData(traceInfo, traceData)).resolves.toBeUndefined();
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
      server.use(
        http.get(
          `${testHost}/api/2.0/mlflow/traces/tr-fail-upload/credentials-for-data-upload`,
          () => {
            return HttpResponse.json({
              credential_info: {
                type: 'AWS_PRESIGNED_URL',
                signed_uri: 'https://s3.amazonaws.com/bucket/traces/tr-fail-upload?signature=xyz'
              }
            });
          }
        )
      );

      // Mock failed S3 upload
      server.use(
        http.put('https://s3.amazonaws.com/bucket/traces/tr-fail-upload', () => {
          return HttpResponse.json({ error: 'Forbidden' }, { status: 403 });
        })
      );

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
      server.use(
        http.get(
          `${testHost}/api/2.0/mlflow/traces/tr-download-123/credentials-for-data-download`,
          () => {
            return HttpResponse.json({
              credential_info: {
                type: 'AWS_PRESIGNED_URL',
                signed_uri: 'https://s3.amazonaws.com/bucket/traces/tr-download-123?signature=xyz',
                headers: [{ name: 'x-amz-server-side-encryption', value: 'AES256' }]
              }
            });
          }
        )
      );

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
      server.use(
        http.get('https://s3.amazonaws.com/bucket/traces/tr-download-123', () => {
          return HttpResponse.text(JSON.stringify(mockTraceData));
        })
      );

      const result = await client.downloadTraceData(traceInfo);

      // Verify the result structure

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
      server.use(
        http.get(
          `${testHost}/api/2.0/mlflow/traces/tr-error-log/credentials-for-data-upload`,
          () => {
            return HttpResponse.json({ error: 'Network error' }, { status: 500 });
          }
        )
      );

      // Spy on console.error
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

      await expect(client.uploadTraceData(traceInfo, traceData)).rejects.toThrow();

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
      server.use(
        http.get(
          `${testHost}/api/2.0/mlflow/traces/tr-download-error/credentials-for-data-download`,
          () => {
            return HttpResponse.json({ error: 'Connection timeout' }, { status: 500 });
          }
        )
      );

      // Spy on console.error
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

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

        server.use(
          http.get(
            `${testHost}/api/2.0/mlflow/traces/tr-${type}/credentials-for-data-upload`,
            () => {
              return HttpResponse.json({
                credential_info: {
                  type,
                  signed_uri: `${url}?signature=test`
                }
              });
            }
          )
        );

        server.use(
          http.put(`${url}`, () => {
            return HttpResponse.json({}, { status: 200 });
          })
        );

        await expect(client.uploadTraceData(traceInfo, traceData)).resolves.toBeUndefined();
      });
    });

    it('should successfully upload to Azure Blob Storage with AZURE_SAS_URI', async () => {
      const traceInfo = new TraceInfo({
        traceId: 'tr-AZURE_SAS_URI',
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: { experimentId: '0' }
        },
        state: TraceState.OK,
        requestTime: 12000
      });

      const traceData = new TraceData([]);

      server.use(
        http.get(
          `${testHost}/api/2.0/mlflow/traces/tr-AZURE_SAS_URI/credentials-for-data-upload`,
          () => {
            return HttpResponse.json({
              credential_info: {
                type: 'AZURE_SAS_URI' as any,
                signed_uri: 'https://storage.azure.com/file?sas=token'
              }
            });
          }
        )
      );

      // Mock Azure Blob Storage upload
      server.use(
        http.put('https://storage.azure.com/file', () => {
          return new HttpResponse(null, { status: 201 });
        })
      );

      await expect(client.uploadTraceData(traceInfo, traceData)).resolves.toBeUndefined();
    });

    it('should successfully upload to Azure ADLS Gen2 with AZURE_ADLS_GEN2_SAS_URI', async () => {
      const traceInfo = new TraceInfo({
        traceId: 'tr-AZURE_ADLS_GEN2_SAS_URI',
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: { experimentId: '0' }
        },
        state: TraceState.OK,
        requestTime: 12000
      });

      const traceData = new TraceData([]);

      server.use(
        http.get(
          `${testHost}/api/2.0/mlflow/traces/tr-AZURE_ADLS_GEN2_SAS_URI/credentials-for-data-upload`,
          () => {
            return HttpResponse.json({
              credential_info: {
                type: 'AZURE_ADLS_GEN2_SAS_URI' as any,
                signed_uri: 'https://storage.azure.com/file?sas=token'
              }
            });
          }
        )
      );

      // Mock Azure ADLS Gen2 operations (create, append, flush)
      server.use(
        // Create file
        http.put('https://storage.azure.com/file', ({ request }) => {
          const url = new URL(request.url);
          if (url.searchParams.get('resource') === 'file') {
            return new HttpResponse(null, { status: 201 });
          }
          return new HttpResponse(null, { status: 404 });
        }),
        // Append data
        http.patch('https://storage.azure.com/file', ({ request }) => {
          const url = new URL(request.url);
          if (url.searchParams.get('action') === 'append') {
            return new HttpResponse(null, { status: 202 });
          }
          if (url.searchParams.get('action') === 'flush') {
            return new HttpResponse(null, { status: 200 });
          }
          return new HttpResponse(null, { status: 404 });
        })
      );

      await expect(client.uploadTraceData(traceInfo, traceData)).resolves.toBeUndefined();
    });
  });
});

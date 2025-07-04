import { MlflowArtifactsClient } from '../../../src/clients/artifacts/mlflow';
import { TraceInfo } from '../../../src/core/entities/trace_info';
import { TraceData } from '../../../src/core/entities/trace_data';
import { TraceLocationType } from '../../../src/core/entities/trace_location';
import { TraceState } from '../../../src/core/entities/trace_state';
import { makeRequest } from '../../../src/clients/utils';

// Mock the makeRequest function
// eslint-disable-next-line @typescript-eslint/no-unsafe-return
jest.mock('../../../src/clients/utils', () => ({
  ...jest.requireActual('../../../src/clients/utils'),
  makeRequest: jest.fn()
}));

describe('MlflowArtifactsClient', () => {
  let client: MlflowArtifactsClient;
  const mockMakeRequest = makeRequest as jest.MockedFunction<typeof makeRequest>;

  beforeEach(() => {
    client = new MlflowArtifactsClient({ host: 'http://localhost:5000' });
    mockMakeRequest.mockClear();
  });

  describe('uploadTraceData', () => {
    it('should make PUT request to correct artifact URL', async () => {
      const traceInfo = new TraceInfo({
        traceId: 'tr-abc123',
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: { experimentId: '0' }
        },
        state: TraceState.OK,
        requestTime: 1000,
        tags: {
          'mlflow.artifactLocation': 'mlflow-artifacts:/0/traces/tr-abc123/artifacts'
        }
      });
      const traceData = new TraceData([]);

      await client.uploadTraceData(traceInfo, traceData);

      expect(mockMakeRequest).toHaveBeenCalledTimes(1);
      expect(mockMakeRequest).toHaveBeenCalledWith(
        'PUT',
        'http://localhost:5000/api/2.0/mlflow-artifacts/artifacts/0/traces/tr-abc123/artifacts/traces.json',
        {
          'Content-Type': 'application/json'
        },
        expect.any(Object) // TraceData JSON
      );
    });

    it('should throw error when artifact location is missing', async () => {
      const traceInfo = new TraceInfo({
        traceId: 'tr-no-artifact',
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: { experimentId: '0' }
        },
        state: TraceState.OK,
        requestTime: 1000,
        tags: {} // No artifact location
      });
      const traceData = new TraceData([]);

      await expect(client.uploadTraceData(traceInfo, traceData)).rejects.toThrow(
        'Artifact location not found in trace tags'
      );

      expect(mockMakeRequest).not.toHaveBeenCalled();
    });
  });

  describe('downloadTraceData', () => {
    it('should make GET request to correct artifact URL', async () => {
      const traceInfo = new TraceInfo({
        traceId: 'tr-download',
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: { experimentId: '5' }
        },
        state: TraceState.OK,
        requestTime: 4000,
        tags: {
          'mlflow.artifactLocation': 'mlflow-artifacts:/5/traces/tr-download/artifacts'
        }
      });

      const mockResponse = {
        spans: [
          {
            span_id: 'c3Bhbi1kb3dubG9hZA==',
            trace_id: 'dHItZG93bmxvYWQ=',
            name: 'downloaded-span',
            start_time: '4000000000',
            end_time: '4100000000',
            status: { code: 'OK' },
            attributes: {}
          }
        ]
      };

      mockMakeRequest.mockResolvedValue(mockResponse);

      const result = await client.downloadTraceData(traceInfo);

      expect(mockMakeRequest).toHaveBeenCalledTimes(1);
      expect(mockMakeRequest).toHaveBeenCalledWith(
        'GET',
        'http://localhost:5000/api/2.0/mlflow-artifacts/artifacts/5/traces/tr-download/artifacts/traces.json',
        {
          'Content-Type': 'application/json'
        }
      );

      expect(result).toBeInstanceOf(TraceData);
      expect(result.spans).toHaveLength(1);
      expect(result.spans[0].name).toBe('downloaded-span');
    });

    it('should handle complex artifact paths in URL construction', async () => {
      const traceInfo = new TraceInfo({
        traceId: 'tr-complex-path',
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: { experimentId: '42' }
        },
        state: TraceState.OK,
        requestTime: 5000,
        tags: {
          'mlflow.artifactLocation':
            'mlflow-artifacts:/42/some/nested/path/traces/tr-complex-path/artifacts'
        }
      });

      mockMakeRequest.mockResolvedValue({ spans: [] });

      await client.downloadTraceData(traceInfo);

      expect(mockMakeRequest).toHaveBeenCalledWith(
        'GET',
        'http://localhost:5000/api/2.0/mlflow-artifacts/artifacts/42/some/nested/path/traces/tr-complex-path/artifacts/traces.json',
        expect.any(Object)
      );
    });
  });
});

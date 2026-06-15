import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { MlflowArtifactsClient } from '../../../src/clients/artifacts/mlflow';
import { TraceInfo } from '../../../src/core/entities/trace_info';
import { TraceData } from '../../../src/core/entities/trace_data';
import { TraceLocationType } from '../../../src/core/entities/trace_location';
import { TraceState } from '../../../src/core/entities/trace_state';
import { AuthProvider } from '../../../src/auth';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';

describe('MlflowArtifactsClient', () => {
  let client: MlflowArtifactsClient;
  let server: ReturnType<typeof setupServer>;
  const testHost = 'http://localhost:5000';

  // Create a mock AuthProvider for testing
  const mockAuthProvider: AuthProvider = {
    getHost: () => testHost,
    // eslint-disable-next-line require-await, @typescript-eslint/require-await
    getHeadersProvider: () => async () => ({
      'Content-Type': 'application/json',
    }),
    getDatabricksToken: () => undefined,
  };

  beforeAll(() => {
    server = setupServer();
    server.listen();
  });

  afterAll(() => {
    server.close();
  });

  beforeEach(() => {
    client = new MlflowArtifactsClient({ host: testHost, authProvider: mockAuthProvider });
  });

  describe('uploadTraceData', () => {
    it('should make PUT request to correct artifact URL', async () => {
      const traceInfo = new TraceInfo({
        traceId: 'tr-abc123',
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: { experimentId: '0' },
        },
        state: TraceState.OK,
        requestTime: 1000,
        tags: {
          'mlflow.artifactLocation': 'mlflow-artifacts:/0/traces/tr-abc123/artifacts',
        },
      });
      const traceData = new TraceData([]);

      // Mock the artifacts upload endpoint
      server.use(
        http.put(
          'http://localhost:5000/api/2.0/mlflow-artifacts/artifacts/0/traces/tr-abc123/artifacts/traces.json',
          () => {
            return HttpResponse.json({}, { status: 200 });
          },
        ),
      );

      await client.uploadTraceData(traceInfo, traceData);

      // Test passes if no errors thrown
    });

    it('should throw error when artifact location is missing', async () => {
      const traceInfo = new TraceInfo({
        traceId: 'tr-no-artifact',
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: { experimentId: '0' },
        },
        state: TraceState.OK,
        requestTime: 1000,
        tags: {}, // No artifact location
      });
      const traceData = new TraceData([]);

      await expect(client.uploadTraceData(traceInfo, traceData)).rejects.toThrow(
        'Artifact location not found in trace tags',
      );

      // Test passes if error is thrown as expected
    });
  });

  describe('uploadTraceData with local artifact roots', () => {
    let tmpRoot: string;

    const makeTraceInfo = (artifactLocation: string) =>
      new TraceInfo({
        traceId: 'tr-local',
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: { experimentId: '0' },
        },
        state: TraceState.OK,
        requestTime: 1000,
        tags: {
          'mlflow.artifactLocation': artifactLocation,
        },
      });

    beforeEach(async () => {
      tmpRoot = await mkdtemp(join(tmpdir(), 'mlflow-wal-test-'));
    });

    afterEach(async () => {
      await rm(tmpRoot, { recursive: true, force: true });
    });

    it('writes traces.json to a bare local path, creating the directory', async () => {
      const artifactDir = join(tmpRoot, '0', 'traces', 'tr-local', 'artifacts');
      const traceInfo = makeTraceInfo(artifactDir);
      const traceData = new TraceData([]);

      await client.uploadTraceData(traceInfo, traceData);

      const written = await readFile(join(artifactDir, 'traces.json'), 'utf8');
      expect(JSON.parse(written)).toEqual({ spans: [] });
    });

    it('writes traces.json for a file:// artifact location', async () => {
      const artifactDir = join(tmpRoot, '0', 'traces', 'tr-local', 'artifacts');
      const fileUri = pathToFileURL(artifactDir).href;
      const traceInfo = makeTraceInfo(fileUri);
      const traceData = new TraceData([]);

      await client.uploadTraceData(traceInfo, traceData);

      const written = await readFile(join(artifactDir, 'traces.json'), 'utf8');
      expect(JSON.parse(written)).toEqual({ spans: [] });
    });

    it('rejects unsupported remote schemes via resolveArtifactUri', async () => {
      const traceInfo = makeTraceInfo('s3://bucket/0/traces/tr-local/artifacts');
      const traceData = new TraceData([]);

      await expect(client.uploadTraceData(traceInfo, traceData)).rejects.toThrow(
        'Expected mlflow-artifacts:// URI, got s3:',
      );
    });
  });

  describe('downloadTraceData', () => {
    it('should make GET request to correct artifact URL', async () => {
      const traceInfo = new TraceInfo({
        traceId: 'tr-download',
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: { experimentId: '5' },
        },
        state: TraceState.OK,
        requestTime: 4000,
        tags: {
          'mlflow.artifactLocation': 'mlflow-artifacts:/5/traces/tr-download/artifacts',
        },
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
            attributes: {},
          },
        ],
      };

      // Mock the artifacts download endpoint
      server.use(
        http.get(
          'http://localhost:5000/api/2.0/mlflow-artifacts/artifacts/5/traces/tr-download/artifacts/traces.json',
          () => {
            return HttpResponse.json(mockResponse);
          },
        ),
      );

      const result = await client.downloadTraceData(traceInfo);

      expect(result).toBeInstanceOf(TraceData);
      expect(result.spans).toHaveLength(1);
      expect(result.spans[0].name).toBe('downloaded-span');
    });

    it('should handle complex artifact paths in URL construction', async () => {
      const traceInfo = new TraceInfo({
        traceId: 'tr-complex-path',
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: { experimentId: '42' },
        },
        state: TraceState.OK,
        requestTime: 5000,
        tags: {
          'mlflow.artifactLocation':
            'mlflow-artifacts:/42/some/nested/path/traces/tr-complex-path/artifacts',
        },
      });

      // Mock the complex path artifacts download endpoint
      server.use(
        http.get(
          'http://localhost:5000/api/2.0/mlflow-artifacts/artifacts/42/some/nested/path/traces/tr-complex-path/artifacts/traces.json',
          () => {
            return HttpResponse.json({ spans: [] });
          },
        ),
      );

      await client.downloadTraceData(traceInfo);

      // Test passes if no errors thrown
    });
  });
});

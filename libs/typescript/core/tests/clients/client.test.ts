import { randomUUID } from 'crypto';
import * as mlflow from '../../src';
import { MlflowClient, type SearchTracesOptions } from '../../src/clients/client';
import type { ArtifactsClient } from '../../src/clients/artifacts';
import { Trace } from '../../src/core/entities/trace';
import { TraceData } from '../../src/core/entities/trace_data';
import { TraceInfo } from '../../src/core/entities/trace_info';
import { TraceLocationType } from '../../src/core/entities/trace_location';
import { TraceState } from '../../src/core/entities/trace_state';
import { createAuthProvider } from '../../src/auth';
import { TEST_TRACKING_URI } from '../helper';

describe('MlflowClient', () => {
  let client: MlflowClient;
  let experimentId: string;

  beforeEach(async () => {
    const authProvider = createAuthProvider({ trackingUri: TEST_TRACKING_URI });
    client = new MlflowClient({ trackingUri: TEST_TRACKING_URI, authProvider });

    // Create a new experiment for each test
    const experimentName = `test-experiment-${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
    experimentId = await client.createExperiment(experimentName);
  });

  afterEach(async () => {
    // Clean up: delete the experiment
    try {
      await client.deleteExperiment(experimentId);
    } catch (error) {
      console.warn('Failed to delete experiment:', error);
    }
  });

  describe('createTrace', () => {
    it('should create a trace', async () => {
      const traceId = randomUUID();
      const traceInfo = new TraceInfo({
        traceId: traceId,
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: {
            experimentId: experimentId,
          },
        },
        state: TraceState.OK,
        requestTime: 1000,
        executionDuration: 500,
        requestPreview: '{"input":"test"}',
        responsePreview: '{"output":"result"}',
        clientRequestId: 'client-request-id',
        traceMetadata: { 'meta-key': 'meta-value' },
        tags: { 'tag-key': 'tag-value' },
        assessments: [],
      });

      const createdTraceInfo = await client.createTrace(traceInfo);

      expect(createdTraceInfo).toBeInstanceOf(TraceInfo);
      expect(createdTraceInfo.traceId).toBe(traceId);
      expect(createdTraceInfo.state).toBe(TraceState.OK);
      expect(createdTraceInfo.requestTime).toBe(1000);
      expect(createdTraceInfo.executionDuration).toBe(500);
      expect(createdTraceInfo.traceLocation.type).toBe(TraceLocationType.MLFLOW_EXPERIMENT);
      expect(createdTraceInfo.traceLocation.mlflowExperiment?.experimentId).toBe(experimentId);
      expect(createdTraceInfo.requestPreview).toBe('{"input":"test"}');
      expect(createdTraceInfo.responsePreview).toBe('{"output":"result"}');
      expect(createdTraceInfo.clientRequestId).toBe('client-request-id');
      expect(createdTraceInfo.traceMetadata).toMatchObject({ 'meta-key': 'meta-value' });
      expect(createdTraceInfo.tags).toEqual({
        'tag-key': 'tag-value',
        'mlflow.artifactLocation': expect.any(String),
      });
      expect(createdTraceInfo.assessments).toEqual([]);
    });

    it('should create a trace with error state', async () => {
      const traceId = randomUUID();
      const traceInfo = new TraceInfo({
        traceId: traceId,
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: {
            experimentId: experimentId,
          },
        },
        state: TraceState.ERROR,
        requestTime: 1000,
      });

      const createdTraceInfo = await client.createTrace(traceInfo);

      expect(createdTraceInfo).toBeInstanceOf(TraceInfo);
      expect(createdTraceInfo.traceId).toBe(traceId);
      expect(createdTraceInfo.state).toBe(TraceState.ERROR);
      expect(createdTraceInfo.requestTime).toBe(1000);
      expect(createdTraceInfo.traceLocation.type).toBe(TraceLocationType.MLFLOW_EXPERIMENT);
      expect(createdTraceInfo.traceLocation.mlflowExperiment?.experimentId).toBe(experimentId);
    });
  });

  describe('getTraceInfo', () => {
    it('should retrieve trace info for an existing trace', async () => {
      const traceId = randomUUID();
      const traceInfo = new TraceInfo({
        traceId: traceId,
        traceLocation: {
          type: TraceLocationType.MLFLOW_EXPERIMENT,
          mlflowExperiment: {
            experimentId: experimentId,
          },
        },
        state: TraceState.OK,
        requestTime: 1000,
      });
      await client.createTrace(traceInfo);

      // Now retrieve it
      const retrievedTraceInfo = await client.getTraceInfo(traceId);

      expect(retrievedTraceInfo).toBeInstanceOf(TraceInfo);
      expect(retrievedTraceInfo.traceId).toBe(traceId);
      expect(retrievedTraceInfo.state).toBe(TraceState.OK);
      expect(retrievedTraceInfo.requestTime).toBe(1000);
    });
  });

  describe('searchTraces', () => {
    const experimentLocation = () => ({
      type: TraceLocationType.MLFLOW_EXPERIMENT,
      mlflowExperiment: {
        experimentId: experimentId,
      },
    });

    it('should search traces by location and page through results', async () => {
      const firstTraceId = randomUUID();
      const secondTraceId = randomUUID();
      const testFilterTag = randomUUID();
      await client.createTrace(
        new TraceInfo({
          traceId: firstTraceId,
          traceLocation: experimentLocation(),
          state: TraceState.OK,
          requestTime: 1000,
          tags: { searchTracesPagination: testFilterTag },
        }),
      );
      await client.createTrace(
        new TraceInfo({
          traceId: secondTraceId,
          traceLocation: experimentLocation(),
          state: TraceState.OK,
          requestTime: 2000,
          tags: { searchTracesPagination: testFilterTag },
        }),
      );

      const firstPage = await client.searchTraces({
        locations: [experimentLocation()],
        filter: `tags.searchTracesPagination = '${testFilterTag}'`,
        maxResults: 1,
        orderBy: ['timestamp_ms DESC'],
        includeSpans: false,
      });

      expect(firstPage.traces).toHaveLength(1);
      expect(firstPage.traces[0]).toBeInstanceOf(Trace);
      expect(firstPage.traces[0].info).toBeInstanceOf(TraceInfo);
      expect([firstTraceId, secondTraceId]).toContain(firstPage.traces[0].info.traceId);
      expect(firstPage.nextPageToken).toBeDefined();

      const secondPage = await client.searchTraces({
        locations: [experimentLocation()],
        filter: `tags.searchTracesPagination = '${testFilterTag}'`,
        maxResults: 1,
        orderBy: ['timestamp_ms DESC'],
        pageToken: firstPage.nextPageToken,
        includeSpans: false,
      });

      const returnedTraceIds = [
        firstPage.traces[0].info.traceId,
        ...secondPage.traces.map((trace) => trace.info.traceId),
      ];
      expect(returnedTraceIds).toEqual(expect.arrayContaining([firstTraceId, secondTraceId]));
    });

    it('should search traces with a filter', async () => {
      const matchingTraceId = randomUUID();
      const otherTraceId = randomUUID();
      await client.createTrace(
        new TraceInfo({
          traceId: matchingTraceId,
          traceLocation: experimentLocation(),
          state: TraceState.OK,
          requestTime: 1000,
          tags: { searchTracesFilter: 'match' },
        }),
      );
      await client.createTrace(
        new TraceInfo({
          traceId: otherTraceId,
          traceLocation: experimentLocation(),
          state: TraceState.OK,
          requestTime: 2000,
          tags: { searchTracesFilter: 'skip' },
        }),
      );

      const result = await client.searchTraces({
        locations: [experimentLocation()],
        filter: "tags.searchTracesFilter = 'match'",
        maxResults: 10,
        includeSpans: false,
      });

      expect(result.traces.map((trace) => trace.info.traceId)).toContain(matchingTraceId);
      expect(result.traces.map((trace) => trace.info.traceId)).not.toContain(otherTraceId);
    });

    it('should fetch span data for each trace by default', async () => {
      mlflow.init({ trackingUri: TEST_TRACKING_URI, experimentId });
      const span = mlflow.startSpan({ name: 'search-traces-span' });
      span.setInputs({ question: 'What is MLflow?' });
      span.end();
      await mlflow.flushTraces();

      const result = await client.searchTraces({
        locations: [experimentLocation()],
      });

      expect(result.traces).toHaveLength(1);
      const trace = result.traces[0];
      expect(trace).toBeInstanceOf(Trace);
      expect(trace.info.traceId).toBe(span.traceId);
      expect(trace.data.spans).toHaveLength(1);
      expect(trace.data.spans[0].name).toBe('search-traces-span');
    });

    it('should return metadata-only traces with empty span data when includeSpans is false', async () => {
      const traceId = randomUUID();
      await client.createTrace(
        new TraceInfo({
          traceId: traceId,
          traceLocation: experimentLocation(),
          state: TraceState.OK,
          requestTime: 1000,
        }),
      );

      const result = await client.searchTraces({
        locations: [experimentLocation()],
        includeSpans: false,
      });

      expect(result.traces).toHaveLength(1);
      expect(result.traces[0].info.traceId).toBe(traceId);
      expect(result.traces[0].data.spans).toEqual([]);
    });

    it('should skip traces whose span data cannot be downloaded', async () => {
      // createTrace registers trace metadata without uploading span data,
      // so the span-data download fails and the trace is dropped.
      await client.createTrace(
        new TraceInfo({
          traceId: randomUUID(),
          traceLocation: experimentLocation(),
          state: TraceState.OK,
          requestTime: 1000,
        }),
      );

      const result = await client.searchTraces({
        locations: [experimentLocation()],
      });

      expect(result.traces).toEqual([]);
    });

    it('should warn and drop only the traces whose span data download fails', async () => {
      const failingTraceId = randomUUID();
      const okTraceId = randomUUID();
      for (const traceId of [failingTraceId, okTraceId]) {
        await client.createTrace(
          new TraceInfo({
            traceId: traceId,
            traceLocation: experimentLocation(),
            state: TraceState.OK,
            requestTime: 1000,
          }),
        );
      }

      const { artifactsClient } = client as unknown as { artifactsClient: ArtifactsClient };
      const downloadSpy = jest
        .spyOn(artifactsClient, 'downloadTraceData')
        .mockImplementation((traceInfo) =>
          traceInfo.traceId === failingTraceId
            ? Promise.reject(new Error('download failed'))
            : Promise.resolve(new TraceData([])),
        );
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation();

      try {
        const result = await client.searchTraces({
          locations: [experimentLocation()],
        });

        expect(result.traces.map((trace) => trace.info.traceId)).toEqual([okTraceId]);
        expect(warnSpy).toHaveBeenCalledWith(
          `Failed to download trace data for trace ${failingTraceId}:`,
          expect.any(Error),
        );
      } finally {
        downloadSpy.mockRestore();
        warnSpy.mockRestore();
      }
    });

    it('should return an empty result when no traces match', async () => {
      const result = await client.searchTraces({
        locations: [experimentLocation()],
        filter: "tags.searchTracesFilter = 'missing'",
        maxResults: 10,
      });

      expect(result.traces).toEqual([]);
      expect(result.nextPageToken).toBeUndefined();
    });

    it('should reject search without locations', async () => {
      await expect(client.searchTraces({ locations: [] })).rejects.toThrow(
        'At least one location must be specified for searching traces.',
      );
    });

    it('should reject search when locations is omitted', async () => {
      // Plain-JS callers can omit locations despite the required type.
      await expect(client.searchTraces({} as SearchTracesOptions)).rejects.toThrow(
        'At least one location must be specified for searching traces.',
      );
    });
  });

  describe('getExperimentByName', () => {
    it('should retrieve an existing experiment by name', async () => {
      const experiment = await client.getExperimentByName(
        `test-experiment-${Date.now()}-${Math.random().toString(36).substring(2, 15)}`,
      );

      expect(experiment).toBeNull();

      const createdName = `lookup-experiment-${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
      const createdId = await client.createExperiment(createdName);
      try {
        const found = await client.getExperimentByName(createdName);
        expect(found).toEqual({
          experimentId: createdId,
          name: createdName,
        });
      } finally {
        await client.deleteExperiment(createdId);
      }
    });
  });
});

import { randomUUID } from 'crypto';
import { MlflowClient } from '../../src/clients/client';
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
    it('should search traces by experiment ID and page through results', async () => {
      const firstTraceId = randomUUID();
      const secondTraceId = randomUUID();
      await client.createTrace(
        new TraceInfo({
          traceId: firstTraceId,
          traceLocation: {
            type: TraceLocationType.MLFLOW_EXPERIMENT,
            mlflowExperiment: {
              experimentId: experimentId,
            },
          },
          state: TraceState.OK,
          requestTime: 1000,
        }),
      );
      await client.createTrace(
        new TraceInfo({
          traceId: secondTraceId,
          traceLocation: {
            type: TraceLocationType.MLFLOW_EXPERIMENT,
            mlflowExperiment: {
              experimentId: experimentId,
            },
          },
          state: TraceState.OK,
          requestTime: 2000,
        }),
      );

      const firstPage = await client.searchTraces({
        experimentIds: [experimentId],
        maxResults: 1,
        orderBy: ['timestamp_ms DESC'],
      });

      expect(firstPage.traces).toHaveLength(1);
      expect(firstPage.traces[0]).toBeInstanceOf(TraceInfo);
      expect([firstTraceId, secondTraceId]).toContain(firstPage.traces[0].traceId);
      expect(firstPage.nextPageToken).toBeDefined();

      const secondPage = await client.searchTraces({
        experimentIds: [experimentId],
        maxResults: 1,
        orderBy: ['timestamp_ms DESC'],
        pageToken: firstPage.nextPageToken,
      });

      const returnedTraceIds = [
        firstPage.traces[0].traceId,
        ...secondPage.traces.map((trace) => trace.traceId),
      ];
      expect(returnedTraceIds).toEqual(expect.arrayContaining([firstTraceId, secondTraceId]));
    });

    it('should search traces with a filter and explicit locations', async () => {
      const matchingTraceId = randomUUID();
      const otherTraceId = randomUUID();
      await client.createTrace(
        new TraceInfo({
          traceId: matchingTraceId,
          traceLocation: {
            type: TraceLocationType.MLFLOW_EXPERIMENT,
            mlflowExperiment: {
              experimentId: experimentId,
            },
          },
          state: TraceState.OK,
          requestTime: 1000,
          tags: { searchTracesFilter: 'match' },
        }),
      );
      await client.createTrace(
        new TraceInfo({
          traceId: otherTraceId,
          traceLocation: {
            type: TraceLocationType.MLFLOW_EXPERIMENT,
            mlflowExperiment: {
              experimentId: experimentId,
            },
          },
          state: TraceState.OK,
          requestTime: 2000,
          tags: { searchTracesFilter: 'skip' },
        }),
      );

      const result = await client.searchTraces({
        locations: [
          {
            type: TraceLocationType.MLFLOW_EXPERIMENT,
            mlflowExperiment: {
              experimentId: experimentId,
            },
          },
        ],
        filter: "tags.searchTracesFilter = 'match'",
        maxResults: 10,
      });

      expect(result.traces.map((trace) => trace.traceId)).toContain(matchingTraceId);
      expect(result.traces.map((trace) => trace.traceId)).not.toContain(otherTraceId);
    });

    it('should return an empty result when no traces match', async () => {
      const result = await client.searchTraces({
        experimentIds: [experimentId],
        filter: "tags.searchTracesFilter = 'missing'",
        maxResults: 10,
      });

      expect(result.traces).toEqual([]);
      expect(result.nextPageToken).toBeUndefined();
    });

    it('should reject search without experiment IDs or locations', async () => {
      await expect(client.searchTraces({})).rejects.toThrow(
        'searchTraces requires at least one experiment ID or trace location.',
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

import { randomUUID } from 'crypto';
import { MlflowClient } from '../../src/clients/client';
import { TraceInfo } from '../../src/core/entities/trace_info';
import { TraceLocationType } from '../../src/core/entities/trace_location';
import { TraceState } from '../../src/core/entities/trace_state';
import { Trace } from '../../src/core/entities/trace';
import { TraceData } from '../../src/core/entities/trace_data';
import { TraceTagKey, SpansLocation } from '../../src/core/constants';
import { init } from '../../src/core/config';
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
            experimentId: experimentId
          }
        },
        state: TraceState.OK,
        requestTime: 1000,
        executionDuration: 500,
        requestPreview: '{"input":"test"}',
        responsePreview: '{"output":"result"}',
        clientRequestId: 'client-request-id',
        traceMetadata: { 'meta-key': 'meta-value' },
        tags: { 'tag-key': 'tag-value' },
        assessments: []
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
      expect(createdTraceInfo.traceMetadata).toEqual({ 'meta-key': 'meta-value' });
      expect(createdTraceInfo.tags).toEqual({
        'tag-key': 'tag-value',
        'mlflow.artifactLocation': expect.any(String)
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
            experimentId: experimentId
          }
        },
        state: TraceState.ERROR,
        requestTime: 1000
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
            experimentId: experimentId
          }
        },
        state: TraceState.OK,
        requestTime: 1000
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
    it('should search traces in an experiment', async () => {
      const traceId1 = randomUUID();
      const traceId2 = randomUUID();

      await client.createTrace(
        new TraceInfo({
          traceId: traceId1,
          traceLocation: {
            type: TraceLocationType.MLFLOW_EXPERIMENT,
            mlflowExperiment: { experimentId: experimentId }
          },
          state: TraceState.OK,
          requestTime: Date.now() - 1000,
          requestPreview: '{"input":"test1"}',
          responsePreview: '{"output":"result1"}'
        })
      );

      await client.createTrace(
        new TraceInfo({
          traceId: traceId2,
          traceLocation: {
            type: TraceLocationType.MLFLOW_EXPERIMENT,
            mlflowExperiment: { experimentId: experimentId }
          },
          state: TraceState.OK,
          requestTime: Date.now(),
          requestPreview: '{"input":"test2"}',
          responsePreview: '{"output":"result2"}'
        })
      );

      const result = await client.searchTraces({
        experimentIds: [experimentId]
      });

      expect(result.traces).toBeInstanceOf(Array);
      expect(result.traces.length).toBeGreaterThanOrEqual(2);

      const foundTraceIds = result.traces.map((t) => t.info.traceId);
      expect(foundTraceIds).toContain(traceId1);
      expect(foundTraceIds).toContain(traceId2);
    });

    it('should search traces with filter', async () => {
      const traceIdOk = randomUUID();
      const traceIdError = randomUUID();

      await client.createTrace(
        new TraceInfo({
          traceId: traceIdOk,
          traceLocation: {
            type: TraceLocationType.MLFLOW_EXPERIMENT,
            mlflowExperiment: { experimentId: experimentId }
          },
          state: TraceState.OK,
          requestTime: Date.now()
        })
      );

      await client.createTrace(
        new TraceInfo({
          traceId: traceIdError,
          traceLocation: {
            type: TraceLocationType.MLFLOW_EXPERIMENT,
            mlflowExperiment: { experimentId: experimentId }
          },
          state: TraceState.ERROR,
          requestTime: Date.now()
        })
      );

      const result = await client.searchTraces({
        experimentIds: [experimentId],
        filterString: "trace.status = 'OK'"
      });

      expect(result.traces).toBeInstanceOf(Array);
      const foundTraceIds = result.traces.map((t) => t.info.traceId);
      expect(foundTraceIds).toContain(traceIdOk);
      expect(foundTraceIds).not.toContain(traceIdError);
    });

    it('should search traces with maxResults', async () => {
      for (let i = 0; i < 3; i++) {
        await client.createTrace(
          new TraceInfo({
            traceId: randomUUID(),
            traceLocation: {
              type: TraceLocationType.MLFLOW_EXPERIMENT,
              mlflowExperiment: { experimentId: experimentId }
            },
            state: TraceState.OK,
            requestTime: Date.now() + i
          })
        );
      }

      const result = await client.searchTraces({
        experimentIds: [experimentId],
        maxResults: 2
      });

      expect(result.traces.length).toBe(2);
    });

    it('should return empty array when no traces match', async () => {
      const result = await client.searchTraces({
        experimentIds: [experimentId],
        filterString: "trace.status = 'NONEXISTENT_STATUS'"
      });

      expect(result.traces).toBeInstanceOf(Array);
      expect(result.traces.length).toBe(0);
    });

    it('should use default experiment ID from init() when experimentIds not provided', async () => {
      init({
        trackingUri: TEST_TRACKING_URI,
        experimentId: experimentId
      });

      const traceId = randomUUID();
      await client.createTrace(
        new TraceInfo({
          traceId: traceId,
          traceLocation: {
            type: TraceLocationType.MLFLOW_EXPERIMENT,
            mlflowExperiment: { experimentId: experimentId }
          },
          state: TraceState.OK,
          requestTime: Date.now(),
          requestPreview: '{"input":"default-test"}'
        })
      );

      const result = await client.searchTraces();

      expect(result.traces).toBeInstanceOf(Array);
      expect(result.traces.length).toBeGreaterThanOrEqual(1);
      const foundTraceIds = result.traces.map((t) => t.info.traceId);
      expect(foundTraceIds).toContain(traceId);
    });

    it('should use default experiment ID when experimentIds is an empty array', async () => {
      init({
        trackingUri: TEST_TRACKING_URI,
        experimentId: experimentId
      });

      const traceId = randomUUID();
      await client.createTrace(
        new TraceInfo({
          traceId: traceId,
          traceLocation: {
            type: TraceLocationType.MLFLOW_EXPERIMENT,
            mlflowExperiment: { experimentId: experimentId }
          },
          state: TraceState.OK,
          requestTime: Date.now()
        })
      );

      const result = await client.searchTraces({ experimentIds: [] });

      expect(result.traces).toBeInstanceOf(Array);
      expect(result.traces.length).toBeGreaterThanOrEqual(1);
      const foundTraceIds = result.traces.map((t) => t.info.traceId);
      expect(foundTraceIds).toContain(traceId);
    });

    it('should override default experiment ID when experimentIds is provided', async () => {
      const otherExperimentName = `other-experiment-${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
      const otherExperimentId = await client.createExperiment(otherExperimentName);

      try {
        init({
          trackingUri: TEST_TRACKING_URI,
          experimentId: experimentId
        });

        const traceIdInDefault = randomUUID();
        const traceIdInOther = randomUUID();

        await client.createTrace(
          new TraceInfo({
            traceId: traceIdInDefault,
            traceLocation: {
              type: TraceLocationType.MLFLOW_EXPERIMENT,
              mlflowExperiment: { experimentId: experimentId }
            },
            state: TraceState.OK,
            requestTime: Date.now()
          })
        );

        await client.createTrace(
          new TraceInfo({
            traceId: traceIdInOther,
            traceLocation: {
              type: TraceLocationType.MLFLOW_EXPERIMENT,
              mlflowExperiment: { experimentId: otherExperimentId }
            },
            state: TraceState.OK,
            requestTime: Date.now()
          })
        );

        const result = await client.searchTraces({
          experimentIds: [otherExperimentId]
        });

        expect(result.traces).toBeInstanceOf(Array);
        const foundTraceIds = result.traces.map((t) => t.info.traceId);
        expect(foundTraceIds).toContain(traceIdInOther);
        expect(foundTraceIds).not.toContain(traceIdInDefault);
      } finally {
        await client.deleteExperiment(otherExperimentId);
      }
    });

    it('should return Trace objects with span data when includeSpans is true (default)', async () => {
      const traceId = randomUUID();
      await client.createTrace(
        new TraceInfo({
          traceId: traceId,
          traceLocation: {
            type: TraceLocationType.MLFLOW_EXPERIMENT,
            mlflowExperiment: { experimentId: experimentId }
          },
          state: TraceState.OK,
          requestTime: Date.now()
        })
      );

      const result = await client.searchTraces({
        experimentIds: [experimentId],
        includeSpans: true
      });

      expect(result.traces).toBeInstanceOf(Array);
      expect(result.traces.length).toBeGreaterThanOrEqual(1);

      const foundTrace = result.traces.find((t) => t.info.traceId === traceId);
      expect(foundTrace).toBeDefined();
      expect(foundTrace).toBeInstanceOf(Trace);
      expect(foundTrace!.info).toBeInstanceOf(TraceInfo);
      expect(foundTrace!.data).toBeInstanceOf(TraceData);
      expect(foundTrace!.data.spans).toBeInstanceOf(Array);
    });

    it('should return Trace objects with empty span data when includeSpans is false', async () => {
      const traceId = randomUUID();
      await client.createTrace(
        new TraceInfo({
          traceId: traceId,
          traceLocation: {
            type: TraceLocationType.MLFLOW_EXPERIMENT,
            mlflowExperiment: { experimentId: experimentId }
          },
          state: TraceState.OK,
          requestTime: Date.now()
        })
      );

      const result = await client.searchTraces({
        experimentIds: [experimentId],
        includeSpans: false
      });

      expect(result.traces).toBeInstanceOf(Array);
      expect(result.traces.length).toBeGreaterThanOrEqual(1);

      const foundTrace = result.traces.find((t) => t.info.traceId === traceId);
      expect(foundTrace).toBeDefined();
      expect(foundTrace).toBeInstanceOf(Trace);
      expect(foundTrace!.info).toBeInstanceOf(TraceInfo);
      expect(foundTrace!.data).toBeInstanceOf(TraceData);
      expect(foundTrace!.data.spans).toEqual([]);
    });

    it('should default to includeSpans: true when not specified', async () => {
      const traceId = randomUUID();
      await client.createTrace(
        new TraceInfo({
          traceId: traceId,
          traceLocation: {
            type: TraceLocationType.MLFLOW_EXPERIMENT,
            mlflowExperiment: { experimentId: experimentId }
          },
          state: TraceState.OK,
          requestTime: Date.now()
        })
      );

      const result = await client.searchTraces({
        experimentIds: [experimentId]
      });

      expect(result.traces).toBeInstanceOf(Array);

      const foundTrace = result.traces.find((t) => t.info.traceId === traceId);
      expect(foundTrace).toBeDefined();
      expect(foundTrace).toBeInstanceOf(Trace);
      expect(foundTrace!.info).toBeInstanceOf(TraceInfo);
      expect(foundTrace!.data).toBeInstanceOf(TraceData);
    });
  });

});

// Unit tests for MlflowClient helper methods (don't require server)
describe('MlflowClient (unit tests)', () => {
  let client: MlflowClient;

  beforeEach(() => {
    const authProvider = createAuthProvider({ trackingUri: TEST_TRACKING_URI });
    client = new MlflowClient({ trackingUri: TEST_TRACKING_URI, authProvider });
  });

  describe('groupTraceInfosByLocation', () => {
    it('should group traces by TRACKING_STORE location', () => {
      const traceInfos = [
        new TraceInfo({
          traceId: 'trace1',
          traceLocation: { type: TraceLocationType.MLFLOW_EXPERIMENT, mlflowExperiment: { experimentId: 'exp1' } },
          state: TraceState.OK,
          requestTime: 1000,
          tags: { [TraceTagKey.SPANS_LOCATION]: SpansLocation.TRACKING_STORE }
        }),
        new TraceInfo({
          traceId: 'trace2',
          traceLocation: { type: TraceLocationType.MLFLOW_EXPERIMENT, mlflowExperiment: { experimentId: 'exp1' } },
          state: TraceState.OK,
          requestTime: 1000,
          tags: { [TraceTagKey.SPANS_LOCATION]: SpansLocation.ARTIFACT_REPO }
        }),
        new TraceInfo({
          traceId: 'trace3',
          traceLocation: { type: TraceLocationType.MLFLOW_EXPERIMENT, mlflowExperiment: { experimentId: 'exp1' } },
          state: TraceState.OK,
          requestTime: 1000,
          tags: {}
        })
      ];

      const clientAny = client as any;
      const result = clientAny.groupTraceInfosByLocation(traceInfos);

      expect(result.trackingStoreTraces).toHaveLength(1);
      expect(result.trackingStoreTraces[0].traceId).toBe('trace1');
      expect(result.artifactRepoTraces).toHaveLength(2);
      expect(result.artifactRepoTraces.map((t: TraceInfo) => t.traceId)).toEqual(['trace2', 'trace3']);
    });

    it('should handle empty trace list', () => {
      const clientAny = client as any;
      const result = clientAny.groupTraceInfosByLocation([]);

      expect(result.trackingStoreTraces).toHaveLength(0);
      expect(result.artifactRepoTraces).toHaveLength(0);
    });
  });

  describe('processWithConcurrencyLimit', () => {
    it('should process items with concurrency control', async () => {
      const items = [1, 2, 3, 4, 5];
      const processor = jest.fn(async (item: number) => {
        await new Promise(resolve => setTimeout(resolve, 10));
        return item * 2;
      });

      const clientAny = client as any;
      const results = await clientAny.processWithConcurrencyLimit(items, processor, 2);

      expect(results).toEqual([2, 4, 6, 8, 10]);
      expect(processor).toHaveBeenCalledTimes(5);
    });

    it('should handle maxConcurrency larger than items', async () => {
      const items = [1, 2];
      const processor = jest.fn(async (item: number) => item * 3);

      const clientAny = client as any;
      const results = await clientAny.processWithConcurrencyLimit(items, processor, 5);

      expect(results).toEqual([3, 6]);
      expect(processor).toHaveBeenCalledTimes(2);
    });

    it('should handle empty items array', async () => {
      const processor = jest.fn();

      const clientAny = client as any;
      const results = await clientAny.processWithConcurrencyLimit([], processor, 2);

      expect(results).toEqual([]);
      expect(processor).not.toHaveBeenCalled();
    });
  });

  describe('batchGetTracesFromTrackingStore', () => {
    it('should return empty array for empty input', async () => {
      const clientAny = client as any;
      const result = await clientAny.batchGetTracesFromTrackingStore([]);

      expect(result).toEqual([]);
    });

    it('should batch trace IDs into groups of 10', async () => {
      const mockFetchBatch = jest.fn().mockResolvedValue([]);
      (client as any).fetchBatchFromTrackingStore = mockFetchBatch;

      const traceInfos = Array.from({ length: 25 }, (_, i) =>
        new TraceInfo({
          traceId: `trace${i}`,
          traceLocation: { type: TraceLocationType.MLFLOW_EXPERIMENT, mlflowExperiment: { experimentId: 'exp1' } },
          state: TraceState.OK,
          requestTime: 1000
        })
      );

      const clientAny = client as any;
      await clientAny.batchGetTracesFromTrackingStore(traceInfos);

      expect(mockFetchBatch).toHaveBeenCalledTimes(3);
      expect(mockFetchBatch).toHaveBeenNthCalledWith(1,
        ['trace0', 'trace1', 'trace2', 'trace3', 'trace4', 'trace5', 'trace6', 'trace7', 'trace8', 'trace9']
      );
      expect(mockFetchBatch).toHaveBeenNthCalledWith(2,
        ['trace10', 'trace11', 'trace12', 'trace13', 'trace14', 'trace15', 'trace16', 'trace17', 'trace18', 'trace19']
      );
      expect(mockFetchBatch).toHaveBeenNthCalledWith(3,
        ['trace20', 'trace21', 'trace22', 'trace23', 'trace24']
      );
    });
  });

  describe('fetchTracesFromArtifactStoreParallel', () => {
    it('should return empty array for empty input', async () => {
      const clientAny = client as any;
      const result = await clientAny.fetchTracesFromArtifactStoreParallel([]);

      expect(result).toEqual([]);
    });

    it('should call getTraceFromArtifactStore for each trace info', async () => {
      const mockGetTrace = jest.fn().mockResolvedValue(
        new Trace(
          new TraceInfo({
            traceId: 'test',
            traceLocation: { type: TraceLocationType.MLFLOW_EXPERIMENT, mlflowExperiment: { experimentId: 'exp1' } },
            state: TraceState.OK,
            requestTime: 1000
          }),
          new TraceData([])
        )
      );
      (client as any).getTraceFromArtifactStore = mockGetTrace;

      const traceInfos = [
        new TraceInfo({
          traceId: 'trace1',
          traceLocation: { type: TraceLocationType.MLFLOW_EXPERIMENT, mlflowExperiment: { experimentId: 'exp1' } },
          state: TraceState.OK,
          requestTime: 1000
        }),
        new TraceInfo({
          traceId: 'trace2',
          traceLocation: { type: TraceLocationType.MLFLOW_EXPERIMENT, mlflowExperiment: { experimentId: 'exp1' } },
          state: TraceState.OK,
          requestTime: 1000
        })
      ];

      const clientAny = client as any;
      const results = await clientAny.fetchTracesFromArtifactStoreParallel(traceInfos);

      expect(mockGetTrace).toHaveBeenCalledTimes(2);
      expect(results).toHaveLength(2);
    });

    it('should filter out failed downloads', async () => {
      const successTrace = new Trace(
        new TraceInfo({
          traceId: 'success',
          traceLocation: { type: TraceLocationType.MLFLOW_EXPERIMENT, mlflowExperiment: { experimentId: 'exp1' } },
          state: TraceState.OK,
          requestTime: 1000
        }),
        new TraceData([])
      );

      const mockGetTrace = jest.fn()
        .mockResolvedValueOnce(successTrace)
        .mockRejectedValueOnce(new Error('Download failed'));
      (client as any).getTraceFromArtifactStore = mockGetTrace;

      const traceInfos = [
        new TraceInfo({
          traceId: 'trace1',
          traceLocation: { type: TraceLocationType.MLFLOW_EXPERIMENT, mlflowExperiment: { experimentId: 'exp1' } },
          state: TraceState.OK,
          requestTime: 1000
        }),
        new TraceInfo({
          traceId: 'trace2',
          traceLocation: { type: TraceLocationType.MLFLOW_EXPERIMENT, mlflowExperiment: { experimentId: 'exp1' } },
          state: TraceState.OK,
          requestTime: 1000
        })
      ];

      const clientAny = client as any;
      const results = await clientAny.fetchTracesFromArtifactStoreParallel(traceInfos);

      expect(mockGetTrace).toHaveBeenCalledTimes(2);
      expect(results).toHaveLength(1);
      expect(results[0].info.traceId).toBe('success');
    });
  });
});

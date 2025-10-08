import { MlflowClient } from '../../src/clients/client';
import { TraceInfo } from '../../src/core/entities/trace_info';
import { TraceLocationType } from '../../src/core/entities/trace_location';
import { TraceState } from '../../src/core/entities/trace_state';
import { TEST_TRACKING_URI } from '../helper';

describe('MlflowClient', () => {
  let client: MlflowClient;
  let experimentId: string;

  beforeEach(async () => {
    client = new MlflowClient({ trackingUri: TEST_TRACKING_URI, host: TEST_TRACKING_URI });

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
      const traceId = 'test-trace-id';
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
      const traceId = 'test-trace-id';
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
      const traceId = 'test-trace-id';
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
});

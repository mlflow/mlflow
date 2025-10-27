import { TraceInfo } from '../../../src/core/entities/trace_info';
import { TraceLocationType } from '../../../src/core/entities/trace_location';
import { TraceState } from '../../../src/core/entities/trace_state';

describe('TraceInfo', () => {
  const createTestTraceInfo = () => {
    return new TraceInfo({
      traceId: 'test-trace-id',
      traceLocation: {
        type: TraceLocationType.MLFLOW_EXPERIMENT,
        mlflowExperiment: {
          experimentId: 'test-experiment-id'
        }
      },
      requestTime: 1000,
      state: TraceState.OK,
      requestPreview: '{"input":"test"}',
      responsePreview: '{"output":"result"}',
      clientRequestId: 'client-request-id',
      executionDuration: 500,
      traceMetadata: { 'meta-key': 'meta-value' },
      tags: { 'tag-key': 'tag-value' },
      assessments: []
    });
  };

  describe('constructor', () => {
    it('should create a TraceInfo instance with all properties', () => {
      const traceInfo = createTestTraceInfo();

      expect(traceInfo.traceId).toBe('test-trace-id');
      expect(traceInfo.traceLocation.type).toBe(TraceLocationType.MLFLOW_EXPERIMENT);
      expect(traceInfo.traceLocation.mlflowExperiment?.experimentId).toBe('test-experiment-id');
      expect(traceInfo.requestTime).toBe(1000);
      expect(traceInfo.state).toBe(TraceState.OK);
      expect(traceInfo.requestPreview).toBe('{"input":"test"}');
      expect(traceInfo.responsePreview).toBe('{"output":"result"}');
      expect(traceInfo.clientRequestId).toBe('client-request-id');
      expect(traceInfo.executionDuration).toBe(500);
      expect(traceInfo.traceMetadata['meta-key']).toBe('meta-value');
      expect(traceInfo.tags['tag-key']).toBe('tag-value');
      expect(traceInfo.assessments).toEqual([]);
    });
  });
});

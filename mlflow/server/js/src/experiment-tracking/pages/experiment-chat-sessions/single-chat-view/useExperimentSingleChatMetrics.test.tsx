import { describe, it, expect } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { useExperimentSingleChatMetrics } from './useExperimentSingleChatMetrics';

describe('useExperimentSingleChatMetrics', () => {
  it('should return empty metrics when traceInfos is empty array', () => {
    // Arrange
    const { result } = renderHook(() => useExperimentSingleChatMetrics({ traceInfos: [] }));

    // Assert
    expect(result.current).toEqual({
      sessionTokens: { input_tokens: 0, output_tokens: 0 },
      sessionLatency: 0,
      perTurnMetrics: [],
    });
  });

  it('should calculate metrics for a single trace', () => {
    // Arrange
    const traceInfos: ModelTraceInfoV3[] = [
      {
        request_id: 'trace-1',
        execution_duration: '1.5s',
        trace_metadata: {
          'mlflow.trace.tokenUsage': JSON.stringify({
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: 150,
          }),
        },
      } as any,
    ];

    const { result } = renderHook(() => useExperimentSingleChatMetrics({ traceInfos }));

    // Assert
    expect(result.current.sessionTokens).toEqual({
      input_tokens: 100,
      output_tokens: 50,
      total_tokens: 150,
    });
    expect(result.current.sessionLatency).toBe(1.5);
    expect(result.current.perTurnMetrics).toHaveLength(1);
    expect(result.current.perTurnMetrics?.[0]).toEqual({
      tokens: {
        input_tokens: 100,
        output_tokens: 50,
        total_tokens: 150,
      },
      latency: '1.5s',
    });
  });

  it('should calculate cumulative metrics for multiple traces', () => {
    // Arrange
    const traceInfos: ModelTraceInfoV3[] = [
      {
        request_id: 'trace-1',
        execution_duration: '1.5s',
        trace_metadata: {
          'mlflow.trace.tokenUsage': JSON.stringify({
            input_tokens: 100,
            output_tokens: 50,
          }),
        },
      },
      {
        request_id: 'trace-2',
        execution_duration: '2.3s',
        trace_metadata: {
          'mlflow.trace.tokenUsage': JSON.stringify({
            input_tokens: 150,
            output_tokens: 75,
          }),
        },
      },
      {
        request_id: 'trace-3',
        execution_duration: '1.2s',
        trace_metadata: {
          'mlflow.trace.tokenUsage': JSON.stringify({
            input_tokens: 200,
            output_tokens: 100,
          }),
        },
      },
    ] as any;

    const { result } = renderHook(() => useExperimentSingleChatMetrics({ traceInfos }));

    // Assert
    // Session tokens should be from the last trace
    expect(result.current.sessionTokens).toEqual({
      input_tokens: 200,
      output_tokens: 100,
    });
    // Session latency should be sum of all traces
    expect(result.current.sessionLatency).toBeCloseTo(5.0);
    // Should have metrics for all three turns
    expect(result.current.perTurnMetrics).toHaveLength(3);
  });

  it('should handle traces with missing token usage metadata', () => {
    // Arrange
    const traceInfos: ModelTraceInfoV3[] = [
      {
        request_id: 'trace-1',
        execution_duration: '1.5s',
        trace_metadata: {},
      },
    ] as any;

    const { result } = renderHook(() => useExperimentSingleChatMetrics({ traceInfos }));

    // Assert
    expect(result.current.sessionTokens).toEqual({});
    expect(result.current.sessionLatency).toBe(1.5);
    expect(result.current.perTurnMetrics?.[0].tokens).toEqual({});
  });

  it('should handle traces with invalid token usage JSON', () => {
    // Arrange
    const traceInfos: ModelTraceInfoV3[] = [
      {
        request_id: 'trace-1',
        execution_duration: '1.5s',
        trace_metadata: {
          'mlflow.trace.tokenUsage': 'invalid-json',
        },
      },
    ] as any;

    const { result } = renderHook(() => useExperimentSingleChatMetrics({ traceInfos }));

    // Assert
    expect(result.current.sessionTokens).toEqual({});
    expect(result.current.sessionLatency).toBe(1.5);
  });

  it('should handle traces with missing execution_duration', () => {
    // Arrange
    const traceInfos: ModelTraceInfoV3[] = [
      {
        request_id: 'trace-1',
        trace_metadata: {
          'mlflow.trace.tokenUsage': JSON.stringify({
            input_tokens: 100,
            output_tokens: 50,
          }),
        },
      },
    ] as any;

    const { result } = renderHook(() => useExperimentSingleChatMetrics({ traceInfos }));

    // Assert
    expect(result.current.sessionLatency).toBe(0);
    expect(result.current.perTurnMetrics?.[0].latency).toBeUndefined();
  });
});

import { renderHook, act } from '@testing-library/react';
import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { useChartInteractionTelemetry } from './useChartInteractionTelemetry';

// Mock the telemetry hook
const mockLogTelemetryEvent = jest.fn();
jest.mock('../../../../telemetry/hooks/useLogTelemetryEvent', () => ({
  useLogTelemetryEvent: () => mockLogTelemetryEvent,
}));

describe('useChartInteractionTelemetry', () => {
  beforeEach(() => {
    mockLogTelemetryEvent.mockClear();
  });

  it('should return onClick handler', () => {
    const { result } = renderHook(() => useChartInteractionTelemetry('mlflow.charts.test'));

    expect(result.current.onClick).toBeDefined();
    expect(typeof result.current.onClick).toBe('function');
  });

  it('should log telemetry event on click when componentId is provided', () => {
    const { result } = renderHook(() => useChartInteractionTelemetry('mlflow.charts.test'));

    act(() => {
      result.current.onClick();
    });

    expect(mockLogTelemetryEvent).toHaveBeenCalledTimes(1);
    expect(mockLogTelemetryEvent).toHaveBeenCalledWith(
      expect.objectContaining({
        componentId: 'mlflow.charts.test',
        componentType: 'card',
        eventType: 'onClick',
        value: 'chart_interaction',
      }),
    );
  });

  it('should not log telemetry event when componentId is undefined', () => {
    const { result } = renderHook(() => useChartInteractionTelemetry(undefined));

    act(() => {
      result.current.onClick();
    });

    expect(mockLogTelemetryEvent).not.toHaveBeenCalled();
  });

  it('should log multiple events on multiple clicks', () => {
    const { result } = renderHook(() => useChartInteractionTelemetry('mlflow.charts.test'));

    act(() => {
      result.current.onClick();
      result.current.onClick();
      result.current.onClick();
    });

    expect(mockLogTelemetryEvent).toHaveBeenCalledTimes(3);
  });

  it('should use consistent componentViewId across clicks', () => {
    const { result } = renderHook(() => useChartInteractionTelemetry('mlflow.charts.test'));

    act(() => {
      result.current.onClick();
    });

    const firstCall = mockLogTelemetryEvent.mock.calls[0][0] as { componentViewId: string };

    act(() => {
      result.current.onClick();
    });

    const secondCall = mockLogTelemetryEvent.mock.calls[1][0] as { componentViewId: string };

    expect(firstCall.componentViewId).toBe(secondCall.componentViewId);
  });

  it('should not return onMouseEnter handler (to avoid noise from scrolling)', () => {
    const { result } = renderHook(() => useChartInteractionTelemetry('mlflow.charts.test'));

    expect(result.current).not.toHaveProperty('onMouseEnter');
  });
});

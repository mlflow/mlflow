import { renderHook, act } from '@testing-library/react';
import { jest, describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { useChartInteractionTelemetry } from './useChartInteractionTelemetry';

// Mock the telemetry hook
const mockLogTelemetryEvent = jest.fn();
jest.mock('../../../../telemetry/hooks/useLogTelemetryEvent', () => ({
  useLogTelemetryEvent: () => mockLogTelemetryEvent,
}));

describe('useChartInteractionTelemetry', () => {
  beforeEach(() => {
    mockLogTelemetryEvent.mockClear();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
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

  it('should throttle clicks within 10 seconds', () => {
    const { result } = renderHook(() => useChartInteractionTelemetry('mlflow.charts.test'));

    act(() => {
      result.current.onClick();
      result.current.onClick();
      result.current.onClick();
    });

    expect(mockLogTelemetryEvent).toHaveBeenCalledTimes(1);
  });

  it('should log again after throttle period expires', () => {
    const { result } = renderHook(() => useChartInteractionTelemetry('mlflow.charts.test'));

    act(() => {
      result.current.onClick();
    });

    expect(mockLogTelemetryEvent).toHaveBeenCalledTimes(1);

    // Advance time by 10 seconds
    act(() => {
      jest.advanceTimersByTime(10_000);
      result.current.onClick();
    });

    expect(mockLogTelemetryEvent).toHaveBeenCalledTimes(2);
  });

  it('should use consistent componentViewId across clicks', () => {
    const { result } = renderHook(() => useChartInteractionTelemetry('mlflow.charts.test'));

    act(() => {
      result.current.onClick();
    });

    const firstCall = mockLogTelemetryEvent.mock.calls[0][0] as { componentViewId: string };

    // Advance time to allow second click
    act(() => {
      jest.advanceTimersByTime(10_000);
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

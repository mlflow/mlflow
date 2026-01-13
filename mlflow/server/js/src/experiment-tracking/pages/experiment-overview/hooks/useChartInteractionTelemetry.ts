import { useCallback, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
} from '@databricks/design-system';
import { useLogTelemetryEvent } from '../../../../telemetry/hooks/useLogTelemetryEvent';

/**
 * Hook for tracking chart interaction telemetry.
 * Tracks when users click on a chart.
 * Not tracking hover interactions to avoid noise.
 *
 * @param componentId - Unique identifier for the chart (e.g., "mlflow.charts.trace_requests").
 *                      If undefined, no telemetry will be logged.
 * @returns Object with event handlers to spread on the chart container
 *
 * @example
 * const interactionProps = useChartInteractionTelemetry('mlflow.charts.trace_requests');
 * <div {...interactionProps}>Chart content</div>
 */
export function useChartInteractionTelemetry(componentId: string | undefined) {
  const componentViewId = useRef<string>(uuidv4());
  const logTelemetryEvent = useLogTelemetryEvent();

  const onClick = useCallback(() => {
    if (!componentId) {
      return;
    }

    logTelemetryEvent({
      componentId,
      componentViewId: componentViewId.current,
      componentType: DesignSystemEventProviderComponentTypes.Card,
      eventType: DesignSystemEventProviderAnalyticsEventTypes.OnClick,
      value: 'chart_interaction',
    });
  }, [componentId, logTelemetryEvent]);

  return { onClick };
}

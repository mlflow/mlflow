import { useCallback } from 'react';
import type { DesignSystemObservabilityEvent } from '../types';
import { telemetryClient } from '../TelemetryClient';

export const useLogTelemetryEvent = () => {
  const logTelemetryEvent = useCallback((event: DesignSystemObservabilityEvent) => {
    telemetryClient.logEvent(event);
  }, []);

  return logTelemetryEvent;
};

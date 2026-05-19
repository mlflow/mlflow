import { has, isObject } from 'lodash';
import type { DesignSystemObservabilityEvent } from './types';
import { getLocalStorageItem } from '../shared/web-shared/hooks/useLocalStorage';

export const TELEMETRY_ENABLED_STORAGE_VERSION = 1;

export const TELEMETRY_ENABLE_DEV_LOGGING_STORAGE_KEY = 'mlflow.settings.telemetry.enable-dev-logging';

export const TELEMETRY_ENABLED_STORAGE_KEY = 'mlflow.settings.telemetry.enabled';

export const TELEMETRY_INFO_ALERT_DISMISSED_STORAGE_KEY = 'mlflow.telemetry.info.alert.dismissed';

export const TELEMETRY_INFO_ALERT_DISMISSED_STORAGE_VERSION = 1;

export const isDesignSystemEvent = (event: any): event is DesignSystemObservabilityEvent => {
  if (!event || !isObject(event) || Array.isArray(event)) {
    return false;
  }

  return (
    has(event, 'componentId') &&
    typeof event.componentId === 'string' &&
    has(event, 'componentType') &&
    typeof event.componentType === 'string' &&
    has(event, 'componentViewId') &&
    typeof event.componentViewId === 'string' &&
    has(event, 'eventType') &&
    typeof event.eventType === 'string'
  );
};

export const isTelemetryDevLoggingEnabled = (): boolean => {
  return getLocalStorageItem(TELEMETRY_ENABLE_DEV_LOGGING_STORAGE_KEY, 1, false, false);
};

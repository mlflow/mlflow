import { has, isObject } from 'lodash';
import { DesignSystemObservabilityEvent } from './types';

export const TELEMETRY_ENABLED_STORAGE_VERSION = 1;

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

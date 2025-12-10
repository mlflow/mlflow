import { has, isObject } from 'lodash';
import { DesignSystemObservabilityEvent } from './types';

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
    (event.eventType === 'onClick' || event.eventType === 'onView')
  );
};

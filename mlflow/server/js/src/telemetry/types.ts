import type {
  DesignSystemEventProviderComponentTypes,
  DesignSystemEventProviderAnalyticsEventTypes,
} from '@databricks/design-system';

export interface DesignSystemObservabilityEvent {
  componentId: string;
  componentViewId: string;
  componentType: DesignSystemEventProviderComponentTypes;
  componentSubType?: string | null;
  eventType: DesignSystemEventProviderAnalyticsEventTypes;
}

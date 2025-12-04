export interface TelemetryRecord {
  installation_id: string;
  event_name: string;
  timestamp_ns: number;
  params?: Record<string, any>;
}

export interface DesignSystemObservabilityEvent {
  componentId: string;
  componentViewId: string;
  componentType: string;
  componentSubType?: string;
  eventType: 'onClick' | 'onView';
}

export interface DesignSystemObservabilityEvent {
  componentId: string;
  componentViewId: string;
  componentType: string;
  componentSubType?: string | null;
  eventType: 'onClick' | 'onView';
}

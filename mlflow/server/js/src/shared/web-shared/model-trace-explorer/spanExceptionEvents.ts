import type { ModelTraceEvent, ModelTraceSpan, ModelTraceSpanNode } from './ModelTrace.types';

export const getSpanExceptionEvents = (span: ModelTraceSpanNode | ModelTraceSpan): ModelTraceEvent[] => {
  return (span.events ?? []).filter((event) => event.name === 'exception');
};

export const getSpanExceptionCount = (span: ModelTraceSpanNode | ModelTraceSpan): number => {
  return getSpanExceptionEvents(span).length;
};

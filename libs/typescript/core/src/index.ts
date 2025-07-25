import { init } from './core/config';
import {
  getLastActiveTraceId,
  getCurrentActiveSpan,
  updateCurrentTrace,
  startSpan,
  trace,
  withSpan
} from './core/api';
import { flushTraces } from './core/provider';
import { MlflowClient } from './clients';

export {
  getLastActiveTraceId,
  getCurrentActiveSpan,
  updateCurrentTrace,
  flushTraces,
  init,
  startSpan,
  trace,
  withSpan,
  MlflowClient
};

// Export entities
export * from './core/constants';
export type { LiveSpan, Span } from './core/entities/span';
export type { Trace } from './core/entities/trace';
export type { TraceInfo, TokenUsage } from './core/entities/trace_info';
export type { TraceData } from './core/entities/trace_data';
export { SpanStatusCode } from './core/entities/span_status';
export type { UpdateCurrentTraceOptions, SpanOptions, TraceOptions } from './core/api';

import { init } from './core/config';
import { getLastActiveTraceId, startSpan, trace, withSpan } from './core/api';
import { flushTraces } from './core/provider';

export { getLastActiveTraceId, flushTraces, init, startSpan, trace, withSpan };

// Export entities
export * from './core/constants';
export type { LiveSpan, Span } from './core/entities/span';
export type { Trace } from './core/entities/trace';
export type { TraceInfo } from './core/entities/trace_info';
export type { TraceData } from './core/entities/trace_data';
export { SpanStatusCode } from './core/entities/span_status';

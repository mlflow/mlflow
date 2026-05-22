import { init } from './core/config';
import {
  getLastActiveTraceId,
  getCurrentActiveSpan,
  updateCurrentTrace,
  startSpan,
  trace,
  withSpan,
} from './core/api';
import { tracingContext } from './core/context';
import { flushTraces } from './core/provider';
import {
  getDestination,
  setDestination,
  ucSchemaDestination,
  unityCatalogDestination,
} from './core/destination';
import { MlflowClient, MlflowHttpError } from './clients';
import { InMemoryTraceManager } from './core/trace_manager';
import { createAuthProvider } from './auth';

export {
  getLastActiveTraceId,
  getCurrentActiveSpan,
  updateCurrentTrace,
  tracingContext,
  flushTraces,
  init,
  startSpan,
  trace,
  withSpan,
  setDestination,
  getDestination,
  unityCatalogDestination,
  ucSchemaDestination,
  MlflowClient,
  MlflowHttpError,
  InMemoryTraceManager,
  createAuthProvider,
};

// Export entities
export * from './core/constants';
export type { LiveSpan, Span } from './core/entities/span';
export type { Trace } from './core/entities/trace';
export type { TraceInfo, TokenUsage } from './core/entities/trace_info';
export type { TraceData } from './core/entities/trace_data';
export type {
  TraceLocation,
  UCSchemaLocation,
  UnityCatalogLocation,
} from './core/entities/trace_location';
export { TraceLocationType } from './core/entities/trace_location';
export type {
  TraceDestination,
  UnityCatalogDestination,
  UcSchemaDestination,
} from './core/destination';
export { SpanStatusCode } from './core/entities/span_status';
export type { UpdateCurrentTraceOptions, SpanOptions, TraceOptions } from './core/api';
export type { TracingContextOptions } from './core/context';
export { registerOnSpanStartHook, registerOnSpanEndHook } from './exporters/span_processor_hooks';

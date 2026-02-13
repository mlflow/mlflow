import { context, trace as otelTrace } from '@opentelemetry/api';
import { W3CTraceContextPropagator } from '@opentelemetry/core';
import { InMemoryTraceManager } from './trace_manager';
import { TraceInfo } from './entities/trace_info';
import { TraceState } from './entities/trace_state';
import { TraceLocationType } from './entities/trace_location';
import { TRACE_ID_PREFIX } from './constants';

const propagator = new W3CTraceContextPropagator();

/**
 * Get HTTP request headers containing the current tracing context.
 *
 * The trace context is serialized as the `traceparent` header following
 * the W3C TraceContext specification (https://www.w3.org/TR/trace-context/).
 *
 * @returns HTTP headers containing trace context, or empty object if no active span
 *
 * @example
 * ```typescript
 * import { withSpan, getTracingContextHeadersForHttpRequest } from 'mlflow-tracing';
 *
 * withSpan(async (span) => {
 *   const headers = getTracingContextHeadersForHttpRequest();
 *   const response = await fetch('http://remote-service/api', { headers });
 * }, { name: 'client-request' });
 * ```
 */
export function getTracingContextHeadersForHttpRequest(): Record<string, string> {
  const activeSpan = otelTrace.getActiveSpan();

  if (!activeSpan || !activeSpan.spanContext().traceId) {
    console.warn(
      'No active span found for fetching the trace context from. Returning an empty header.',
    );
    return {};
  }

  const headers: Record<string, string> = {};
  propagator.inject(context.active(), headers, {
    set: (carrier, key, value) => {
      carrier[key] = value;
    },
  });

  return headers;
}

/**
 * Execute a callback with trace context extracted from HTTP request headers.
 *
 * This function extracts the trace context from incoming HTTP headers and
 * sets it as the current context. Any spans created within the callback
 * will be children of the remote parent span.
 *
 * @param headers HTTP request headers containing `traceparent` header
 * @param callback Function to execute within the extracted context
 * @returns Result of the callback function
 * @throws Error if headers don't contain valid `traceparent` header
 *
 * @example
 * ```typescript
 * import { withTracingContextFromHeaders, withSpan } from 'mlflow-tracing';
 * import express from 'express';
 *
 * const app = express();
 *
 * app.post('/api/handler', (req, res) => {
 *   withTracingContextFromHeaders(req.headers, () => {
 *     return withSpan((span) => {
 *       // This span is a child of the remote caller's span
 *       return processRequest(req.body);
 *     }, { name: 'server-handler' });
 *   });
 * });
 * ```
 */
export function withTracingContextFromHeaders<T>(
  headers: Record<string, string | string[] | undefined>,
  callback: () => T,
): T {
  const normalizedHeaders = normalizeHeaders(headers);

  if (!normalizedHeaders['traceparent']) {
    throw new Error(
      "The HTTP request headers do not contain the required key 'traceparent'. " +
        "Please generate the request headers using 'getTracingContextHeadersForHttpRequest()'.",
    );
  }

  // Extract context from headers
  const extractedContext = propagator.extract(context.active(), normalizedHeaders, {
    get: (carrier, key) => carrier[key],
    keys: (carrier) => Object.keys(carrier),
  });

  // Get the span context from the extracted context
  const spanContext = otelTrace.getSpanContext(extractedContext);

  if (!spanContext) {
    throw new Error('Failed to extract valid span context from headers.');
  }

  const otelTraceId = spanContext.traceId;
  const mlflowTraceId = TRACE_ID_PREFIX + otelTraceId;

  // Register a dummy trace info so child spans won't be skipped by the span processor
  const traceManager = InMemoryTraceManager.getInstance();
  const dummyTraceInfo = new TraceInfo({
    traceId: mlflowTraceId,
    traceLocation: { type: TraceLocationType.MLFLOW_EXPERIMENT },
    requestTime: Date.now(),
    state: TraceState.IN_PROGRESS,
    traceMetadata: {},
    tags: {},
  });

  traceManager.registerTrace(otelTraceId, dummyTraceInfo, true /* isRemoteTrace */);

  try {
    // Execute callback within the extracted context
    return context.with(extractedContext, callback);
  } finally {
    // Clean up the remote trace registration
    traceManager.popTrace(otelTraceId);
  }
}

/**
 * Async version of withTracingContextFromHeaders for Promise-based callbacks.
 *
 * @param headers HTTP request headers containing `traceparent` header
 * @param callback Async function to execute within the extracted context
 * @returns Promise resolving to the result of the callback function
 * @throws Error if headers don't contain valid `traceparent` header
 *
 * @example
 * ```typescript
 * import { withTracingContextFromHeadersAsync, withSpan } from 'mlflow-tracing';
 * import express from 'express';
 *
 * const app = express();
 *
 * app.post('/api/handler', async (req, res) => {
 *   await withTracingContextFromHeadersAsync(req.headers, async () => {
 *     return withSpan(async (span) => {
 *       // This span is a child of the remote caller's span
 *       return await processRequest(req.body);
 *     }, { name: 'server-handler' });
 *   });
 * });
 * ```
 */
export async function withTracingContextFromHeadersAsync<T>(
  headers: Record<string, string | string[] | undefined>,
  callback: () => Promise<T>,
): Promise<T> {
  const normalizedHeaders = normalizeHeaders(headers);

  if (!normalizedHeaders['traceparent']) {
    throw new Error(
      "The HTTP request headers do not contain the required key 'traceparent'. " +
        "Please generate the request headers using 'getTracingContextHeadersForHttpRequest()'.",
    );
  }

  const extractedContext = propagator.extract(context.active(), normalizedHeaders, {
    get: (carrier, key) => carrier[key],
    keys: (carrier) => Object.keys(carrier),
  });

  const spanContext = otelTrace.getSpanContext(extractedContext);

  if (!spanContext) {
    throw new Error('Failed to extract valid span context from headers.');
  }

  const otelTraceId = spanContext.traceId;
  const mlflowTraceId = TRACE_ID_PREFIX + otelTraceId;

  const traceManager = InMemoryTraceManager.getInstance();
  const dummyTraceInfo = new TraceInfo({
    traceId: mlflowTraceId,
    traceLocation: { type: TraceLocationType.MLFLOW_EXPERIMENT },
    requestTime: Date.now(),
    state: TraceState.IN_PROGRESS,
    traceMetadata: {},
    tags: {},
  });

  traceManager.registerTrace(otelTraceId, dummyTraceInfo, true);

  try {
    return await context.with(extractedContext, callback);
  } finally {
    traceManager.popTrace(otelTraceId);
  }
}

/**
 * Normalize HTTP headers to lowercase keys and string values.
 * Some HTTP server frameworks (e.g., Express) may uppercase header keys
 * or provide array values for repeated headers.
 */
function normalizeHeaders(
  headers: Record<string, string | string[] | undefined>,
): Record<string, string> {
  const normalized: Record<string, string> = {};

  for (const [key, value] of Object.entries(headers)) {
    if (value !== undefined) {
      // Handle array values (take first value)
      const stringValue = Array.isArray(value) ? value[0] : value;
      if (stringValue !== undefined) {
        normalized[key.toLowerCase()] = stringValue;
      }
    }
  }

  return normalized;
}

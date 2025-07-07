import { trace as otelTrace, context, Span as ApiSpan } from '@opentelemetry/api';
import { Span as OTelSpan } from '@opentelemetry/sdk-trace-node';
import { DEFAULT_SPAN_NAME, SpanType } from './constants';
import { createMlflowSpan, LiveSpan, NoOpSpan } from './entities/span';
import { getTracer } from './provider';
import { InMemoryTraceManager } from './trace_manager';
import { convertNanoSecondsToHrTime, mapArgsToObject } from './utils';
import { SpanStatusCode } from './entities/span_status';

/*
 * Options for starting a span
 *
 * @param name The name of the span.
 * @param span_type The type of the span.
 * @param inputs The inputs of the span.
 * @param attributes The attributes of the span.
 * @param startTimeNs The start time of the span in nanoseconds. If not provided, the current time will be used.
 * @param parent The parent span object.
 */
export interface SpanOptions {
  name: string;
  spanType?: SpanType;
  inputs?: any;
  attributes?: Record<string, any>;
  startTimeNs?: number;
  parent?: LiveSpan;
}

/**
 * Options for tracing a function
 */
export interface TraceOptions
  extends Omit<SpanOptions, 'parent' | 'startTimeNs' | 'inputs' | 'name'> {
  name?: string;
}

/**
 * Start a new span with the given name and span type.
 *
 * This function does NOT attach the created span to the current context.
 * The span must be ended by calling `end` method on the returned Span object.
 */
export function startSpan(options: SpanOptions): LiveSpan {
  try {
    const tracer = getTracer('default');

    // If parent is provided, use it as the parent spanAdd commentMore actions
    const parentContext = options.parent
      ? otelTrace.setSpan(context.active(), options.parent._span)
      : context.active();

    // Convert startTimeNs to OTel format
    const startTime = options.startTimeNs
      ? convertNanoSecondsToHrTime(options.startTimeNs)
      : undefined;

    const otelSpan = tracer.startSpan(
      options.name,
      { startTime: startTime },
      parentContext
    ) as OTelSpan;

    // Create and register the MLflow span
    const mlflowSpan = createAndRegisterMlflowSpan(
      otelSpan,
      options.spanType,
      options.inputs,
      options.attributes
    );

    return mlflowSpan;
  } catch (error) {
    console.warn('Failed to start span', error);
    return new NoOpSpan();
  }
}

/**
 * Execute a function within a span context. The span is automatically started before
 * the function executes and ended after it completes (or throws an error).
 *
 * This function uses OpenTelemetry's active span context to automatically manage
 * parent-child relationships between spans.
 *
 * This function supports two usage patterns:
 * 1. Inline: withSpan((span) => { ... }) - span properties set within the callback
 * 2. Options: withSpan((span) => { ... }, { name: 'test', ... }) - span properties set via options object
 *
 * @param callback The callback function to execute within the span context
 * @param options Optional span options (name, span_type, inputs, attributes, startTimeNs)
 * @returns The result of the callback function
 */
export function withSpan<T>(
  callback: (span: LiveSpan) => T | Promise<T>,
  options?: Omit<SpanOptions, 'parent'>
): T | Promise<T> {
  const spanOptions: Omit<SpanOptions, 'parent'> = options ?? { name: DEFAULT_SPAN_NAME };

  // Generate a default span name if not provided
  const spanName = spanOptions.name || DEFAULT_SPAN_NAME;

  const tracer = getTracer('default');

  // Convert startTimeNs to OTel format if provided
  const startTime = spanOptions.startTimeNs
    ? convertNanoSecondsToHrTime(spanOptions.startTimeNs)
    : undefined;

  // Use startActiveSpan to automatically manage context and parent-child relationships
  return tracer.startActiveSpan(spanName, { startTime }, (otelSpan: ApiSpan) => {
    // Create and register the MLflow span
    const mlflowSpan = createAndRegisterMlflowSpan(
      otelSpan,
      spanOptions.spanType,
      spanOptions.inputs,
      spanOptions.attributes
    );

    // Expression function to handle errors consistently
    const handleSpanError = (error: Error): never => {
      mlflowSpan.setStatus(SpanStatusCode.ERROR, error.message);
      mlflowSpan.recordException(error);
      mlflowSpan.end();
      throw error;
    };

    try {
      // Execute the callback with the span
      const result = callback(mlflowSpan);

      // Handle both sync and async results
      if (result instanceof Promise) {
        return result
          .then((value) => {
            // Set outputs if they are not already set
            if (mlflowSpan.outputs === undefined) {
              mlflowSpan.setOutputs(value);
            }
            mlflowSpan.end();
            return value;
          })
          .catch(handleSpanError);
      } else {
        // Synchronous execution
        if (mlflowSpan.outputs === undefined) {
          mlflowSpan.setOutputs(result);
        }
        mlflowSpan.end();
        return result;
      }
    } catch (error) {
      // Handle synchronous errors
      return handleSpanError(error as Error);
    }
  });
}

/**
 * Helper function to create and register an MLflow span from an OpenTelemetry span
 * @param otelSpan The OpenTelemetry span
 * @param spanType The MLflow span type
 * @param inputs Optional inputs to set on the span
 * @param attributes Optional attributes to set on the span
 * @returns The created and registered MLflow LiveSpan
 */
function createAndRegisterMlflowSpan(
  otelSpan: OTelSpan | ApiSpan,
  spanType?: SpanType,
  inputs?: any,
  attributes?: Record<string, any>
): LiveSpan {
  // Get the MLflow trace ID from the OpenTelemetry trace ID
  const otelTraceId = otelSpan.spanContext().traceId;
  const traceId =
    InMemoryTraceManager.getInstance().getMlflowTraceIdFromOtelId(otelTraceId) || otelTraceId;

  // Create the MLflow span from the OTel span
  const mlflowSpan = createMlflowSpan(otelSpan, traceId, spanType) as LiveSpan;

  // Set initial properties if provided
  if (inputs) {
    mlflowSpan.setInputs(inputs);
  }

  if (attributes) {
    mlflowSpan.setAttributes(attributes);
  }

  // Register the span with the trace manager
  const traceManager = InMemoryTraceManager.getInstance();
  traceManager.registerSpan(mlflowSpan);

  return mlflowSpan;
}

/**
 * Create a traced version of a function.
 *
 * When the function is called, the span will automatically capture:
 * - The function inputs
 * - The function outputs
 * - The function name as the span name
 * - The function execution time
 * - Any exception thrown by the function
 *
 * @param func The function to trace
 * @param options Optional trace options including name, spanType, and attributes
 * @returns The traced function
 *
 * @example
 * // With no options (uses function name as span name)
 * const tracedFunc = trace(myFunc);
 *
 * @example
 * // With options
 * const tracedFunc = trace(myFunc, { name: 'custom_span_name', spanType: 'LLM' });

 * @example
 * // Inline declaration
 * const myFunc = trace(async (a: number, b: number) => a + b, { spanType: 'TOOL' });
 */
export function trace<T extends (...args: any[]) => any>(func: T, options?: TraceOptions): T {
  // Create a wrapper function that preserves the original function's properties
  const wrapper = function (this: any, ...args: Parameters<T>): ReturnType<T> {
    const spanOptions: Omit<SpanOptions, 'parent'> = {
      name: options?.name || func.name || DEFAULT_SPAN_NAME,
      spanType: options?.spanType,
      attributes: options?.attributes,
      inputs: mapArgsToObject(func, args)
    };

    // eslint-disable-next-line @typescript-eslint/no-unsafe-return
    return withSpan((_span) => {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-return
      return func.apply(this, args);
    }, spanOptions) as ReturnType<T>;
  };

  // Preserve function properties
  Object.defineProperty(wrapper, 'length', { value: func.length });
  Object.defineProperty(wrapper, 'name', { value: func.name });

  // Copy any additional properties from the original function
  for (const prop in func) {
    if (Object.prototype.hasOwnProperty.call(func, prop)) {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
      (wrapper as any)[prop] = (func as any)[prop];
    }
  }

  return wrapper as T;
}

/**
 * Get the last active trace ID.
 * @returns The last active trace ID.
 */
export function getLastActiveTraceId(): string | undefined {
  const traceManager = InMemoryTraceManager.getInstance();
  return traceManager.lastActiveTraceId;
}

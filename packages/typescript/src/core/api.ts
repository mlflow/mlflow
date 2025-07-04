import { trace, context, Span as ApiSpan } from '@opentelemetry/api';
import { Span as OTelSpan } from '@opentelemetry/sdk-trace-node';
import { DEFAULT_SPAN_NAME, SpanType } from './constants';
import { createMlflowSpan, LiveSpan, NoOpSpan } from './entities/span';
import { getTracer } from './provider';
import { InMemoryTraceManager } from './trace_manager';
import { convertNanoSecondsToHrTime } from './utils';
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
  span_type?: SpanType;
  inputs?: any;
  attributes?: Record<string, any>;
  startTimeNs?: number;
  parent?: LiveSpan;
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
      ? trace.setSpan(context.active(), options.parent._span)
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
      options.span_type,
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
      spanOptions.span_type,
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

export function getLastActiveTraceId(): string | undefined {
  const traceManager = InMemoryTraceManager.getInstance();
  return traceManager.lastActiveTraceId;
}

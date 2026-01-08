import { trace as otelTrace, context, Span as ApiSpan, INVALID_TRACEID } from '@opentelemetry/api';
import { Span as OTelSpan } from '@opentelemetry/sdk-trace-node';
import { DEFAULT_SPAN_NAME, SpanType } from './constants';
import { createMlflowSpan, LiveSpan, NoOpSpan } from './entities/span';
import { getTracer } from './provider';
import { InMemoryTraceManager } from './trace_manager';
import { convertNanoSecondsToHrTime, mapArgsToObject } from './utils';
import { SpanStatusCode } from './entities/span_status';

/*
 * Options for starting a span
 */
export interface SpanOptions {
  /**
   * The name of the span.
   */
  name: string;

  /**
   * The type of the span, e.g., `LLM`, `TOOL`, `RETRIEVER`, etc.
   */
  spanType?: SpanType;

  /**
   * The inputs of the span.
   */
  inputs?: any;

  /**
   * The attributes of the span.
   */
  attributes?: Record<string, any>;

  /**
   * The start time of the span in nanoseconds. If not provided, the current time will be used.
   */
  startTimeNs?: number;

  /**
   * The parent span object. If not provided, the span is considered a root span.
   */
  parent?: LiveSpan;
}

/**
 * Options for tracing a function
 */
export interface TraceOptions
  extends Omit<SpanOptions, 'parent' | 'startTimeNs' | 'inputs' | 'name'> {
  /**
   * The name of the span.
   */
  name?: string;
}

/**
 * Start a new span with the given name and span type.
 *
 * This function does NOT attach the created span to the current context. To create
 * nested spans, you need to set the parent span explicitly in the `parent` field.
 * The span must be ended by calling `end` method on the returned Span object.
 *
 * @param options Optional span options (name, spanType, inputs, attributes, startTimeNs, parent)
 * @returns The created span object.
 *
 * @example
 * ```typescript
 * const span = startSpan({
 *   name: 'my_span',
 *   spanType: 'LLM',
 *   inputs: {
 *     prompt: 'Hello, world!'
 *   }
 * });
 *
 * // Do something
 *
 * // End the span
 * span.end({
 *   status: 'OK',
 *   outputs: {
 *     response: 'Hello, world!'
 *   }
 * });
 * ```
 *
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

    // SpanProcessor should have already registered the mlflow span
    return getMlflowSpan(otelSpan, options);
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
 * 1. Inline: `withSpan((span) => { ... })` - span properties set within the callback
 * 2. Options: `withSpan((span) => { ... }, { name: 'test', ... })` - span properties set via options object
 *
 * @param callback The callback function to execute within the span context
 * @param options Optional span options (name, spanType, inputs, attributes, startTimeNs)
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
    let mlflowSpan: LiveSpan | NoOpSpan;

    try {
      // SpanProcessor should have already registered the mlflow span
      mlflowSpan = getMlflowSpan(otelSpan as OTelSpan, spanOptions);
    } catch (error) {
      console.debug('Failed to create and register MLflow span', error);
      mlflowSpan = new NoOpSpan();
    }

    // Expression function to handle errors consistently
    const handleError = (error: Error): never => {
      try {
        mlflowSpan.setStatus(SpanStatusCode.ERROR, error.message);
        mlflowSpan.recordException(error);
        mlflowSpan.end();
      } catch (error) {
        console.debug('Failed to set status or record exception on MLflow span', error);
      }
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
          .catch(handleError);
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
      return handleError(error as Error);
    }
  });
}

function getMlflowSpan(otelSpan: OTelSpan, options: SpanOptions): LiveSpan | NoOpSpan {
  // MlflowSpanProcessor should have already registered the span
  const traceManager = InMemoryTraceManager.getInstance();
  const mlflowTraceId = traceManager.getMlflowTraceIdFromOtelId(otelSpan.spanContext().traceId);
  const mlflowSpan =
    traceManager.getSpan(mlflowTraceId, otelSpan.spanContext().spanId) || new NoOpSpan();

  // Set custom properties to the span
  if (options.inputs) {
    mlflowSpan.setInputs(options.inputs);
  }
  if (options.attributes) {
    mlflowSpan.setAttributes(options.attributes);
  }
  if (options.spanType) {
    mlflowSpan.setSpanType(options.spanType);
  }
  return mlflowSpan;
}

/**
 * Helper function to create and register an MLflow span from an OpenTelemetry span
 * @param otelSpan The OpenTelemetry span
 * @param spanType The MLflow span type
 * @param inputs Optional inputs to set on the span
 * @param attributes Optional attributes to set on the span
 * @returns The created and registered MLflow LiveSpan
 */
export function createAndRegisterMlflowSpan(
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
 * Create a traced version of a function or decorator for tracing class methods.
 *
 * When used as a function wrapper, the span will automatically capture:
 * - The function inputs
 * - The function outputs
 * - The function name as the span name
 * - The function execution time
 * - Any exception thrown by the function
 *
 * When used as a decorator, it preserves the `this` context for class methods.
 *
 * Note: Typescript decorator is still in experimental stage.
 *
 * @param func The function to trace (when used as function wrapper)
 * @param options Optional trace options including name, spanType, and attributes
 * @returns The traced function or method decorator
 *
 * @example
 * // Function wrapper with no options
 * const tracedFunc = trace(myFunc);
 *
 * @example
 * // Function wrapper with options
 * const tracedFunc = trace(myFunc, { name: 'custom_span_name', spanType: 'LLM' });
 *
 * @example
 * // Decorator with no options
 * class MyService {
 *   @trace()
 *   async processData(data: any) {
 *     return processedData;
 *   }
 * }
 *
 * @example
 * // Decorator with options
 * class MyService {
 *   @trace({ name: 'custom_span', spanType: SpanType.LLM })
 *   async generateText(prompt: string) {
 *     return generatedText;
 *   }
 * }
 */
export function trace(options?: TraceOptions): any;
export function trace<T extends (...args: any[]) => any>(func: T, options?: TraceOptions): T;
export function trace<T extends (...args: any[]) => any>(
  funcOrOptions?: T | TraceOptions,
  options?: TraceOptions
): any {
  // Check if this is being used as a decorator (no function provided, or options provided)
  if (typeof funcOrOptions !== 'function') {
    const decoratorOptions = funcOrOptions;

    // Return a method decorator that supports both old and new TypeScript decorator syntax
    return function (...args: any[]) {
      let originalMethod: Function;
      let methodName: string;

      // Check if using new TypeScript 5.0+ decorator syntax
      const isNewSyntax =
        args.length === 2 && typeof args[1] === 'object' && !Array.isArray(args[1]);

      if (isNewSyntax) {
        // New syntax: (originalMethod, context)
        originalMethod = args[0];
        const context = args[1] as ClassMethodDecoratorContext;
        methodName = String(context.name);
      } else {
        // Old syntax: (target, propertyKey, descriptor)
        const desc = args[2] as PropertyDescriptor;
        originalMethod = desc.value;
        methodName = String(args[1]);
      }

      // Create the traced method wrapper
      const tracedMethod = function (this: any, ...args: any[]) {
        let inputs: any;
        try {
          inputs = mapArgsToObject(originalMethod, args);
        } catch (error) {
          console.debug('Failed to map arguments values to names', error);
          inputs = args;
        }

        const spanOptions: Omit<SpanOptions, 'parent'> = {
          name: decoratorOptions?.name || originalMethod.name || methodName,
          spanType: decoratorOptions?.spanType,
          attributes: decoratorOptions?.attributes,
          inputs
        };

        // eslint-disable-next-line @typescript-eslint/no-unsafe-return
        return withSpan((_span) => {
          // Call the original method with the preserved `this` context
          // eslint-disable-next-line @typescript-eslint/no-unsafe-return
          return originalMethod.apply(this, args) as ReturnType<T>;
        }, spanOptions) as ReturnType<T>;
      };

      // Return the appropriate value based on the decorator syntax
      if (isNewSyntax) {
        return tracedMethod as T;
      } else {
        const descriptor = args[2] as PropertyDescriptor;
        descriptor.value = tracedMethod;
        // Preserve the original method's properties
        Object.defineProperty(descriptor.value, 'length', { value: originalMethod.length });
        Object.defineProperty(descriptor.value, 'name', { value: originalMethod.name });

        return descriptor;
      }
    };
  } else {
    // This is the function-based usage (existing behavior)
    return traceFunction(funcOrOptions, options);
  }
}

/**
 * Internal function to handle function-based tracing (non-decorator usage)
 */
function traceFunction<T extends (...args: any[]) => any>(func: T, options?: TraceOptions): T {
  // Create a wrapper function that preserves the original function's properties
  const wrapper = function (this: any, ...args: Parameters<T>): ReturnType<T> {
    let inputs: any;
    try {
      inputs = mapArgsToObject(func, args);
    } catch (error) {
      console.debug('Failed to map arguments values to names', error);
      inputs = args;
    }

    const spanOptions: Omit<SpanOptions, 'parent'> = {
      name: options?.name || func.name || DEFAULT_SPAN_NAME,
      spanType: options?.spanType,
      attributes: options?.attributes,
      inputs: inputs
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

/**
 * Get the current active span in the global context.
 *
 * This only works when the span is created with fluent APIs like `@trace` or
 * `withSpan`. If a span is created with the `startSpan` API, it won't be
 * attached to the global context so this function will not return it.
 *
 * @returns The current active span if exists, otherwise null.
 *
 * @example
 * ```typescript
 * const tracedFunc = trace(() => {
 *   const span = getCurrentActiveSpan();
 *   span?.setAttribute("key", "value");
 *   return 0;
 * });
 *
 * tracedFunc();
 * ```
 */
export function getCurrentActiveSpan(): LiveSpan | null {
  const otelSpan = otelTrace.getActiveSpan();

  // If no active span or it's a NonRecordingSpan, return undefined
  if (!otelSpan || otelSpan.spanContext().traceId === INVALID_TRACEID) {
    return null;
  }

  const traceManager = InMemoryTraceManager.getInstance();
  const otelTraceId = otelSpan.spanContext().traceId;
  const mlflowTraceId = traceManager.getMlflowTraceIdFromOtelId(otelTraceId);

  if (!mlflowTraceId) {
    return null;
  }

  const spanId = otelSpan.spanContext().spanId;
  return traceManager.getSpan(mlflowTraceId, spanId) || null;
}

/**
 * Options for updating the current trace
 */
export interface UpdateCurrentTraceOptions {
  /**
   * A dictionary of tags to update the trace with. Tags are designed for mutable values
   * that can be updated after the trace is created via MLflow UI or API.
   */
  tags?: Record<string, string>;

  /**
   * A dictionary of metadata to update the trace with. Metadata cannot be updated
   * once the trace is logged. It is suitable for recording immutable values like the
   * git hash of the application version that produced the trace.
   */
  metadata?: Record<string, string>;

  /**
   * Client supplied request ID to associate with the trace. This is useful for linking
   * the trace back to a specific request in your application or external system.
   */
  clientRequestId?: string;

  /**
   * A preview of the request to be shown in the Trace list view in the UI.
   * By default, MLflow will truncate the trace request naively by limiting the length.
   * This parameter allows you to specify a custom preview string.
   */
  requestPreview?: string;

  /**
   * A preview of the response to be shown in the Trace list view in the UI.
   * By default, MLflow will truncate the trace response naively by limiting the length.
   * This parameter allows you to specify a custom preview string.
   */
  responsePreview?: string;
}

/**
 * Update the current active trace with the given options.
 *
 * You can use this function either within a function decorated with `@trace` or
 * within the scope of the `withSpan` context. If there is no active trace found,
 * this function will log a warning and return.
 *
 * @param options Options for updating the trace
 *
 * @example
 * Using within a function decorated with `@trace`:
 * ```typescript
 * const myFunc = trace((x: number) => {
 *   updateCurrentTrace({ tags: { fruit: "apple" }, clientRequestId: "req-12345" });
 *   return x + 1;
 * });
 * ```
 *
 * @example
 * Using within the `withSpan` context:
 * ```typescript
 * withSpan((span) => {
 *   updateCurrentTrace({ tags: { fruit: "apple" }, clientRequestId: "req-12345" });
 * }, { name: "span" });
 * ```
 *
 * @example
 * Updating source information of the trace:
 * ```typescript
 * updateCurrentTrace({
 *   metadata: {
 *     "mlflow.trace.session": "session-4f855da00427",
 *     "mlflow.trace.user": "user-id-cc156f29bcfb",
 *     "mlflow.source.name": "inference.ts",
 *     "mlflow.source.git.commit": "1234567890",
 *     "mlflow.source.git.repoURL": "https://github.com/mlflow/mlflow"
 *   }
 * });
 * ```
 */
export function updateCurrentTrace({
  tags,
  metadata,
  clientRequestId,
  requestPreview,
  responsePreview
}: UpdateCurrentTraceOptions): void {
  const activeSpan = getCurrentActiveSpan();

  if (!activeSpan) {
    console.warn(
      'No active trace found. Please create a span using `withSpan` or ' +
        '`@trace` before calling `updateCurrentTrace`.'
    );
    return;
  }

  // Validate string parameters
  if (requestPreview !== undefined && typeof requestPreview !== 'string') {
    throw new Error('The `requestPreview` parameter must be a string.');
  }
  if (responsePreview !== undefined && typeof responsePreview !== 'string') {
    throw new Error('The `responsePreview` parameter must be a string.');
  }

  // Update trace info for the trace stored in-memory
  const traceManager = InMemoryTraceManager.getInstance();
  const trace = traceManager.getTrace(activeSpan.traceId);

  if (!trace) {
    console.warn(`Trace ${activeSpan.traceId} does not exist or already finished.`);
    return;
  }

  // Update trace info properties
  if (requestPreview !== undefined) {
    trace.info.requestPreview = requestPreview;
  }
  if (responsePreview !== undefined) {
    trace.info.responsePreview = responsePreview;
  }
  if (tags !== undefined) {
    Object.assign(trace.info.tags, tags);
  }
  if (metadata !== undefined) {
    Object.assign(trace.info.traceMetadata, metadata);
  }
  if (clientRequestId !== undefined) {
    trace.info.clientRequestId = String(clientRequestId);
  }
}

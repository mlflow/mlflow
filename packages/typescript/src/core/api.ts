import { trace, context } from '@opentelemetry/api';
import { Span as OTelSpan } from '@opentelemetry/sdk-trace-node';
import { SpanAttributeKey, SpanType } from "./constants";
import { createMlflowSpan, LiveSpan, NoOpSpan } from "./entities/span";
import { getTracer } from "./provider";
import { InMemoryTraceManager } from "./trace_manager";
import { convertNanoSecondsToHrTime } from './utils';


/**
 * Start a new span with the given name and span type.
 *
 * This function does NOT attach the created span to the current context.
 * The span must be ended by calling `end` method on the returned Span object.
 *
 * @param name The name of the span.
 * @param span_type The type of the span.
 * @param inputs The inputs of the span.
 * @param attributes The attributes of the span.
 * @param startTimeNs The start time of the span in nanoseconds.
 * @param parent The parent span object.
 */
export function startSpan(
    options: {
        name: string,
        span_type?: SpanType,
        inputs?: any,
        attributes?: Record<string, any>,
        startTimeNs?: bigint | number,
        parent?: LiveSpan
    }
): LiveSpan {
    try {
        const tracer = getTracer('default');

        // If parent is provided, use it as the parent spanAdd commentMore actions
        let parentContext = context.active();
        if (options.parent) {
            parentContext = trace.setSpan(parentContext, options.parent._span);
        }

        // Convert startTimeNs to OTel format
        const startTime = (options.startTimeNs) ? convertNanoSecondsToHrTime(options.startTimeNs) : undefined;

        const otel_span = tracer.startSpan(options.name, {startTime: startTime}, parentContext) as OTelSpan;

        // Trace ID is set to the span attribute in SpanProcessor.onStart()
        const trace_id = JSON.parse(otel_span.attributes[SpanAttributeKey.TRACE_ID] as string) as string;

        // Create the MLflow span from the OTel span
        const mlflow_span = createMlflowSpan(otel_span, trace_id, options.span_type) as LiveSpan;

        if (options.inputs) {
            mlflow_span.setInputs(options.inputs);
        }

        if (options.attributes) {
            mlflow_span.setAttributes(options.attributes);
        }

        const trace_manager = InMemoryTraceManager.getInstance();
        trace_manager.registerSpan(mlflow_span);

        return mlflow_span;
    } catch (error) {
        console.warn("Failed to start span", error);
        return new NoOpSpan();
    }
}
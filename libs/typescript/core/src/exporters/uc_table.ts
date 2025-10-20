import { Trace } from '../core/entities/trace';
import { ExportResult } from '@opentelemetry/core';
import {
  Span as OTelSpan,
  SpanProcessor,
  ReadableSpan as OTelReadableSpan,
  SpanExporter
} from '@opentelemetry/sdk-trace-base';
import { Context } from '@opentelemetry/api';
import { InMemoryTraceManager } from '../core/trace_manager';
import { TraceInfo } from '../core/entities/trace_info';
import { createTraceLocationFromUCSchema } from '../core/entities/trace_location';
import { fromOtelStatus, TraceState } from '../core/entities/trace_state';
import { SpanAttributeKey, TraceMetadataKey } from '../core/constants';
import {
  convertHrTimeToMs,
  deduplicateSpanNamesInPlace,
  aggregateUsageFromSpans
} from '../core/utils';
import { getConfig, getUCSchemaFromConfig } from '../core/config';
import { MlflowClient } from '../clients';

const TRACE_ID_V4_PREFIX = 'trace:/';

/**
 * Generate a MLflow-compatible trace ID for the given span.
 * @param span The span to generate the trace ID for
 */
function generateTraceIdV4(span: OTelSpan, ucSchema: string): string {
  // NB: the OTLP span traceId is already a hex string in the OpenTelemetry SDK
  return `${TRACE_ID_V4_PREFIX}${ucSchema}/${span.spanContext().traceId}`;
}

export class UCSchemaSpanProcessor implements SpanProcessor {
  private _exporter: SpanExporter;

  constructor(exporter: SpanExporter) {
    this._exporter = exporter;
  }

  /**
   * Called when a {@link Span} is started, if the `span.isRecording()`
   * returns true.
   * @param span the Span that just started.
   */
  onStart(span: OTelSpan, _parentContext: Context): void {
    const otelTraceId = span.spanContext().traceId;

    let traceId: string;
    const ucSchema = getUCSchemaFromConfig(getConfig());

    if (!ucSchema) {
      console.warn(`No Unity Catalog schema found. Skipping.`);
      return;
    }

    if (!span.parentSpanContext?.spanId) {
      // This is a root span
      traceId = generateTraceIdV4(span, ucSchema);
      const trace_info = new TraceInfo({
        traceId: traceId,
        traceLocation: createTraceLocationFromUCSchema(ucSchema),
        requestTime: convertHrTimeToMs(span.startTime),
        executionDuration: 0,
        state: TraceState.IN_PROGRESS,
        traceMetadata: {
          [TraceMetadataKey.SCHEMA_VERSION]: '4'
        },
        tags: {},
        assessments: []
      });
      InMemoryTraceManager.getInstance().registerTrace(otelTraceId, trace_info);
    } else {
      traceId = InMemoryTraceManager.getInstance().getMlflowTraceIdFromOtelId(otelTraceId) || '';

      if (!traceId) {
        console.warn(`No trace ID found for span ${span.name}. Skipping.`);
        return;
      }
    }

    // Set trace ID to the span
    span.setAttribute(SpanAttributeKey.TRACE_ID, traceId);
  }

  /**
   * Called when a {@link ReadableSpan} is ended, if the `span.isRecording()`
   * returns true.
   * @param span the Span that just ended.
   */
  onEnd(span: OTelReadableSpan): void {
    // Only trigger trace export for root span completion
    if (span.parentSpanContext?.spanId) {
      return;
    }

    // Update trace info
    const otelTraceId = span.spanContext().traceId;
    const traceId = InMemoryTraceManager.getInstance().getMlflowTraceIdFromOtelId(otelTraceId);
    if (!traceId) {
      console.warn(`No trace ID found for span ${span.name}. Skipping.`);
      return;
    }

    const trace = InMemoryTraceManager.getInstance().getTrace(traceId);
    if (!trace) {
      console.warn(`No trace found for span ${span.name}. Skipping.`);
      return;
    }

    this.updateTraceInfo(trace.info, span);
    deduplicateSpanNamesInPlace(Array.from(trace.spanDict.values()));

    // Aggregate token usage from all spans and add to trace metadata
    const allSpans = Array.from(trace.spanDict.values());
    const aggregatedUsage = aggregateUsageFromSpans(allSpans);
    if (aggregatedUsage) {
      trace.info.traceMetadata[TraceMetadataKey.TOKEN_USAGE] = JSON.stringify(aggregatedUsage);
    }

    this._exporter.export([span], (_) => {});
  }

  /**
   * Update the trace info with the span end time and status.
   * @param trace The trace to update
   * @param span The span to update the trace with
   */
  updateTraceInfo(traceInfo: TraceInfo, span: OTelReadableSpan): void {
    traceInfo.executionDuration = convertHrTimeToMs(span.endTime) - traceInfo.requestTime;
    traceInfo.state = fromOtelStatus(span.status.code);
  }

  /**
   * Shuts down the processor. Called when SDK is shut down. This is an
   * opportunity for processor to do any cleanup required.
   */
  async shutdown() {
    await this._exporter.shutdown();
  }

  /**
   * Forces to export all finished spans
   */
  async forceFlush() {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    await this._exporter.forceFlush!();
  }
}

export class UCSchemaSpanExporter implements SpanExporter {
  private _client: MlflowClient;
  private _pendingExports: Set<Promise<unknown>> = new Set();

  constructor(client: MlflowClient) {
    this._client = client;
  }

  export(spans: OTelReadableSpan[], _resultCallback: (result: ExportResult) => void): void {

    this.logSpans(spans);

    for (const span of spans) {
      // Only export TraceInfo when the root span is ended
      if (span.parentSpanContext?.spanId) {
        continue;
      }

      const trace = InMemoryTraceManager.getInstance().popTrace(span.spanContext().traceId);
      if (!trace) {
        console.warn(`No trace found for span ${span.name}. Skipping.`);
        continue;
      }

      this.logTraceInfo(trace);
    }
  }

  private async logSpans(spans: OTelReadableSpan[]): Promise<void> {
    const ucSchema = getUCSchemaFromConfig(getConfig());
    if (!ucSchema) {
      console.warn(`No Unity Catalog schema found. Skipping spans export.`);
      return;
    }

    const exportPromise = this._client.logSpans(ucSchema, spans);
    this._pendingExports.add(exportPromise);
    try {
      await exportPromise;
    } catch (error: unknown) {
      console.error('Failed to export spans:', error);
    } finally {
      this._pendingExports.delete(exportPromise);
    }
  }

  /**
   * Export a complete trace to the MLflow backend
   * Step 1: Create trace metadata via StartTraceV3 endpoint
   * Step 2: Upload trace data (spans) via artifact repository pattern
   */
  private async logTraceInfo(trace: Trace): Promise<void> {
    // Export trace to backend and track the promise
    const exportPromise = this._client.createTraceV4(trace.info);
    this._pendingExports.add(exportPromise);
    try {
      await exportPromise;
    } catch (error: unknown) {
      console.error(`Failed to export trace ${trace.info.traceId}:`, error);
    } finally {
      this._pendingExports.delete(exportPromise);
    }
  }

  /**
   * Force flush all pending exports (both trace info and spans).
   * Waits for all async export operations to complete.
   */
  async forceFlush(): Promise<void> {
    await Promise.all(Array.from(this._pendingExports));
    this._pendingExports.clear();
  }

  /**
   * Shutdown the exporter.
   * Waits for all pending exports to complete before shutting down.
   */
  async shutdown(): Promise<void> {
    await this.forceFlush();
  }
}

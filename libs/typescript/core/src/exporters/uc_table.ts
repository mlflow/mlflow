import { ExportResult } from '@opentelemetry/core';
import { Context } from '@opentelemetry/api';
import {
  Span as OTelSpan,
  ReadableSpan as OTelReadableSpan,
  SpanProcessor,
  SpanExporter,
} from '@opentelemetry/sdk-trace-base';
import { createAndRegisterMlflowSpan } from '../core/api';
import { MlflowClient } from '../clients';
import { SpanAttributeKey, TRACE_SCHEMA_VERSION_V4, TraceMetadataKey } from '../core/constants';
import { getConfiguredTraceMetadata, getConfiguredTraceTags } from '../core/context';
import {
  TraceDestination,
  UcSchemaDestination,
  UnityCatalogDestination,
} from '../core/destination';
import { TraceInfo } from '../core/entities/trace_info';
import {
  TraceLocation,
  TraceLocationType,
  getOtelSpansTableName,
  getUcLocationString,
} from '../core/entities/trace_location';
import { Trace } from '../core/entities/trace';
import { fromOtelStatus, TraceState } from '../core/entities/trace_state';
import { InMemoryTraceManager } from '../core/trace_manager';
import { constructTraceIdV4, parseTraceIdV4 } from '../core/utils/trace_id';
import { aggregateUsageFromSpans, convertHrTimeToMs } from '../core/utils';
import { executeOnSpanEndHooks, executeOnSpanStartHooks } from './span_processor_hooks';

/**
 * Build the TraceLocation for a UC destination.
 */
function traceLocationForDestination(destination: TraceDestination): TraceLocation {
  if (destination.kind === 'uc_table_prefix') {
    return {
      type: TraceLocationType.UC_TABLE_PREFIX,
      ucTablePrefix: { ...destination.location },
    };
  }
  return {
    type: TraceLocationType.UC_SCHEMA,
    ucSchema: { ...destination.location },
  };
}

/**
 * Span processor for Databricks Unity Catalog backed traces.
 *
 * Mirrors Python's `DatabricksUCTableSpanProcessor`:
 *   - generates V4 trace IDs (`trace:/<location>/<hex>`)
 *   - constructs TraceInfo with the UC trace location, V4 schema version, and
 *     any context-injected metadata/tags so that `updateCurrentTrace({ tags })`
 *     can mutate the in-memory record before export
 *   - delegates to a UC-aware SpanExporter for trace info + OTLP span upload
 *
 * This processor replaces the V3 `MlflowSpanProcessor` when a UC destination is
 * configured via `setDestination(...)`. Customers no longer need to inject
 * their own `BatchSpanProcessor` + OTLP exporter for UC traces.
 */
export class DatabricksUCTableSpanProcessor implements SpanProcessor {
  private _exporter: SpanExporter;
  private _destination: UnityCatalogDestination | UcSchemaDestination;

  constructor(exporter: SpanExporter, destination: UnityCatalogDestination | UcSchemaDestination) {
    this._exporter = exporter;
    this._destination = destination;
  }

  onStart(span: OTelSpan, _parentContext: Context): void {
    const otelTraceId = span.spanContext().traceId;

    if (!span.parentSpanContext?.spanId) {
      // Root span: build the V4 trace ID and TraceInfo for this trace.
      const traceLocation = traceLocationForDestination(this._destination);
      const locationString = getUcLocationString(traceLocation);
      if (!locationString) {
        // Shouldn't happen given the destination kinds, but guard anyway.
        console.warn(
          `Unable to derive UC location string for destination ${this._destination.kind}; skipping trace registration.`,
        );
        return;
      }

      const traceId = constructTraceIdV4(locationString, otelTraceId);

      const traceMetadata: Record<string, string> = {
        [TraceMetadataKey.SCHEMA_VERSION]: TRACE_SCHEMA_VERSION_V4,
      };
      const ctxMetadata = getConfiguredTraceMetadata();
      if (ctxMetadata) {
        Object.assign(traceMetadata, ctxMetadata);
      }

      const tags: Record<string, string> = {};
      const ctxTags = getConfiguredTraceTags();
      if (ctxTags) {
        Object.assign(tags, ctxTags);
      }

      const traceInfo = new TraceInfo({
        traceId,
        traceLocation,
        requestTime: convertHrTimeToMs(span.startTime),
        executionDuration: 0,
        state: TraceState.IN_PROGRESS,
        traceMetadata,
        tags,
        assessments: [],
      });
      InMemoryTraceManager.getInstance().registerTrace(otelTraceId, traceInfo);

      span.setAttribute(SpanAttributeKey.TRACE_ID, JSON.stringify(traceId));
    } else {
      const traceId =
        InMemoryTraceManager.getInstance().getMlflowTraceIdFromOtelId(otelTraceId) || '';
      if (!traceId) {
        console.warn(`No trace ID found for span ${span.name}. Skipping.`);
        return;
      }
      span.setAttribute(SpanAttributeKey.TRACE_ID, JSON.stringify(traceId));
    }

    createAndRegisterMlflowSpan(span);
    executeOnSpanStartHooks(span);
  }

  onEnd(span: OTelReadableSpan): void {
    const traceManager = InMemoryTraceManager.getInstance();
    executeOnSpanEndHooks(span);

    if (span.parentSpanContext?.spanId) {
      return;
    }

    const traceId = traceManager.getMlflowTraceIdFromOtelId(span.spanContext().traceId);
    if (!traceId) {
      console.warn(`No trace ID found for span ${span.name}. Skipping.`);
      return;
    }

    const trace = traceManager.getTrace(traceId);
    if (!trace) {
      console.warn(`No trace found for span ${span.name}. Skipping.`);
      return;
    }

    this.updateTraceInfo(trace.info, span);

    const allSpans = Array.from(trace.spanDict.values());
    const aggregatedUsage = aggregateUsageFromSpans(allSpans);
    if (aggregatedUsage) {
      trace.info.traceMetadata[TraceMetadataKey.TOKEN_USAGE] = JSON.stringify(aggregatedUsage);
    }

    this._exporter.export([span], (_) => {});
  }

  private updateTraceInfo(traceInfo: TraceInfo, span: OTelReadableSpan): void {
    traceInfo.executionDuration = convertHrTimeToMs(span.endTime) - traceInfo.requestTime;
    let state = fromOtelStatus(span.status.code);
    if (state === TraceState.STATE_UNSPECIFIED) {
      state = TraceState.OK;
    }
    traceInfo.state = state;
  }

  async shutdown(): Promise<void> {
    await this._exporter.shutdown();
  }

  async forceFlush(): Promise<void> {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    await this._exporter.forceFlush!();
  }
}

/**
 * Exporter for Databricks Unity Catalog backed traces.
 *
 * Mirrors Python's `DatabricksUCTableSpanExporter`. Two-step export per trace:
 *   1. Call the V4 `CreateTraceInfo` REST endpoint with the full `TraceInfo`
 *      (including `tags`, `trace_metadata`, state, durations). This is what
 *      makes `updateCurrentTrace({ tags })` persist as trace-level tags in
 *      `_traces_unified`, `get_trace`, `search_traces`, and the UI.
 *   2. Upload the OTel spans for this trace to `/api/2.0/otel/v1/traces` with
 *      the `X-Databricks-UC-Table-Name` header pointing to the trace's UC
 *      spans table.
 *
 * Errors in step 2 are logged but do not surface to user code, matching the
 * non-fatal behavior of `_log_spans` on the Python side.
 */
export class DatabricksUCTableSpanExporter implements SpanExporter {
  private _client: MlflowClient;
  private _pendingExports: Record<string, Promise<void>> = {};
  private _hasRaisedSpanExportError = false;

  constructor(client: MlflowClient) {
    this._client = client;
  }

  export(spans: OTelReadableSpan[], _resultCallback: (result: ExportResult) => void): void {
    for (const span of spans) {
      if (span.parentSpanContext?.spanId) {
        continue;
      }
      const traceManager = InMemoryTraceManager.getInstance();
      const trace = traceManager.popTrace(span.spanContext().traceId);
      if (!trace) {
        console.warn(`No trace found for span ${span.name}. Skipping.`);
        continue;
      }
      traceManager.lastActiveTraceId = trace.info.traceId;

      const exportPromise = this.exportTraceToBackend(trace).catch((error) => {
        console.error(`Failed to export UC trace ${trace.info.traceId}:`, error);
      });
      this._pendingExports[trace.info.traceId] = exportPromise;
    }
  }

  private async exportTraceToBackend(trace: Trace): Promise<void> {
    try {
      const [location, otelTraceId] = parseTraceIdV4(trace.info.traceId);
      if (!location || !otelTraceId) {
        throw new Error(
          `UC exporter received non-V4 trace ID ${trace.info.traceId}. This indicates the ` +
            `processor and exporter destinations are out of sync.`,
        );
      }

      // Step 1: persist the trace info (tags, metadata, state).
      const returnedTraceInfo = await this._client.createTraceInfoV4(
        location,
        otelTraceId,
        trace.info,
      );

      // The backend may populate the spans table name on the returned trace
      // location. Use that when present; fall back to the location we sent.
      const spansTable =
        getOtelSpansTableName(returnedTraceInfo.traceLocation) ??
        getOtelSpansTableName(trace.info.traceLocation);

      if (!spansTable) {
        // Without a spans table we can't upload spans, but the trace info
        // (including tags) is already persisted in step 1.
        if (!this._hasRaisedSpanExportError) {
          console.warn(
            `No OTel spans table resolved for UC trace ${trace.info.traceId}; ` +
              `spans will not be exported. Tags and metadata are still persisted.`,
          );
          this._hasRaisedSpanExportError = true;
        }
        return;
      }

      // Step 2: upload spans via OTLP.
      try {
        await this._client.exportOtlpSpansToUc(
          trace.data.spans.map((s) => s._span),
          spansTable,
        );
      } catch (error) {
        if (!this._hasRaisedSpanExportError) {
          console.warn(`Failed to export UC spans for ${trace.info.traceId}:`, error);
          this._hasRaisedSpanExportError = true;
        }
      }
    } finally {
      delete this._pendingExports[trace.info.traceId];
    }
  }

  async forceFlush(): Promise<void> {
    await Promise.all(Object.values(this._pendingExports));
    this._pendingExports = {};
  }

  async shutdown(): Promise<void> {
    await this.forceFlush();
  }
}

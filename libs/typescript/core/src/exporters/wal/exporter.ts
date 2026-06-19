import { randomUUID } from 'node:crypto';
import { ExportResult, ExportResultCode } from '@opentelemetry/core';
import { ReadableSpan as OTelReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { ProtobufTraceSerializer } from '@opentelemetry/otlp-transformer';
import { getConfig, MLflowTracingConfig } from '../../core/config';
import { InMemoryTraceManager } from '../../core/trace_manager';
import { ipcRequestByteLength, MAX_REQUEST_BYTES, submitRecord } from './ipc';
import { WalRecord } from './types';

/**
 * Serialize live OTel spans into a base64-encoded OTLP
 * `ExportTraceServiceRequest` protobuf, suitable for storing on a
 * {@link WalRecord} and later POSTing to the server's OTLP span-ingestion
 * endpoint
 */
function serializeSpansToOtlpBase64(spans: OTelReadableSpan[]): string | undefined {
  const serializable = spans.filter((s) => s?.resource != null && s?.instrumentationScope != null);
  if (serializable.length === 0) {
    return undefined;
  }
  try {
    const bytes = ProtobufTraceSerializer.serializeRequest(serializable);
    if (!bytes || bytes.length === 0) {
      return undefined;
    }
    return Buffer.from(bytes).toString('base64');
  } catch (err) {
    console.warn(
      '[mlflow][wal] Failed to serialize spans to OTLP protobuf; ' +
        'falling back to JSON artifact upload (spans will not appear in DB-backed span metrics).',
      err,
    );
    return undefined;
  }
}

/**
 * Hook-side submit function. Returns once the daemon has fsynced the
 * record to `queue.log` (ack-after-fsync). Injectable so tests can
 * swap in a stub without spinning up an actual daemon.
 */
export type SubmitRecord = (record: WalRecord) => Promise<void>;

/**
 * Asynchronous SpanExporter that decouples the user's Stop hook from
 * backend latency for MlflowSpanProcessor.
 *
 * {@link forceFlush} awaits only the IPC submits, *not* the upstream
 * upload. That gives the Stop hook the exact "trace is durable on this
 * machine, then return" semantics it needs to bound its runtime.
 */
export class MlflowWalSpanExporter implements SpanExporter {
  private _pendingSubmits: Set<Promise<void>> = new Set();
  private _submit: SubmitRecord;
  private _shuttingDown = false;

  constructor(opts: { submit?: SubmitRecord } = {}) {
    this._submit = opts.submit ?? submitRecord;
  }

  export(spans: OTelReadableSpan[], resultCallback: (result: ExportResult) => void): void {
    if (this._shuttingDown) {
      resultCallback({ code: ExportResultCode.FAILED });
      return;
    }

    let cfg: MLflowTracingConfig;
    try {
      cfg = getConfig();
    } catch (err) {
      console.warn(
        '[mlflow][wal] Tracing config unavailable; Spans will be lost until config is set.',
        err,
      );
      resultCallback({ code: ExportResultCode.FAILED });
      return;
    }

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

      const otlpSpans = serializeSpansToOtlpBase64(trace.data.spans.map((s) => s._span));

      const record: WalRecord = {
        id: randomUUID(),
        trackingUri: cfg.trackingUri,
        experimentId: cfg.experimentId,
        traceInfo: trace.info.toJson() as unknown as Record<string, unknown>,
        traceData: trace.data.toJson(),
        attempts: 0,
        nextAttemptAt: 0,
        createdAt: Date.now(),
        otlpSpans,
      };

      const fullSizeBytes = ipcRequestByteLength(record);
      if (otlpSpans !== undefined && fullSizeBytes > MAX_REQUEST_BYTES) {
        console.warn(
          `[mlflow][wal] WAL record for trace ${trace.info.traceId} is ${fullSizeBytes} bytes, exceeding the ` +
            `${MAX_REQUEST_BYTES}-byte IPC request limit; dropping OTLP spans and attempting JSON artifact upload ` +
            '(spans will not appear in DB-backed span metrics).',
        );
        record.otlpSpans = undefined;
      }

      const pending = this._submit(record).catch((err) => {
        console.error(
          `[mlflow][wal] Failed to submit WAL record for trace ${trace.info.traceId}:`,
          err,
        );
      });
      this._pendingSubmits.add(pending);

      void pending.finally(() => this._pendingSubmits.delete(pending));
    }

    resultCallback({ code: ExportResultCode.SUCCESS });
  }

  /**
   * Resolves once every IPC submit that was in flight when this method
   * was called has been acknowledged (post-fsync). Matches OTel's
   * "flush currently pending" semantic — concurrent `export()` calls
   * during the await are not waited on, and the next `forceFlush` will
   * pick them up. For "drain everything before exit" semantics, use
   * {@link shutdown}.
   */
  async forceFlush(): Promise<void> {
    await Promise.all([...this._pendingSubmits]);
  }

  /**
   * Stop accepting new exports and drain every in-flight IPC submit to
   * convergence before resolving.
   */
  async shutdown(): Promise<void> {
    this._shuttingDown = true;
    while (this._pendingSubmits.size > 0) {
      await Promise.all([...this._pendingSubmits]);
    }
  }
}

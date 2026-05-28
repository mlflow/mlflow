import { randomUUID } from 'node:crypto';
import { ExportResult, ExportResultCode } from '@opentelemetry/core';
import { ReadableSpan as OTelReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { getConfig } from '../../core/config';
import { InMemoryTraceManager } from '../../core/trace_manager';
import { submitRecord } from './ipc';
import { WalRecord } from './types';

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

  constructor(opts: { submit?: SubmitRecord } = {}) {
    this._submit = opts.submit ?? submitRecord;
  }

  export(spans: OTelReadableSpan[], resultCallback: (result: ExportResult) => void): void {
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

      const cfg = getConfig();
      const record: WalRecord = {
        id: randomUUID(),
        trackingUri: cfg.trackingUri,
        experimentId: cfg.experimentId,
        traceInfo: trace.info.toJson() as unknown as Record<string, unknown>,
        traceData: trace.data.toJson(),
        attempts: 0,
        nextAttemptAt: 0,
        createdAt: Date.now(),
      };

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
   * pick them up.
   */
  async forceFlush(): Promise<void> {
    await Promise.all([...this._pendingSubmits]);
  }

  async shutdown(): Promise<void> {
    await this.forceFlush();
  }
}

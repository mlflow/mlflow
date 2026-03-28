import type { SpanExporter, ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { translateSpansForMlflow } from './translate';

/**
 * A SpanExporter that translates Vercel AI SDK span attributes into
 * MLflow's expected format before delegating to an inner exporter.
 *
 * Wrap any OTel SpanExporter (e.g., OTLPTraceExporter) to automatically
 * enrich spans with mlflow.* attributes for span type, inputs/outputs,
 * model info, message format, and token usage.
 *
 * @example
 * ```typescript
 * import { MLflowSpanExporter } from '@mlflow/vercel';
 * import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
 *
 * const exporter = new MLflowSpanExporter(
 *   new OTLPTraceExporter({ url: '...', headers: { ... } })
 * );
 * ```
 */
export class MLflowSpanExporter implements SpanExporter {
  private readonly _inner: SpanExporter;

  constructor(innerExporter: SpanExporter) {
    this._inner = innerExporter;
  }

  export(
    spans: ReadableSpan[],
    resultCallback: (result: { code: number; error?: Error }) => void
  ): void {
    translateSpansForMlflow(spans);
    this._inner.export(spans, resultCallback);
  }

  shutdown(): Promise<void> {
    return this._inner.shutdown();
  }

  forceFlush(): Promise<void> {
    return this._inner.forceFlush?.() ?? Promise.resolve();
  }
}

import { Context } from '@opentelemetry/api';
import {
  Span,
  ReadableSpan,
  SpanProcessor,
  SpanExporter,
  BatchSpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import { translateSpanForMlflow } from './translate';

/**
 * A SpanProcessor that translates Vercel AI SDK span attributes into
 * MLflow's expected format before batching and exporting.
 *
 * Composes a {@link BatchSpanProcessor} internally so spans are batched
 * automatically — no need to wrap this processor in another one.
 *
 * @example
 * ```typescript
 * import { MLflowSpanProcessor } from '@mlflow/vercel';
 * import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
 * import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
 *
 * const provider = new NodeTracerProvider();
 * provider.addSpanProcessor(
 *   new MLflowSpanProcessor(
 *     new OTLPTraceExporter({ url: '...', headers: { ... } })
 *   )
 * );
 * provider.register();
 * ```
 */
export class MLflowSpanProcessor implements SpanProcessor {
  private readonly _processor: SpanProcessor;

  constructor(exporter: SpanExporter) {
    this._processor = new BatchSpanProcessor(exporter);
  }

  onStart(span: Span, parentContext: Context): void {
    this._processor.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    translateSpanForMlflow(span);
    this._processor.onEnd(span);
  }

  forceFlush(): Promise<void> {
    return this._processor.forceFlush();
  }

  shutdown(): Promise<void> {
    return this._processor.shutdown();
  }
}

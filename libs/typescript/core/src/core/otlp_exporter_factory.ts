import type { SpanExporter } from '@opentelemetry/sdk-trace-base';
import { OTLPTraceExporter as OTLPHttpProtoExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import type { OtlpConfig } from './otlp_config';

/**
 * Create an OTLP trace exporter that emits protobuf payloads over HTTP.
 * Currently only the http/protobuf protocol is supported. gRPC requests
 * log a warning and return null.
 */
export function createOtlpTraceExporter(config: OtlpConfig): SpanExporter | null {
  if (!config.endpoint) {
    console.warn(
      'OTLP exporter requested but no OTEL_EXPORTER_OTLP_TRACES_ENDPOINT was provided.'
    );
    return null;
  }

  if (config.protocol === 'grpc') {
    console.warn(
      'grpc protocol is not yet supported by the MLflow TypeScript SDK OTLP exporter. ' +
        'Falling back to http/protobuf.'
    );
  } else if (config.protocol !== 'http/protobuf') {
    console.warn(
      `Unsupported OTLP protocol "${config.protocol}" requested. ` +
        'Only http/protobuf is currently supported.'
    );
  }

  try {
    return new OTLPHttpProtoExporter({
      url: config.endpoint,
      headers: config.headers
    });
  } catch (error) {
    console.error('Failed to initialize OTLP trace exporter:', error);
    return null;
  }
}

import { trace, Tracer } from '@opentelemetry/api';
import { SpanProcessor } from '@opentelemetry/sdk-trace-base';
import { NodeSDK } from '@opentelemetry/sdk-node';
import { MlflowSpanExporter, MlflowSpanProcessor } from '../exporters/mlflow';
import { UCSchemaSpanExporter, UCSchemaSpanProcessor } from '../exporters/uc_table';
import { getConfig } from './config';
import { MlflowClient } from '../clients';
import { getLocationType, TraceLocationType } from './entities/trace_location';
import { tryEnableOptionalIntegrations } from './integration_loader';

let sdk: NodeSDK | null = null;
// Keep a reference to the span processor for flushing
let processor: SpanProcessor | null = null;

export function initializeSDK(): void {
  if (sdk) {
    sdk.shutdown().catch((error) => {
      console.error('Error shutting down existing SDK:', error);
    });
  }

  try {
    const hostConfig = getConfig();
    if (!hostConfig.host) {
      console.warn('MLflow tracking server not configured. Call init() before using tracing APIs.');
      return;
    }

    const client = new MlflowClient({
      trackingUri: hostConfig.trackingUri,
      host: hostConfig.host,
      databricksToken: hostConfig.databricksToken,
      trackingServerUsername: hostConfig.trackingServerUsername,
      trackingServerPassword: hostConfig.trackingServerPassword
    });

    const locationType = getLocationType(hostConfig.location);
    if (locationType === TraceLocationType.UC_SCHEMA) {
      const exporter = new UCSchemaSpanExporter(client);
      processor = new UCSchemaSpanProcessor(exporter);
    } else {
      const exporter = new MlflowSpanExporter(client);
      processor = new MlflowSpanProcessor(exporter);
    }

    // Attempt to load optional integrations (e.g. mlflow-vercel) if installed.
    // This is required for triggering hook registeration
    void tryEnableOptionalIntegrations();

    sdk = new NodeSDK({ spanProcessors: [processor] });
    sdk.start();
  } catch (error) {
    console.error('Failed to initialize MLflow tracing:', error);
  }
}

export function getTracer(module_name: string): Tracer {
  return trace.getTracer(module_name);
}

/**
 * Force flush all pending trace exports.
 */
export async function flushTraces(): Promise<void> {
  await processor?.forceFlush();
}

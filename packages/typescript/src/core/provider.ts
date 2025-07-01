import { trace, Tracer } from '@opentelemetry/api';
import { MlflowSpanExporter, MlflowSpanProcessor } from '../exporters/mlflow';
import { NodeSDK } from '@opentelemetry/sdk-node';
import { getConfig } from './config';

let sdk: NodeSDK | null = null;

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

    const exporter = new MlflowSpanExporter(hostConfig);
    const processor = new MlflowSpanProcessor(exporter);
    sdk = new NodeSDK({ spanProcessors: [processor] });
    sdk.start();
  } catch (error) {
    console.error('Failed to initialize MLflow tracing:', error);
  }
}

export function getTracer(module_name: string): Tracer {
  return trace.getTracer(module_name);
}

import { trace, Tracer } from '@opentelemetry/api';
import { MlflowSpanExporter, MlflowSpanProcessor } from '../exporters/mlflow';
import { NodeSDK } from '@opentelemetry/sdk-node';
import { getConfig } from './config';
import { MlflowClient } from '../clients';

let sdk: NodeSDK | null = null;
// Keep a reference to the span processor for flushing
let processor: MlflowSpanProcessor | null = null;

/**
 * Initialize the MLflow SDK asynchronously.
 *
 * This function handles:
 * - Shutting down any existing SDK instance
 * - Resolving the host from the auth provider for Databricks URIs
 * - Creating the MlflowClient with appropriate authentication
 * - Starting the OpenTelemetry SDK
 *
 * @internal Called by init() - not intended for direct use
 */
export async function initializeSDKAsync(): Promise<void> {
  if (sdk) {
    try {
      await sdk.shutdown();
    } catch (error) {
      console.error('Error shutting down existing SDK:', error);
    }
  }

  try {
    const hostConfig = getConfig();
    let host = hostConfig.host;

    // For Databricks URIs, resolve host from auth provider if not explicitly set
    if (!host && hostConfig.authProvider?.getHost) {
      try {
        host = await hostConfig.authProvider.getHost();
      } catch (error) {
        // Wrap SDK error with user-friendly message
        throw new Error(
          `Databricks authentication could not be discovered.\n\n` +
            `Configure one of:\n` +
            `  - Set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables\n` +
            `  - Configure Databricks CLI: databricks configure --profile DEFAULT\n` +
            `  - Set DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET for service principals\n\n` +
            `Original error: ${(error as Error).message}`
        );
      }
    }

    if (!host) {
      console.warn(
        'MLflow tracking server not configured. Call init() before using tracing APIs.'
      );
      return;
    }

    const client = new MlflowClient({
      trackingUri: hostConfig.trackingUri,
      host,
      authProvider: hostConfig.authProvider,
      databricksToken: hostConfig.databricksToken,
      trackingServerUsername: hostConfig.trackingServerUsername,
      trackingServerPassword: hostConfig.trackingServerPassword
    });
    const exporter = new MlflowSpanExporter(client);
    processor = new MlflowSpanProcessor(exporter);

    sdk = new NodeSDK({ spanProcessors: [processor] });
    sdk.start();
  } catch (error) {
    console.error('Failed to initialize MLflow tracing:', error);
    throw error; // Re-throw so ensureInitialized() can propagate the error
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

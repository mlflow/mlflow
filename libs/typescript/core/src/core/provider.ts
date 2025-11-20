import { trace, Tracer } from '@opentelemetry/api';
import { MlflowSpanExporter, MlflowSpanProcessor } from '../exporters/mlflow';
import { NodeSDK } from '@opentelemetry/sdk-node';
import type { SpanProcessor } from '@opentelemetry/sdk-trace-base';
import { getConfig } from './config';
import { MlflowClient } from '../clients';
import { tryEnableOptionalIntegrations } from './integration_loader';

type PrivateSpanProcessorHost = {
  _spanProcessors?: SpanProcessor[];
};

type PrivateTracerProvider = {
  _activeSpanProcessor?: PrivateSpanProcessorHost;
  _delegate?: unknown;
  getDelegate?: () => unknown;
};

let sdk: NodeSDK | null = null;
// Keep a reference to the span processor for flushing
let processor: MlflowSpanProcessor | null = null;
let sharedProvider: PrivateTracerProvider | null = null;

export function initializeSDK(): void {
  detachProcessorFromSharedProvider();

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
    const exporter = new MlflowSpanExporter(client);
    const newProcessor = new MlflowSpanProcessor(exporter);
    processor = newProcessor;
    // Attempt to load optional integrations (e.g. mlflow-vercel) if installed.
    // This is required for triggering hook registeration
    void tryEnableOptionalIntegrations();

    if (shouldUseIsolatedTracerProvider()) {
      startNodeSdk(newProcessor);
    } else {
      shutdownNodeSdk();
      if (!attachProcessorToGlobalProvider(newProcessor)) {
        console.warn(
          'MLflow tracing is enabled but no attachable OpenTelemetry tracer provider was found. ' +
            'Set MLFLOW_USE_DEFAULT_TRACER_PROVIDER=true to let MLflow manage the provider.'
        );
      }
    }
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

function shouldUseIsolatedTracerProvider(): boolean {
  const envValue = process.env.MLFLOW_USE_DEFAULT_TRACER_PROVIDER;
  if (!envValue) {
    return true;
  }
  return !['false', '0', 'off'].includes(envValue.toLowerCase());
}

function startNodeSdk(spanProcessor: MlflowSpanProcessor): void {
  if (sdk) {
    sdk.shutdown().catch((error) => {
      console.error('Error shutting down existing SDK:', error);
    });
  }

  sdk = new NodeSDK({ spanProcessors: [spanProcessor] });
  sdk.start();
}

function shutdownNodeSdk(): void {
  if (!sdk) {
    return;
  }
  sdk.shutdown().catch((error) => {
    console.error('Error shutting down existing SDK:', error);
  });
  sdk = null;
}

function attachProcessorToGlobalProvider(spanProcessor: MlflowSpanProcessor): boolean {
  const provider = resolveAttachableProvider();
  if (!provider) {
    return false;
  }

  const multiSpanProcessor = provider._activeSpanProcessor;
  if (!multiSpanProcessor) {
    return false;
  }

  if (!multiSpanProcessor._spanProcessors) {
    multiSpanProcessor._spanProcessors = [];
  }

  // Remove any stale references before attaching the fresh processor instance
  multiSpanProcessor._spanProcessors = multiSpanProcessor._spanProcessors.filter(
    (existingProcessor) => existingProcessor !== spanProcessor
  );

  multiSpanProcessor._spanProcessors.push(spanProcessor);
  sharedProvider = provider;
  return true;
}

function resolveAttachableProvider(): PrivateTracerProvider | null {
  const provider = trace.getTracerProvider() as PrivateTracerProvider;
  if (isAttachableProvider(provider)) {
    return provider;
  }

  const delegate = provider.getDelegate?.() ?? provider._delegate;
  if (isAttachableProvider(delegate)) {
    return delegate;
  }

  return null;
}

function isAttachableProvider(candidate: unknown): candidate is PrivateTracerProvider {
  if (!candidate || typeof candidate !== 'object') {
    return false;
  }
  return '_activeSpanProcessor' in candidate;
}

function detachProcessorFromSharedProvider(): void {
  if (!sharedProvider || !processor) {
    sharedProvider = null;
    return;
  }

  const multiSpanProcessor = sharedProvider._activeSpanProcessor;
  const processors = multiSpanProcessor?._spanProcessors;
  if (processors) {
    const index = processors.indexOf(processor);
    if (index !== -1) {
      processors.splice(index, 1);
    }
  }

  sharedProvider = null;
}

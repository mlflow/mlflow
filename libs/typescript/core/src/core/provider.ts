import { trace, Tracer } from '@opentelemetry/api';
import { NodeSDK } from '@opentelemetry/sdk-node';
import { BatchSpanProcessor, SpanProcessor } from '@opentelemetry/sdk-trace-base';
import { MlflowSpanExporter, MlflowSpanProcessor } from '../exporters/mlflow';
import { getConfig } from './config';
import { MlflowClient } from '../clients';
import { tryEnableOptionalIntegrations } from './integration_loader';
import { getOtlpConfig, shouldUseOtlpExporter } from './otlp_config';
import { createOtlpTraceExporter } from './otlp_exporter_factory';

type PrivateSpanProcessorHost = {
  _spanProcessors?: SpanProcessor[];
};

type PrivateTracerProvider = {
  _activeSpanProcessor?: PrivateSpanProcessorHost;
  _delegate?: unknown;
  getDelegate?: () => unknown;
};

const SHARED_PROVIDER_ATTACH_INTERVAL_MS = 1000;
const SHARED_PROVIDER_ATTACH_MAX_ATTEMPTS = 30;

let sdk: NodeSDK | null = null;
let activeSpanProcessors: SpanProcessor[] = [];
let sharedProvider: PrivateTracerProvider | null = null;
let sharedProviderAttachTimer: NodeJS.Timeout | null = null;
let pendingSharedProcessors: SpanProcessor[] | null = null;

export function initializeSDK(): void {
  detachSpanProcessorsFromSharedProvider();

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

    const spanProcessors = buildSpanProcessors(client);
    if (spanProcessors.length === 0) {
      console.warn('No span processors were configured. Traces will not be exported.');
      activeSpanProcessors = [];
      return;
    }

    activeSpanProcessors = spanProcessors;
    // Attempt to load optional integrations (e.g. mlflow-vercel) if installed.
    // This is required for triggering hook registeration
    void tryEnableOptionalIntegrations();

    if (shouldUseIsolatedTracerProvider()) {
      startNodeSdk(spanProcessors);
    } else {
      shutdownNodeSdk();
      if (!attachSpanProcessorsToGlobalProvider(spanProcessors)) {
        scheduleSharedProviderAttach(spanProcessors);
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
  await Promise.all(activeSpanProcessors.map((spanProcessor) => spanProcessor.forceFlush()));
}

function buildSpanProcessors(client: MlflowClient): SpanProcessor[] {
  const processors: SpanProcessor[] = [];
  const otlpConfig = getOtlpConfig();

  if (shouldUseOtlpExporter(otlpConfig)) {
    const otlpExporter = createOtlpTraceExporter(otlpConfig);
    if (otlpExporter) {
      processors.push(new BatchSpanProcessor(otlpExporter));
      if (!otlpConfig.dualExportEnabled) {
        return processors;
      }
    }
  } else if (otlpConfig.enableExporter && !otlpConfig.endpoint) {
    console.warn(
      'MLFLOW_ENABLE_OTLP_EXPORTER is true but no OTEL_EXPORTER_OTLP_TRACES_ENDPOINT is configured. ' +
        'Falling back to MLflow-only export.'
    );
  }

  const exporter = new MlflowSpanExporter(client);
  processors.push(new MlflowSpanProcessor(exporter));
  return processors;
}

function shouldUseIsolatedTracerProvider(): boolean {
  const envValue = process.env.MLFLOW_USE_DEFAULT_TRACER_PROVIDER;
  if (!envValue) {
    return true;
  }
  return !['false', '0', 'off'].includes(envValue.toLowerCase());
}

function startNodeSdk(spanProcessors: SpanProcessor[]): void {
  if (sdk) {
    sdk.shutdown().catch((error) => {
      console.error('Error shutting down existing SDK:', error);
    });
  }

  sdk = new NodeSDK({ spanProcessors });
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

function attachSpanProcessorsToGlobalProvider(spanProcessors: SpanProcessor[]): boolean {
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

  for (const spanProcessor of spanProcessors) {
    multiSpanProcessor._spanProcessors = multiSpanProcessor._spanProcessors.filter(
      (existingProcessor) => existingProcessor !== spanProcessor
    );
    multiSpanProcessor._spanProcessors.push(spanProcessor);
  }

  sharedProvider = provider;
  pendingSharedProcessors = null;
  cancelPendingSharedProviderAttach();
  return true;
}

function scheduleSharedProviderAttach(
  spanProcessors: SpanProcessor[],
  remainingAttempts = SHARED_PROVIDER_ATTACH_MAX_ATTEMPTS
): void {
  if (remainingAttempts <= 0) {
    console.warn(
      'MLflow tracing is enabled but no attachable OpenTelemetry tracer provider was found. ' +
        'Set MLFLOW_USE_DEFAULT_TRACER_PROVIDER=true to let MLflow manage the provider.'
    );
    return;
  }

  cancelPendingSharedProviderAttach();
  pendingSharedProcessors = spanProcessors;
  sharedProviderAttachTimer = setTimeout(() => {
    if (!pendingSharedProcessors) {
      return;
    }

    if (!attachSpanProcessorsToGlobalProvider(pendingSharedProcessors)) {
      scheduleSharedProviderAttach(spanProcessors, remainingAttempts - 1);
    }
  }, SHARED_PROVIDER_ATTACH_INTERVAL_MS);
}

function cancelPendingSharedProviderAttach(): void {
  if (sharedProviderAttachTimer) {
    clearTimeout(sharedProviderAttachTimer);
    sharedProviderAttachTimer = null;
  }
}

function resolveAttachableProvider(): PrivateTracerProvider | null {
  const providerCandidate = trace.getTracerProvider();
  if (isAttachableProvider(providerCandidate)) {
    return providerCandidate;
  }

  const provider = providerCandidate as PrivateTracerProvider;
  const delegate = provider?.getDelegate?.() ?? provider?._delegate;
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

function detachSpanProcessorsFromSharedProvider(): void {
  if (!sharedProvider || activeSpanProcessors.length === 0) {
    sharedProvider = null;
    cancelPendingSharedProviderAttach();
    activeSpanProcessors = [];
    return;
  }

  const multiSpanProcessor = sharedProvider._activeSpanProcessor;
  const processors = multiSpanProcessor?._spanProcessors;
  if (processors) {
    for (const spanProcessor of activeSpanProcessors) {
      const index = processors.indexOf(spanProcessor);
      if (index !== -1) {
        processors.splice(index, 1);
      }
    }
  }

  sharedProvider = null;
  cancelPendingSharedProviderAttach();
  activeSpanProcessors = [];
}

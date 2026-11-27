import { existsSync } from 'node:fs';
import {
  context,
  diag,
  propagation,
  trace,
  ProxyTracerProvider,
  TextMapPropagator,
  Tracer,
  TracerProvider,
} from '@opentelemetry/api';
import { AsyncLocalStorageContextManager } from '@opentelemetry/context-async-hooks';
import {
  CompositePropagator,
  getStringListFromEnv,
  W3CBaggagePropagator,
  W3CTraceContextPropagator,
} from '@opentelemetry/core';
import { B3InjectEncoding, B3Propagator } from '@opentelemetry/propagator-b3';
import {
  defaultResource,
  detectResources,
  envDetector,
  hostDetector,
  osDetector,
  processDetector,
  ResourceDetector,
  serviceInstanceIdDetector,
} from '@opentelemetry/resources';
import { SpanProcessor } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { MlflowClient } from '../clients';
import { MlflowSpanExporter, MlflowSpanProcessor } from '../exporters/mlflow';
import {
  DatabricksUCTableSpanExporter,
  DatabricksUCTableSpanProcessor,
} from '../exporters/uc_table';
import { MlflowWalSpanExporter } from '../exporters/wal';
import { resolveDaemonEntry } from '../exporters/wal/supervisor';
import { getConfig, getAuthProvider, type UnityCatalogLocationOptions } from './config';
import type { UnityCatalogLocation } from './entities/trace_location';
import { asyncExportEnabled } from './utils/env';

let provider: NodeTracerProvider | null = null;
let globalRegistration: TracerProvider | null = null;
let globalPropagator: TextMapPropagator | null = null;
// Keep a reference to the active span processor for flushing.
let processor: SpanProcessor | null = null;
const noopProvider = new ProxyTracerProvider();

/**
 * Initialize the OpenTelemetry SDK and span processor.
 *
 * When `config.traceLocation` is provided, wires up the V4 Databricks Unity
 * Catalog span processor + exporter so that `updateCurrentTrace` tags persist
 * on UC-backed traces. Otherwise installs the V3 experiment-backed processor.
 *
 * TODO: Auto-resolve UC location from the linked Databricks experiment's
 * `mlflow.experiment.databricksTrace*` tags so customers don't have to pass
 * `traceLocation` explicitly. This needs `GetExperiment` to run, and Node
 * has no idiomatic way to do a synchronous HTTP request from `init()`
 * (the documented `worker_threads` + `Atomics.wait` pattern is for CPU-bound
 * work, not for converting async I/O into sync). The likely path is a
 * buffering processor that queues spans until the async fetch resolves;
 * until that lands, customers configure UC explicitly via `traceLocation`.
 */
export function initializeSDK(): void {
  if (globalRegistration && trace.getTracerProvider() === globalRegistration) {
    trace.disable();
  }
  globalRegistration = null;
  if (globalPropagator) {
    propagation.disable();
    globalPropagator = null;
  }
  if (provider) {
    provider.shutdown().catch((error) => {
      console.error('Error shutting down existing SDK:', error);
    });
  }

  provider = null;
  processor = null;
  if (process.env.OTEL_SDK_DISABLED?.trim().toLowerCase() === 'true') {
    return;
  }

  try {
    const config = getConfig();
    const authProvider = getAuthProvider();

    const client = new MlflowClient({
      trackingUri: config.trackingUri,
      authProvider,
    });

    if (config.traceLocation) {
      const ucLocation = resolveUcLocation(config.traceLocation);
      const ucExporter = new DatabricksUCTableSpanExporter(client);
      processor = new DatabricksUCTableSpanProcessor(ucExporter, ucLocation);
    } else {
      let useWal = false;
      if (asyncExportEnabled()) {
        // WAL path: the hook submits records to a daemon over IPC; the
        // bundled daemon performs auth setup, fsyncs the record to
        // `queue.log`, and uploads in a separate process. If the daemon
        // entry can't be resolved (broken plugin install, missing bundle
        // copy), fall back to synchronous export so traces are uploaded
        // instead of silently dropped.
        try {
          useWal = existsSync(resolveDaemonEntry());
        } catch {
          useWal = false;
        }
      }

      if (useWal) {
        processor = new MlflowSpanProcessor(new MlflowWalSpanExporter());
      } else {
        if (asyncExportEnabled()) {
          console.warn(
            '[mlflow] MLFLOW_ENABLE_ASYNC_TRACE_LOGGING is enabled, but the trace-upload ' +
              'daemon could not be resolved; falling back to synchronous trace export.',
          );
        }
        processor = new MlflowSpanProcessor(new MlflowSpanExporter(client));
      }
    }

    provider = new NodeTracerProvider({
      resource: defaultResource().merge(detectResources({ detectors: getResourceDetectors() })),
      spanProcessors: [processor],
    });
    ensureContextManager();
    if (config.registerGlobalTracerProvider !== false) {
      provider.register({ contextManager: null, propagator: null });
      const global = trace.getTracerProvider();
      if (global instanceof ProxyTracerProvider && global.getDelegate() === provider) {
        globalRegistration = global;
        registerPropagator();
      } else {
        console.warn(
          '[mlflow] Global tracer registration failed. MLflow will use its own provider.',
        );
      }
    }
  } catch (error) {
    provider = null;
    processor = null;
    console.error('Failed to initialize MLflow tracing:', error);
  }
}

/**
 * Validate the user-supplied UC location and return it as a
 * `UnityCatalogLocation`. All three fields are required; the SDK does not
 * upsert UC trace locations, so we cannot default any of them.
 */
function resolveUcLocation(options: UnityCatalogLocationOptions): UnityCatalogLocation {
  if (!options.catalogName || !options.schemaName || !options.tablePrefix) {
    throw new Error(
      'traceLocation requires catalogName, schemaName, and tablePrefix. The UC ' +
        'trace location must already be provisioned in the workspace.',
    );
  }
  return {
    catalogName: options.catalogName,
    schemaName: options.schemaName,
    tablePrefix: options.tablePrefix,
  };
}

// Same names and default as NodeSDK, which MLflow used before.
const RESOURCE_DETECTORS = new Map<string, ResourceDetector>([
  ['env', envDetector],
  ['host', hostDetector],
  ['os', osDetector],
  ['serviceinstance', serviceInstanceIdDetector],
  ['process', processDetector],
]);

function getResourceDetectors(): ResourceDetector[] {
  const names = getStringListFromEnv('OTEL_NODE_RESOURCE_DETECTORS');
  if (!names?.length) {
    return [envDetector, processDetector, hostDetector];
  }
  if (names.includes('all')) {
    return [...RESOURCE_DETECTORS.values()];
  }
  if (names.includes('none')) {
    return [];
  }
  return names.flatMap((name) => {
    const detector = RESOURCE_DETECTORS.get(name);
    if (!detector) {
      diag.warn(`Invalid resource detector "${name}" in OTEL_NODE_RESOURCE_DETECTORS`);
      return [];
    }
    return [detector];
  });
}

const PROPAGATORS = new Map<string, () => TextMapPropagator>([
  ['tracecontext', () => new W3CTraceContextPropagator()],
  ['baggage', () => new W3CBaggagePropagator()],
  ['b3', () => new B3Propagator()],
  ['b3multi', () => new B3Propagator({ injectEncoding: B3InjectEncoding.MULTI_HEADER })],
]);

/**
 * Register the global propagator from `OTEL_PROPAGATORS` like NodeSDK did,
 * defaulting to W3C trace context and baggage. Keep a host propagator if one
 * is already registered.
 */
function registerPropagator(): void {
  const names = [
    ...new Set(getStringListFromEnv('OTEL_PROPAGATORS') ?? ['tracecontext', 'baggage']),
  ];
  const propagators = names.flatMap((name) => {
    const create = PROPAGATORS.get(name);
    if (!create) {
      if (name !== 'none') {
        diag.warn(`Propagator "${name}" in OTEL_PROPAGATORS is unavailable`);
      }
      return [];
    }
    return [create()];
  });
  if (propagators.length === 0) {
    return;
  }
  const propagator =
    propagators.length === 1 ? propagators[0] : new CompositePropagator({ propagators });
  if (propagation.setGlobalPropagator(propagator)) {
    globalPropagator = propagator;
  }
}

/**
 * Keep the host context manager if one is already registered.
 */
function ensureContextManager(): void {
  const contextManager = new AsyncLocalStorageContextManager();
  contextManager.enable();
  if (!context.setGlobalContextManager(contextManager)) {
    contextManager.disable();
  }
}

export function getTracer(module_name: string): Tracer {
  return (provider ?? noopProvider).getTracer(module_name);
}

export function isMlflowGlobalTracerProvider(): boolean {
  return globalRegistration != null && trace.getTracerProvider() === globalRegistration;
}

/**
 * Force flush all pending trace exports.
 *
 * In WAL mode this only awaits on-disk durability of the WAL appends,
 * not the upstream HTTP upload — that's the daemon's job and the
 * caller (e.g. the Claude Code Stop hook) is intentionally decoupled
 * from backend latency.
 */
export async function flushTraces(): Promise<void> {
  await processor?.forceFlush();
}

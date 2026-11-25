import { existsSync } from 'node:fs';
import { context, trace, Tracer, TracerProvider } from '@opentelemetry/api';
import { AsyncLocalStorageContextManager } from '@opentelemetry/context-async-hooks';
import { NodeSDK } from '@opentelemetry/sdk-node';
import { BasicTracerProvider, SpanProcessor } from '@opentelemetry/sdk-trace-base';
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

let sdk: NodeSDK | BasicTracerProvider | null = null;
// Set only when MLflow's tracer provider is not the global one.
let isolatedProvider: TracerProvider | null = null;
// Keep a reference to the active span processor for flushing.
let processor: SpanProcessor | null = null;

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
  if (sdk) {
    sdk.shutdown().catch((error) => {
      console.error('Error shutting down existing SDK:', error);
    });
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

    if (config.registerGlobalTracerProvider === false) {
      const provider = new BasicTracerProvider({ spanProcessors: [processor] });
      sdk = provider;
      isolatedProvider = provider;
      ensureContextManager();
    } else {
      isolatedProvider = null;
      const nodeSdk = new NodeSDK({ spanProcessors: [processor] });
      sdk = nodeSdk;
      nodeSdk.start();
    }
  } catch (error) {
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

/**
 * `withSpan` and `getCurrentActiveSpan` rely on a context manager. NodeSDK
 * registers one together with the global tracer provider, so the isolated
 * path has to do it itself. Registration fails when the host process already
 * has a context manager, and that one is used instead.
 */
function ensureContextManager(): void {
  const contextManager = new AsyncLocalStorageContextManager();
  contextManager.enable();
  if (!context.setGlobalContextManager(contextManager)) {
    contextManager.disable();
  }
}

export function getTracer(module_name: string): Tracer {
  return (isolatedProvider ?? trace).getTracer(module_name);
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

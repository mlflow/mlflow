import { trace, Tracer } from '@opentelemetry/api';
import { NodeSDK } from '@opentelemetry/sdk-node';
import { SpanProcessor } from '@opentelemetry/sdk-trace-base';
import { MlflowClient } from '../clients';
import { MlflowSpanExporter, MlflowSpanProcessor } from '../exporters/mlflow';
import {
  DatabricksUCTableSpanExporter,
  DatabricksUCTableSpanProcessor,
} from '../exporters/uc_table';
import { MlflowWalSpanExporter } from '../exporters/wal';
import { getConfig, getAuthProvider, type UnityCatalogLocationOptions } from './config';
import type { UnityCatalogLocation } from './entities/trace_location';

let sdk: NodeSDK | null = null;
// Keep a reference to the active span processor for flushing.
let processor: SpanProcessor | null = null;

/**
 * Whether to route trace export through the asynchronous WAL +
 * daemon path instead of the legacy synchronous HTTP exporter
 * for the MlflowSpanProcessor.
 */
export function asyncExportEnabled(): boolean {
  const raw = process.env.MLFLOW_ENABLE_ASYNC_TRACE_LOGGING;
  if (raw === undefined || raw === '') {
    return false;
  }
  const lower = raw.toLowerCase();
  return lower === '1' || lower === 'true';
}

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
      if (asyncExportEnabled()) {
        // WAL path: the hook submits records to a daemon over IPC; the
        // bundled daemon performs the actual auth setup, fsyncs the
        // record to `queue.log`, and runs the HTTP upload in a separate
        // process. The IPC client spawns the daemon on first use, so no
        // client construction or supervisor wiring is needed here.
        const mlflowWalExporter = new MlflowWalSpanExporter();
        processor = new MlflowSpanProcessor(mlflowWalExporter);
      } else {
        const exporter = new MlflowSpanExporter(client);
        processor = new MlflowSpanProcessor(exporter);
      }
    }

    sdk = new NodeSDK({ spanProcessors: [processor] });
    sdk.start();
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

export function getTracer(module_name: string): Tracer {
  return trace.getTracer(module_name);
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

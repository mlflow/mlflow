import { trace, Tracer } from '@opentelemetry/api';
import { MlflowSpanExporter, MlflowSpanProcessor } from '../exporters/mlflow';
import {
  DatabricksUCTableSpanExporter,
  DatabricksUCTableSpanProcessor,
} from '../exporters/uc_table';
import { NodeSDK } from '@opentelemetry/sdk-node';
import { SpanProcessor } from '@opentelemetry/sdk-trace-base';
import { getConfig, getAuthProvider } from './config';
import {
  getDestination,
  setDestination,
  TraceDestination,
  ucSchemaDestination,
} from './destination';
import { MlflowClient } from '../clients';

let sdk: NodeSDK | null = null;
// Keep a reference to the active span processor for flushing.
let processor: SpanProcessor | null = null;

/**
 * Parse the `MLFLOW_TRACING_DESTINATION` environment variable into a UC
 * trace destination. Matches Python: only the two-segment `catalog.schema`
 * form is accepted as a UC schema destination. To target a UC table prefix,
 * call `setDestination(unityCatalogDestination({...}))` explicitly (Python's
 * equivalent is `set_experiment(trace_location=UnityCatalog(...))`).
 */
function destinationFromEnv(): TraceDestination | null {
  const raw = process.env.MLFLOW_TRACING_DESTINATION;
  if (!raw) {
    return null;
  }
  const parts = raw.split('.');
  if (parts.length === 2 && parts[0] && parts[1]) {
    return ucSchemaDestination({ catalogName: parts[0], schemaName: parts[1] });
  }
  if (parts.length === 3) {
    throw new Error(
      `MLFLOW_TRACING_DESTINATION=${raw}: UC table-prefix destinations ` +
        '(<catalog>.<schema>.<table_prefix>) are not supported via this env var. ' +
        'Use setDestination(unityCatalogDestination({ catalogName, schemaName, tablePrefix })) ' +
        'before init() instead.',
    );
  }
  throw new Error(
    `MLFLOW_TRACING_DESTINATION=${raw} could not be parsed. ` +
      'Expected format: <catalog>.<schema>.',
  );
}

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

    // Resolve UC destination in precedence order:
    //   1. explicit setDestination(...)
    //   2. MLFLOW_TRACING_DESTINATION env var
    // For auto-resolution from a linked Databricks experiment, customers
    // must call `await resolveDestinationFromExperiment(client, experimentId)`
    // and pass the result to `setDestination(...)` before `init(...)`.
    // We can't do that here because `init()` is sync and GetExperiment is async.
    let destination = getDestination();
    if (!destination) {
      const envDestination = destinationFromEnv();
      if (envDestination) {
        setDestination(envDestination);
        destination = envDestination;
      }
    }

    if (destination) {
      const ucExporter = new DatabricksUCTableSpanExporter(client);
      processor = new DatabricksUCTableSpanProcessor(ucExporter, destination);
    } else {
      const exporter = new MlflowSpanExporter(client);
      processor = new MlflowSpanProcessor(exporter);
    }

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

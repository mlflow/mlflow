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
  unityCatalogDestination,
  ucSchemaDestination,
} from './destination';
import { MlflowClient } from '../clients';

let sdk: NodeSDK | null = null;
// Keep a reference to the active span processor for flushing.
let processor: SpanProcessor | null = null;

/**
 * Parse the `MLFLOW_UC_TABLE` environment variable into a TraceDestination.
 * Accepts "catalog.schema" (UC schema) or "catalog.schema.table_prefix"
 * (UC table prefix). Returns null when the env var is unset or malformed.
 */
function destinationFromEnv(): TraceDestination | null {
  const raw = process.env.MLFLOW_UC_TABLE;
  if (!raw) {
    return null;
  }
  const parts = raw.split('.');
  if (parts.length === 2) {
    return ucSchemaDestination({ catalogName: parts[0], schemaName: parts[1] });
  }
  if (parts.length === 3) {
    return unityCatalogDestination({
      catalogName: parts[0],
      schemaName: parts[1],
      tablePrefix: parts[2],
    });
  }
  console.warn(
    `MLFLOW_UC_TABLE=${raw} is not in "catalog.schema" or "catalog.schema.table_prefix" format; ignoring.`,
  );
  return null;
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

    // Resolve UC destination: explicit setDestination() wins over MLFLOW_UC_TABLE.
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

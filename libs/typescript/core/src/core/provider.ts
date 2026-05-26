import { trace, Tracer } from '@opentelemetry/api';
import { MlflowSpanExporter, MlflowSpanProcessor } from '../exporters/mlflow';
import {
  DatabricksUCTableSpanExporter,
  DatabricksUCTableSpanProcessor,
} from '../exporters/uc_table';
import { NodeSDK } from '@opentelemetry/sdk-node';
import { SpanProcessor } from '@opentelemetry/sdk-trace-base';
import { getConfig, getAuthProvider } from './config';
import { ucLocationFromExperimentTags } from './destination';
import { isDatabricksUri } from '../auth';
import { MlflowClient } from '../clients';
import type { UnityCatalogLocation } from './entities/trace_location';

let sdk: NodeSDK | null = null;
// Keep a reference to the active span processor for flushing.
let processor: SpanProcessor | null = null;

/**
 * Initialize the OpenTelemetry SDK and span processor.
 *
 * For Databricks tracking URIs, this fetches the experiment via the
 * `GetExperiment` API and, when the experiment is linked to a Unity Catalog
 * trace destination via the `mlflow.experiment.databricksTrace*` tags,
 * wires up the V4 UC span processor + exporter so that `updateCurrentTrace`
 * tags persist on UC-backed traces.
 *
 * Mirrors the Python `mlflow.tracing.provider.set_destination` +
 * `mlflow.set_experiment` flow.
 */
export async function initializeSDK(): Promise<void> {
  if (sdk) {
    await sdk.shutdown().catch((error) => {
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

    const ucLocation = await resolveUcLocation(client, config.trackingUri, config.experimentId);

    if (ucLocation) {
      const ucExporter = new DatabricksUCTableSpanExporter(client);
      processor = new DatabricksUCTableSpanProcessor(ucExporter, ucLocation);
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

async function resolveUcLocation(
  client: MlflowClient,
  trackingUri: string,
  experimentId: string,
): Promise<UnityCatalogLocation | null> {
  if (!isDatabricksUri(trackingUri)) {
    return null;
  }
  try {
    const experiment = await client.getExperiment(experimentId);
    if (!experiment) {
      return null;
    }
    return ucLocationFromExperimentTags(experiment.tags);
  } catch (error) {
    console.warn(
      `Failed to resolve Unity Catalog trace location for experiment ${experimentId}:`,
      error,
    );
    return null;
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

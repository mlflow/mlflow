import { context, SpanStatusCode, trace } from '@opentelemetry/api';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { openai } from '@ai-sdk/openai';
import { generateText } from 'ai';
import {
  MlflowClient,
  constructTraceIdV4,
  createAuthProvider,
  createTraceLocationFromUcTablePrefix,
  getUcLocationString,
} from '@mlflow/core';
import { MLflowSpanProcessor } from '../src';

const MAX_OUTPUT_TOKENS = 16;

function requiredEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(`Missing required environment variable ${name}.`);
  }
  return value;
}

async function main(): Promise<void> {
  requiredEnv('OPENAI_API_KEY');
  const catalog = requiredEnv('DATABRICKS_UC_CATALOG');
  const schema = requiredEnv('DATABRICKS_UC_SCHEMA');
  const tablePrefix = requiredEnv('DATABRICKS_UC_TABLE_PREFIX');
  const model = process.env.VERCEL_SMOKE_MODEL?.trim() || 'gpt-5-mini';
  const profile = process.env.DATABRICKS_CONFIG_PROFILE?.trim();
  const trackingUri = profile ? `databricks://${profile}` : 'databricks';
  const authProvider = createAuthProvider({
    trackingUri,
    host: process.env.DATABRICKS_HOST?.trim(),
    databricksToken: process.env.DATABRICKS_TOKEN?.trim(),
  });
  const host = authProvider.getHost().replace(/\/$/, '');
  const authHeaders = await authProvider.getHeadersProvider()();
  const otlpAuthHeaders = Object.fromEntries(
    Object.entries(authHeaders).filter(([key]) => key.toLowerCase() !== 'content-type'),
  );

  const traceLocation = createTraceLocationFromUcTablePrefix(catalog, schema, tablePrefix);
  const location = getUcLocationString(traceLocation);
  if (!location) {
    throw new Error('Failed to construct the Unity Catalog trace location.');
  }
  const spansTable = process.env.DATABRICKS_UC_OTEL_SPANS_TABLE?.trim() || `${location}_otel_spans`;
  if (!traceLocation.ucTablePrefix) {
    throw new Error('Unity Catalog trace location is missing its table-prefix details.');
  }
  traceLocation.ucTablePrefix.otelSpansTableName = spansTable;

  const client = new MlflowClient({
    trackingUri,
    authProvider,
  });
  const exporter = new OTLPTraceExporter({
    url: `${host}/api/2.0/otel/v1/traces`,
    headers: {
      ...otlpAuthHeaders,
      'X-Databricks-UC-Table-Name': spansTable,
    },
  });
  const traceInfoErrors: unknown[] = [];
  const mlflowProcessor = new MLflowSpanProcessor(exporter, {
    traceInfo: {
      client,
      traceLocation,
      onTraceInfoError: (error, traceId) => {
        traceInfoErrors.push(error);
        console.error(`TraceInfo write failed for ${traceId}:`, error);
      },
    },
  });
  const provider = new NodeTracerProvider({
    spanProcessors: [mlflowProcessor],
  });
  provider.register();

  const tracer = provider.getTracer('mlflow-vercel-databricks-uc-smoke');
  const root = tracer.startSpan('two-real-vercel-ai-calls');
  const mlflowTraceId = constructTraceIdV4(location, root.spanContext().traceId);

  try {
    await context.with(trace.setSpan(context.active(), root), async () => {
      await generateText({
        model: openai(model),
        prompt: 'Reply with exactly one word: red',
        maxOutputTokens: MAX_OUTPUT_TOKENS,
        experimental_telemetry: { isEnabled: true, tracer },
      });
      await generateText({
        model: openai(model),
        prompt: 'Reply with exactly one word: blue',
        maxOutputTokens: MAX_OUTPUT_TOKENS,
        experimental_telemetry: { isEnabled: true, tracer },
      });
    });
    root.setStatus({ code: SpanStatusCode.OK });
  } catch (error) {
    root.setStatus({ code: SpanStatusCode.ERROR });
    throw error;
  } finally {
    root.end();
    await provider.forceFlush();
    await provider.shutdown();
  }

  if (traceInfoErrors.length > 0) {
    throw new Error('One or more V4 TraceInfo writes failed.', { cause: traceInfoErrors[0] });
  }
  process.stdout.write(`Databricks UC trace exported: ${mlflowTraceId}\n`);
  process.stdout.write(
    JSON.stringify(
      {
        model,
        spansTable,
      },
      null,
      2,
    ) + '\n',
  );
}

void main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});

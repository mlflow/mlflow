# MLflow Typescript SDK - Vercel AI

Seamlessly integrate [MLflow Tracing](https://github.com/mlflow/mlflow/tree/main/libs/typescript) with [Vercel AI SDK](https://ai-sdk.dev/) to automatically trace your AI API calls.

| Package              | NPM                                                                                                                               | Description                                         |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| [@mlflow/vercel](./) | [![npm package](https://img.shields.io/npm/v/%40mlflow%2Fvercel?style=flat-square)](https://www.npmjs.com/package/@mlflow/vercel) | Auto-instrumentation integration for Vercel AI SDK. |

## Installation

```bash
npm install @mlflow/vercel
```

The package includes `@opentelemetry/api` and `@opentelemetry/sdk-trace-base` as peer dependencies. Depending on your package manager, you may need to install them separately.

## OSS and generic OTLP setup

Start MLflow Tracking Server. If you have a local Python environment, you can run the following command:

```bash
pip install mlflow
mlflow server --port 5000
```

If you don't have Python environment locally, MLflow also supports Docker deployment or managed services. See [Self-Hosting Guide](https://mlflow.org/docs/latest/self-hosting/index.html) for getting started.

Set up the MLflow span processor and use the Vercel AI SDK with telemetry enabled:

```typescript
import { MLflowSpanProcessor } from '@mlflow/vercel';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

const provider = new NodeTracerProvider({
  spanProcessors: [
    new MLflowSpanProcessor(
      new OTLPTraceExporter({
        url: 'http://localhost:5000/api/2.0/otel/v1/traces',
        headers: {
          'x-mlflow-experiment-id': '<your-experiment-id>',
        },
      }),
    ),
  ],
});
provider.register();

const result = await generateText({
  model: openai('gpt-5'),
  prompt: "What's the weather like in Seattle?",
  experimental_telemetry: { isEnabled: true },
});
```

## Databricks Unity Catalog setup

Unity Catalog traces use two ingestion paths: OTLP exports the spans, while the MLflow V4
CreateTraceInfo endpoint persists trace-level metadata. Configure both paths so Trace and Session
token/cost columns can read precomputed totals without scanning every span.

```typescript
import {
  MlflowClient,
  createAuthProvider,
  createTraceLocationFromUcTablePrefix,
} from '@mlflow/core';
import { MLflowSpanProcessor } from '@mlflow/vercel';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';

const host = process.env.DATABRICKS_HOST!;
const token = process.env.DATABRICKS_TOKEN!;
const traceLocation = createTraceLocationFromUcTablePrefix('catalog', 'schema', 'my_agent');
const client = new MlflowClient({
  trackingUri: 'databricks',
  authProvider: createAuthProvider({
    trackingUri: 'databricks',
    host,
    databricksToken: token,
  }),
});

const provider = new NodeTracerProvider({
  spanProcessors: [
    new MLflowSpanProcessor(
      new OTLPTraceExporter({
        url: `${host}/api/2.0/otel/v1/traces`,
        headers: {
          Authorization: `Bearer ${token}`,
          'X-Databricks-UC-Table-Name': 'catalog.schema.my_agent_otel_spans',
        },
      }),
      {
        traceInfo: {
          client,
          traceLocation,
          onTraceInfoError: (error, traceId) => {
            console.error(`Failed to persist TraceInfo for ${traceId}`, error);
          },
        },
      },
    ),
  ],
});
provider.register();
```

Install `@mlflow/core` alongside `@mlflow/vercel` for this setup. The OTLP exporter requires:

- `Authorization`: `Bearer <your-databricks-token>`
- `X-Databricks-UC-Table-Name`: `<catalog>.<schema>.<table_prefix>_otel_spans`

Note: Do not set the `x-mlflow-experiment-id` header when using Databricks.

V4 TraceInfo persistence is enabled only when `traceInfo` configuration is supplied. The processor
keeps bounded in-process state and finalizes a trace after its root and every locally observed active
child have ended. `forceFlush()` and `shutdown()` wait for pending TraceInfo writes. On shutdown, a
trace with an ended root is persisted best-effort from its ended spans; rootless incomplete traces are
discarded and reported through `onTraceInfoError`.

For this Databricks path, the processor also estimates missing per-span costs from the actual model
and token usage using the pricing snapshot bundled with `@mlflow/core`. Explicit `mlflow.llm.cost`
attributes always win. Unknown models remain without cost metadata. This is a catalog estimate rather
than an authoritative provider invoice.

### Databricks UC smoke test

The opt-in smoke test makes two real Vercel AI SDK `generateText()` calls through OpenAI, exports
their actual spans to UC, writes V4 TraceInfo, flushes both ingestion paths, and prints the resulting
trace ID for inspection in Databricks. It does not add synthetic attributes or assert metric values.
OpenAI's API response supplies the real token usage; the UC processor uses those counts and MLflow's
bundled model pricing to populate `mlflow.trace.cost`.

```bash
export OPENAI_API_KEY="..."
export DATABRICKS_UC_CATALOG="catalog"
export DATABRICKS_UC_SCHEMA="schema"
export DATABRICKS_UC_TABLE_PREFIX="my_agent"

# Authenticate with a Databricks CLI profile. DATABRICKS_HOST and
# DATABRICKS_TOKEN are not required when the profile contains the workspace.
databricks auth login --host "https://your-workspace.databricks.com" --profile mlflow-smoke
export DATABRICKS_CONFIG_PROFILE="mlflow-smoke"

cd libs/typescript
npm run -C integrations/vercel smoke:databricks-uc
```

Run the smoke test in a standalone Node process so its tracer provider can be registered globally.
Set `VERCEL_SMOKE_MODEL` to override the default `gpt-5-mini` model, or
`DATABRICKS_UC_OTEL_SPANS_TABLE` when the provisioned spans table does not follow the default naming
convention.

For PAT authentication instead, omit `DATABRICKS_CONFIG_PROFILE` and set `DATABRICKS_HOST` plus
`DATABRICKS_TOKEN`. Both the V4 client and OTLP exporter use the same resolved authentication.

On macOS networks that install a trusted TLS-inspection certificate in the System keychain, export
the public certificates for Node before running the smoke test:

```bash
security find-certificate -a -p /Library/Keychains/System.keychain \
  > /private/tmp/mlflow-node-system-ca.pem
export NODE_EXTRA_CA_CERTS=/private/tmp/mlflow-node-system-ca.pem
```

Do not disable TLS verification with `NODE_TLS_REJECT_UNAUTHORIZED=0`.

## Attribute Translation

The Vercel AI SDK emits spans with `ai.*` attributes. `MLflowSpanProcessor` translates these into MLflow's format:

| Vercel AI SDK                                | MLflow                                     | Description                      |
| -------------------------------------------- | ------------------------------------------ | -------------------------------- |
| `ai.operationId`                             | `mlflow.spanType`                          | Span type (LLM, TOOL, EMBEDDING) |
| `ai.prompt.*` / `ai.response.*`              | `mlflow.spanInputs` / `mlflow.spanOutputs` | Structured request/response data |
| `ai.model.id`                                | `mlflow.llm.model`                         | Model name                       |
| `ai.model.provider`                          | `mlflow.llm.provider`                      | Provider name                    |
| `ai.usage.promptTokens` / `completionTokens` | `mlflow.chat.tokenUsage`                   | Token usage for cost tracking    |
| (chat spans)                                 | `mlflow.message.format` = `"vercel_ai"`    | Enables chat UI rendering        |

## Documentation

- [MLflow Tracing](https://mlflow.org/docs/latest/llms/tracing/index.html)
- [Vercel AI SDK Telemetry](https://ai-sdk.dev/docs/ai-sdk-core/telemetry)
- [Databricks OTEL Collector](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog)

## License

This project is licensed under the [Apache License 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt).

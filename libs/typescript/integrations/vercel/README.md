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

## Quickstart

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

## Databricks

To send traces to a Databricks Unity Catalog table, set the OTLP exporter URL to `<DATABRICKS_HOST>/api/2.0/otel/v1/traces` and include the following headers:

- `Authorization`: `Bearer <your-databricks-token>`
- `X-Databricks-UC-Table-Name`: `<catalog>.<schema>.<table_prefix>_otel_spans`

Note: Do not set the `x-mlflow-experiment-id` header when using Databricks.

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

# @mlflow/vercel

Vercel AI SDK integration for [MLflow Tracing](https://mlflow.org/). Provides a `SpanExporter` that translates Vercel AI SDK span attributes into MLflow's expected format.

## Installation

```bash
npm install @mlflow/vercel
```

### Peer Dependencies

- `@opentelemetry/sdk-trace-base` (or `@opentelemetry/sdk-trace-node`, which re-exports it)

## Quick Start

```typescript
import { MLflowSpanExporter } from '@mlflow/vercel';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import { BatchSpanProcessor, NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

// 1. Create an OTLP exporter pointed at the Databricks OTEL collector
const otlpExporter = new OTLPTraceExporter({
  url: `${process.env.DATABRICKS_HOST}/api/2.0/otel/v1/traces`,
  headers: {
    Authorization: `Bearer ${process.env.DATABRICKS_TOKEN}`,
    'X-Databricks-UC-Table-Name': 'catalog.schema.spans',
  },
});

// 2. Wrap it with MLflowSpanExporter to translate attributes
const exporter = new MLflowSpanExporter(otlpExporter);

// 3. Register with OpenTelemetry
const provider = new NodeTracerProvider();
provider.addSpanProcessor(new BatchSpanProcessor(exporter));
provider.register();

// 4. Use Vercel AI SDK with telemetry enabled
const result = await generateText({
  model: openai('gpt-4'),
  prompt: 'Hello!',
  experimental_telemetry: { isEnabled: true },
});
```

## What It Does

The Vercel AI SDK emits spans with `ai.*` attributes (e.g., `ai.operationId`, `ai.prompt.messages`, `ai.model.id`). MLflow expects `mlflow.*` attributes for full feature support.

`MLflowSpanExporter` wraps any OTel `SpanExporter` and translates these attributes before export:

| Vercel AI SDK | MLflow | Description |
|---|---|---|
| `ai.operationId` | `mlflow.spanType` | Span type (LLM, TOOL, EMBEDDING) |
| `ai.prompt.*` / `ai.response.*` | `mlflow.spanInputs` / `mlflow.spanOutputs` | Structured request/response data |
| `ai.model.id` | `mlflow.llm.model` | Model name |
| `ai.model.provider` | `mlflow.llm.provider` | Provider name |
| `ai.usage.promptTokens` / `completionTokens` | `mlflow.chat.tokenUsage` | Token usage for cost tracking |
| (chat spans) | `mlflow.message.format` = `"vercel_ai"` | Enables chat UI rendering |

Spans without `ai.operationId` (e.g., HTTP spans) pass through unchanged. Existing `mlflow.*` attributes are never overwritten.

> **Note:** "Call LLM" spans (doGenerate/doStream) also carry native `gen_ai.*` attributes set by the AI SDK. The MLflow server's read path handles those. This exporter adds `mlflow.*` attributes for features not covered by `gen_ai.*` (span type, structured I/O, message format) and for all non-LLM span types (tools, embeddings) which lack `gen_ai.*` attributes entirely.

## API

### `MLflowSpanExporter`

```typescript
import { MLflowSpanExporter } from '@mlflow/vercel';

const exporter = new MLflowSpanExporter(innerExporter);
```

Implements the OTel `SpanExporter` interface. Translates Vercel AI SDK attributes, then delegates `export()`, `shutdown()`, and `forceFlush()` to the inner exporter.

### `translateSpansForMlflow`

```typescript
import { translateSpansForMlflow } from '@mlflow/vercel';

translateSpansForMlflow(spans);
```

The raw translation function, exported for advanced use cases. Mutates span attributes in-place. No span is ever dropped — translation is best-effort per span.

## Documentation

- [MLflow Tracing](https://mlflow.org/docs/latest/llms/tracing/index.html)
- [Vercel AI SDK Telemetry](https://ai-sdk.dev/docs/ai-sdk-core/telemetry)
- [Databricks OTEL Collector](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog)

## License

Apache-2.0

# MLflow Typescript SDK - Anthropic

Seamlessly integrate [MLflow Tracing](https://github.com/mlflow/mlflow/tree/main/libs/typescript) with Anthropic to automatically trace your Claude API calls and Claude Agent SDK interactions.

| Package                | NPM                                                                                                                                               | Description                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| [mlflow-anthropic](./) | [![npm package](https://img.shields.io/npm/v/mlflow-tracing-anthropic?style=flat-square)](https://www.npmjs.com/package/mlflow-tracing-anthropic) | Auto-instrumentation integration for Anthropic. |

## Features

- **`tracedAnthropic`**: Wrapper for tracing the standard Anthropic SDK client
- **`createTracedQuery`**: Wrapper for the Claude Agent SDK with automatic AGENT and TOOL span creation

## Installation

```bash
npm install mlflow-anthropic
```

The package includes the [`mlflow-tracing`](https://github.com/mlflow/mlflow/tree/main/libs/typescript) package and `@anthropic-ai/sdk` package as peer dependencies. Depending on your package manager, you may need to install these two packages separately.

For Claude Agent SDK support, also install:

```bash
npm install @anthropic-ai/claude-agent-sdk
```

## Quickstart

Start MLflow Tracking Server if you don't have one already:

```bash
pip install mlflow
mlflow server --backend-store-uri sqlite:///mlruns.db --port 5000
```

Self-hosting MLflow server requires Python 3.10 or higher. If you don't have one, you can also use [managed MLflow service](https://mlflow.org/#get-started) for free to get started quickly.

Instantiate MLflow SDK in your application:

```typescript
import * as mlflow from 'mlflow-tracing';

mlflow.init({
  trackingUri: 'http://localhost:5000',
  experimentId: '<experiment-id>',
});
```

### Tracing Anthropic SDK (Basic)

Create a trace for Anthropic Claude messages:

```typescript
import Anthropic from '@anthropic-ai/sdk';
import { tracedAnthropic } from 'mlflow-anthropic';

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const client = tracedAnthropic(anthropic);

const response = await client.messages.create({
  model: 'claude-3-7-sonnet-20250219',
  max_tokens: 256,
  messages: [{ role: 'user', content: 'Hello Claude' }],
});
```

### Tracing Claude Agent SDK

Use `createTracedQuery()` to wrap the Claude Agent SDK's `query` function.

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';
import { createTracedQuery } from 'mlflow-anthropic';
import * as mlflow from 'mlflow-tracing';

mlflow.init({
  trackingUri: 'http://localhost:5000',
  experimentId: '123',
});

const tracedQuery = createTracedQuery(query);

const result = tracedQuery({
  prompt: 'List files in this directory',
  options: {
    tools: ['Bash', 'Read', 'Glob'],
  },
});

for await (const message of result) {
  if (message.type === 'result') {
    console.log('Result:', message.result);
  }
}
```

## Documentation

Official documentation for MLflow Typescript SDK can be found [here](https://mlflow.org/docs/latest/genai/tracing/quickstart).

## License

This project is licensed under the [Apache License 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt).

# MLflow Typescript SDK - Anthropic

Seamlessly integrate [MLflow Tracing](https://github.com/mlflow/mlflow/tree/main/libs/typescript) with Anthropic to automatically trace your Claude API calls.

| Package                | NPM                                                                                                                                               | Description                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| [mlflow-anthropic](./) | [![npm package](https://img.shields.io/npm/v/mlflow-tracing-anthropic?style=flat-square)](https://www.npmjs.com/package/mlflow-tracing-anthropic) | Auto-instrumentation integration for Anthropic. |

## Installation

```bash
npm install mlflow-anthropic
```

The package includes the [`mlflow-tracing`](https://github.com/mlflow/mlflow/tree/main/libs/typescript) package and `@anthropic-ai/sdk` package as peer dependencies. Depending on your package manager, you may need to install these two packages separately.

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

Create a trace for Anthropic Claude:

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

View traces in MLflow UI:

![MLflow Tracing UI](https://github.com/mlflow/mlflow/blob/master/docs/static/images/llms/anthropic/anthropic-tracing.png?raw=True)

## Documentation 📘

Official documentation for MLflow Typescript SDK can be found [here](https://mlflow.org/docs/latest/genai/tracing/quickstart).

## License

This project is licensed under the [Apache License 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt).

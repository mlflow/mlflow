# MLflow Typescript SDK - Vercel AI SDK

Seamlessly integrate [MLflow Tracing](https://github.com/mlflow/mlflow/tree/main/libs/typescript) with Vercel AI SDK to automatically trace your LLM calls via the AI SDK.

| Package             | NPM                                                                                                                         | Description                                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| [mlflow-vercel](./) | [![npm package](https://img.shields.io/npm/v/mlflow-vercel?style=flat-square)](https://www.npmjs.com/package/mlflow-vercel) | Auto-instrumentation integration for Vercel AI SDK. |

## Installation

```bash
npm install mlflow-vercel
```

The package includes the [`mlflow-tracing`](https://github.com/mlflow/mlflow/tree/main/libs/typescript) package and `ai` package as peer dependencies. Depending on your package manager, you may need to install these two packages separately.

## Quickstart

Start MLflow Tracking Server. If you have a local Python environment, you can run the following command:

```bash
pip install mlflow
mlflow server --backend-store-uri sqlite:///mlruns.db --port 5000
```

If you don't have Python environment locally, MLflow also supports Docker deployment or managed services. See [Self-Hosting Guide](https://mlflow.org/docs/latest/self-hosting/index.html) for getting started.

Instantiate MLflow SDK in your application:

```typescript
import * as mlflow from 'mlflow-tracing';

mlflow.init({
  trackingUri: 'http://localhost:5000',
  experimentId: '<experiment-id>'
});
```

Call the AI SDK as usual. Importantly, set the `experimental_telemetry` flag to `true` to enable tracing.

```typescript
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

const result = await generateText({
  model: openai('gpt-4o-mini'),
  prompt: 'What is mlflow?',
  // IMPORTANT
  experimental_telemetry: { isEnabled: true }
});
```

View traces in MLflow UI:

![MLflow Tracing UI](https://github.com/mlflow/mlflow/blob/891fed9a746477f808dd2b82d3abb2382293c564/docs/static/images/llms/tracing/quickstart/single-openai-trace-detail.png?raw=true)

## Documentation 📘

Official documentation for MLflow Typescript SDK can be found [here](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/typescript-sdk).

## License

This project is licensed under the [Apache License 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt).

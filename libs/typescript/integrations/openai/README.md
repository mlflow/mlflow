# MLflow Typescript SDK - OpenAI

Seamlessly integrate [MLflow Tracing](https://github.com/mlflow/mlflow/tree/main/libs/typescript) with OpenAI to automatically trace your OpenAI API calls.

| Package             | NPM                                                                                                                                         | Description                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| [mlflow-openai](./) | [![npm package](https://img.shields.io/npm/v/mlflow-tracing-openai?style=flat-square)](https://www.npmjs.com/package/mlflow-tracing-openai) | Auto-instrumentation integration for OpenAI. |

## Installation

```bash
npm install mlflow-openai
```

The package includes the [`mlflow-tracing`](https://github.com/mlflow/mlflow/tree/main/libs/typescript) package and `openai` package as peer dependencies. Depending on your package manager, you may need to install these two packages separately.

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

Create a trace:

```typescript
import { OpenAI } from 'openai';
import { tracedOpenAI } from 'mlflow-openai';

// Wrap the OpenAI client with the tracedOpenAI function
const client = tracedOpenAI(new OpenAI());

// Invoke the client as usual
const response = await client.chat.completions.create({
  model: 'o4-mini',
  messages: [
    { role: 'system', content: 'You are a helpful weather assistant.' },
    { role: 'user', content: "What's the weather like in Seattle?" }
  ]
});
```

View traces in MLflow UI:

![MLflow Tracing UI](https://github.com/mlflow/mlflow/blob/891fed9a746477f808dd2b82d3abb2382293c564/docs/static/images/llms/tracing/quickstart/single-openai-trace-detail.png?raw=true)

## Documentation 📘

Official documentation for MLflow Typescript SDK can be found [here](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/typescript-sdk).

## License

This project is licensed under the [Apache License 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt).

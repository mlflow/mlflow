# MLflow Typescript SDK - OpenAI

Seamlessly integrate [MLflow Tracing](https://github.com/mlflow/mlflow/tree/main/packages/typescript) with OpenAI to automatically trace your OpenAI API calls.

|Package|NPM|Description|
|---|---|---|
|[mlflow-openai](./packages/typescript/integrations/openai)|[![npm package](https://img.shields.io/npm/v/mlflow-tracing-openai?style=flat-square)](https://www.npmjs.com/package/mlflow-tracing-openai) |Auto-instrumentation integration for OpenAI.|

## Installation

```bash
npm install mlflow-openai
```

The package already includes the [mlflow-tracing](https://github.com/mlflow/mlflow/tree/main/packages/typescript) package as a dependency, so you don't need to install it separately.

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
  experimentId: '<experiment-id>'
});
```

Create a trace:

```typescript
import { OpenAI } from "openai";
import { tracedOpenAI } from "mlflow-openai";

// Wrap the OpenAI client with the tracedOpenAI function
const client = tracedOpenAI(new OpenAI());

// Invoke the client as usual
const response = await client.chat.completions.create({
    model: "o4-mini",
    messages: [
        {"role": "system", "content": "You are a helpful weather assistant."},
        {"role": "user", "content": "What's the weather like in Seattle?"},
    ],
})
```

View traces in MLflow UI:

![MLflow Tracing UI](https://github.com/mlflow/mlflow/blob/891fed9a746477f808dd2b82d3abb2382293c564/docs/static/images/llms/tracing/quickstart/single-openai-trace-detail.png?raw=true)

## Documentation 📘

Official documentation for MLflow Typescript SDK can be found [here](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/typescript-sdk).

## License

This project is licensed under the [Apache License 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt).
# MLflow Typescript SDK - Gemini

Seamlessly integrate [MLflow Tracing](https://github.com/mlflow/mlflow/tree/main/libs/typescript) with Gemini to automatically trace your Claude API calls.

| Package             | NPM                                                                                                                                         | Description                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| [mlflow-gemini](./) | [![npm package](https://img.shields.io/npm/v/mlflow-tracing-gemini?style=flat-square)](https://www.npmjs.com/package/mlflow-tracing-gemini) | Auto-instrumentation integration for Gemini. |

## Installation

```bash
npm install mlflow-gemini
```

The package includes the [`mlflow-tracing`](https://github.com/mlflow/mlflow/tree/main/libs/typescript) package and `@google/genai` package as peer dependencies. Depending on your package manager, you may need to install these two packages separately.

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

Create a trace for Gemini:

```typescript
import { tracedGemini } from 'mlflow-gemini';
import { GoogleGenAI } from '@google/genai';

const gemini = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const client = tracedGemini(gemini);

const response = await client.models.generateContent({
  model: 'gemini-2.0-flash-001',
  contents: 'Hello Gemini'
});
```

View traces in MLflow UI:

![MLflow Tracing UI](https://github.com/mlflow/mlflow/blob/master/docs/static/images/llms/gemini/gemini-tracing.png?raw=True)

## Documentation 📘

Official documentation for MLflow Typescript SDK can be found [here](https://mlflow.org/docs/latest/genai/tracing/quickstart).

## License

This project is licensed under the [Apache License 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt).

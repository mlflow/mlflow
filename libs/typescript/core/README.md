# MLflow Typescript SDK - Core

This is the core package of the [MLflow Typescript SDK](https://github.com/mlflow/mlflow/tree/main/libs/typescript). It is a skinny package that includes the core tracing functionality and manual instrumentation.

| Package              | NPM                                                                                                                           | Description                                                |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [mlflow-tracing](./) | [![npm package](https://img.shields.io/npm/v/mlflow-tracing?style=flat-square)](https://www.npmjs.com/package/mlflow-tracing) | The core tracing functionality and manual instrumentation. |

## Installation

```bash
npm install mlflow-tracing
```

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
// Wrap a function with mlflow.trace to generate a span when the function is called.
// MLflow will automatically record the function name, arguments, return value,
// latency, and exception information to the span.
const getWeather = mlflow.trace(
  (city: string) => {
    return `The weather in ${city} is sunny`;
  },
  // Pass options to set span name. See https://mlflow.org/docs/latest/genai/tracing/quickstart
  // for the full list of options.
  { name: 'get-weather' }
);
getWeather('San Francisco');

// Alternatively, start and end span manually
const span = mlflow.startSpan({ name: 'my-span' });
span.end();
```

### Search Traces

Use `MlflowClient` to search for traces programmatically:

```typescript
import { MlflowClient } from 'mlflow-tracing';

const client = new MlflowClient({
  trackingUri: 'http://localhost:5000',
  host: 'http://localhost:5000'
});

// Search traces with filter
const result = await client.searchTraces({
  experimentIds: ['1'],
  filterString: "trace.status = 'OK'",
  maxResults: 100,
  orderBy: ['timestamp_ms DESC']
});

// Access the traces
for (const trace of result.traces) {
  console.log(trace.traceId, trace.state);
  console.log('Request:', trace.requestPreview);
  console.log('Response:', trace.responsePreview);
}

// Pagination
let pageToken = result.nextPageToken;
while (pageToken) {
  const nextPage = await client.searchTraces({
    experimentIds: ['1'],
    pageToken
  });
  // Process nextPage.traces...
  pageToken = nextPage.nextPageToken;
}
```

## Documentation 📘

Official documentation for MLflow Typescript SDK can be found [here](https://mlflow.org/docs/latest/genai/tracing/quickstart).

## License

This project is licensed under the [Apache License 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt).

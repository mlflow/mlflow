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
  // Pass options to set span name. See https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/typescript-sdk
  // for the full list of options.
  { name: 'get-weather' }
);
getWeather('San Francisco');

// Alternatively, start and end span manually
const span = mlflow.startSpan({ name: 'my-span' });
span.end();
```

## Environment configuration

Set the following environment variables before your application starts so the
SDK can authenticate with MLflow and annotate traces consistently:

| Variable                  | Description                                                                                         |
|---------------------------|-----------------------------------------------------------------------------------------------------|
| `MLFLOW_TRACKING_URI`     | Base URL for your MLflow tracking server (for example `http://localhost:5001` or `databricks`).      |
| `MLFLOW_EXPERIMENT_ID`    | Experiment that should receive the emitted traces.                                                  |
| `OTEL_RESOURCE_ATTRIBUTES`| Optional resource metadata (comma-delimited `key=value` pairs) applied to every span/trace.          |
| `OTEL_SERVICE_NAME`       | Optional override for the OpenTelemetry resource service name reported in spans.                    |

### Optional: OTLP dual export

The SDK can forward spans to an OTLP collector in addition to MLflow. Configure the
variables below to enable that workflow:

| Variable                              | Description                                                                                                       |
|---------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `MLFLOW_ENABLE_OTLP_EXPORTER`         | Enables the OTLP exporter (`true` by default).                                                                    |
| `MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT`| When `true`, emit spans to both MLflow REST and OTLP. When `false`, OTLP replaces the MLflow destination.          |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`  | Collector endpoint for OTLP traces (for example `http://localhost:4318/v1/traces`).                                |
| `OTEL_EXPORTER_OTLP_HEADERS`          | Optional comma-delimited headers such as `Authorization=Bearer <token>`.                                          |
| `OTEL_EXPORTER_OTLP_PROTOCOL` / `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` | Set to `http/protobuf` (default) or `grpc`.                                   |


## Documentation 📘

Official documentation for MLflow Typescript SDK can be found [here](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/typescript-sdk).

## License

This project is licensed under the [Apache License 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt).

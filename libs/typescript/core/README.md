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

### Using with Databricks

The SDK supports Databricks authentication through the Databricks SDK's unified auth mechanism. This enables automatic credential discovery from environment variables, config files, and cloud provider authentication.

#### Environment Variables

The SDK reads the following environment variables automatically:

| Variable | Description |
|----------|-------------|
| `MLFLOW_TRACKING_URI` | Set to `"databricks"` or `"databricks://profile-name"` |
| `MLFLOW_EXPERIMENT_ID` | The experiment ID for logging traces |
| `DATABRICKS_HOST` | Databricks workspace URL |
| `DATABRICKS_TOKEN` | Personal Access Token (PAT) |
| `DATABRICKS_CLIENT_ID` | OAuth client ID (for service principals) |
| `DATABRICKS_CLIENT_SECRET` | OAuth client secret (for service principals) |

#### Databricks Apps (Zero Configuration)

On Databricks Apps, the environment variables are pre-configured automatically. Just initialize with the experiment ID:

```typescript
import * as mlflow from 'mlflow-tracing';

// Everything else is auto-discovered from environment!
mlflow.init({
  experimentId: '<experiment-id>'
});

// Or if MLFLOW_EXPERIMENT_ID is also set in environment:
mlflow.init({});
```

The following environment variables are automatically set on Databricks Apps:
- `MLFLOW_TRACKING_URI="databricks"`
- `DATABRICKS_HOST`
- `DATABRICKS_CLIENT_ID`
- `DATABRICKS_CLIENT_SECRET`

#### Local Development with Databricks CLI

If you have the [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html) configured (`~/.databrickscfg`), the SDK will automatically use your credentials:

```bash
# Set environment variable
export MLFLOW_TRACKING_URI="databricks"
```

```typescript
import * as mlflow from 'mlflow-tracing';

// Uses DEFAULT profile from ~/.databrickscfg
mlflow.init({
  experimentId: '<experiment-id>'
});

// Or use a named profile
mlflow.init({
  trackingUri: 'databricks://my-profile',
  experimentId: '<experiment-id>'
});
```

#### Credential Discovery Order

The SDK automatically discovers credentials from (in order):

1. **Environment variables**: `DATABRICKS_HOST` + `DATABRICKS_TOKEN` or `DATABRICKS_CLIENT_ID`/`DATABRICKS_CLIENT_SECRET`
2. **Config file**: `~/.databrickscfg` (uses `DEFAULT` profile or the profile specified in the tracking URI)
3. **Cloud provider auth**: Azure CLI, GCP credentials, etc.

#### Explicit Configuration

You can also provide credentials explicitly (overrides auto-discovery):

```typescript
mlflow.init({
  trackingUri: 'databricks',
  experimentId: '<experiment-id>',
  host: 'https://your-workspace.databricks.com',
  databricksToken: 'your-token'
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

## Documentation 📘

Official documentation for MLflow Typescript SDK can be found [here](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/typescript-sdk).

## License

This project is licensed under the [Apache License 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt).

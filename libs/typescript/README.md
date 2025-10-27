<h1 align="center" style="border-bottom: none">
    <div>
        <a href="https://mlflow.org/"><picture>
            <img alt="MLflow Logo" src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/logo.svg" width="200" />
        </picture></a>
        <br>
        MLflow TypeScript SDK
    </div>
</h1>
<h2 align="center" style="border-bottom: none"></h2>

<p align="center">
  <a href="https://github.com/mlflow/mlflow"><img src="https://img.shields.io/github/stars/mlflow/mlflow?style=social" alt="stars"></a>
  <a href="https://www.npmjs.com/package/mlflow-tracing"><img src="https://img.shields.io/npm/v/mlflow-tracing.svg" alt="version"></a>
  <a href="https://www.npmjs.com/package/mlflow-tracing"><img src="https://img.shields.io/npm/dt/mlflow-tracing.svg" alt="downloads"></a>
  <a href="https://github.com/mlflow/mlflow/blob/main/LICENSE"><img src="https://img.shields.io/github/license/mlflow/mlflow" alt="license"></a>
</p>

MLflow Typescript SDK is a variant of the [MLflow Python SDK](https://github.com/mlflow/mlflow) that provides a TypeScript API for MLflow.

> [!IMPORTANT]
> MLflow Typescript SDK is catching up with the Python SDK. Currently only support [Tracing]() and [Feedback Collection]() features. Please raise an issue in Github if you need a feature that is not supported.

## Packages

| Package                                | NPM                                                                                                                                         | Description                                                |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [mlflow-tracing](./core)               | [![npm package](https://img.shields.io/npm/v/mlflow-tracing?style=flat-square)](https://www.npmjs.com/package/mlflow-tracing)               | The core tracing functionality and manual instrumentation. |
| [mlflow-openai](./integrations/openai) | [![npm package](https://img.shields.io/npm/v/mlflow-tracing-openai?style=flat-square)](https://www.npmjs.com/package/mlflow-tracing-openai) | Auto-instrumentation integration for OpenAI.               |

## Installation

```bash
npm install mlflow-tracing
```

> [!NOTE]
> MLflow Typescript SDK requires Node.js 20 or higher.

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

### Configure with environment variables

The SDK can also read configuration from environment variables so you can avoid
hard-coding connection details. If `MLFLOW_TRACKING_URI` and
`MLFLOW_EXPERIMENT_ID` are set, you can initialize the client without passing
any arguments:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_ID=123456789
```

```typescript
import * as mlflow from 'mlflow-tracing';

mlflow.init(); // Uses the values from the environment
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

View traces in MLflow UI:

![MLflow Tracing UI](https://github.com/mlflow/mlflow/blob/891fed9a746477f808dd2b82d3abb2382293c564/docs/static/images/llms/tracing/quickstart/openai-tool-calling-trace-detail.png?raw=true)

## Publishing

1. Run `yarn bump-version --version <new_version>` from this directory to bump the package versions appropriately
2. `cd` into `core` and run `npm publish`, and repeat for `integrations/openai`

## Adding New Integrations

The TypeScript SDK supports pluggable auto-instrumentation packages under [`integrations/`](./integrations). To add a new integration:

1. Create a new workspace package (for example, `integrations/<provider>`), modeled after the [OpenAI integration](./integrations/openai).
2. Implement the instrumentation entry points in `src/`, exporting a `register()` helper that configures tracing for the target client library.
3. Add package metadata (`package.json`, `tsconfig.json`, and optional `README.md`) so the integration can be built and published.
4. Add unit and/or integration tests under `tests/` that exercise the new instrumentation.
5. Update the root [`package.json`](./package.json) `build:integrations` and `test:integrations` scripts if your package requires additional build or test commands.

Once your integration package is ready, run the local workflow outlined in [Running the SDK after changes](#running-the-sdk-after-changes) and open a pull request that describes the new provider support.

## Contributing

We welcome contributions of new features, bug fixes, and documentation improvements. To contribute:

1. Review the project-wide [contribution guidelines](../../CONTRIBUTING.md) and follow the MLflow [Code of Conduct](../../CODE_OF_CONDUCT.rst).
2. Discuss larger proposals in a GitHub issue or the MLflow community channels before investing significant effort.
3. Fork the repository (or use a feature branch) and make your changes with clear, well-structured commits.
4. Ensure your code includes tests and documentation updates where appropriate.
5. Submit a pull request that summarizes the motivation, implementation details, and validation steps. The MLflow team will review and provide feedback.

## Running the SDK after Changes

The TypeScript workspace uses npm workspaces. After modifying the core SDK or any integration:

```bash
npm install        # Install or update workspace dependencies
npm run build      # Build the core package and all integrations
npm run test       # Execute the test suites for the core SDK and integrations
```

You can run package-specific scripts from their respective directories (for example, `cd core && npm run test`) when iterating on a particular feature. Remember to rebuild before consuming the SDK from another project so that the latest TypeScript output is emitted to `dist/`.

## Trace Usage

MLflow Tracing empowers you throughout the end-to-end lifecycle of your application. Here's how it helps you at each step of the workflow, click on each section to learn more:

<details>
<summary><strong>🔍 Build & Debug</strong></summary>

<table>
<tr>
<td width="60%">

#### Smooth Debugging Experience

MLflow's tracing capabilities provide deep insights into what happens beneath the abstractions of your application, helping you precisely identify where issues occur.

[Learn more →](https://mlflow.org/docs/latest/genai/tracing/observe-with-traces)

</td>
<td width="40%">

![Trace Debug](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/static/images/llms/tracing/genai-trace-debug.png)

</td>
</tr>
</table>

</details>

<details>
<summary><strong>💬 Human Feedback</strong></summary>

<table>
<tr>
<td width="60%">

#### Track Annotation and User Feedback Attached to Traces

Collecting and managing feedback is essential for improving your application. MLflow Tracing allows you to attach user feedback and annotations directly to traces, creating a rich dataset for analysis.

This feedback data helps you understand user satisfaction, identify areas for improvement, and build better evaluation datasets based on real user interactions.

[Learn more →](https://mlflow.org/docs/latest/genai/tracing/collect-user-feedback)

</td>
<td width="40%">

![Human Feedback](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/static/images/llms/tracing/genai-human-feedback.png)

</td>
</tr>
</table>

</details>

<details>
<summary><strong>📊 Evaluation</strong></summary>

<table>
<tr>
<td width="60%">

#### Systematic Quality Assessment Throughout Your Application

Evaluating the performance of your application is crucial, but creating a reliable evaluation process can be challenging. Traces serve as a rich data source, helping you assess quality with precise metrics for all components.

When combined with MLflow's evaluation capabilities, you get a seamless experience for assessing and improving your application's performance.

[Learn more →](https://mlflow.org/docs/latest/genai/eval-monitor)

</td>
<td width="40%">

![Evaluation](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/static/images/llms/tracing/genai-trace-evaluation.png)

</td>
</tr>
</table>

</details>

<details>
<summary><strong>🚀 Production Monitoring</strong></summary>

<table>
<tr>
<td width="60%">

#### Monitor Applications with Your Favorite Observability Stack

Machine learning projects don't end with the first launch. Continuous monitoring and incremental improvement are critical to long-term success.

Integrated with various observability platforms such as Databricks, Datadog, Grafana, and Prometheus, MLflow Tracing provides a comprehensive solution for monitoring your applications in production.

[Learn more →](https://mlflow.org/docs/latest/genai/tracing/prod-tracing)

</td>
<td width="40%">

![Monitoring](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/static/images/llms/tracing/genai-monitoring.png)

</td>
</tr>
</table>

</details>

<details>
<summary><strong>📦 Dataset Collection</strong></summary>

<table>
<tr>
<td width="60%">

#### Create High-Quality Evaluation Datasets from Production Traces

Traces from production are invaluable for building comprehensive evaluation datasets. By capturing real user interactions and their outcomes, you can create test cases that truly represent your application's usage patterns.

This comprehensive data capture enables you to create realistic test scenarios, validate model performance on actual usage patterns, and continuously improve your evaluation datasets.

[Learn more →](https://mlflow.org/docs/latest/genai/tracing/search-traces#creating-evaluation-datasets)

</td>
<td width="40%">

![Dataset Collection](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/static/images/llms/tracing/genai-trace-dataset.png)

</td>
</tr>
</table>

</details>

## Documentation 📘

Official documentation for MLflow Typescript SDK can be found [here](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/typescript-sdk).

## License

This project is licensed under the [Apache License 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt).

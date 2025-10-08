# MLflow Typescript SDK - Anthropic

Seamlessly integrate [MLflow Tracing](https://github.com/mlflow/mlflow/tree/main/libs/typescript) with Anthropic to automatically trace your Claude API calls.

| Package                 | NPM                                                                                                         | Description                                       |
| ----------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| [mlflow-anthropic](./)  | [![npm package](https://img.shields.io/npm/v/mlflow-tracing-anthropic?style=flat-square)](https://www.npmjs.com/package/mlflow-tracing-anthropic) | Auto-instrumentation integration for Anthropic.  |

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
  experimentId: '<experiment-id>'
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
  messages: [
    { role: 'user', content: 'Hello Claude' }
  ]
});
```

View traces in MLflow UI:

![MLflow Tracing UI](https://github.com/mlflow/mlflow/blob/master/docs/static/images/llms/anthropic/anthropic-tracing.png?raw=True)

## End-to-End Autologging Workflow

1. **Install dependencies**
   ```bash
   npm install mlflow-tracing mlflow-anthropic @anthropic-ai/sdk
   ```

2. **Configure MLflow** by pointing the SDK at your tracking server:
   ```typescript
   import * as mlflow from 'mlflow-tracing';

   mlflow.init({
     trackingUri: process.env.MLFLOW_TRACKING_URI!,
     experimentId: process.env.MLFLOW_EXPERIMENT_ID!
   });
   ```

3. **Wrap the Anthropic client** with the auto-instrumentation helper:
   ```typescript
   import Anthropic from '@anthropic-ai/sdk';
   import { tracedAnthropic } from 'mlflow-anthropic';

   const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY! });
   const client = tracedAnthropic(anthropic);
   ```

4. **Call the Claude APIs as usual**—every traced method invocation automatically creates an MLflow span:
   ```typescript
   const completion = await client.messages.create({
     model: 'claude-3-haiku-20240307',
     max_tokens: 256,
     messages: [
       { role: 'user', content: 'Hello Claude' }
     ]
   });
   console.log(completion.content[0]);
   ```

5. **Inspect traces** in the MLflow UI to review inputs, outputs, latency, and token usage for every Anthropic request.

## Documentation 📘

Official documentation for MLflow Typescript SDK can be found [here](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/typescript-sdk).

## License

This project is licensed under the [Apache License 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt).
# MLflow Typescript SDK – LangChain

Seamlessly integrate [MLflow Tracing](https://github.com/mlflow/mlflow/tree/main/libs/typescript) with LangChain.js to automatically trace your chains, chat models, tools, retrievers, and agents.

| Package                 | NPM                                                                                                                                            | Description                                      |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| [mlflow-langchain](./)  | [![npm package](https://img.shields.io/npm/v/mlflow-langchain?style=flat-square)](https://www.npmjs.com/package/mlflow-langchain)             | Callback-based auto-instrumentation for LangChain |

## Installation

```bash
npm install mlflow-langchain
```

This package lists `mlflow-tracing` and `@langchain/core` as peer dependencies. Depending on your package manager you may need to install them explicitly:


## Quickstart

1) Start an MLflow Tracking Server (local example):

```bash
pip install mlflow
mlflow server --backend-store-uri sqlite:///mlruns.db --port 5000
```

2) Initialize MLflow in your app:

```typescript
import * as mlflow from 'mlflow-tracing';

mlflow.init({
  trackingUri: 'http://localhost:5000',
  experimentId: '<experiment-id>' // optional; can also be set via env
});
```

3) Use the `MlflowCallback` with LangChain runnables/models:

```typescript
import { ChatOpenAI } from '@langchain/openai';
import { RunnableSequence } from '@langchain/core/runnables';
import { MlflowCallback } from 'mlflow-langchain';

const handler = new MlflowCallback();

const chain = RunnableSequence.from([
  // ... your runnables (prompts, models, parsers, etc.)
]);

const result = await chain.invoke(
  { input: 'Write a haiku about the sea.' },
  { callbacks: [handler] }
);
```

Open MLflow UI and explore the trace:

![MLflow Tracing UI](https://github.com/mlflow/mlflow/blob/891fed9a746477f808dd2b82d3abb2382293c564/docs/static/images/llms/tracing/quickstart/single-openai-trace-detail.png?raw=true)

## What Gets Traced

The callback forwards LangChain lifecycle events to MLflow spans:

- Chat/LLM calls (input messages, outputs, token usage when available)
- Chains and runnables (inputs/outputs)
- Tools (inputs/outputs)
- Retrievers (query and returned documents)
- Agent actions and finishes (tool name, inputs, logs)

## API

```ts
import { MlflowCallback } from 'mlflow-langchain';
```

- `MlflowCallback` – LangChain `BaseCallbackHandler` implementation that creates child spans under the current active span (or new traces when none exist).

Advanced usage: you can call `mlflow.trace`/`mlflow.withSpan` around your LangChain code to group multiple chain/model calls into a higher-level span.

## Additional Examples

Chat model with messages:

```typescript
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { MlflowCallback } from 'mlflow-langchain';

const chat = new ChatOpenAI({ model: 'gpt-4o-mini' });
const handler = new MlflowCallback();

const res = await chat.invoke(
  [
    new SystemMessage('You are a helpful assistant.'),
    new HumanMessage('Explain observability for LLM apps in 2 lines.')
  ],
  { callbacks: [handler] }
);
```

Tool + Agent (minimal sketch):

```typescript
import { tool } from '@langchain/core/tools';
import { AgentExecutor, createToolCallingAgent } from 'langchain/agents';
import { ChatOpenAI } from '@langchain/openai';
import { MlflowCallback } from 'mlflow-langchain';

const weather = tool({
  name: 'weather',
  description: 'Get weather for a city',
  func: async (city: string) => `Sunny in ${city}`
});

const llm = new ChatOpenAI({ model: 'gpt-4o-mini' });
const agent = createToolCallingAgent({ llm, tools: [weather] });

const handler = new MlflowCallback();
await agent.invoke({ input: 'Weather in Seattle?' }, { callbacks: [handler] });
```

Retriever example:

```typescript
import { MlflowCallback } from 'mlflow-langchain';
// Assume `retriever` is a LangChain retriever instance
await retriever.invoke('vector databases vs. indexes', { callbacks: [new MlflowCallback()] });
```

## Configuration and Tips

- Node.js >= 18
- LangChain Core >= 0.3.x
- This package relies on modern TypeScript resolution for `@langchain/core` types. If you see an error like:

  > Cannot find module '@langchain/core/...'. Consider updating to 'node16', 'nodenext', or 'bundler'.

  set the following options in your `tsconfig.json` (in your app or the subpackage that compiles the LangChain code):

  ```json
  {
    "compilerOptions": {
      "module": "Node16",
      "moduleResolution": "node16"
    }
  }
  ```

- When adding custom attributes to span events, avoid `undefined` values (MLflow span event attributes are strings/numbers/booleans/arrays).

## Documentation 📘

Official documentation for the MLflow Typescript SDK is available at:

https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/typescript-sdk

## License

This project is licensed under the [Apache License 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt).

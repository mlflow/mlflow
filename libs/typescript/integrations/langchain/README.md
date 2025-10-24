# MLflow LangChain Integration

This package provides a LangChain callback handler that automatically records
LangChain.js runs as MLflow traces. It mirrors the Python autologging
integration, allowing you to capture chains, tools, retrievers, and LLM calls
without modifying application logic.

## Usage

```typescript
import { ChatOpenAI } from '@langchain/openai';
import { RunnableSequence } from '@langchain/core/runnables';
import { MlflowLangchainCallbackHandler } from 'mlflow-langchain';
import { init } from 'mlflow-tracing';

init({ trackingUri: 'http://localhost:5000' });

const handler = new MlflowLangchainCallbackHandler();

const sequence = RunnableSequence.from([/* your runnables */]).withConfig({
  callbacks: [handler]
});

await sequence.invoke({ prompt: 'Hello' });
```

Refer to the design note in `.agent/langchain-tracing/design.md` for the
implementation plan and outstanding items.

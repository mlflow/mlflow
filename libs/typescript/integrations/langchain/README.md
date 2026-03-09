# MLflow LangChain Integration

Auto-instrumentation for [LangChain](https://js.langchain.com/) chat models with MLflow Tracing.

## Installation

```bash
npm install @mlflow/langchain @mlflow/core @langchain/core
```

## Usage

```typescript
import * as mlflow from '@mlflow/core';
import { tracedModel } from '@mlflow/langchain';
import { ChatAnthropic } from '@langchain/anthropic';

mlflow.init({
  trackingUri: 'http://localhost:5000',
  experimentId: '<experiment-id>',
});

const model = tracedModel(new ChatAnthropic({ model: 'claude-sonnet-4-5-20250514' }));

// Both invoke() and stream() are automatically traced
const result = await model.invoke([{ role: 'user', content: 'Hello!' }]);
```

## Supported Models

- `ChatAnthropic` (`@langchain/anthropic`)
- `ChatOpenAI` (`@langchain/openai`)
- `ChatXAI` (`@langchain/xai`)
- Any `BaseChatModel` subclass with `invoke()` / `stream()` methods

## Features

- Traces `invoke()` and `stream()` calls as LLM spans
- Captures input messages and output content
- Extracts token usage from `usage_metadata`
- Auto-detects message format from model class name (with manual override option)
- Preserves tracing through `bindTools()` calls

## Options

`tracedModel` accepts an optional second argument:

```typescript
// Explicit message format (useful when bundlers mangle class names)
const model = tracedModel(new ChatAnthropic({ model: 'claude-sonnet-4-5-20250514' }), {
  messageFormat: 'anthropic', // 'anthropic' | 'openai' | 'gemini' | 'langchain'
});
```

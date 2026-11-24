# MLflow TypeScript SDK - TypeSafe AI

Use [MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/) with the
[TypeSafe AI JavaScript SDK](https://github.com/typesafe-ai/typesafe-sdk-js) to automatically
trace System One decisions.

## Installation

```bash
npm install @mlflow/core @mlflow/typesafe @typesafe-ai/sdk
```

## Quickstart

Start an MLflow Tracking Server, then initialize MLflow once when your application starts:

```typescript
import * as mlflow from '@mlflow/core';

mlflow.init({
  trackingUri: 'http://localhost:5000',
  experimentId: '<experiment-id>',
});
```

Wrap the TypeSafe client and call it normally:

```typescript
import { tracedTypeSafe } from '@mlflow/typesafe';
import { choice, noul, TypeSafeClient } from '@typesafe-ai/sdk';

const client = tracedTypeSafe(new TypeSafeClient());

const result = await client.systemOne({
  state: {
    question: 'What is MLflow?',
    answer: 'A platform for managing the machine learning lifecycle.',
  },
  questions: {
    relevant: noul('Is the answer relevant to the question?'),
    quality: choice('How good is the answer?', {
      poor: null,
      good: null,
      excellent: null,
    }),
  },
});

console.log(result.answers.relevant.noul);
```

The integration traces `systemOne()` calls, including their state, questions, structured answers,
model, token usage, latency, request ID, and errors. Transport options such as API keys, headers,
timeouts, retries, and abort signals are not recorded. Other SDK operations, including
`models.list()`, are not traced.

## License

This project is licensed under the [Apache License 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt).

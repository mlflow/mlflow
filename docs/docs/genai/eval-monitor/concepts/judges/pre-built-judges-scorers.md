---
description: >
  Learn about MLflow's prebuilt LLM judges for common GenAI evaluation use cases including safety, relevance, groundedness, correctness, and context sufficiency with SDK examples and scorer integration.
last_update:
  date: 2025-05-18
---

# Predefined judges & scorers

## Overview

MLflow provides research-backed judges, wrapped as predefined scorers, for common quality checks available as SDKs.

:::important
While the judges can be used as standalone APIs, they must be wrapped in [Scorers](/genai/eval-monitor/concepts/scorers) for use by the [Evaluation Harness](/genai/eval-monitor/concepts/eval-harness) and [production monitoring service](/genai/eval-monitor/concepts/production-monitoring). MLflow provides predefined implementations of scorers but you can also create custom scorers that use the judge's APIs for more advanced use cases.
:::

:::list-table

- - **Judge**
  - **Key Inputs**
  - **Requires ground truth**
  - **What it evaluates?**
  - **Available in predefined scorers**
- - [`is_context_relevant`](/genai/eval-monitor/concepts/judges/is_context_relevant)
  - `request`, `context`
  - No
  - Is the `context` directly relevant to the user's `request` without deviating into unrelated topics?
  - [`RelevanceToQuery`](/genai/eval-monitor/concepts/judges/is_context_relevant#1-relevancetoquery-scorer)<br/>[`RetrievalRelevance`](/genai/eval-monitor/concepts/judges/is_context_relevant#2-retrievalrelevance-scorer)
- - [`is_safe`](/genai/eval-monitor/concepts/judges/is_safe)
  - `content`
  - No
  - Does the `content` (not) contain harmful, offensive, or toxic material?
  - [`Safety`](/genai/eval-monitor/concepts/judges/is_safe#using-the-prebuilt-scorer)
- - [`is_grounded`](/genai/eval-monitor/concepts/judges/is_grounded)
  - `request`, `response`, `context`
  - No
  - Is the `response` to the `request` grounded in the information provided in the `context` (e.g., the app is not hallucinating a response)?
  - [`RetrievalGroundedness`](/genai/eval-monitor/concepts/judges/is_grounded#using-the-prebuilt-scorer)
- - [`is_correct`](/genai/eval-monitor/concepts/judges/is_correct)
  - `request`, `response`, `expected_facts`
  - Yes
  - Is the `response` to the `request` correct as compared to the provided ground truth `expected_facts`?
  - [`Correctness`](/genai/eval-monitor/concepts/judges/is_correct#using-the-prebuilt-scorer)
- - [`is_context_sufficient`](/genai/eval-monitor/concepts/judges/is_context_sufficient)
  - `request`, `context`, `expected_facts`
  - Yes
  - Does the `context` provide all necessary information to generate a response that includes the ground truth `expected_facts` for the given `request`?
  - [`RetrievalSufficiency`](/genai/eval-monitor/concepts/judges/is_context_sufficient#using-the-prebuilt-scorer)

:::

## Prerequisites for running the examples

1. Install MLflow and required packages

   ```bash
   pip install --upgrade "mlflow[databricks]>=3.1.0"
   ```

2. Create an MLflow experiment by following the [setup your environment quickstart](/mlflow3/genai/getting-started/connect-environment).

## 3 ways to use prebuilt judges

There are 3 ways to use the prebuilt judges.

### 1. Directly via the SDK

Calling the judges directly via the SDK allows you to intergate the judges directly into your application. For example, you might want to check the groundedness of a response before returning the response back to your user.

Below is an example of using the `is_grounded` judge SDK. Refer to each judge's page for additional examples.

```python
from mlflow.genai.judges import is_grounded

result = is_grounded(
    request="What is the capital of France?",
    response="Paris",
    context="Paris is the capital of France.",
)
# result is...
# mlflow.entities.Assessment.Feedback(
#     rationale="The response asks 'What is the capital of France?' and answers 'Paris'. The retrieved context states 'Paris is the capital of France.' This directly supports the answer given in the response.",
#     feedback=FeedbackValue(value=<CategoricalRating.YES: 'yes'>)
# )

result = is_grounded(
    request="What is the capital of France?",
    response="Paris",
    context="Paris is known for its Eiffel Tower.",
)

# result is...
# mlflow.entities.Assessment.Feedback(
#     rationale="The retrieved context states that 'Paris is known for its Eiffel Tower,' but it does not mention that Paris is the capital of France. Therefore, the response is not fully supported by the retrieved context.",
#     feedback=FeedbackValue(value=<CategoricalRating.NO: 'no'>)
# )
```

### 2. Using via the prebuilt scorers

For simpler applications, you can get started with evaluation using MLflow's predefined scorers.

Below is an example of using the `Correctness` predefined scorer. Refer to each judge's page for additional examples and the required Trace data schema to use its predefined scorer.

```python
eval_dataset = [
    {
        "inputs": {"query": "What is the capital of France?"},
        "outputs": {
            "response": "Paris is the magnificent capital city of France, a stunning metropolis known worldwide for its iconic Eiffel Tower, rich cultural heritage, beautiful architecture, world-class museums like the Louvre, and its status as one of Europe's most important political and economic centers. As the capital city, Paris serves as the seat of France's government and is home to numerous important national institutions."
        },
        "expectations": {
            "expected_facts": ["Paris is the capital of France."],
        },
    },
]


from mlflow.genai.scorers import Correctness


eval_results = mlflow.genai.evaluate(data=eval_dataset, scorers=[Correctness])
```

:::danger SCREENSHOT PLACEHOLDER
🔴 Record gif once the bug with expected_facts population is fixed
:::

### 3. Using in custom [Scorers](/genai/eval-monitor/concepts/scorers)

As your application logic and evaluation criteria gets more complex, you need more control over the data passed to the judge, or your application's trace does not meet the predefined scorer's requirements, you can wrap the judge's SDK in a [custom scorer](/genai/eval-monitor/custom-scorers)

Below is an example for wrapping the `is_grounded` judge SDK in a custom scorer.

```python
from mlflow.genai.judges import is_grounded
from mlflow.genai.scorers import scorer

eval_dataset = [
    {
        "inputs": {"query": "What is the capital of France?"},
        "outputs": {
            "response": "Paris",
            "retrieved_context": [
                {
                    "content": "Paris is the capital of France.",
                    "source": "wikipedia",
                }
            ],
        },
    },
]

@scorer
def is_grounded_scorer(inputs: Dict[Any, Any], outputs: Dict[Any, Any]):
    return is_grounded(
        request=inputs["query"],
        response=outputs["response"],
        context=outputs["retrieved_context"],
    )

eval_results = mlflow.genai.evaluate(data=eval_dataset, scorers=[is_grounded_scorer])
```

![Evaluation results](https://assets.docs.databricks.com/_images/mlflow3/mlflow-screenshots/prebuilt-judge-custom-scorer.gif)

## Next Steps

- [Use predefined scorers in evaluation](/genai/eval-monitor/predefined-judge-scorers) - Get started with built-in quality metrics
- [Create custom judges](/genai/eval-monitor/custom-judge/index) - Build judges tailored to your specific needs
- [Run evaluations](/genai/eval-monitor/evaluate-app) - Apply judges to systematically assess your application

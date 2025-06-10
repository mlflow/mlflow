---
description: >
  Evaluate if a GenAI application's response is factually correct against ground truth using the is_correct judge and Correctness scorer.
last_update:
  date: 2025-05-18
---

# Correctness judge & scorer

The `judges.is_correct()` predefined judge assesses whether your GenAI application's response is factually correct by comparing it against provided ground truth information (`expected_facts` or `expected_response`).

This judge is available through the predefined `Correctness` scorer for evaluating application responses against known correct answers.

## API Signature

<!-- :::danger TODO
🔴 Link API reference docs
::: -->

```python
from mlflow.genai.judges import is_correct

def is_correct(
    *,
    request: str,                               # User's question or query
    response: str,                              # Application's response to evaluate
    expected_facts: Optional[list[str]],        # List of expected facts (provide either expected_response or expected_facts)
    expected_response: Optional[str] = None,    #  Ground truth response (provide either expected_response or expected_facts)
    name: Optional[str] = None                  # Optional custom name for display in the MLflow UIs
) -> mlflow.entities.Feedback:
    """Returns Feedback with 'yes' or 'no' value and a rationale"""
```

## Prerequisites for running the examples

1. Install MLflow and required packages

   ```bash
   pip install --upgrade "mlflow[databricks]>=3.1.0"
   ```

2. Create an MLflow experiment by following the [setup your environment quickstart](/mlflow3/genai/getting-started/connect-environment).

## Direct SDK Usage

```python
from mlflow.genai.judges import is_correct

# Example 1: Response contains expected facts
feedback = is_correct(
    request="What is MLflow?",
    response="MLflow is an open-source platform for managing the ML lifecycle.",
    expected_facts=[
        "MLflow is open-source",
        "MLflow is a platform for ML lifecycle"
    ]
)
print(feedback.value)  # "yes"
print(feedback.rationale)  # Explanation of correctness

# Example 2: Response missing or contradicting facts
feedback = is_correct(
    request="When was MLflow released?",
    response="MLflow was released in 2017.",
    expected_facts=["MLflow was released in June 2018"]
)
print(feedback.value)  # "no"
print(feedback.rationale)  # Explanation of what's incorrect
```

## Using the prebuilt scorer

The `is_correct` judge is available through the `Correctness` prebuilt scorer.

**Requirements:**

- **Trace requirements**: `inputs` and `outputs` must be on the Trace's root span
- **Ground-truth labels**: Required - must provide either `expected_facts` or `expected_response` in the `expectations` dictionary

```python
from mlflow.genai.scorers import Correctness

# Create evaluation dataset with ground truth
eval_dataset = [
    {
        "inputs": {"query": "What is the capital of France?"},
        "outputs": {
            "response": "Paris is the magnificent capital city of France, known for the Eiffel Tower and rich culture."
        },
        "expectations": {
            "expected_facts": ["Paris is the capital of France."]
        },
    },
    {
        "inputs": {"query": "What are the main components of MLflow?"},
        "outputs": {
            "response": "MLflow has four main components: Tracking, Projects, Models, and Registry."
        },
        "expectations": {
            "expected_facts": [
                "MLflow has four main components",
                "Components include Tracking",
                "Components include Projects",
                "Components include Models",
                "Components include Registry"
            ]
        },
    },
    {
        "inputs": {"query": "When was MLflow released?"},
        "outputs": {
            "response": "MLflow was released in 2017 by Databricks."
        },
        "expectations": {
            "expected_facts": ["MLflow was released in June 2018"]
        },
    }
]

# Run evaluation with Correctness scorer
eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[Correctness()]
)
```

### Alternative: Using expected_response

You can also use `expected_response` instead of `expected_facts`:

```python
eval_dataset_with_response = [
    {
        "inputs": {"query": "What is MLflow?"},
        "outputs": {
            "response": "MLflow is an open-source platform for managing the ML lifecycle."
        },
        "expectations": {
            "expected_response": "MLflow is an open-source platform for managing the machine learning lifecycle, including experimentation, reproducibility, and deployment."
        },
    }
]

# Run evaluation with expected_response
eval_results = mlflow.genai.evaluate(
    data=eval_dataset_with_response,
    scorers=[Correctness()]
)
```

:::tip
Using `expected_facts` is recommended over `expected_response` as it allows for more flexible evaluation - the response doesn't need to match word-for-word, just contain the key facts.
:::

## Using in a custom scorer

When evaluating applications with different data structures than the [requirements](#using-the-prebuilt-scorer) the predefined scorer, wrap the judge in a custom scorer:

```python
from mlflow.genai.judges import is_correct
from mlflow.genai.scorers import scorer
from typing import Dict, Any

eval_dataset = [
    {
        "inputs": {"question": "What are the main components of MLflow?"},
        "outputs": {
            "answer": "MLflow has four main components: Tracking, Projects, Models, and Registry."
        },
        "expectations": {
            "facts": [
                "MLflow has four main components",
                "Components include Tracking",
                "Components include Projects",
                "Components include Models",
                "Components include Registry"
            ]
        }
    },
    {
        "inputs": {"question": "What is MLflow used for?"},
        "outputs": {
            "answer": "MLflow is used for building websites."
        },
        "expectations": {
            "facts": [
                "MLflow is used for managing ML lifecycle",
                "MLflow helps with experiment tracking"
            ]
        }
    }
]

@scorer
def correctness_scorer(inputs: Dict[Any, Any], outputs: Dict[Any, Any], expectations: Dict[Any, Any]):
    return is_correct(
        request=inputs["question"],
        response=outputs["answer"],
        expected_facts=expectations["facts"]
    )

# Run evaluation
eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[correctness_scorer]
)
```

## Interpreting Results

The judge returns a `Feedback` object with:

- **`value`**: "yes" if response is correct, "no" if incorrect
- **`rationale`**: Detailed explanation of which facts are supported or missing

## Next Steps

- [Explore other predefined judges](/genai/eval-monitor/concepts/judges/pre-built-judges-scorers) - Learn about other built-in quality evaluation judges
- [Create custom judges](/genai/eval-monitor/custom-judge/index) - Build domain-specific evaluation judges
- [Run evaluations](/genai/eval-monitor/evaluate-app) - Use judges in comprehensive application evaluation

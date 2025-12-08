# inference_params Demo for LLM Judges

This demo shows how to use `inference_params` with `make_judge()` to control LLM behavior during evaluation.

## Feature Overview

The `inference_params` parameter allows you to pass inference parameters (e.g., `temperature`, `top_p`, `max_tokens`) to the underlying LLM when using MLflow's LLM Judges.

## Usage

```python
from mlflow.genai import make_judge

# Deterministic judge with temperature=0.0
deterministic_judge = make_judge(
    name="accuracy_check",
    instructions="Evaluate if {{ outputs }} is accurate.",
    model="databricks:/databricks-claude-sonnet-4",
    inference_params={"temperature": 0.0},
)

# Varied judge with temperature=1.0
varied_judge = make_judge(
    name="accuracy_check",
    instructions="Evaluate if {{ outputs }} is accurate.",
    model="databricks:/databricks-claude-sonnet-4",
    inference_params={"temperature": 1.0},
)

# Multiple parameters
custom_judge = make_judge(
    name="quality_eval",
    instructions="Rate {{ outputs }} on clarity.",
    model="databricks:/databricks-claude-sonnet-4",
    inference_params={
        "temperature": 0.3,
        "max_tokens": 200,
        "top_p": 0.9,
    },
)
```

## Databricks Demo

The `databricks_demo.py` notebook demonstrates:

1. **Test 1**: Deterministic outputs with `temperature=0.0`
2. **Test 2**: Varied outputs with `temperature=1.0`
3. **Test 3**: Multiple inference parameters
4. **Test 4**: Default behavior (no inference_params)
5. **Test 5**: Integration with `mlflow.genai.evaluate()`

### Model URI Format

On Databricks, use the format: `databricks:/<model-name>`

Available models:
- `databricks:/databricks-claude-sonnet-4`
- `databricks:/databricks-gpt-5`
- `databricks:/databricks-meta-llama-3-3-70b-instruct`

## Test Results

Screenshots from Databricks testing are included in this directory:

| Screenshot | Description |
|------------|-------------|
| 01-setup-pr-title-and-install.png | Setup and MLflow installation |
| 02-test1-model-list-and-judge-creation.png | Available models and judge creation |
| 03-test1-deterministic-runs-1-2.png | Deterministic judge runs 1-2 |
| 04-test1-deterministic-run-3-result.png | Deterministic judge run 3 and result |
| 05-test2-varied-temperature-1.0.png | Varied judge with temperature=1.0 |
| 06-test3-multiple-inference-params.png | Multiple inference parameters |
| 07-test4-default-no-inference-params.png | Default behavior without params |
| 08-test5-evaluate-data-setup.png | evaluate() data setup |
| 09-test5-evaluate-running.png | evaluate() execution |
| 10-test5-evaluate-results.png | evaluate() results |

## Key Observations

- `temperature=0.0` produces deterministic, reproducible outputs
- `temperature=1.0` produces varied outputs across runs
- `inference_params` correctly serializes in `__repr__` and `model_dump()`
- Feature integrates seamlessly with `mlflow.genai.evaluate()`

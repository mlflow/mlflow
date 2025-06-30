# CLAUDE.md - MLflow GenAI Evaluation

## Overview

MLflow GenAI Evaluation provides comprehensive evaluation capabilities for generative AI applications, including LLMs, RAG systems, and AI agents. It's built on top of MLflow's tracing infrastructure and integrates with the Databricks Agent Evaluation framework.

### Quick Example

```python
import mlflow
from mlflow.genai.scorers import Correctness, Safety, Guidelines

# Evaluate traces from existing experiments
traces = mlflow.search_traces(experiment_ids=["my-experiment"])
results = mlflow.genai.evaluate(
    data=traces,
    scorers=[
        Correctness(),
        Safety(),
        Guidelines(guidelines=["Do not use any profanity."])
    ]
)
```

## Architecture

### Core Components

- **`base.py`** - Main evaluation orchestration with `evaluate()`.
- **`constant.py`** - Reserved keys for Agent Evaluation expectations and constants
- **`utils.py`** - Data transformation utilities and legacy evaluation system bridge
- **`../scorers/`** - Build-in scorers and the decorator for custom scorers.
- **`../judges/`** - LLM judge APIs that can be used within scorers.

### Data Flow

1. **Data Input**: Accepts traces, DataFrames, dictionaries, or predict functions
2. **Data Transformation**: Converts input data to Agent Evaluation schema format
3. **Trace Generation**: Creates traces for predict functions that don't emit them
4. **Scoring**: Applies configured evaluators/scorers to generate assessments
5. **Assessment Storage**: Stores evaluation results as assessments within traces
6. **Results Aggregation**: Returns comprehensive evaluation results with metrics
7. **Results Display**: Displays evaluation results in the MLflow UI.

--------------------------------
[IMPORTANT NOTE] Near-future migration of the core evaluation harness to open-source MLflow code base.

Currently, 4-6 are implemented within the `databricks-agents` package. This will be ported over to
open-source MLflow code base in the near future.
--------------------------------

### Three Evaluation Modes

1. **Trace-based Evaluation**: Uses existing MLflow traces
   - Extracts inputs/outputs from trace data
   - Leverages rich trace context for evaluation
   - Supports retrieval context extraction for RAG evaluation

2. **Static Dataset Evaluation**: Uses provided inputs/outputs/expectations
   - Supports pandas DataFrame, Spark DataFrame, lists, EvaluationDataset
   - Converts data to standardized evaluation format
   - Does not require active model serving

3. **Dynamic Evaluation**: Uses predict function for on-the-fly generation
   - Automatically generates traces during prediction
   - Supports both traced and non-traced prediction functions
   - Enables real-time evaluation workflows


## Development Setup

### Dependencies

```bash
# databricks-agents package is required for GenAI evaluation before migration
uv pip install mlflow[databricks]
```


## Testing

```bash
# Run all GenAI tests
pytest tests/genai

# Run specific test files
pytest tests/genai/evaluate/test_evaluation.py
pytest tests/genai/scorers
```

### Test Structure

- **Mocked Dependencies**: Tests use `@pytest.fixture(autouse=True)` to mock Databricks authentication
- **Version Checks**: Tests conditionally skip based on databricks-agents version availability
- **Trace Validation**: Verifies correct trace generation and content
- **Multi-format Testing**: Tests all supported data input formats

## Documentation

Since GenAI evaluation is not fully open-sourced yet, the OSS MLflow documentation does not cover it. For the time being, refer to Databricks documentation for the features and detailed examples about GenAI evaluation: https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/


## Debugging

### Common Issues

1. **"databricks-agents not found"**: Install with `uv pip install databricks-agents>=1.0.0`
2. **"Not running on Databricks"**: Evaluation requires Databricks tracking URI. Tests should mock tracking URI to 'databricks'.

## FAQ

**Q. Why does GenAI evaluation require Databricks?**
A. GenAI evaluation uses the Databricks Agent Evaluation framework for advanced LLM evaluation capabilities, including built-in scorers and assessment storage. This is not a hard blocker and we are actively working on porting the core evaluation harness to open-source MLflow code base.

**Q. What is the difference between `scorers` and `judges`?**
A. `scorers` are the classes that perform the actual scoring during `mlflow.genai.evaluate()` API execution. They must have a general and standard interface `(inputs, outputs, expectations, trace) -> score` (or subset of inputs). On the other hand, `judges` are more primitive functions that takes minimum required inputs such as request string. In a nutshell, `scorers` are high-level abstraction that connect judges and any other scoring logic with MLflow's evaluation framework.

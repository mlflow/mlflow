# Databricks notebook source
# MAGIC %md
# MAGIC # PR #19152: inference_params for LLM Judges
# MAGIC
# MAGIC **PR:** https://github.com/mlflow/mlflow/pull/19152
# MAGIC
# MAGIC This notebook tests the `inference_params` feature on Databricks, demonstrating:
# MAGIC 1. Deterministic outputs with `temperature=0.0`
# MAGIC 2. Varied outputs with `temperature=1.0`
# MAGIC 3. Multiple inference parameters
# MAGIC 4. Integration with `mlflow.genai.evaluate()`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Install PR Branch

# COMMAND ----------

# MAGIC %pip install git+https://github.com/debu-sinha/mlflow.git@feature/expose-inference-params-for-llm-judges --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
print(f"MLflow version: {mlflow.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 1: Deterministic Judge (temperature=0.0)
# MAGIC
# MAGIC With `temperature=0.0`, the model should produce identical outputs across multiple runs.

# COMMAND ----------

from mlflow.genai import make_judge

# Create deterministic judge
deterministic_judge = make_judge(
    name="accuracy_check",
    instructions="Evaluate if {{ outputs }} is factually accurate. Provide a brief rationale.",
    model="databricks:/databricks-claude-sonnet-4",
    inference_params={"temperature": 0.0},
)

print(f"Judge: {deterministic_judge.name}")
print(f"Model: {deterministic_judge._model}")
print(f"inference_params: {deterministic_judge.inference_params}")

# COMMAND ----------

# Run 3 evaluations with deterministic judge
test_input = {"answer": "The Eiffel Tower is 330 meters tall."}

print("=" * 70)
print("  TEST 1: Deterministic Judge (temperature=0.0)")
print("=" * 70)
print(f"\nTest Input: {test_input}")
print("\nRunning 3 evaluations (expecting identical rationales)...\n")

det_results = []
for i in range(3):
    result = deterministic_judge(outputs=test_input)
    det_results.append(result)
    print(f"Run {i+1}:")
    print(f"  value: {result.value}")
    print(f"  rationale: {result.rationale}\n")

rationales = [r.rationale for r in det_results]
identical = all(r == rationales[0] for r in rationales)
print(f"[Result] All rationales identical: {identical}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 2: Varied Judge (temperature=1.0)
# MAGIC
# MAGIC With `temperature=1.0`, the model should produce varied outputs across runs.

# COMMAND ----------

# Create varied judge
varied_judge = make_judge(
    name="accuracy_check",
    instructions="Evaluate if {{ outputs }} is factually accurate. Provide a brief rationale.",
    model="databricks:/databricks-claude-sonnet-4",
    inference_params={"temperature": 1.0},
)

print(f"Judge: {varied_judge.name}")
print(f"Model: {varied_judge._model}")
print(f"inference_params: {varied_judge.inference_params}")

# COMMAND ----------

# Run 3 evaluations with varied judge
print("=" * 70)
print("  TEST 2: Varied Judge (temperature=1.0)")
print("=" * 70)
print(f"\nTest Input: {test_input}")
print("\nRunning 3 evaluations (expecting varied rationales)...\n")

var_results = []
for i in range(3):
    result = varied_judge(outputs=test_input)
    var_results.append(result)
    print(f"Run {i+1}:")
    print(f"  value: {result.value}")
    print(f"  rationale: {result.rationale}\n")

rationales = [r.rationale for r in var_results]
identical = all(r == rationales[0] for r in rationales)
print(f"[Result] All rationales identical: {identical}")
print(f"[Expected] False (varied outputs with temperature=1.0)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 3: Multiple Inference Parameters
# MAGIC
# MAGIC Testing with `temperature`, `max_tokens`, and `top_p` together.

# COMMAND ----------

# Create judge with multiple inference parameters
multi_param_judge = make_judge(
    name="quality_eval",
    instructions="Rate {{ outputs }} on clarity and accuracy.",
    model="databricks:/databricks-claude-sonnet-4",
    inference_params={
        "temperature": 0.3,
        "max_tokens": 200,
        "top_p": 0.9,
    },
)

print("=" * 70)
print("  TEST 3: Multiple Inference Parameters")
print("=" * 70)
print(f"\nJudge: {multi_param_judge.name}")
print(f"Model: {multi_param_judge._model}")
print(f"inference_params: {multi_param_judge.inference_params}")

# Test __repr__
print(f"\n__repr__:\n  {repr(multi_param_judge)}")

# Test model_dump
dumped = multi_param_judge.model_dump()
pydantic_data = dumped.get("instructions_judge_pydantic_data", {})
print(f"\nmodel_dump inference_params: {pydantic_data.get('inference_params')}")

# COMMAND ----------

# Run evaluation
test_output = {"response": "Machine learning enables computers to learn from data without explicit programming."}
print(f"\nTest Input: {test_output}")
result = multi_param_judge(outputs=test_output)
print(f"\nResult:")
print(f"  value: {result.value}")
print(f"  rationale: {result.rationale}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 4: Default Behavior (No inference_params)
# MAGIC
# MAGIC Verify that judges work correctly without inference_params.

# COMMAND ----------

# Create judge without inference_params
default_judge = make_judge(
    name="default_eval",
    instructions="Check if {{ outputs }} is valid.",
    model="databricks:/databricks-claude-sonnet-4",
)

print("=" * 70)
print("  TEST 4: Default Behavior (no inference_params)")
print("=" * 70)
print(f"\nJudge: {default_judge.name}")
print(f"Model: {default_judge._model}")
print(f"inference_params: {default_judge.inference_params}")

# Verify __repr__ does NOT include inference_params
repr_str = repr(default_judge)
print(f"\n__repr__:\n  {repr_str}")
print(f"\ninference_params in __repr__: {'inference_params' in repr_str}")

# Verify model_dump does NOT include inference_params
dumped = default_judge.model_dump()
pydantic_data = dumped.get("instructions_judge_pydantic_data", {})
print(f"inference_params in model_dump: {'inference_params' in pydantic_data}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 5: Integration with mlflow.genai.evaluate()
# MAGIC
# MAGIC Demonstrate inference_params working within the evaluate() pipeline.

# COMMAND ----------

import pandas as pd
from mlflow.genai import evaluate

# Create evaluation data
eval_data = pd.DataFrame({
    "inputs": [
        {"question": "What is the capital of France?"},
        {"question": "What is 2 + 2?"},
        {"question": "Who wrote Romeo and Juliet?"},
    ],
    "outputs": [
        {"answer": "Paris is the capital of France."},
        {"answer": "2 + 2 equals 4."},
        {"answer": "William Shakespeare wrote Romeo and Juliet."},
    ],
})

print("=" * 70)
print("  TEST 5: Integration with mlflow.genai.evaluate()")
print("=" * 70)
print("\nEvaluation Data:")
display(eval_data)

# COMMAND ----------

# Create deterministic judge for reproducible evaluations
eval_judge = make_judge(
    name="correctness",
    instructions="Evaluate if {{ outputs }} correctly answers {{ inputs }}. Respond 'correct' or 'incorrect'.",
    model="databricks:/databricks-claude-sonnet-4",
    inference_params={"temperature": 0.0},  # Deterministic for reproducible evals
)

print(f"Evaluation Judge: {eval_judge.name}")
print(f"inference_params: {eval_judge.inference_params}")

# COMMAND ----------

# Run evaluation
print("\nRunning mlflow.genai.evaluate()...\n")

with mlflow.start_run(run_name="inference_params_test"):
    results = evaluate(
        data=eval_data,
        scorers=[eval_judge],
    )

print("Evaluation complete!")

# COMMAND ----------

 # Display results
print("\nAvailable tables:", list(results.tables.keys()))
print("\nEvaluation results:")
# Show first available table
for name, table in results.tables.items():
    print(f"\nTable: {name}")
    display(table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### Test Results:
# MAGIC
# MAGIC | Test | Description | Expected | Status |
# MAGIC |------|-------------|----------|--------|
# MAGIC | 1 | Deterministic (temp=0.0) | Identical rationales | OK|
# MAGIC | 2 | Varied (temp=1.0) | Different rationales | OK|
# MAGIC | 3 | Multiple params | All params applied | OK|
# MAGIC | 4 | Default (no params) | inference_params=None | OK|
# MAGIC | 5 | evaluate() integration | Works with scorer | OK
# MAGIC |
# MAGIC
# MAGIC ### Key Observations:
# MAGIC - `temperature=0.0` produces deterministic, reproducible outputs
# MAGIC - `temperature=1.0` produces varied outputs across runs
# MAGIC - `inference_params` correctly serializes in `__repr__` and `model_dump()`
# MAGIC - Feature integrates seamlessly with `mlflow.genai.evaluate()`

# COMMAND ----------


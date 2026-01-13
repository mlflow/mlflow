---
name: agent-evaluation
description: Use this when you need to IMPROVE or OPTIMIZE an existing LLM agent's performance - including improving tool selection accuracy, answer quality, reducing costs, or fixing issues where the agent gives wrong/incomplete responses. Evaluates agents systematically using MLflow evaluation with datasets, scorers, and tracing. Covers end-to-end evaluation workflow or individual components (tracing setup, dataset creation, scorer definition, evaluation execution).
allowed-tools: Read, Write, Bash, Grep, Glob, WebFetch
---

# Agent Evaluation with MLflow

Comprehensive guide for evaluating GenAI agents with MLflow. Use this skill for the complete evaluation workflow or individual components - tracing setup, environment configuration, dataset creation, scorer definition, or evaluation execution. Each section can be used independently based on your needs.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Documentation Access Protocol](#documentation-access-protocol)
3. [Setup Overview](#setup-overview)
4. [Evaluation Workflow](#evaluation-workflow)
5. [Bundled Resources](#bundled-resources)

## Quick Start

**Setup (prerequisite)**: Install MLflow 3.8+, configure environment, integrate tracing

**Evaluation workflow in 4 steps**:

1. **Understand**: Run agent, inspect traces, understand purpose
2. **Define**: Select/create scorers for quality criteria
3. **Dataset**: ALWAYS discover existing datasets first, only create new if needed
4. **Evaluate**: Run agent on dataset, apply scorers, analyze results

## Command Conventions

**Always use `uv run` for MLflow and Python commands:**

```bash
uv run mlflow --version          # MLflow CLI commands
uv run python scripts/xxx.py     # Python script execution
uv run python -c "..."           # Python one-liners
```

This ensures commands run in the correct environment with proper dependencies.

## Documentation Access Protocol

**All MLflow documentation must be accessed through llms.txt:**

1. Start at: `https://mlflow.org/docs/latest/llms.txt`
2. Query llms.txt for your topic with specific prompt
3. If llms.txt references another doc, use WebFetch with that URL
4. Do not use WebSearch - use WebFetch with llms.txt first

**This applies to all steps**, especially:

- Dataset creation (read GenAI dataset docs from llms.txt)
- Scorer registration (check MLflow docs for scorer APIs)
- Evaluation execution (understand mlflow.genai.evaluate API)

## Pre-Flight Validation

Validate environment before starting:

```bash
uv run mlflow --version  # Should be >=3.8.0
uv run python -c "import mlflow; print(f'MLflow {mlflow.__version__} installed')"
```

If MLflow is missing or version is <3.8.0, see Setup Overview below.

## Setup Overview

Before evaluation, complete these three setup steps:

1. **Install MLflow** (version >=3.8.0)
2. **Configure environment** (tracking URI and experiment)
3. **Integrate tracing** (autolog and @mlflow.trace decorators)
   - ⚠️ **MANDATORY**: Read `references/tracing-integration.md` documentation BEFORE implementing
   - ✓ **VERIFY**: Run validation script AFTER implementing

⚠️ **Tracing must work before evaluation.** If tracing fails, stop and troubleshoot.

**Checkpoint - verify before proceeding:**

- [ ] MLflow >=3.8.0 installed
- [ ] MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_ID set
- [ ] Autolog enabled and @mlflow.trace decorators added
- [ ] Test run creates a trace (verify trace ID is not None)

**For complete setup instructions:** See `references/setup-guide.md`

## Evaluation Workflow

### Step 1: Understand Agent Purpose

1. Invoke agent with sample input
2. Inspect MLflow trace (especially LLM prompts describing agent purpose)
3. Print your understanding and ask user for verification
4. **Wait for confirmation before proceeding**

### Step 2: Define Quality Scorers

1. Check existing scorers: `uv run mlflow scorers list --experiment-id $MLFLOW_EXPERIMENT_ID`
2. Discover built-in scorers: `uv run mlflow scorers list -b`
3. Identify gaps and register additional scorers if needed
4. Test scorers on sample trace before full evaluation

**For scorer selection and registration:** See `references/scorers.md`
**For CLI constraints (yes/no format, template variables):** See `references/scorers-constraints.md`

### Step 3: Prepare Evaluation Dataset

**ALWAYS discover existing datasets first** to prevent duplicate work:

1. **Run dataset discovery** (mandatory):

   ```bash
   uv run python scripts/list_datasets.py
   ```

2. **Present findings to user**:

   - Show all discovered datasets with their characteristics (size, topics covered)
   - If datasets found, highlight most relevant options based on agent type

3. **Ask user about existing datasets**:

   - "I found [N] existing evaluation dataset(s). Do you want to use one of these? (y/n)"
   - If yes: Ask which dataset to use and record the dataset name
   - If no: Proceed to step 4

4. **Create new dataset only if user declined existing ones**:
   ```bash
   uv run python scripts/create_dataset_template.py
   ```
   Review and execute the generated script.

**IMPORTANT**: Do not skip dataset discovery. Always run `list_datasets.py` first, even if you plan to create a new dataset. This prevents duplicate work and ensures users are aware of existing evaluation datasets.

**For complete dataset guide:** See `references/dataset-preparation.md`

### Step 4: Run Evaluation

1. Generate traces:

   ```bash
   uv run python scripts/run_evaluation_template.py
   ```

   Review and execute the generated script.

2. Apply scorers:

   ```bash
   uv run mlflow traces evaluate \
     --trace-ids <comma_separated_trace_ids> \
     --scorers <scorer1>,<scorer2>,... \
     --output json > evaluation_results.json
   ```

3. Analyze results:
   ```bash
   uv run python scripts/analyze_results.py evaluation_results.json
   ```
   Generates `evaluation_report.md` with pass rates, failure patterns, and recommendations.

## Bundled Resources

This skill includes scripts and reference documentation to support the evaluation workflow.

### Scripts (scripts/)

Executable automation for common operations:

**Validation Scripts:**

- **validate_environment.py**: Environment validation (mlflow doctor + custom checks)

  - **Use**: Pre-flight check before starting
  - Checks MLflow version, env vars, connectivity

- **validate_auth.py**: Authentication testing

  - **Use**: Before expensive operations
  - Tests Databricks/local auth, LLM provider

- **validate_tracing_static.py**: Static tracing validation (NO auth needed)

  - **Use**: Step 3.4 Stage 1
  - Code analysis only - fast validation

- **validate_tracing_runtime.py**: Runtime tracing validation (REQUIRES auth, BLOCKING)
  - **Use**: Step 3.4 Stage 2
  - Runs agent to verify traces are captured

**Setup & Configuration:**

- **setup_mlflow.py**: Interactive environment configuration
  - **Use**: Step 2 (Configure Environment)
  - Handles tracking URI and experiment ID setup

**Dataset Management:**

- **list_datasets.py**: Dataset discovery and comparison

  - **Use**: Step 3 - MANDATORY first step
  - Lists, compares, recommends datasets with diversity metrics
  - Always run before considering dataset creation

- **create_dataset_template.py**: Dataset creation code generator
  - **Use**: Step 3 - ONLY if user declines existing datasets
  - Generates customized dataset creation script
  - **IMPORTANT**: Generated code uses `mlflow.genai.datasets` APIs and prompts you to inspect agent function signature to match parameters exactly

**Evaluation:**

- **run_evaluation_template.py**: Evaluation execution code generator

  - **Use**: Step 4.1 (Generate Traces)
  - Generates evaluation script using `mlflow.genai.evaluate()`
  - **IMPORTANT**: Loads dataset using `mlflow.genai.datasets.search_datasets()` - never manually recreates data

- **analyze_results.py**: Results analysis and insights
  - **Use**: Step 4.3 (After applying scorers)
  - Pattern detection, recommendations, report generation

### References (references/)

Detailed guides loaded as needed:

- **setup-guide.md** (~180 lines)

  - **When to read**: During Setup (before evaluation)
  - **Covers**: MLflow installation, environment configuration, tracing integration
  - Complete setup instructions with checkpoints

- **tracing-integration.md** (~450 lines)

  - **When to read**: During Step 3 of Setup (Integrate Tracing)
  - **Covers**: Autolog, decorators, session tracking, verification
  - Complete implementation guide with code examples

- **dataset-preparation.md** (~320 lines)

  - **When to read**: During Evaluation Step 3 (Prepare Dataset)
  - **Covers**: Dataset schema, APIs, creation, Unity Catalog
  - Full workflow with Databricks considerations

- **scorers.md** (~430 lines)

  - **When to read**: During Evaluation Step 2 (Define Scorers)
  - **Covers**: Built-in vs custom, registration, testing, design patterns
  - Comprehensive scorer guide

- **scorers-constraints.md** (~150 lines)

  - **When to read**: When registering custom scorers with CLI
  - **Covers**: Template variable constraints, yes/no format, common mistakes
  - Critical CLI requirements and examples

- **troubleshooting.md** (~460 lines)
  - **When to read**: When encountering errors at any step
  - **Covers**: Environment, tracing, dataset, evaluation, scorer issues
  - Organized by phase with error/cause/solution format

### Assets (assets/)

Output templates (not loaded to context):

- **evaluation_report_template.md**
  - **Use**: Step 4.3 (Analyze Results)
  - Structured template for evaluation report generation

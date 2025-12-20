---
name: agent-evaluation
description: For developers building custom GenAI/LLM agents - understand your agent's quality and identify improvements using MLflow evaluation. Covers end-to-end workflow or individual components (tracing setup, dataset creation, scorer definition, evaluation execution). Use for comprehensive evaluation or specific parts as needed.
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

**Evaluation workflow in 4 steps**:
1. **Setup**: Install MLflow 3.8+, configure environment, integrate tracing
2. **Understand**: Run agent, inspect traces, understand purpose
3. **Define**: Select/create scorers for quality criteria
4. **Evaluate**: Run agent on dataset, apply scorers, analyze results

## Documentation Access Protocol

**MANDATORY: All documentation must be accessed through llms.txt**

1. Start at: `https://mlflow.org/docs/latest/llms.txt`
2. Query llms.txt for your topic with specific prompt
3. If llms.txt references another doc, extract and fetch that URL
4. Only use WebSearch if llms.txt truly doesn't cover the topic

This applies to ALL steps in the evaluation process, not just CLI commands.

## Pre-Flight Validation

**CRITICAL: Run environment validation FIRST before starting evaluation**

```bash
python scripts/validate_environment.py
```

This checks:
- ✓ MLflow installed (>=3.8.0)
- ✓ Environment variables set
- ✓ Basic connectivity

If validation passes, proceed with setup. If it fails, follow error messages to fix issues.

## Setup Overview

Before evaluation, complete these setup steps **IN ORDER**.

### Step 1: Install MLflow

Check if MLflow >=3.8.0 is installed:

```bash
mlflow --version
```

If not installed or version too old:

```bash
uv pip install mlflow>=3.8.0
```

### Step 2: Configure Environment

Run the interactive setup helper:

```bash
python scripts/setup_mlflow.py
```

This guides you through:
- Setting `MLFLOW_TRACKING_URI` (Databricks or local server)
- Setting `MLFLOW_EXPERIMENT_ID` (select existing or create new)
- Authentication (Databricks) or server startup (local)

**Manual configuration** (if needed):

```bash
export MLFLOW_TRACKING_URI="databricks://DEFAULT"  # or http://127.0.0.1:5050
export MLFLOW_EXPERIMENT_ID="123456"
```

### Step 3: Integrate MLflow Tracing

**CRITICAL**: Tracing must work before evaluation. **DO NOT PROCEED IF TRACING FAILS**.

Complete these steps **IN ORDER**:

#### Step 3.1: Enable Autolog

Add autolog for your agent's library (LangChain, LangGraph, OpenAI, etc.):

```python
import mlflow
mlflow.langchain.autolog()  # Place in __init__.py before agent imports
```

#### Step 3.2: Add @mlflow.trace Decorators

Decorate **ALL** entry point functions:

```python
import mlflow

@mlflow.trace  # <-- ADD THIS
def run_agent(query: str, llm_provider: LLMProvider) -> str:
    # Agent code here
    ...
```

Verify decorators present:

```bash
grep -B 2 "def run_agent\|def stream_agent" src/*/agent/*.py
```

#### Step 3.3: Capture Session ID (Optional)

If agent supports conversations, capture session_id:

```python
@mlflow.trace
def run_agent(query: str, session_id: str | None = None) -> str:
    if session_id is None:
        session_id = str(uuid.uuid4())

    trace_id = mlflow.get_last_active_trace_id()
    if trace_id:
        mlflow.set_trace_tag(trace_id, "session_id", session_id)

    # Rest of function...
```

#### Step 3.4: Verify Complete Tracing (Two-Stage Validation)

**Stage 1: Static Validation** (NO auth required - fast):

```bash
python scripts/validate_tracing_static.py
```

This checks code structure without running the agent:
- Autolog calls present
- @mlflow.trace decorators on entry points
- Import order correct
- Session ID capture code present

**Stage 2: Runtime Validation** (REQUIRES auth - BLOCKING):

```bash
python scripts/validate_tracing_runtime.py
```

**CRITICAL**: This actually runs your agent with LLM. If this fails, **STOP THE WORKFLOW** and work with the user to fix authentication/LLM issues before continuing.

**CHECKPOINT - DO NOT PROCEED** until:
- [ ] Static validation passes
- [ ] Authentication working (`python scripts/validate_auth.py`)
- [ ] Runtime validation passes
- [ ] Test trace shows complete hierarchy (decorator + autolog spans)

**For detailed tracing setup**: See `references/tracing-integration.md`

## Evaluation Workflow

### Step 1: Understand Agent Purpose

1. Invoke agent with sample input
2. Inspect MLflow trace (especially LLM prompts describing agent purpose)
3. Print your understanding and ask user for verification
4. **Wait for confirmation before proceeding**

### Step 2: Define Quality Scorers

1. Check existing registered scorers in the experiment:
   ```bash
   uv run mlflow scorers list --experiment-id $MLFLOW_EXPERIMENT_ID
   ```

2. Review MLflow's built-in scorers via llms.txt documentation
   - Search for "judges", "scorers", "LLM-as-a-judge", "evaluation"
   - Check if built-in scorers cover your criteria
   - Note which require ground truth and verify trace structure assumptions

3. Evaluate if existing scorers are sufficient for your quality criteria. If not, identify gaps.

4. Present proposed scorers to user for confirmation

5. **Register scorers** if needed:

   - **Built-in scorers**: Register using Python
   - **Custom scorers**: Register using `mlflow scorers register-llm-judge` CLI (use **pass/fail format**)

   **For detailed instructions**: See `references/scorers.md`

6. Test scorers on sample trace:
   ```bash
   uv run mlflow traces evaluate \
     --output json \
     --scorers <scorer_name> \
     --trace-ids <test_trace_id>
   ```

**For scorer design patterns**: See `references/scorers.md`

### Step 3: Prepare Evaluation Dataset

**IMPORTANT: Check existing datasets FIRST before creating new ones**

#### Step 3.1: Discover and Compare Datasets

```bash
python scripts/list_datasets.py
```

This script:
- Lists all datasets in the experiment
- Shows query counts and diversity metrics
- Displays sample queries
- Recommends best dataset for evaluation
- Allows interactive selection or creation of new dataset

**Always use this script first** - prevents duplicate work and helps make informed decisions about dataset selection.

#### Step 3.2: Create New Dataset (If Needed)

If no suitable dataset exists, use the template generator:

```bash
python scripts/create_dataset_template.py
```

This guides you through dataset creation with proper naming (including Unity Catalog for Databricks) and sample queries.

**Databricks**: Use fully-qualified UC table name (`catalog.schema.table`), no tags.

**For complete dataset guide**: See `references/dataset-preparation.md`

### Step 4: Run Evaluation

#### Step 4.1: Generate Traces

Use the template generator:

```bash
python scripts/run_evaluation_template.py
```

This creates a customized script that:
- Loads your dataset
- Runs agent on each query
- Collects trace IDs

Review and execute the generated script.

#### Step 4.2: Apply Scorers

```bash
mlflow traces evaluate \
  --trace-ids <comma_separated_trace_ids> \
  --scorers <scorer1>,<scorer2>,... \
  --output json > evaluation_results.json
```

**Note**: JSON output follows some free-text output that can be ignored.

#### Step 4.3: Analyze Results

Use the analysis script to generate actionable insights:

```bash
python scripts/analyze_results.py evaluation_results.json
```

This automatically:
- Calculates pass rates per scorer
- Detects failure patterns (PR queries, how-to questions, multi-failure queries)
- Identifies query correlations
- Generates prioritized recommendations
- Creates markdown evaluation report

**Output**: `evaluation_report.md` with comprehensive analysis and next steps

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
  - **Use**: Step 3.1 (ALWAYS run first)
  - Lists, compares, recommends datasets with diversity metrics

- **create_dataset_template.py**: Dataset creation code generator
  - **Use**: Step 3.2 (Only if no suitable dataset exists)
  - Generates customized dataset creation script

**Evaluation:**
- **run_evaluation_template.py**: Evaluation execution code generator
  - **Use**: Step 4.1 (Generate Traces)
  - Generates customized evaluation execution script

- **analyze_results.py**: Results analysis and insights
  - **Use**: Step 4.3 (After applying scorers)
  - Pattern detection, recommendations, report generation

### References (references/)

Detailed guides loaded as needed:

- **tracing-integration.md** (~180 lines)
  - **When to read**: During Step 3 (Integrate Tracing)
  - **Covers**: Autolog, decorators, session tracking, verification
  - Complete implementation guide with code examples

- **dataset-preparation.md** (~160 lines)
  - **When to read**: During Step 3 (Prepare Dataset)
  - **Covers**: Checking datasets, comparison, creation, Unity Catalog
  - Full workflow with Databricks considerations

- **scorers.md** (~120 lines)
  - **When to read**: During Step 2 (Define Scorers)
  - **Covers**: Built-in vs custom, pass/fail format, registration, testing
  - Design patterns and best practices

- **troubleshooting.md** (~100 lines)
  - **When to read**: When encountering errors at any step
  - **Covers**: Environment, tracing, dataset, evaluation, scorer issues
  - Organized by phase with error/cause/solution format

### Assets (assets/)

Output templates (not loaded to context):

- **evaluation_report_template.md**
  - **Use**: Step 4.3 (Analyze Results)
  - Structured template for evaluation report generation

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
5. [References](#references)

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

**CRITICAL: Separate stderr from stdout when capturing CLI output:**

When saving CLI command output to files for parsing (JSON, CSV, etc.), always redirect stderr separately to avoid mixing logs with structured data:

```bash
# WRONG - mixes progress bars and logs with JSON output
uv run mlflow traces evaluate ... --output json > results.json

# CORRECT - separates stderr from JSON output
uv run mlflow traces evaluate ... --output json 2>/dev/null > results.json

# ALTERNATIVE - save both separately for debugging
uv run mlflow traces evaluate ... --output json > results.json 2> evaluation.log
```

**When to separate streams:**
- Any command with `--output json` flag
- Commands that output structured data (CSV, JSON, XML)
- When piping output to parsing tools (`jq`, `grep`, etc.)

**When NOT to separate:**
- Interactive commands where you want to see progress
- Debugging scenarios where logs provide context
- Commands that only output unstructured text

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

## Discovering Agent Structure

**Each project has unique structure.** Use dynamic exploration instead of assumptions:

### Find Agent Entry Points
```bash
# Search for main agent functions
grep -r "def.*agent" . --include="*.py"
grep -r "def (run|stream|handle|process)" . --include="*.py"

# Check common locations
ls main.py app.py src/*/agent.py 2>/dev/null

# Look for API routes
grep -r "@app\.(get|post)" . --include="*.py"  # FastAPI/Flask
grep -r "def.*route" . --include="*.py"
```

### Find Tracing Integration
```bash
# Find autolog calls
grep -r "mlflow.*autolog" . --include="*.py"

# Find trace decorators
grep -r "@mlflow.trace" . --include="*.py"

# Check imports
grep -r "import mlflow" . --include="*.py"
```

### Understand Project Structure
```bash
# Check entry points in package config
cat pyproject.toml setup.py 2>/dev/null | grep -A 5 "scripts\|entry_points"

# Read project documentation
cat README.md docs/*.md 2>/dev/null | head -100

# Explore main directories
ls -la src/ app/ agent/ 2>/dev/null
```

## Setup Overview

Before evaluation, complete these three setup steps:

1. **Install MLflow** (version >=3.8.0)
2. **Configure environment** (tracking URI and experiment)
   - **Guide**: Follow `references/setup-guide.md` Steps 1-2
3. **Integrate tracing** (autolog and @mlflow.trace decorators)
   - ⚠️ **MANDATORY**: Follow `references/tracing-integration.md` - the authoritative tracing guide
   - ✓ **VERIFY**: Run `scripts/validate_agent_tracing.py` after implementing

⚠️ **Tracing must work before evaluation.** If tracing fails, stop and troubleshoot.

**Checkpoint - verify before proceeding:**

- [ ] MLflow >=3.8.0 installed
- [ ] MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_ID set
- [ ] Autolog enabled and @mlflow.trace decorators added
- [ ] Test run creates a trace (verify trace ID is not None)

**Validation scripts:**
```bash
uv run python scripts/validate_environment.py  # Check MLflow install, env vars, connectivity
uv run python scripts/validate_auth.py         # Test authentication before expensive operations
```

**For complete setup instructions:** See `references/setup-guide.md`

## Evaluation Workflow

### Step 1: Understand Agent Purpose

1. Invoke agent with sample input
2. Inspect MLflow trace (especially LLM prompts describing agent purpose)
3. Print your understanding and ask user for verification
4. **Wait for confirmation before proceeding**

### Step 2: Define Quality Scorers

1. **Discover built-in scorers using documentation protocol:**
   - Query `https://mlflow.org/docs/latest/llms.txt` for "What built-in LLM judges or scorers are available?"
   - Read scorer documentation to understand their purpose and requirements
   - Note: Do NOT use `mlflow scorers list -b` - use documentation instead for accurate information

2. **Check registered scorers in your experiment:**
   ```bash
   uv run mlflow scorers list -x $MLFLOW_EXPERIMENT_ID
   ```

3. Identify quality dimensions for your agent and select appropriate scorers
4. Register scorers and test on sample trace before full evaluation

**For scorer selection and registration:** See `references/scorers.md`
**For CLI constraints (yes/no format, template variables):** See `references/scorers-constraints.md`

### Step 3: Prepare Evaluation Dataset

**ALWAYS discover existing datasets first** to prevent duplicate work:

1. **Run dataset discovery** (mandatory):

   ```bash
   uv run python scripts/list_datasets.py  # Lists, compares, recommends datasets
   uv run python scripts/list_datasets.py --format json  # Machine-readable output
   uv run python scripts/list_datasets.py --help  # All options
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
   # Generates dataset creation script from test cases file
   uv run python scripts/create_dataset_template.py --test-cases-file test_cases.txt
   uv run python scripts/create_dataset_template.py --help  # See all options
   ```
   Generated code uses `mlflow.genai.datasets` APIs - review and execute the script.

**IMPORTANT**: Do not skip dataset discovery. Always run `list_datasets.py` first, even if you plan to create a new dataset. This prevents duplicate work and ensures users are aware of existing evaluation datasets.

**For complete dataset guide:** See `references/dataset-preparation.md`

### Step 4: Run Evaluation

1. Generate traces:

   ```bash
   # Generates evaluation script (auto-detects agent module, entry point, dataset)
   uv run python scripts/run_evaluation_template.py
   uv run python scripts/run_evaluation_template.py --help  # Override auto-detection
   ```

   Generated script uses `mlflow.genai.evaluate()` - review and execute it.

2. Apply scorers:

   ```bash
   # IMPORTANT: Redirect stderr to avoid mixing logs with JSON output
   uv run mlflow traces evaluate \
     --trace-ids <comma_separated_trace_ids> \
     --scorers <scorer1>,<scorer2>,... \
     --output json 2>/dev/null > evaluation_results.json
   ```

3. Analyze results:
   ```bash
   # Pattern detection, failure analysis, recommendations
   uv run python scripts/analyze_results.py evaluation_results.json
   ```
   Generates `evaluation_report.md` with pass rates and improvement suggestions.

## References

Detailed guides in `references/` (load as needed):

- **setup-guide.md** - Environment setup (MLflow install, tracking URI configuration)
- **tracing-integration.md** - Authoritative tracing guide (autolog, decorators, session tracking, verification)
- **dataset-preparation.md** - Dataset schema, APIs, creation, Unity Catalog
- **scorers.md** - Built-in vs custom scorers, registration, testing
- **scorers-constraints.md** - CLI requirements for custom scorers (yes/no format, templates)
- **troubleshooting.md** - Common errors by phase with solutions

Scripts are self-documenting - run with `--help` for usage details.

# MLflow Environment Setup Guide

Complete guide for setting up MLflow environment before agent evaluation.

## Table of Contents

1. [Step 1: Install MLflow](#step-1-install-mlflow)
2. [Step 2: Configure Environment](#step-2-configure-environment)
3. [Step 3: Integrate MLflow Tracing](#step-3-integrate-mlflow-tracing)

## Overview

Before evaluation, complete these setup steps in order.

## Step 1: Install MLflow

Check if MLflow >=3.8.0 is installed:

```bash
uv run mlflow --version
```

If not installed or version too old:

```bash
uv pip install mlflow>=3.8.0
```

## Step 2: Configure Environment

### Quick Setup (Recommended - 90% of cases)

**Auto-detects Databricks or local MLflow server:**

Run these commands to auto-configure MLflow:

```bash
# 1. Detect tracking server type
if databricks current-user me &> /dev/null; then
    # Databricks detected
    export MLFLOW_TRACKING_URI="databricks"
    export DB_USER=$(databricks current-user me --output json | grep -o '"value":"[^"]*"' | head -1 | cut -d'"' -f4)
    export PROJECT_NAME=$(basename $(pwd))
    export EXP_NAME="/Users/$DB_USER/${PROJECT_NAME}-evaluation"
    echo "✓ Detected Databricks"
    echo "  User: $DB_USER"
    echo "  Experiment: $EXP_NAME"
else
    # Local or other server
    export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
    export PROJECT_NAME=$(basename $(pwd))
    export EXP_NAME="${PROJECT_NAME}-evaluation"
    echo "✓ Using local MLflow server"
    echo "  URI: $MLFLOW_TRACKING_URI"
    echo "  Experiment: $EXP_NAME"
    echo ""
    echo "  Note: If MLflow server isn't running, start it with:"
    echo "    mlflow server --host 127.0.0.1 --port 5000 &"
fi

# 2. Find existing or create new experiment
export EXP_ID=$(uv run python -c "
import mlflow
mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI')
experiments = mlflow.search_experiments(
    filter_string=\"name = '$EXP_NAME'\",
    max_results=1
)
if experiments:
    print(experiments[0].experiment_id)
else:
    print(mlflow.create_experiment('$EXP_NAME'))
")

# 3. Display configuration
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ MLflow Configuration Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Tracking URI:   $MLFLOW_TRACKING_URI"
echo "Experiment ID:  $EXP_ID"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Export for use in subsequent steps
export MLFLOW_EXPERIMENT_ID="$EXP_ID"
```

**Alternative**: Use the setup script with auto-detection:

```bash
uv run python scripts/setup_mlflow.py
# Auto-detects Databricks or local MLflow, creates experiment if needed
# Outputs: export MLFLOW_TRACKING_URI="..." and export MLFLOW_EXPERIMENT_ID="..."
```

**After running the above commands**, automatically detect and update the agent's configuration:

1. **Detect configuration mechanism** by checking for:
   - `.env` file (most common)
   - `config.py` or `settings.py` with Settings/Config class
   - Other configuration files

2. **Update configuration automatically**:
   - If `.env` exists: Append `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_ID`
   - If config class exists: Add `mlflow_tracking_uri` and `mlflow_experiment_id` fields
   - If neither exists: Set environment variables in agent initialization code

3. **Verify configuration** by importing the agent and checking values load correctly:
   ```bash
   uv run python -c "
   import os
   import mlflow

   tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
   experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID')

   if tracking_uri and experiment_id:
       print(f'✓ MLFLOW_TRACKING_URI: {tracking_uri}')
       print(f'✓ MLFLOW_EXPERIMENT_ID: {experiment_id}')
       mlflow.set_tracking_uri(tracking_uri)
       exp = mlflow.get_experiment(experiment_id)
       print(f'✓ Connected to experiment: {exp.name}')
   else:
       print('⚠ Environment variables not set - check agent configuration')
   "
   ```
```

**If the quick setup succeeds**, you're done! Skip to Step 3.

**If the quick setup fails**, proceed to Manual Setup below.

---

### Manual Setup (10% edge cases)

**Note**: The `setup_mlflow.py` script now includes auto-detection for most scenarios. Manual setup is primarily needed for edge cases where auto-detection cannot work.

Use manual setup if:
- Using a custom remote MLflow server (not Databricks, not localhost)
- Non-standard port or hostname for local server
- Quick setup and `setup_mlflow.py` both failed
- Need more control over experiment naming or configuration

#### Step 2.1: Set Tracking URI

Choose your tracking server type:

```bash
# For Databricks
export MLFLOW_TRACKING_URI="databricks"

# For local server (start server first: mlflow server --host 127.0.0.1 --port 5000 &)
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"

# For other remote server
export MLFLOW_TRACKING_URI="<your_server_uri>"
```

#### Step 2.2: Find or Create Experiment

**Option A: Use Existing Experiment**

Find an experiment to use (efficient lookup by name):

```bash
export EXP_NAME="<your_experiment_name>"
export EXP_ID=$(uv run python -c "
import mlflow
mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI')
experiments = mlflow.search_experiments(
    filter_string=\"name = '$EXP_NAME'\",
    max_results=1
)
print(experiments[0].experiment_id if experiments else 'NOT_FOUND')
")

if [ "$EXP_ID" = "NOT_FOUND" ]; then
    echo "Experiment not found: $EXP_NAME"
    exit 1
fi

export MLFLOW_EXPERIMENT_ID="$EXP_ID"
```

**Option B: Create New Experiment**

```bash
# For Databricks - must use /Users/<username>/<name> format
export EXP_NAME="/Users/<your_email>/<experiment_name>"
uv run mlflow experiments create --experiment-name "$EXP_NAME"

# For local - simple name works
export EXP_NAME="<experiment_name>"
uv run mlflow experiments create --experiment-name "$EXP_NAME"

# Get the experiment ID
export EXP_ID=$(uv run python -c "
import mlflow
mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI')
exp = mlflow.get_experiment_by_name('$EXP_NAME')
print(exp.experiment_id)
")
export MLFLOW_EXPERIMENT_ID="$EXP_ID"
```

#### Step 2.3: Persist Configuration

Add to .env file:

```bash
cat >> .env << EOF

# MLflow Configuration
MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
MLFLOW_EXPERIMENT_ID=$MLFLOW_EXPERIMENT_ID
EOF
```

Add to config.py Settings class (if not present):

```python
# MLflow Configuration
mlflow_tracking_uri: Optional[str] = Field(
    default=None,
    description="MLflow tracking URI (e.g., 'databricks', 'http://localhost:5000')",
)
mlflow_experiment_id: Optional[str] = Field(
    default=None,
    description="MLflow experiment ID for logging traces and evaluation results",
)
```

#### Step 2.4: Verify Configuration

```bash
uv run python -c "
from config import settings
assert settings.mlflow_tracking_uri, 'MLFLOW_TRACKING_URI not loaded'
assert settings.mlflow_experiment_id, 'MLFLOW_EXPERIMENT_ID not loaded'
print('✓ MLflow configuration verified')
"
```

## Setup Complete - Environment Configured

To complete the setup, verify:

- [ ] MLflow >=3.8.0 installed (`uv run mlflow --version`)
- [ ] MLFLOW_TRACKING_URI set (points to your tracking server)
- [ ] MLFLOW_EXPERIMENT_ID set (experiment exists and is accessible)

# MLflow Demo Data Framework

This module provides a framework for generating demo data that helps users explore MLflow's GenAI features immediately. Demo data serves two primary audiences:

1. **Consideration stage users** - People evaluating MLflow who want to see features before committing
2. **Post-adoption users** - People who have chosen MLflow and want to discover features through exploration

## User Entry Points

### 1. `mlflow demo` CLI Command

A dedicated command for frictionless onboarding during the consideration stage:

```bash
mlflow demo                    # Launch demo server, open browser to demo experiment
mlflow demo --no-browser       # Launch without opening browser
mlflow demo --port 5001        # Custom port
```

**Behavior:**

- Creates a temporary, self-contained environment (SQLite in temp directory)
- Generates demo data automatically on startup
- Opens browser directly to the MLflow Demo experiment
- Auto-cleanup on exit

This enables simple messaging in documentation:

> "Try MLflow's GenAI features now: `uvx mlflow demo` or `pip install mlflow && mlflow demo`"

### 2. Launch Demo Button (Home Page)

For users who start `mlflow server` normally:

- Prominent banner on the home page for first-time visitors
- "Launch Demo" button generates the demo experiment and navigates to it
- Feature cards allow navigation to different sections of the demo experiment
- Banner can be dismissed and remembers preference via localStorage

## Design Principles

1. **Minimal friction** - Demo data is accessible with one command or one click
2. **Idempotent** - Safe to run multiple times; skips generation if data already exists
3. **Versioned** - Each generator has a version; stale data is automatically cleaned up and regenerated
4. **Reserved namespace** - Demo entities use reserved names (`MLflow Demo` experiment, `mlflow-demo.*` prompts)

## Versioning

Each generator has a `version` class attribute. When demo data is generated, the version is stored as a tag on the MLflow Demo experiment.

### Why Versioning Matters

Versioning ensures demo data stays compatible with the current MLflow version. This is critical when:

1. **API changes** - If MLflow's tracing or evaluation APIs change, old demo data may become invalid or display incorrectly
2. **New features** - When new UI features require additional data fields to be present in demo entities
3. **Bug fixes** - If demo data was generated with bugs that have since been fixed
4. **Schema changes** - When the structure of traces, assessments, or other entities evolves

Without versioning, users on existing deployments would see broken or incomplete demos after upgrading MLflow.

### How It Works

On demo generation:

1. `is_generated()` checks if data exists AND version matches the current generator version
2. If version mismatch: `delete_demo()` is called to clean up old data, then `generate()` creates fresh data
3. After generation: `store_version()` saves the current version as an experiment tag

### When to Bump Version

Bump the version when making changes to demo data that require regeneration:

- Changing the structure of generated traces/spans
- Adding new required fields to assessments or evaluations
- Modifying prompt templates
- Any change that makes old demo data incompatible with the current UI

## Creating a New Generator

Create a new file in `mlflow/demo/generators/` that inherits from `BaseDemoGenerator`:

```python
from mlflow.demo.base import (
    BaseDemoGenerator,
    DemoFeature,
    DemoResult,
    DEMO_EXPERIMENT_NAME,
)


class MyFeatureDemoGenerator(BaseDemoGenerator):
    name = DemoFeature.TRACES
    version = 1  # Bump when demo format changes

    def generate(self) -> DemoResult:
        # Create demo data using MLflow APIs
        return DemoResult(
            feature=self.name,
            entity_ids=["trace-id-1", "trace-id-2"],
            navigation_url="/path/to/view",
        )

    def _data_exists(self) -> bool:
        # Check if demo data already exists
        return False

    def delete_demo(self) -> None:
        # Cleanup for version upgrades
        pass
```

Register new generators in `mlflow/demo/generators/__init__.py`.

## Naming Conventions

All demo entities must use reserved namespaces to avoid conflicts with user data:

| Entity Type | Naming Convention                        | Example                                  |
| ----------- | ---------------------------------------- | ---------------------------------------- |
| Experiment  | `DEMO_EXPERIMENT_NAME` constant          | `"MLflow Demo"`                          |
| Prompts     | `{DEMO_PROMPT_PREFIX}.<category>.<name>` | `"mlflow-demo.prompts.customer-support"` |
| Run tags    | `mlflow.demo.*`                          | `mlflow.demo.version.traces`             |

Import constants from `mlflow.demo.base`:

```python
from mlflow.demo.base import DEMO_EXPERIMENT_NAME, DEMO_PROMPT_PREFIX
```

## Testing

Run demo tests with:

```bash
uv run pytest tests/demo/ -v
```

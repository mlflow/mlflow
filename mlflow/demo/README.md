# MLflow Demo Data Framework

This module generates demo data for MLflow's GenAI features. For user-facing documentation, see the MLflow docs.

## User Entry Points

1. **CLI:** `mlflow demo` - Launches a temporary server with demo data
2. **Home Page:** "Launch Demo" button - Adds demo data to an existing server
3. **Settings:** "Clear Demo Data" - Removes all demo data

## Architecture

```
mlflow/demo/
├── base.py                  # BaseDemoGenerator, DemoFeature enum, DemoResult
├── registry.py              # DemoRegistry for managing generators
└── generators/
    ├── __init__.py          # Registers generators (order matters!)
    ├── prompts.py           # Prompt versions and aliases
    ├── traces.py            # Sample traces with various patterns
    ├── evaluation.py        # Evaluation runs and datasets
    └── scorers.py           # Registered LLM judges
```

**Generator order matters** - some generators depend on others (e.g., traces depend on prompts, evaluation depends on traces).

## API Endpoints

| Endpoint                             | Method | Description                     |
| ------------------------------------ | ------ | ------------------------------- |
| `/ajax-api/3.0/mlflow/demo/generate` | POST   | Generate demo data (idempotent) |
| `/ajax-api/3.0/mlflow/demo/delete`   | POST   | Hard delete all demo data       |

## Adding a New Generator

1. Add feature to `DemoFeature` enum in `base.py`
2. Create generator class extending `BaseDemoGenerator`
3. Register in `generators/__init__.py` (respect dependency order)

```python
class MyFeatureDemoGenerator(BaseDemoGenerator):
    name = DemoFeature.MY_FEATURE
    version = 1

    def generate(self) -> DemoResult:
        # Create demo data, return DemoResult with navigation_url
        ...

    def _data_exists(self) -> bool:
        # Return True if demo data exists
        ...

    def delete_demo(self) -> None:
        # Clean up demo data (optional, has default no-op)
        ...
```

## Naming Conventions

| Entity Type | Convention                       | Example                   |
| ----------- | -------------------------------- | ------------------------- |
| Experiment  | `DEMO_EXPERIMENT_NAME` constant  | `"MLflow Demo"`           |
| Prompts     | `{DEMO_PROMPT_PREFIX}.<name>`    | `"mlflow-demo.prompts.*"` |
| Scorers     | `{DEMO_PROMPT_PREFIX}.scorers.*` | `"mlflow-demo.scorers.*"` |
| Metadata    | `mlflow.demo.*`                  | `mlflow.demo.version`     |

## Versioning

Each generator has a `version` attribute. When the version changes, old demo data is automatically deleted and regenerated. Bump versions when changes make old demo data incompatible with the current UI.

## Testing

```bash
uv run pytest tests/demo/ -v
```

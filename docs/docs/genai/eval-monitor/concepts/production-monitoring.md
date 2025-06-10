---
description: 'Production monitoring in MLflow - continuous quality assessment with real-world patterns and code examples'
last_update:
  date: 2025-05-18
---

# Production Monitoring

::include[beta]

Production monitoring enables continuous quality assessment of your GenAI applications by automatically running scorers on live traffic. The monitoring service runs every 15 minutes, evaluating a configurable sample of traces using the same scorers you use in development.

## How it works

When you enable production monitoring for an MLflow experiment:

1. **Automatic execution** - A background job runs every 15 minutes (after initial setup)
2. **Scorer evaluation** - Each configured scorer runs on a sample of your production traces
3. **Feedback attachment** - Results are attached as [feedback](/mlflow3/genai/tracing/data-model#feedback) to each evaluated trace
4. **Data archival** - All traces (not just sampled ones) are written to a Delta Table in Unity Catalog for analysis

The monitoring service ensures consistent evaluation using the same scorers from development, providing automated quality assessment without manual intervention.

:::warning
Currently, production monitoring only supports [predefined scorers](/genai/eval-monitor/predefined-judge-scorers). Contact your Databricks account representative if you need to run custom code-based or LLM-based scorers in production.
:::

## API Reference

### create_external_monitor

Creates a monitor for a GenAI application served outside Databricks. Once created, the monitor begins automatically evaluating traces according to the configured assessment suite.

```python
# These packages are automatically installed with mlflow[databricks]
from databricks.agents.monitoring import create_external_monitor

create_external_monitor(
    *,
    catalog_name: str,
    schema_name: str,
    assessments_config: AssessmentsSuiteConfig | dict,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
) -> ExternalMonitor
```

#### Parameters

| Parameter            | Type                               | Description                                                                                          |
| -------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `catalog_name`       | `str`                              | Unity Catalog catalog name where the trace archive table will be created                             |
| `schema_name`        | `str`                              | Unity Catalog schema name where the trace archive table will be created                              |
| `assessments_config` | `AssessmentsSuiteConfig` or `dict` | Configuration for the suite of assessments to run on traces                                          |
| `experiment_id`      | `str` or `None`                    | ID of MLflow experiment to associate with the monitor. Defaults to the currently active experiment   |
| `experiment_name`    | `str` or `None`                    | Name of MLflow experiment to associate with the monitor. Defaults to the currently active experiment |

#### Returns

`ExternalMonitor` - The created monitor object containing experiment ID, configuration, and monitoring URLs

#### Example

```python
import mlflow
from databricks.agents.monitoring import create_external_monitor, AssessmentsSuiteConfig, BuiltinJudge, GuidelinesJudge

# Create a monitor with multiple scorers
external_monitor = create_external_monitor(
    catalog_name="workspace",
    schema_name="default",
    assessments_config=AssessmentsSuiteConfig(
        sample=0.5,  # Sample 50% of traces
        assessments=[
            BuiltinJudge(name="safety"),
            BuiltinJudge(name="relevance_to_query"),
            BuiltinJudge(name="groundedness", sample_rate=0.2),  # Override sampling for this scorer
            GuidelinesJudge(
                guidelines={
                    "mlflow_only": [
                        "If the request is unrelated to MLflow, the response must refuse to answer."
                    ],
                    "professional_tone": [
                        "The response must maintain a professional and helpful tone."
                    ]
                }
            ),
        ],
    ),
)

print(f"Monitor created for experiment: {external_monitor.experiment_id}")
print(f"View traces at: {external_monitor.monitoring_page_url}")
```

### get_external_monitor

Retrieves an existing monitor for a GenAI application served outside Databricks.

```python
# These packages are automatically installed with mlflow[databricks]
from databricks.agents.monitoring import get_external_monitor

get_external_monitor(
    *,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
) -> ExternalMonitor
```

#### Parameters

| Parameter         | Type            | Description                                               |
| ----------------- | --------------- | --------------------------------------------------------- |
| `experiment_id`   | `str` or `None` | ID of the MLflow experiment associated with the monitor   |
| `experiment_name` | `str` or `None` | Name of the MLflow experiment associated with the monitor |

#### Returns

`ExternalMonitor` - The retrieved monitor object

#### Raises

- `ValueError` - When neither experiment_id nor experiment_name is provided
- `NoMonitorFoundError` - When no monitor is found for the given experiment

#### Example

```python
from databricks.agents.monitoring import get_external_monitor

# Get monitor by experiment ID
monitor = get_external_monitor(experiment_id="123456789")

# Get monitor by experiment name
monitor = get_external_monitor(experiment_name="my-genai-app-experiment")

# Access monitor configuration
print(f"Sampling rate: {monitor.assessments_config.sample}")
print(f"Archive table: {monitor.trace_archive_table}")
```

### update_external_monitor

Updates the configuration of an existing monitor. The configuration is completely replaced (not merged) with the new values.

```python
# These packages are automatically installed with mlflow[databricks]
from databricks.agents.monitoring import update_external_monitor

update_external_monitor(
    *,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
    assessments_config: AssessmentsSuiteConfig | dict,
) -> ExternalMonitor
```

#### Parameters

| Parameter            | Type                               | Description                                                                   |
| -------------------- | ---------------------------------- | ----------------------------------------------------------------------------- |
| `experiment_id`      | `str` or `None`                    | ID of the MLflow experiment associated with the monitor                       |
| `experiment_name`    | `str` or `None`                    | Name of the MLflow experiment associated with the monitor                     |
| `assessments_config` | `AssessmentsSuiteConfig` or `dict` | Updated configuration that will completely replace the existing configuration |

#### Returns

`ExternalMonitor` - The updated monitor object

#### Raises

- `ValueError` - When assessments_config is not provided

### delete_external_monitor

Deletes the monitor for a GenAI application served outside Databricks.

```python
# These packages are automatically installed with mlflow[databricks]
from databricks.agents.monitoring import delete_external_monitor

delete_external_monitor(
    *,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
) -> None
```

#### Parameters

| Parameter         | Type            | Description                                               |
| ----------------- | --------------- | --------------------------------------------------------- |
| `experiment_id`   | `str` or `None` | ID of the MLflow experiment associated with the monitor   |
| `experiment_name` | `str` or `None` | Name of the MLflow experiment associated with the monitor |

#### Example

```python
from databricks.agents.monitoring import delete_external_monitor

# Delete monitor by experiment ID
delete_external_monitor(experiment_id="123456789")

# Delete monitor by experiment name
delete_external_monitor(experiment_name="my-genai-app-experiment")
```

## Configuration Classes

### AssessmentsSuiteConfig

Configuration for a suite of assessments to be run on traces from a GenAI application.

```python
# These packages are automatically installed with mlflow[databricks]
from databricks.agents.monitoring import AssessmentsSuiteConfig

@dataclasses.dataclass
class AssessmentsSuiteConfig:
    sample: float | None = None
    paused: bool | None = None
    assessments: list[AssessmentConfig] | None = None
```

#### Attributes

| Attribute     | Type                               | Description                                                                                                |
| ------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `sample`      | `float` or `None`                  | Global sampling rate between 0.0 (exclusive) and 1.0 (inclusive). Individual assessments can override this |
| `paused`      | `bool` or `None`                   | Whether the monitoring is paused                                                                           |
| `assessments` | `list[AssessmentConfig]` or `None` | List of assessments to run on traces                                                                       |

#### Methods

##### from_dict

Creates an AssessmentsSuiteConfig from a dictionary representation.

```python
@classmethod
def from_dict(cls, data: dict) -> AssessmentsSuiteConfig
```

##### get_guidelines_judge

Returns the first GuidelinesJudge from the assessments list, or None if not found.

```python
def get_guidelines_judge(self) -> GuidelinesJudge | None
```

#### Example

```python
from databricks.agents.monitoring import AssessmentsSuiteConfig, BuiltinJudge, GuidelinesJudge

# Create configuration with multiple assessments
config = AssessmentsSuiteConfig(
    sample=0.3,  # Sample 30% of all traces
    assessments=[
        BuiltinJudge(name="safety"),
        BuiltinJudge(name="relevance_to_query", sample_rate=0.5),  # Override to 50%
        GuidelinesJudge(
            guidelines={
                "accuracy": ["The response must be factually accurate"],
                "completeness": ["The response must fully address the user's question"]
            }
        )
    ]
)

# Create from dictionary
config_dict = {
    "sample": 0.3,
    "assessments": [
        {"name": "safety"},
        {"name": "relevance_to_query", "sample_rate": 0.5}
    ]
}
config = AssessmentsSuiteConfig.from_dict(config_dict)
```

### BuiltinJudge

Configuration for a built-in judge to be run on traces.

```python
# These packages are automatically installed with mlflow[databricks]
from databricks.agents.monitoring import BuiltinJudge

@dataclasses.dataclass
class BuiltinJudge:
    name: Literal["safety", "groundedness", "relevance_to_query", "chunk_relevance"]
    sample_rate: float | None = None
```

#### Attributes

| Attribute     | Type              | Description                                                                                                           |
| ------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------- |
| `name`        | `str`             | Name of the built-in judge. Must be one of: `"safety"`, `"groundedness"`, `"relevance_to_query"`, `"chunk_relevance"` |
| `sample_rate` | `float` or `None` | Optional override sampling rate for this specific judge (0.0 to 1.0)                                                  |

#### Available Built-in Judges

- **`safety`** - Detects harmful or toxic content in responses
- **`groundedness`** - Assesses if responses are grounded in retrieved context (RAG applications)
- **`relevance_to_query`** - Checks if responses address the user's request
- **`chunk_relevance`** - Evaluates relevance of each retrieved chunk (RAG applications)

### GuidelinesJudge

Configuration for a guideline adherence judge to evaluate custom business rules.

```python
# These packages are automatically installed with mlflow[databricks]
from databricks.agents.monitoring import GuidelinesJudge

@dataclasses.dataclass
class GuidelinesJudge:
    guidelines: dict[str, list[str]]
    sample_rate: float | None = None
    name: Literal["guideline_adherence"] = "guideline_adherence"  # Set automatically
```

#### Attributes

| Attribute     | Type                   | Description                                                           |
| ------------- | ---------------------- | --------------------------------------------------------------------- |
| `guidelines`  | `dict[str, list[str]]` | Dictionary mapping guideline names to lists of guideline descriptions |
| `sample_rate` | `float` or `None`      | Optional override sampling rate for this judge (0.0 to 1.0)           |

#### Example

```python
from databricks.agents.monitoring import GuidelinesJudge

# Create guidelines judge with multiple business rules
guidelines_judge = GuidelinesJudge(
    guidelines={
        "data_privacy": [
            "The response must not reveal any personal customer information",
            "The response must not include internal system details"
        ],
        "brand_voice": [
            "The response must maintain a professional yet friendly tone",
            "The response must use 'we' instead of 'I' when referring to the company"
        ],
        "accuracy": [
            "The response must only provide information that can be verified",
            "The response must acknowledge uncertainty when appropriate"
        ]
    },
    sample_rate=0.8  # Evaluate 80% of traces with these guidelines
)
```

### ExternalMonitor

Represents a monitor for a GenAI application served outside of Databricks.

```python
@dataclasses.dataclass
class ExternalMonitor:
    experiment_id: str
    assessments_config: AssessmentsSuiteConfig
    trace_archive_table: str | None
    _checkpoint_table: str
    _legacy_ingestion_endpoint_name: str

    @property
    def monitoring_page_url(self) -> str
```

#### Attributes

| Attribute             | Type                     | Description                                              |
| --------------------- | ------------------------ | -------------------------------------------------------- |
| `experiment_id`       | `str`                    | ID of the MLflow experiment associated with this monitor |
| `assessments_config`  | `AssessmentsSuiteConfig` | Configuration for assessments being run                  |
| `trace_archive_table` | `str` or `None`          | Unity Catalog table where traces are archived            |
| `monitoring_page_url` | `str`                    | URL to view monitoring results in the MLflow UI          |

## Next steps

- [Set up production monitoring](/genai/eval-monitor/run-scorer-in-prod) - Step-by-step guide to enable monitoring
- [Build evaluation datasets](/genai/eval-monitor/build-eval-dataset) - Use monitoring results to improve quality
- [Predefined scorers reference](/genai/eval-monitor/predefined-judge-scorers) - Available built-in judges

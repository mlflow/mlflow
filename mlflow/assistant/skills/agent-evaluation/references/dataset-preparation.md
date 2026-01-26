# Evaluation Dataset Preparation Guide

Complete guide for creating and managing MLflow evaluation datasets for agent evaluation.

## Read MLflow Documentation First

Before creating any dataset, follow the documentation protocol and read the MLflow GenAI dataset documentation:

- Query llms.txt for "evaluation datasets", "dataset schema", "mlflow.genai.datasets"
- Understand required record schema and APIs

## Examine Your Agent's Function Signature

Before writing dataset code:

1. Open your agent's entry point function
2. Read its function signature to identify parameter names
3. Note which parameters come from the dataset vs. your code

**Example:**

```python
# Load config
config = load_config()

# Agent code
def run_agent(
    query: str, llm_provider: LLMProvider, session_id: str | None = None
) -> str:
    ...

run_agent(query="what can you help me with", llm_provider=config.provider)
```

**Parameter analysis:**

- `query` (str) → **FROM DATASET** - this goes in `inputs` dict
- `llm_provider` (LLMProvider) → **FROM CODE** - provided by agent's config in code, not needed in dataset
- `session_id` (str, optional) → **FROM CODE** - optional, not needed in dataset

**Therefore, your dataset inputs MUST be:**

```python
{"inputs": {"query": "your question here"}}  # ✅ CORRECT - matches parameter name
```

**NOT:**

```python
{"inputs": {"request": "..."}}  # ❌ WRONG - no 'request' parameter
{"inputs": {"question": "..."}}  # ❌ WRONG - no 'question' parameter
{"inputs": {"prompt": "..."}}  # ❌ WRONG - no 'prompt' parameter
```

---

## Required Schema & APIs

### Record Schema

Every dataset record represents a test case and has this structure:

```python
{
    "inputs": dict,  # REQUIRED - parameters for your agent function
    "expectations": dict,  # OPTIONAL - ground truth for evaluation
    "tags": dict,  # OPTIONAL - metadata for filtering
}
```

**CRITICAL**: The `inputs` dict keys MUST EXACTLY match your agent's function parameter names (as identified in Step 2 above).

### Required APIs - NEVER/ALWAYS Rules

#### ❌ NEVER Use These:

- `mlflow.data.from_pandas()` - Wrong namespace, for non-GenAI datasets
- `mlflow.log_input()` - Wrong approach for evaluation datasets
- Manual DataFrame creation without GenAI APIs
- Searching MLflow runs to "recreate" datasets
- Hardcoding dataset queries in evaluation scripts

#### ✅ ALWAYS Use These:

**Core Operations:**

```python
from mlflow.genai.datasets import create_dataset, search_datasets

# CREATE dataset
dataset = create_dataset(name="my-dataset", experiment_id=["1"])

# ADD records
records = [{"inputs": {"query": "test"}}]
dataset.merge_records(records)

# LOAD dataset
datasets = search_datasets(filter_string="name = 'my-dataset'")
dataset = datasets[0]

# USE in evaluation
results = mlflow.genai.evaluate(
    data=dataset, predict_fn=predict_fn, scorers=[RelevanceToQuery()]
)
```

**For complete workflows:**

- Check existing datasets: `python scripts/list_datasets.py` (auto-lists all)
- Create new dataset: `python scripts/create_dataset_template.py --test-cases-file test_cases.txt`

---

## Table of Contents

1. [Understanding MLflow GenAI Datasets](#understanding-mlflow-genai-datasets)
2. [Checking Existing Datasets](#checking-existing-datasets) (Use list_datasets.py script)
3. [Creating New Datasets](#creating-new-datasets) (Use create_dataset_template.py script)
4. [Databricks Unity Catalog Considerations](#databricks-unity-catalog-considerations)
5. [Best Practices](#best-practices)

## Understanding MLflow GenAI Datasets

**IMPORTANT**: MLflow has generic datasets, but **GenAI datasets for agent evaluation are different**.

### What are GenAI Evaluation Datasets?

GenAI evaluation datasets are specialized datasets for evaluating language model applications and agents. They:

- Have a specific schema with `inputs` and optional `expectations`
- Are managed through the MLflow GenAI datasets SDK
- Can be associated with experiments
- Support pagination and search

### Dataset Schema

See the [Required Schema & APIs](#required-schema--apis) section above for the complete record schema definition.

## Checking Existing Datasets

Before creating a new dataset, check if suitable datasets already exist.

### Use the Dataset Discovery Script

```bash
uv run python scripts/list_datasets.py  # Table format (default)
# Or for machine-readable output:
uv run python scripts/list_datasets.py --format json
```

**This script automatically:**

- Lists all datasets in your experiment
- Calculates diversity metrics (record count, query length range)
- Shows sample queries from each dataset
- Recommends the best dataset based on size and diversity
- Allows interactive selection

**For manual dataset access**, use the APIs shown in [Required Schema & APIs](#required-schema--apis).

## Creating New Datasets

If no suitable dataset exists, create a new one.

### ⚠️ Before Creating Dataset ⚠️

**Complete these steps FIRST:**

1. ✅ Read MLflow GenAI dataset documentation (see top of file)
2. ✅ Examine your agent's function signature
3. ✅ Know exact parameter names to use in `inputs` dict

### Use the Dataset Template Generator

```bash
uv run python scripts/create_dataset_template.py --test-cases-file test_cases.txt

# For Databricks Unity Catalog:
uv run python scripts/create_dataset_template.py \
  --test-cases-file test_cases.txt \
  --catalog main --schema ml --table eval_v1
```

**The script will:**

1. Detect your environment
2. Guide you through naming conventions:
   - **OSS**: Simple names like `mlflow-agent-eval-v1`
   - **Databricks**: UC table names like `main.default.mlflow_agent_eval_v1`
3. Help create 10+ diverse sample queries interactively
4. Generate a complete Python script using correct APIs
5. Optionally execute the script to create the dataset

**The generated script handles:**

- Correct API usage (`mlflow.genai.datasets` namespace)
- Environment-specific requirements (tags for OSS, UC tables for Databricks)
- Input validation and error handling

### For Manual Creation

If you prefer manual creation, follow the API patterns in [Required Schema & APIs](#required-schema--apis):

1. Use `create_dataset()` with correct name format
2. Prepare records with `inputs` dict matching your agent parameters
3. Add records with `dataset.merge_records()`
4. Verify with `dataset.to_df()`

**See Databricks Unity Catalog Considerations section** if using Databricks.

## Databricks Unity Catalog Considerations

When using Databricks as your tracking URI, special considerations apply.

### Requirements

**1. Fully-Qualified Table Name**

- Format: `<catalog>.<schema>.<table>`
- Example: `main.default.mlflow_agent_eval_v1`
- Cannot use simple names like `my_dataset`

**2. Tags Not Supported**

- Do NOT include `tags` parameter in `create_dataset()`
- Tags are managed by Unity Catalog

**3. Search Not Supported**

- Cannot use `search_datasets()` API reliably
- Use Unity Catalog tools to find tables
- Access datasets directly by name with `get_dataset()`

### Getting Unity Catalog Table Name

**Option 1: Use the script**

```bash
uv run python scripts/create_dataset_template.py --test-cases-file test_cases.txt
```

**Option 2: List with Databricks CLI**

List catalogs:

```bash
databricks catalogs list
```

List schemas in a catalog:

```bash
databricks schemas list <catalog_name>
```

**Option 3: Use Default**
Suggest the default location:

```
main.default.mlflow_agent_eval_v1
```

Where:

- `main`: Default catalog
- `default`: Default schema
- `mlflow_agent_eval_v1`: Your table name (include version)

### Code Pattern

When creating datasets for Databricks:

```python
# Use fully-qualified UC table name, no tags
dataset = create_dataset(
    name="main.default.mlflow_agent_eval_v1",
    experiment_id="<experiment_id>",
    # Note: No tags parameter
)
```

See [Required Schema & APIs](#required-schema--apis) for complete API examples.

## Best Practices

### Query Diversity

Create a **representative test set** covering different aspects:

**Variety dimensions:**

- **Complexity**: Simple ("What is X?") to complex ("How do I do X and Y while avoiding Z?")
- **Length**: Short (5-10 words) to long (20+ words, multi-part)
- **Topics**: Cover all agent capabilities and edge cases
- **Query types**: Questions, requests, comparisons, examples

**Example diverse set:**

```python
[
    {"inputs": {"query": "What is MLflow?"}},  # Simple, short, basic
    {"inputs": {"query": "How do I log a model?"}},  # Action-oriented
    {
        "inputs": {"query": "What's the difference between experiments and runs?"}
    },  # Comparison
    {
        "inputs": {"query": "Show me an example of using autolog with LangChain"}
    },  # Example request
    {
        "inputs": {
            "query": "How can I track hyperparameters, metrics, and artifacts in a single run?"
        }
    },  # Complex, multi-part
]
```

See generated script output for more examples.

### Sample Size

- **Minimum**: 10 queries (for initial testing)
- **Recommended**: 20-50 queries (for comprehensive evaluation)
- **Balance**: Coverage vs execution time/cost

More queries = better coverage but longer evaluation time and higher LLM costs.

### Versioning

- **Include version in name**: `mlflow_agent_eval_v1`, `mlflow_agent_eval_v2`
- **Document changes**: What's different in each version
- **Keep old versions**: For comparison and reproducibility
- **Use tags** (OSS only): `{"version": "2.0", "changes": "Added edge cases"}`

### Quality Over Quantity

- **Realistic queries**: Match actual user questions
- **Clear questions**: Well-formed, unambiguous
- **Representative**: Cover production use cases
- **Avoid duplicates**: Each query should test something different

### Iteration

1. **Start small**: 10-15 queries for initial evaluation
2. **Analyze results**: See what fails, what's missing
3. **Expand**: Add queries to cover gaps
4. **Refine**: Improve existing queries based on agent behavior
5. **Version**: Create new version with improvements

---

**For troubleshooting dataset creation issues**, see `references/troubleshooting.md`

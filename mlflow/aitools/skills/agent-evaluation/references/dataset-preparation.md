# Evaluation Dataset Preparation Guide

Complete guide for creating and managing MLflow evaluation datasets for agent evaluation.

## Table of Contents

1. [Understanding MLflow GenAI Datasets](#understanding-mlflow-genai-datasets)
2. [Checking Existing Datasets](#checking-existing-datasets)
3. [Comparing Datasets for Selection](#comparing-datasets-for-selection)
4. [Creating New Datasets](#creating-new-datasets)
5. [Databricks Unity Catalog Considerations](#databricks-unity-catalog-considerations)
6. [Best Practices](#best-practices)

## Understanding MLflow GenAI Datasets

**IMPORTANT**: MLflow has generic datasets, but **GenAI datasets for agent evaluation are different**.

### What are GenAI Evaluation Datasets?

GenAI evaluation datasets are specialized datasets for evaluating language model applications and agents. They:
- Have a specific schema with `inputs` and optional `expectations`
- Are managed through the MLflow GenAI datasets SDK
- Can be associated with experiments
- Support pagination and search (in OSS MLflow)

### Documentation

**Always consult the official MLflow documentation**:
1. Read `https://mlflow.org/docs/latest/llms.txt`
2. Search for "evaluation dataset", "genai dataset", "dataset schema"
3. Follow links to detailed documentation

### Dataset Schema

Records must follow this format:

```python
[
    {
        "inputs": {"query": "What is MLflow?"},
        "expectations": {"answer": "MLflow is an open source platform..."}  # Optional
    },
    {
        "inputs": {"query": "How do I log a model?"},
        # expectations are optional
    },
    ...
]
```

**Key points**:
- `inputs`: Required dict containing inputs to your agent (e.g., `{"query": "..."}`)
- `expectations`: Optional dict with expected outputs or ground truth

## Checking Existing Datasets

Before creating a new dataset, check if suitable datasets already exist.

### Search for Datasets

```python
from mlflow import MlflowClient

client = MlflowClient()

# Search for datasets in specific experiments
datasets = client.search_datasets(
    experiment_ids=["<experiment_id>"],
    max_results=10
)

# Print dataset information
if datasets:
    print(f"Found {len(datasets)} dataset(s):\n")
    for dataset in datasets:
        print(f"  Name: {dataset.name}")
        print(f"  ID: {dataset.dataset_id}")
        print()
else:
    print("No datasets found in this experiment.")

# Get next page if available
if datasets.token:
    next_page = client.search_datasets(
        experiment_ids=["<experiment_id>"],
        page_token=datasets.token
    )
```

### Important: Field Access Limitations

**OSS MLflow**: Can access all fields (`name`, `dataset_id`, `experiment_ids`, `tags`)

**Databricks**: **ONLY** access `name` and `dataset_id` fields.

Do **NOT** access `experiment_ids` or `tags` as these may fail with:
```
"Evaluation dataset APIs is not supported in Databricks environments"
```

### Load Dataset Details

To see the actual records in a dataset:

```python
from mlflow.genai.datasets import get_dataset

# Load dataset by name
dataset = get_dataset("<dataset_name>")

# Convert to DataFrame for analysis
df = dataset.to_df()

print(f"Dataset: {dataset.dataset_id}")
print(f"Records: {len(df)}")
print()

# Show sample records
print("Sample records:")
for i, row in df.head(5).iterrows():
    inputs = row['inputs']
    query = inputs.get('query', str(inputs))
    print(f"  {i+1}. {query[:80]}{'...' if len(query) > 80 else ''}")
```

## Comparing Datasets for Selection

If multiple datasets exist, compare them to select the most appropriate for evaluation.

### Comparison Criteria

**1. Size (Record Count)**
- More records = better coverage
- Minimum: 10 records
- Recommended: 20-50 records

**2. Query Diversity**
- Variety in query types (simple vs complex)
- Range of query lengths (short vs long)
- Different topics or capabilities tested

**3. Quality**
- Realistic queries matching actual use cases
- Clear, well-formed questions
- Representative of production workload

### Analyze Dataset Diversity

```python
import pandas as pd

# Load dataset
dataset = get_dataset("<dataset_name>")
df = dataset.to_df()

# Extract queries
queries = []
for idx, row in df.iterrows():
    inputs = row['inputs']
    query = inputs.get('query', inputs.get('question', str(inputs)))
    queries.append(query)

# Calculate diversity metrics
query_lengths = [len(q) for q in queries]

print(f"Dataset: <dataset_name>")
print(f"  Total records: {len(queries)}")
print(f"  Query length range: {min(query_lengths)}-{max(query_lengths)} chars")
print(f"  Average length: {sum(query_lengths)/len(query_lengths):.1f} chars")
print()

# Show variety of samples
print("Sample queries (showing variety):")
# Sort by length to show range
sorted_queries = sorted(zip(queries, query_lengths), key=lambda x: x[1])
samples = [
    sorted_queries[0],  # Shortest
    sorted_queries[len(sorted_queries)//3],  # Short-medium
    sorted_queries[len(sorted_queries)//2],  # Medium
    sorted_queries[2*len(sorted_queries)//3],  # Medium-long
    sorted_queries[-1]  # Longest
]

for i, (query, length) in enumerate(samples, 1):
    print(f"  {i}. [{length} chars] {query[:80]}{'...' if length > 80 else ''}")
```

### Recommendation Logic

**Prefer datasets with BOTH high count AND high diversity:**

1. **Clear winner**: Dataset is both larger AND more diverse → recommend it
2. **Trade-off**: One is larger but less diverse, another is smaller but more diverse → show comparison and ask user
3. **Similar quality**: Multiple good datasets → ask user based on preferences

**Present comparison with clear reasoning:**

```
Dataset Comparison:

1. mlflow_agent_eval_v1
   - Records: 32
   - Diversity: HIGH (mix of short/long, simple/complex queries)
   - Topics: Documentation, PRs, releases, troubleshooting
   - Recommendation: ✓ RECOMMENDED (best balance)

2. mlflow_agent_eval_old
   - Records: 15
   - Diversity: MEDIUM (mostly short queries)
   - Topics: Documentation only
   - Recommendation: Consider for targeted doc evaluation

Recommended: mlflow_agent_eval_v1
Reason: Larger size (32 vs 15) and higher diversity
```

## Creating New Datasets

If no suitable dataset exists, create a new one.

### Quick Start with Script

Use the template generator script:

```bash
python .claude/skills/agent-evaluation/scripts/create_dataset_template.py
```

This will:
1. Detect your environment (OSS MLflow vs Databricks)
2. Guide you through naming (including UC table name for Databricks)
3. Help create sample queries interactively
4. Generate a complete Python script
5. Execute the script to create the dataset

### Manual Creation Workflow

If you prefer to create the dataset manually:

#### Step 1: Prepare Environment

```python
import os
from mlflow.genai.datasets import create_dataset

# Ensure environment is configured
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")

print(f"Tracking URI: {tracking_uri}")
print(f"Experiment ID: {experiment_id}")
```

#### Step 2: Determine Dataset Name

**For Databricks (Unity Catalog)**:
- Must use fully-qualified name: `<catalog>.<schema>.<table>`
- Example: `main.default.mlflow_agent_eval_v1`
- See [Databricks Unity Catalog Considerations](#databricks-unity-catalog-considerations)

**For OSS MLflow**:
- Use descriptive name: `mlflow-agent-eval-v1`
- Can include version suffix

#### Step 3: Create Dataset

```python
# Create the dataset and associate with experiment
dataset = create_dataset(
    name="<dataset_name>",  # See naming guidance above
    experiment_id="<experiment_id>",
    tags={  # Optional - ONLY for OSS MLflow
        "version": "1.0",
        "purpose": "agent_evaluation",
    } if not is_databricks else None
)

print(f"✓ Dataset created: {dataset.dataset_id}")
```

#### Step 4: Prepare Sample Inputs

Create 10+ diverse queries that test different aspects of your agent:

```python
records = [
    {
        "inputs": {"query": "What is MLflow?"}
    },
    {
        "inputs": {"query": "How do I log a model in MLflow?"}
    },
    {
        "inputs": {"query": "What's the difference between mlflow.log_param and mlflow.log_metric?"}
    },
    {
        "inputs": {"query": "Show me an example of using mlflow.langchain.autolog()"}
    },
    {
        "inputs": {"query": "What LLM judges are available in MLflow for evaluation?"}
    },
    # ... add 5+ more diverse queries
]
```

**Guidelines**:
- Minimum 10 queries
- Mix simple and complex questions
- Vary query lengths (short, medium, long)
- Cover different agent capabilities
- Include edge cases if relevant

#### Step 5: Add Records to Dataset

```python
# Merge records into the dataset
dataset = dataset.merge_records(records)

print(f"✓ Added {len(records)} records to dataset")
```

#### Step 6: Verify Dataset

```python
print("\n" + "=" * 60)
print("Dataset Created Successfully!")
print("=" * 60)
print(f"Dataset ID: {dataset.dataset_id}")
print(f"Number of records: {len(dataset.records)}")
print()

# Show sample records
print("Sample records:")
for i, record in enumerate(dataset.records[:5], 1):
    query = record["inputs"].get("query", str(record["inputs"]))
    print(f"  {i}. {query[:80]}{'...' if len(query) > 80 else ''}")
```

### Complete Example

```python
#!/usr/bin/env python3
"""Create MLflow evaluation dataset."""

import os
from mlflow.genai.datasets import create_dataset

# Configuration
os.environ["MLFLOW_TRACKING_URI"] = "databricks://DEFAULT"
os.environ["MLFLOW_EXPERIMENT_ID"] = "123456"

DATASET_NAME = "main.default.mlflow_agent_eval_v1"  # Databricks UC table
EXPERIMENT_ID = "123456"

# Create dataset
dataset = create_dataset(
    name=DATASET_NAME,
    experiment_id=EXPERIMENT_ID,
)

# Prepare records
records = [
    {"inputs": {"query": "What is MLflow?"}},
    {"inputs": {"query": "How do I log models?"}},
    {"inputs": {"query": "What are MLflow experiments?"}},
    {"inputs": {"query": "How do I use mlflow.langchain.autolog()?"}},
    {"inputs": {"query": "What evaluation scorers does MLflow provide?"}},
    {"inputs": {"query": "How can I track LLM calls in MLflow?"}},
    {"inputs": {"query": "What's the difference between runs and experiments?"}},
    {"inputs": {"query": "How do I deploy models with MLflow?"}},
    {"inputs": {"query": "Can MLflow track hyperparameters?"}},
    {"inputs": {"query": "What is the MLflow Model Registry?"}},
]

# Add records
dataset = dataset.merge_records(records)

# Verify
print(f"✓ Dataset created: {dataset.dataset_id}")
print(f"✓ Records: {len(dataset.records)}")
```

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

**Option 1: Ask User**
Use the interactive script:
```bash
python .claude/skills/agent-evaluation/scripts/create_dataset_template.py
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

### Databricks-Specific Code

```python
from mlflow.genai.datasets import create_dataset

# For Databricks: No tags, UC table name
dataset = create_dataset(
    name="main.default.mlflow_agent_eval_v1",  # Fully-qualified UC table
    experiment_id="<experiment_id>",
    # No tags parameter - not supported in Databricks
)

records = [
    {"inputs": {"query": "Sample query 1"}},
    {"inputs": {"query": "Sample query 2"}},
    # ...
]

dataset = dataset.merge_records(records)
```

## Best Practices

### Query Diversity

**Include variety**:
- **Simple queries**: "What is X?"
- **Complex queries**: "How do I do X and Y while avoiding Z?"
- **Short queries**: 5-10 words
- **Long queries**: 20+ words, multi-part questions
- **Different topics**: Cover all agent capabilities

**Example diverse set**:
```python
queries = [
    "What is MLflow?",  # Simple, short, basic
    "How do I log a model?",  # Simple, action-oriented
    "What's the difference between experiments and runs?",  # Comparison
    "Show me an example of using autolog with LangChain",  # Example request
    "How can I track hyperparameters, metrics, and artifacts in a single run?",  # Complex, multi-part
    "What LLM evaluation judges are available and when should I use each one?",  # Complex, requires analysis
]
```

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

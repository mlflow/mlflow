"""
Generate a template script for creating MLflow evaluation datasets.

This script creates a customized Python script for dataset creation,
handling both OSS MLflow and Databricks Unity Catalog scenarios.
"""

import argparse
import os
import subprocess
import sys


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate dataset creation script")
    parser.add_argument("--dataset-name", help="Dataset name (for non-Databricks)")
    parser.add_argument("--catalog", help="UC catalog name (for Databricks)")
    parser.add_argument("--schema", help="UC schema name (for Databricks)")
    parser.add_argument("--table", help="UC table name (for Databricks)")
    parser.add_argument("--test-cases-file", help="File with test cases (one per line, minimum 10)")
    parser.add_argument(
        "--non-interactive", action="store_true", help="Fail if args missing (no prompts)"
    )
    return parser.parse_args()


def load_test_cases_from_file(file_path: str) -> list[str]:
    """Load test cases from file (one per line)."""
    try:
        with open(file_path) as f:
            test_cases = [line.strip() for line in f if line.strip()]

        if len(test_cases) < 10:
            print(f"✗ File has only {len(test_cases)} test cases (minimum: 10)")
            sys.exit(1)

        print(f"✓ Loaded {len(test_cases)} test cases from {file_path}")
        return test_cases

    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        sys.exit(1)


def detect_tracking_uri() -> tuple[str, bool]:
    """Detect tracking URI and whether it's Databricks."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")

    is_databricks = tracking_uri.startswith("databricks://") or tracking_uri == "databricks"

    if not tracking_uri:
        print("⚠ MLFLOW_TRACKING_URI not set")
        print("  Using: http://127.0.0.1:5050 (local)")
        tracking_uri = "http://127.0.0.1:5050"
        is_databricks = False

    print(f"Tracking URI: {tracking_uri}")
    print(f"Environment: {'Databricks' if is_databricks else 'OSS MLflow'}")

    return tracking_uri, is_databricks


def list_databricks_catalogs() -> list[str]:
    """List available Databricks catalogs."""
    try:
        result = subprocess.run(
            ["databricks", "catalogs", "list"], capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split("\n")
        # Simple parsing - just get first column
        catalogs = []
        for line in lines[2:]:  # Skip header
            if line.strip():
                parts = line.split()
                if parts:
                    catalogs.append(parts[0])
        return catalogs
    except Exception as e:
        print(f"  ✗ Could not list catalogs: {e}")
        return []


def list_databricks_schemas(catalog: str) -> list[str]:
    """List schemas in a Databricks catalog."""
    try:
        result = subprocess.run(
            ["databricks", "schemas", "list", catalog], capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split("\n")
        # Simple parsing
        schemas = []
        for line in lines[2:]:  # Skip header
            if line.strip():
                parts = line.split()
                if parts:
                    # Schema name is usually in format catalog.schema
                    schema_full = parts[0]
                    if "." in schema_full:
                        schemas.append(schema_full.split(".")[1])
                    else:
                        schemas.append(schema_full)
        return schemas
    except Exception as e:
        print(f"  ✗ Could not list schemas: {e}")
        return []


def get_databricks_table_name(
    args_catalog: str | None = None,
    args_schema: str | None = None,
    args_table: str | None = None,
    non_interactive: bool = False,
) -> str:
    """Interactive selection of Unity Catalog table name.

    Args:
        args_catalog: Catalog name from CLI arguments
        args_schema: Schema name from CLI arguments
        args_table: Table name from CLI arguments
        non_interactive: If True, fail instead of prompting for input
    """
    # If all parts provided via CLI, use them
    if args_catalog and args_schema and args_table:
        full_name = f"{args_catalog}.{args_schema}.{args_table}"
        print(f"\n✓ Using table: {full_name}")
        return full_name

    # Non-interactive mode requires all args
    if non_interactive:
        print("✗ --catalog, --schema, and --table required in non-interactive mode")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Databricks Unity Catalog Table Name")
    print("=" * 60)

    print("\nFor Databricks, you need a fully-qualified table name:")
    print("Format: <catalog>.<schema>.<table>")
    print()

    # Get catalog (from CLI arg or prompt)
    if args_catalog:
        catalog = args_catalog
        print(f"Using catalog: {catalog}")
    else:
        # Try to list catalogs
        print("Fetching catalogs...")
        catalogs = list_databricks_catalogs()

        if catalogs:
            print("\nAvailable catalogs:")
            for i, cat in enumerate(catalogs, 1):
                print(f"  {i}. {cat}")

            while True:
                try:
                    choice = input(f"\nSelect catalog (1-{len(catalogs)}) or enter name: ").strip()
                    if choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(catalogs):
                            catalog = catalogs[idx]
                            break
                    else:
                        catalog = choice
                        break
                except ValueError:
                    print("Invalid selection")
        else:
            catalog = input("Enter catalog name: ").strip()

        print(f"\nSelected catalog: {catalog}")

    # Get schema (from CLI arg or prompt)
    if args_schema:
        schema = args_schema
        print(f"Using schema: {schema}")
    else:
        # Try to list schemas
        print("\nFetching schemas...")
        schemas = list_databricks_schemas(catalog)

        if schemas:
            print("\nAvailable schemas:")
            for i, sch in enumerate(schemas, 1):
                print(f"  {i}. {sch}")

            while True:
                try:
                    choice = input(f"\nSelect schema (1-{len(schemas)}) or enter name: ").strip()
                    if choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(schemas):
                            schema = schemas[idx]
                            break
                    else:
                        schema = choice
                        break
                except ValueError:
                    print("Invalid selection")
        else:
            schema = input("Enter schema name: ").strip()

        print(f"\nSelected schema: {schema}")

    # Get table (from CLI arg or prompt)
    if args_table:
        table = args_table
        print(f"Using table: {table}")
    else:
        default_table = "mlflow_agent_eval_v1"
        table = input(f"\nEnter table name [{default_table}]: ").strip()
        if not table:
            table = default_table

    full_name = f"{catalog}.{schema}.{table}"
    print(f"\nFull table name: {full_name}")

    return full_name


def generate_sample_queries(
    test_cases_file: str | None = None, non_interactive: bool = False
) -> list[str]:
    """Interactive creation of sample evaluation queries.

    Args:
        test_cases_file: File with test cases (one per line)
        non_interactive: If True, fail instead of prompting for input
    """
    # If test cases file provided, load from file
    if test_cases_file:
        return load_test_cases_from_file(test_cases_file)

    # Non-interactive mode requires test cases file
    if non_interactive:
        print("✗ --test-cases-file required in non-interactive mode")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Sample Evaluation Test Cases")
    print("=" * 60)

    print("\nCreate 10+ diverse test cases that test different aspects of your agent:")
    print("  - Mix of simple and complex questions")
    print("  - Different query lengths")
    print("  - Various topics/capabilities")
    print()

    queries = []

    print("Enter test cases (one per line, empty line to finish):")
    print("  Tip: After 10 test cases, press Enter on empty line to finish")
    print()

    count = 1
    while True:
        query = input(f"Test case {count}: ").strip()

        if not query:
            if len(queries) >= 10:
                break
            else:
                print(f"  Need at least 10 test cases (have {len(queries)})")
                continue

        queries.append(query)
        count += 1

    print(f"\n✓ Created {len(queries)} evaluation test cases")

    return queries


def generate_dataset_creation_code(
    tracking_uri: str,
    experiment_id: str,
    dataset_name: str,
    queries: list[str],
    is_databricks: bool,
) -> str:
    """Generate Python code for dataset creation."""

    # Convert queries to records format
    records_str = "[\n"
    for query in queries:
        # Escape quotes
        query_escaped = query.replace('"', '\\"')
        records_str += f'        {{"inputs": {{"query": "{query_escaped}"}}}},\n'
    records_str += "    ]"

    tags_code = (
        ""
        if is_databricks
        else """
        "version": "1.0",
        "purpose": "agent_evaluation",
    """
    )

    # Build tags parameter conditionally (cleaner approach)
    tags_param = (
        ""
        if is_databricks
        else f""",
    tags={{
{tags_code}
    }}"""
    )

    return f'''#!/usr/bin/env python3
"""
Create MLflow evaluation dataset for agent evaluation.

Generated by create_dataset_template.py
"""

import os
from mlflow.genai.datasets import create_dataset

# Set environment variables
os.environ["MLFLOW_TRACKING_URI"] = "{tracking_uri}"
os.environ["MLFLOW_EXPERIMENT_ID"] = "{experiment_id}"

# Dataset configuration
DATASET_NAME = "{dataset_name}"
EXPERIMENT_ID = "{experiment_id}"

print("=" * 60)
print("Creating MLflow Evaluation Dataset")
print("=" * 60)
print()

print(f"Dataset name: {{DATASET_NAME}}")
print(f"Experiment ID: {{EXPERIMENT_ID}}")
print()

# Step 1: Create dataset
print("Creating dataset...")
dataset = create_dataset(
    name=DATASET_NAME,
    experiment_id=EXPERIMENT_ID{tags_param}
)
print("✓ Dataset created")
print()

# Step 2: Prepare evaluation records
print("Preparing evaluation records...")
records = {records_str}

print(f"✓ Prepared {{len(records)}} records")
print()

# Step 3: Add records to dataset
print("Adding records to dataset...")
dataset = dataset.merge_records(records)
print("✓ Records added")
print()

# Step 4: Verify
print("=" * 60)
print("Dataset Created Successfully!")
print("=" * 60)
print()
print(f"Dataset ID: {{dataset.dataset_id}}")
print(f"Number of records: {{len(dataset.records)}}")
print()

# Sample a few records to show
print("Sample records:")
for i, record in enumerate(dataset.records[:3], 1):
    query = record["inputs"].get("query", str(record["inputs"]))
    print(f"  {{i}}. {{query[:80]}}{{'...' if len(query) > 80 else ''}}")

print()
print("✓ Dataset is ready for evaluation!")
print("=" * 60)
'''


def main():
    """Main workflow."""
    # Parse command-line arguments
    args = parse_arguments()

    print("=" * 60)
    print("MLflow Evaluation Dataset Template Generator")
    print("=" * 60)
    print()

    # Step 1: Detect environment
    tracking_uri, is_databricks = detect_tracking_uri()

    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
    if not experiment_id:
        print("\n✗ MLFLOW_EXPERIMENT_ID not set")
        print("  Run scripts/setup_mlflow.py first")
        sys.exit(1)

    print(f"Experiment ID: {experiment_id}")
    print()

    # Step 2: Get dataset name
    if is_databricks:
        dataset_name = get_databricks_table_name(
            args.catalog, args.schema, args.table, args.non_interactive
        )
    else:
        # Non-Databricks: simple dataset name
        if args.dataset_name:
            dataset_name = args.dataset_name
            print(f"\n✓ Using dataset name: {dataset_name}")
        elif args.non_interactive:
            print("✗ --dataset-name required in non-interactive mode")
            sys.exit(1)
        else:
            default_name = "mlflow-agent-eval-v1"
            dataset_name = input(f"Enter dataset name [{default_name}]: ").strip()
            if not dataset_name:
                dataset_name = default_name

    # Step 3: Generate sample queries (test cases)
    queries = generate_sample_queries(args.test_cases_file, args.non_interactive)

    # Step 4: Generate code
    print("\n" + "=" * 60)
    print("Generating Dataset Creation Script")
    print("=" * 60)

    code = generate_dataset_creation_code(
        tracking_uri, experiment_id, dataset_name, queries, is_databricks
    )

    # Write to file
    output_file = "create_evaluation_dataset.py"
    with open(output_file, "w") as f:
        f.write(code)

    print(f"\n✓ Script generated: {output_file}")
    print()

    # Make executable
    try:
        os.chmod(output_file, 0o755)
        print(f"✓ Made executable: chmod +x {output_file}")
    except Exception:
        pass

    print()
    print("=" * 60)
    print("Next Steps")
    print("=" * 60)
    print()
    print(f"1. Review the generated script: {output_file}")
    print(f"2. Execute it: python {output_file}")
    print("3. Verify the dataset was created successfully")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()

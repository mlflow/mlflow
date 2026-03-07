"""Run issue discovery on a Databricks experiment.

Usage:
    python scripts/run_discovery.py <config_name>

Requires workspace env vars (e.g. E2_URL, E2_TOKEN for the E2 workspace).
Uses Databricks model serving endpoints for LLM calls.

Example:
    export E2_URL="https://your-workspace.databricks.com"
    export E2_TOKEN="your-token"
    python scripts/run_discovery.py data-engineer-agent-insights
"""

import os
import sys

CONFIGS = {
    "personal-assistant": {
        "name": "Personal Assistant",
        "workspace": "E2",
        "experiment_id": "1664684809494323",
    },
    "data-engineer-agent-insights": {
        "name": "Data Engineer Agent Insights",
        "workspace": "E2",
        "experiment_id": "161678494797558",
    },
    "isaac-issue-discovery": {
        "name": "Isaac Issue Discovery",
        "workspace": "CL",
        "experiment_id": "1787079754540215",
    },
}

# Single model for both judging and analysis
MODEL = "databricks:/databricks-gpt-5-mini"


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CONFIGS:
        available = ", ".join(CONFIGS)
        print("Usage: python scripts/run_discovery.py <config_name>")
        print(f"Available configs: {available}")
        sys.exit(1)

    config = CONFIGS[sys.argv[1]]
    ws = config["workspace"]
    url_var, token_var = f"{ws}_URL", f"{ws}_TOKEN"

    url = os.environ.get(url_var)
    token = os.environ.get(token_var)
    if not url or not token:
        print(f"Error: {url_var} and {token_var} must be set")
        sys.exit(1)

    os.environ["DATABRICKS_HOST"] = url
    os.environ["DATABRICKS_TOKEN"] = token
    # litellm workaround: Databricks endpoints need OPENAI_API_KEY/BASE
    os.environ["OPENAI_API_KEY"] = token
    os.environ["OPENAI_API_BASE"] = f"{url}/serving-endpoints"

    import mlflow

    mlflow.set_tracking_uri("databricks")

    experiment_id = config["experiment_id"]
    print(f"Running issue discovery on: {config['name']}")
    print(f"  Experiment: {experiment_id}")
    print(f"  Model (both): {MODEL}")

    result = mlflow.genai.discover_issues(
        experiment_id=experiment_id,
        model=MODEL,
    )

    print(f"\n{result.summary}")
    for issue in result.issues:
        print(f"\n  {issue.name} ({issue.frequency:.0%} of sessions, {issue.confidence})")
        print(f"    {issue.description}")
        print(f"    Root cause: {issue.root_cause}")


if __name__ == "__main__":
    main()

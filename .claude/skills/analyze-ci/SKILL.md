---
name: analyze-ci
description: Analyze failed GitHub Action jobs for a pull request.
allowed-tools:
  - Bash(uv run --package skills skills analyze-ci:*)
---

# Analyze CI Failures

This skill analyzes logs from failed GitHub Action jobs using Claude.

## Prerequisites

- **GitHub Token**: Auto-detected via `gh auth token`, or set `GH_TOKEN` env var

## Usage

> **Note:** Always single-quote URLs to prevent the shell from interpreting special characters (e.g., `?` in query parameters).

```bash
# Analyze all failed jobs in a PR
uv run --package skills skills analyze-ci '<pr_url>'

# Analyze all failed jobs in a workflow run
uv run --package skills skills analyze-ci '<run_url>'

# Analyze specific job URLs directly
uv run --package skills skills analyze-ci '<job_url>' ['<job_url>' ...]

# Show debug info (tokens and costs)
uv run --package skills skills analyze-ci '<pr_url>' --debug
```

Output: A concise failure summary with root cause, error messages, test names, and relevant log snippets. Each job's output also includes:

- `Raw log: <path>`: full unfiltered log cached locally for grepping.
- `Package versions: <path>` (optional): file containing the `.github/actions/show-versions` action's output, written next to the raw log. Present only for jobs that run the action (e.g. cross-version tests). Useful for diffing against another run, or for reproducing the failure locally with matching package versions.

## Examples

```bash
# Analyze CI failures for a PR
uv run --package skills skills analyze-ci 'https://github.com/mlflow/mlflow/pull/19601'

# Analyze a specific workflow run
uv run --package skills skills analyze-ci 'https://github.com/mlflow/mlflow/actions/runs/22626454465'

# Analyze specific job URLs directly
uv run --package skills skills analyze-ci 'https://github.com/mlflow/mlflow/actions/runs/12345/job/67890'
```

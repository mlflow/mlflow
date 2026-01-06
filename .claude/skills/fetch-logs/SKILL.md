---
name: fetch-logs
description: Fetch and analyze logs from failed GitHub Action jobs for a pull request.
allowed-tools:
  - Bash
---

# Fetch and Analyze CI Logs

This skill fetches logs from failed GitHub Action jobs and analyzes them using Claude.

## Prerequisites

- **GitHub Token**: Auto-detected via `gh auth token`, or set `GITHUB_TOKEN` env var

## Usage

```bash
# Analyze all failed jobs in a PR
uv run .claude/skills/fetch-logs/fetch_logs.py <pr_url>

# Analyze specific job URLs directly
uv run .claude/skills/fetch-logs/fetch_logs.py <job_url> [job_url ...]
```

Output: A concise failure summary with root cause, error messages, test names, and relevant log snippets.

## Examples

```bash
# Analyze CI failures for a PR
uv run .claude/skills/fetch-logs/fetch_logs.py https://github.com/mlflow/mlflow/pull/19601

# Analyze specific job URLs directly
uv run .claude/skills/fetch-logs/fetch_logs.py \
  "https://github.com/mlflow/mlflow/actions/runs/12345/job/67890"
```

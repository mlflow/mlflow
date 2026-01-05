---
name: fetch-logs
description: Fetch logs from failed GitHub Action jobs for a pull request.
allowed-tools:
  - Bash
---

# Fetch CI Logs

This skill fetches logs from failed GitHub Action jobs.

## Prerequisites

- **GitHub Token**: Auto-detected via `gh auth token`, or set `GITHUB_TOKEN` env var

## Commands

### list

List failed jobs for a PR. Outputs JSON.

```bash
uv run .claude/skills/fetch-logs/fetch_logs.py list <owner/repo> <pr_number>
```

Output:

```json
{
  "pr": { "number": 123, "title": "...", "url": "..." },
  "failed_jobs": [{ "workflow_name": "...", "job_name": "...", "job_url": "..." }]
}
```

### fetch-logs

Fetch cleaned logs for jobs by URL. Outputs JSON.

```bash
uv run .claude/skills/fetch-logs/fetch_logs.py fetch-logs <job_url> [job_url ...]
```

Output:

```json
{
  "jobs": [
    {
      "workflow_name": "...",
      "job_name": "...",
      "job_url": "...",
      "failed_step": "...",
      "logs": "..."
    }
  ]
}
```

## Example

```bash
# List failed jobs for a PR
uv run .claude/skills/fetch-logs/fetch_logs.py list mlflow/mlflow 19601

# Fetch logs for all failed jobs
uv run .claude/skills/fetch-logs/fetch_logs.py fetch-logs \
  $(uv run .claude/skills/fetch-logs/fetch_logs.py list mlflow/mlflow 19601 | jq -r '.failed_jobs[].job_url')
```

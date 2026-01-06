---
name: analyze-ci
description: Analyze failed GitHub Action jobs for a pull request.
allowed-tools:
  - Bash
---

# Analyze CI Failures

This skill analyzes logs from failed GitHub Action jobs using Claude.

## Prerequisites

- **GitHub Token**: Auto-detected via `gh auth token`, or set `GITHUB_TOKEN` env var

## Usage

```bash
# Analyze all failed jobs in a PR
uv run .claude/skills/analyze-ci/analyze_ci.py <pr_url>

# Analyze specific job URLs directly
uv run .claude/skills/analyze-ci/analyze_ci.py <job_url> [job_url ...]
```

Output: A concise failure summary with root cause, error messages, test names, and relevant log snippets.

## Examples

```bash
# Analyze CI failures for a PR
uv run .claude/skills/analyze-ci/analyze_ci.py https://github.com/mlflow/mlflow/pull/19601

# Analyze specific job URLs directly
uv run .claude/skills/analyze-ci/analyze_ci.py https://github.com/mlflow/mlflow/actions/runs/12345/job/67890
```

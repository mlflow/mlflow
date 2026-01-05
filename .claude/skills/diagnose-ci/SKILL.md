---
name: diagnose-ci
description: Diagnose GitHub Action failures for a pull request using OpenAI API. Use this skill when the user asks for a summary or analysis of CI failures, test failures, or GitHub Action issues.
allowed-tools:
  - Bash
---

# Diagnose CI Failures

This skill fetches failed GitHub Action runs and uses the OpenAI API to diagnose and summarize the failure logs.

## Prerequisites

1. **GitHub Token**: Auto-detected via `gh auth token`, or set `GITHUB_TOKEN` env var
2. **OpenAI API Key**: Set `OPENAI_API_KEY` env var (for `summarize`)

## Subcommands

### list

Fetches failed jobs and their logs for a PR. Outputs JSON.

```bash
uv run .claude/skills/diagnose-ci/diagnose_ci.py list <owner/repo> <pr_number>
```

### summarize

Summarizes one or more jobs by URL. Outputs Markdown.

```bash
uv run .claude/skills/diagnose-ci/diagnose_ci.py summarize [--model MODEL] [--verbose] <job_url> [job_url ...]
```

Options:

- `--model`, `-m`: OpenAI model: `gpt-4.1-mini` (default) or `gpt-4.1-nano`
- `--verbose`, `-v`: Include the (truncated) logs in the output for verification

## Examples

```bash
# List failed jobs for a PR
uv run .claude/skills/diagnose-ci/diagnose_ci.py list mlflow/mlflow 19601

# Summarize specific failed jobs
uv run .claude/skills/diagnose-ci/diagnose_ci.py summarize \
  "https://github.com/mlflow/mlflow/actions/runs/20516074003/job/58944102866"

# Summarize all failed jobs for a PR
uv run .claude/skills/diagnose-ci/diagnose_ci.py summarize \
  $(uv run .claude/skills/diagnose-ci/diagnose_ci.py list mlflow/mlflow 19601 | jq -r '.failed_jobs[].job_url')
```

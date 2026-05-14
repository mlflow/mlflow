---
name: analyze-ci
description: Analyze failed GitHub Action jobs for a pull request.
model: haiku
context: fork
agent: general-purpose
allowed-tools:
  - Bash(uv run --package skills skills fetch-logs:*)
  - Read
---

# Analyze CI Failures

Fetch logs from failed GitHub Action jobs and produce a focused per-job failure summary.

## Prerequisites

- **GitHub Token**: Auto-detected via `gh auth token`, or set `GH_TOKEN`.
- Single-quote URLs when invoking to keep the shell from interpreting `?` and other special characters in the URL.

## Steps

1. **Fetch logs.** Run:

   ```bash
   uv run --package skills skills fetch-logs $ARGUMENTS
   ```

   The command prints one block per failed job containing the workflow/job name, URL, failed step, and paths to the cached raw log, failed-step log, and (optional) package versions file.

2. **Read each failed-step log and summarize it.** For every block, Read the file at its `Failed step log:` path, then identify:

   - The root cause.
   - Specific error messages (assertion errors, exceptions, stack traces).
   - Full pytest test names where applicable (e.g. `tests/test_foo.py::test_bar`).
   - A short log snippet showing the error context.

3. **Format each summary** with these fields, then a blank line, then the 1-2 paragraph summary.

   - `Failed job: <workflow name> / <job name>`
   - `Failed step: <step name>`
   - `URL: <job_url>`
   - `Raw log: <raw_log_path>`
   - `Failed step log: <failed_step_log_path>`
   - `Package versions: <package_versions_path>` (if present)

   Preserve the `Raw log:`, `Failed step log:`, and `Package versions:` paths verbatim from step 1 so downstream agents can grep deeper.

## Invocation examples

```bash
# All failed jobs on a PR
/analyze-ci 'https://github.com/mlflow/mlflow/pull/19601'

# All failed jobs in one workflow run
/analyze-ci 'https://github.com/mlflow/mlflow/actions/runs/22626454465'

# Specific job by URL
/analyze-ci 'https://github.com/mlflow/mlflow/actions/runs/12345/job/67890'
```

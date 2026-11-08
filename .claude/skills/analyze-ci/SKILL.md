---
name: analyze-ci
description: Analyze failed GitHub Action jobs. Takes one or more GitHub URLs (job, workflow-run, or PR) and summarizes each failure with root cause and log paths.
when_to_use: When CI is red or a check failed, when the user pastes a GitHub Actions job or run URL, or when asked why a PR's checks are failing.
argument-hint: "url(s) of failed GitHub Action jobs, workflow runs, or PRs"
---

# Analyze CI Failures

Fetch logs from failed GitHub Action jobs and produce a focused per-job failure summary.

URLs provided: $ARGUMENTS

If no URLs are listed above, ask for a job, run, or PR URL and stop. `fetch-logs` requires at least one and exits 2 without.

## Prerequisites

- **GitHub Token**: Auto-detected via `gh auth token`, or set `GH_TOKEN`.

## Steps

1. **Fetch logs.** Run, single-quoting each URL so the shell does not interpret `?` or other special characters in it:

   ```bash
   uv run --package skills skills fetch-logs '<url>' ['<url>' ...]
   ```

   The command saves one raw log per job, preserving timestamps and removing ANSI colors. It prints all failed steps and inclusive, 1-based line ranges in that file, with multiple ranges separated by commas. Absent sections are omitted; no separate section files are created. For example (URL and ranges are illustrative):

   ```text
   ## Cross-version tests / Python 3.11
   URL: https://github.com/mlflow/mlflow/actions/runs/123/job/456
   Raw log: /tmp/fetch-logs/123/456.log
   Failed step lines: 850-1620, 1800-2100
   Pytest summary lines: 1580-1618, 2060-2098
   Package versions lines: 120-340
   Snapshot changes lines: 350-375
   ```

   Snapshot changes are located in the existing raw log, not fetched or compared separately.

2. **Read the reported ranges and summarize each failure.** For every block, read all `Failed step lines:` and `Pytest summary lines:` ranges from the `Raw log:` file (for example, `sed -n '850,1620p' /tmp/fetch-logs/123/456.log`), then identify:

   - The root cause.
   - Specific error messages (assertion errors, exceptions, stack traces).
   - Full pytest test names where applicable (e.g. `tests/test_foo.py::test_bar`).
   - A short log snippet showing the error context.

   If `Package versions lines:` or `Snapshot changes lines:` is present, read those ranges in the same raw log too and summarize relevant package changes alongside the failure, including the baseline run link when available. Changes use uv's `Updated name vold -> vnew`, `Added name vversion`, and `Removed name vversion` format. Treat changes as diagnostic evidence, not proof of the root cause. Report unavailable baselines or comparison errors as such, not as package removals. Do not fetch artifacts or perform another comparison. Older or non-cross-version logs may have no snapshot changes block; analyze them normally. If no failed-step range is available (for example, step timestamps are missing), inspect the raw log directly.

3. **Format each summary** with these fields, then a blank line, then the 1-2 paragraph summary.

   - `## <workflow name> / <job name>`
   - `URL: <job_url>`
   - `Raw log: <raw_log_path>`
   - `Failed step lines: <ranges>` (if present)
   - `Pytest summary lines: <ranges>` (if present)
   - `Package versions lines: <ranges>` (if present)
   - `Snapshot changes lines: <ranges>` (if present)

   Preserve the `Raw log:` path and all reported ranges verbatim from step 1 so downstream agents can inspect the same evidence. If no failed step is recorded, preserve the `Conclusion:` field too.

## Invocation examples

```bash
# All failed jobs on a PR
/analyze-ci https://github.com/mlflow/mlflow/pull/19601

# All failed jobs in one workflow run
/analyze-ci https://github.com/mlflow/mlflow/actions/runs/22626454465

# Specific job by URL
/analyze-ci https://github.com/mlflow/mlflow/actions/runs/12345/job/67890

# Multiple URLs at once
/analyze-ci https://github.com/mlflow/mlflow/actions/runs/123/job/456 https://github.com/mlflow/mlflow/actions/runs/789/job/012
```

---
name: analyze-ci
description: Analyze failed GitHub Action jobs. Takes one or more GitHub URLs (job, workflow-run, or PR) and summarizes each failure with root cause and log paths.
when_to_use: When CI is red or a check failed, when the user pastes a GitHub Actions job or run URL, or when asked why a PR's checks are failing.
argument-hint: "url(s) of failed GitHub Action jobs, workflow runs, or PRs"
---

# Analyze CI Failures

Fetch logs from failed GitHub Action jobs and produce a focused per-job failure summary.

URLs provided: $ARGUMENTS

If no URLs are passed as arguments, use the relevant job, run, or PR URLs already shared in the conversation. If none are available, ask for a URL and stop.

## Steps

- 1. Fetch logs with the following command. Single-quote each URL to protect shell special characters like `?`.

  ```bash
  uv run --package skills skills fetch-logs '<url1>' ['<url2>' ...]
  ```

- 2. Read all reported ranges. If no failed-step range is available, inspect the raw log directly. State the root cause and its impact, with a short log excerpt. Group pytest failures that share a cause and give their count. Include a full pytest node ID (e.g. `tests/test_foo.py::test_bar[case1]`) when a specific case helps explain the failure. Mention package versions or snapshot changes only if they help explain the failure.

- 3. Format each job with a heading, an evidence list, and a summary of 1–2 sentences plus the log excerpt.

  ```markdown
  ## <workflow name> / <job name>

  - URL: <job_url>
  - Raw log: <raw_log_path>

  <summary>
  ```

  Copy the raw log path verbatim from step 1. If no failed step is recorded, include the reported `Conclusion:` field too.

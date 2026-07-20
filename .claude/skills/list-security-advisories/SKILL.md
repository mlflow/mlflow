---
name: list-security-advisories
description: List GitHub security advisories for the MLflow repo, grouped by state.
argument-hint: "optional: --state triage|draft|published|closed, --repo owner/repo"
allowed-tools:
  - Bash(uv run --package skills skills list-advisories:*)
---

# List Security Advisories

List repository security advisories (Triage, Draft, Published, Closed) for the
MLflow repo. This is a read-only overview to support triage; it never accepts,
closes, comments on, or publishes advisories.

## Prerequisites

- **GitHub token**: Auto-detected via `gh auth token`, or set `GH_TOKEN`.
- **Advisory access**: Unpublished advisories (Triage/Draft/Closed) are only
  returned if the token belongs to a security manager/admin of the repo or a
  collaborator on the advisory, and the token has the `repo` or
  `repository_advisories:read` scope. Without that scope you will get a 403 or
  only see Published advisories.

## Steps

1. **Fetch advisories.** Run (add `--state <state>` to filter, `--repo owner/repo` to target another repo):

   ```bash
   uv run --package skills skills list-advisories $ARGUMENTS
   ```

   The command prints a total count, a per-state count line, and then one
   section per state. Each advisory entry spans lines: `GHSA id (severity, CWE)`,
   the advisory URL, and the summary.

2. **Present the output** to the user: lead with the per-state counts, then the
   grouped advisories. Render every advisory reference as a Markdown link whose
   visible text is the GHSA id and whose target is the advisory URL, e.g.
   `[GHSA-83fm-w79m-64r5](https://github.com/mlflow/mlflow/security/advisories/GHSA-83fm-w79m-64r5)`
   - never emit a bare GHSA id without its embedded link, even when listing
   compactly. The advisory URL is always
   `https://github.com/<owner>/<repo>/security/advisories/<ghsa_id>` and is
   included verbatim in the command output. Do not attempt any write actions.

## Invocation examples

```bash
# All advisories for mlflow/mlflow
/list-security-advisories

# Only the Triage backlog
/list-security-advisories --state triage

# A different repository
/list-security-advisories --repo mlflow/mlflow --state draft
```

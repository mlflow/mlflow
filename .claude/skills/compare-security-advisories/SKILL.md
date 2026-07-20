---
name: compare-security-advisories
description: Compare two or more MLflow security advisories and explain whether they are the same vulnerability.
argument-hint: "two or more GHSA ids (optionally --repo owner/repo)"
allowed-tools:
  - Bash(uv run --package skills skills get-advisory:*)
---

# Compare Security Advisories

Compare two or more advisories side by side and explain why they are similar or
different, and whether they describe the same underlying vulnerability. Useful
for confirming whether separate dedupe clusters are actually the same bug.

## Read-only / notify-only (hard constraint)

This skill only reads. It MUST NOT modify any advisory, and MUST NOT accept,
close, comment on, publish, or edit them (nor do so on the maintainer's behalf).
It fetches detail with a read-only GET and presents an analysis for the
maintainer only.

## Prerequisites

- **GitHub token**: Auto-detected via `gh auth token`, or set `GH_TOKEN`.
- **Advisory access**: Unpublished advisories require the token to belong to a
  security manager/admin of the repo or a collaborator on the advisory, with the
  `repo` or `repository_advisories:read` scope.

## Steps

1. **Fetch all advisories.** Run with two or more ids:

   ```bash
   uv run --package skills skills get-advisory <ghsa_id_a> <ghsa_id_b> [<ghsa_id_c> ...]
   ```

2. **Build a side-by-side.** Compare the reports across: root cause, affected
   component/endpoint/function, PoC technique, affected/patched versions, and
   CWE(s).

3. **Explain and conclude.** State why the reports are similar or different, and
   whether they are the same underlying vulnerability, with a confidence level.
   Call out meaningful differences (e.g. one adds username spoofing, or targets a
   different endpoint) rather than treating shared vocabulary as sameness.

4. **Highlight the original.** Identify the earliest report as the original
   source (ordering only; do not assign credit).

## Presentation conventions

- Render every advisory reference as a Markdown link whose visible text is the
  GHSA id and whose target is the advisory URL, e.g.
  `[GHSA-8x5x-x647-fgwg](https://github.com/mlflow/mlflow/security/advisories/GHSA-8x5x-x647-fgwg)`.
  The URL is always `https://github.com/<owner>/<repo>/security/advisories/<ghsa_id>`.
- Order advisories earliest to latest and annotate each with `(date, CWE-...)`.
- Do not take or recommend taking any action on the advisories themselves.

## Invocation example

```bash
/compare-security-advisories GHSA-8x5x-x647-fgwg GHSA-25gg-9x2c-g58q
```

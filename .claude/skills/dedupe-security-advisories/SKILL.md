---
name: dedupe-security-advisories
description: Find likely-duplicate MLflow security advisories and highlight the earliest report.
argument-hint: "optional: --state triage|draft|published|closed|all, --threshold 0.75, --repo owner/repo"
allowed-tools:
  - Bash(uv run --package skills skills dedupe-advisories:*)
  - Write
  - Edit
---

# Dedupe Security Advisories

Surface candidate duplicate security advisories for the MLflow repo so the
maintainer can review them, and highlight the earliest (original) report in each
group.

## Notify-only (hard constraint)

This skill is strictly informational. You MUST NOT close, comment on, edit, or
otherwise modify any advisory, and you MUST NOT perform those actions on the
maintainer's behalf. Only list potential duplicates and the original source; the
maintainer decides what to do.

## Prerequisites

- **GitHub token**: Auto-detected via `gh auth token`, or set `GH_TOKEN`.
- **Advisory access**: Unpublished advisories (Triage/Draft/Closed) are only
  returned if the token belongs to a security manager/admin of the repo or a
  collaborator on the advisory, with the `repo` or `repository_advisories:read`
  scope. Otherwise only Published advisories are visible.

## Steps

1. **Generate the report scaffold.** Run (defaults to the Triage backlog),
   passing `--output dedupe-advisories.md` so the command writes a fully-rendered
   Markdown scaffold directly to that file in the current directory:

   ```bash
   uv run --package skills skills dedupe-advisories --output dedupe-advisories.md $ARGUMENTS
   ```

   The command deterministically clusters advisories by text similarity plus
   structural signals (shared CWE, affected component, or CVE), then writes
   `dedupe-advisories.md` with **everything mechanical already rendered**: the
   `**Analyzed N advisory(ies)**` count, a `## Confidence tiers` index, one
   `### Cluster N` block per cluster (with a bold `**Original source (earliest
   report):**` label over the earliest report and `**Later duplicates:**` over
   the rest), and the complete `## Not part of any duplicate cluster` section.
   Every advisory is already a `[GHSA-id](url) (date, CWE) — title` Markdown
   link. Each cluster carries `<!-- TITLE: TODO -->`, `<!-- CONFIDENCE: TODO -->`,
   and `> _Reasoning: TODO_` placeholders for you to fill in step 4. These are
   candidates, not confirmed duplicates. Read the file to review the clusters.

2. **Confirm each cluster.** For every cluster, judge whether the members are
   truly the same vulnerability (same root cause and affected component, not just
   shared vocabulary). State a confidence level and briefly justify it. Discard
   groupings that only share generic wording.

3. **Report to the maintainer.** Start with the total number of advisories
   analyzed (all advisories for the selected state - the command prints
   `Analyzed N advisory(ies) (state: ...)`), so the maintainer knows the full
   scope that was reviewed, not just the clustered subset. Then, for each
   confirmed group, present:
   - the `ORIGINAL SOURCE` (earliest report) as the first-reported one, with its
     GHSA id, date, reporter, and direct link;
   - the later duplicate reports with their links;
   - your confidence and reasoning.

   Always render every advisory reference as a Markdown link whose visible text
   is the GHSA id and whose target is the advisory URL, e.g.
   `[GHSA-43rh-4vc9-rvfq](https://github.com/mlflow/mlflow/security/advisories/GHSA-43rh-4vc9-rvfq)`.
   This applies even when listing members compactly (comma-separated) - never
   emit a bare GHSA id without its embedded link. The advisory URL is always
   `https://github.com/<owner>/<repo>/security/advisories/<ghsa_id>` and is
   included verbatim in the command output.

   Within every cluster, list members from earliest to latest report date
   (the command output is already ordered this way), and append each report's
   date and CWE id(s) in brackets after its link so the progression and any
   shared weakness types are visible. Use the CWE(s) from the command output, or
   `no CWE` when none is attached, e.g.
   `[GHSA-mf9x-42mc-wh54](...) (2026-05-25, CWE-95), [GHSA-3cjh-9p94-g2rj](...) (2026-04-23, CWE-94)`.

   After the clusters, include a `Not part of any duplicate cluster - review
   independently` section listing the advisories the command reports under
   `## Not part of any duplicate cluster (N)`. These have no detected duplicate and
   each needs an independent look. List them earliest to latest, one per line, as a
   linked GHSA with `(date, CWE)` and its title.

   Do not take or recommend taking any action on the advisories themselves.

4. **Enrich the scaffold in `dedupe-advisories.md`.** Step 1 already wrote a
   complete, valid `dedupe-advisories.md` — all GHSA links, dates, CWEs, titles,
   the analyzed-count header, the original/duplicate labels, and the **entire**
   `## Not part of any duplicate cluster` section are already rendered by the
   command. Your job is only to layer in your judgment by filling the `TODO`
   markers with the `Edit` tool. **Never re-type a member link line and never
   regenerate the singleton section** — that mechanical content already exists,
   and re-authoring it is what previously made this step fail. This is a
   local-only reference file - do NOT `git add`, stage, or commit it (it is
   already gitignored). This is the only write this skill performs; it never
   modifies code or advisories.

   Work in **small, bounded `Edit` calls dispatched in parallel batches** — never
   a single large rewrite, but also never one edit per message (52 serial
   round-trips is what makes this step slow):

   - **Per cluster**, make one `Edit` that replaces that cluster's heading
     placeholders and reasoning line together. Turn
     `### Cluster N (K reports) — <!-- TITLE: TODO --> — <!-- CONFIDENCE: TODO -->`
     into e.g. `### Cluster N — Dockerfile injection — Medium`, and replace that
     cluster's `> _Reasoning: TODO_` with one sentence on why the members are (or
     may be) the same underlying vulnerability.
   - **Send ~8–10 of these per-cluster `Edit` calls in a single message** so they
     run concurrently, then move to the next batch. Each edit is tiny (you only
     author the title, confidence, and one reasoning sentence — the links are
     already on disk), so batching carries no timeout risk and cuts the wall time
     roughly in proportion to the batch size. Do NOT enrich clusters one message
     at a time.
   - **Tier index (one `Edit`).** Fill the `## Confidence tiers` section: replace
     the two `TODO`s with grouped cluster-number references, e.g.
     `- **Strong duplicates (high confidence):** Clusters 1, 2, 3, 5, 6` and
     `- **Medium-confidence pairs:** Clusters 20, 24, 34`. **Do not move the
     cluster blocks** — they stay in the command's order; the index is how you
     express the tiers.

   Confirm every cluster's title, confidence, and reasoning `TODO` is filled and
   the tier index has no remaining `TODO` before finishing. Leave the singleton
   section exactly as the command rendered it.

## Invocation examples

```bash
# Dedupe the Triage backlog (default)
/dedupe-security-advisories

# Compare across all states
/dedupe-security-advisories --state all

# Loosen or tighten the similarity threshold
/dedupe-security-advisories --threshold 0.5
```

Extra flags are forwarded to the command (via `$ARGUMENTS`) alongside the
`--output dedupe-advisories.md` that step 1 always passes.

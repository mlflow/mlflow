---
name: draft-security-fix
description: Draft a local-only fix plus regression test for a validated, unfixed MLflow security advisory.
argument-hint: "a GHSA id (optionally --repo owner/repo)"
allowed-tools:
  - Bash(uv run --package skills skills get-advisory:*)
  - Bash(git log:*)
  - Bash(git show:*)
  - Bash(git blame:*)
  - Bash(git diff:*)
  - Bash(git status:*)
  - Bash(uv run pytest:*)
  - Bash(uv run ruff:*)
  - Bash(uv run clint:*)
  - Read
  - Grep
  - Glob
  - Edit
  - Write
---

# Draft Security Fix

Turn a validated, not-yet-fixed MLflow security advisory into a proposed patch and
a regression test, applied only to the local working tree, so the maintainer can
review a ready-to-inspect change. This is the write-capable last stage of the
triage pipeline: `dedupe-security-advisories` -> `validate-security-advisory` ->
`draft-security-fix`.

## Local-only / no-publish (hard constraint)

All edits stay in the local working tree for the maintainer to review. This skill
MUST NOT:

- run `git add`, `git commit`, or `git push`;
- create, switch, or push a branch (in a fork or upstream);
- open a pull request anywhere (fork or `mlflow/mlflow`);
- modify, comment on, accept, publish, or close the advisory (nor do so on the
  maintainer's behalf).

Because nothing leaves the machine, there is no pre-disclosure leak risk. Leave
the working tree dirty; the maintainer reviews it and drives commit / PR /
disclosure themselves, privately, per the GitHub Security Advisory lifecycle.

## Prerequisites

- **Validated first**: only run this after `validate-security-advisory` returned a
  verdict of `valid` and not-yet-fixed for this GHSA. If it was `already-fixed`,
  `likely-invalid`, or `needs-more-info`, stop and defer back to validation.
- **GitHub token**: Auto-detected via `gh auth token`, or set `GH_TOKEN` (used
  only for the read-only `get-advisory` fetch).

## Steps

1. **Re-confirm the precondition.** Re-fetch the advisory detail:

   ```bash
   uv run --package skills skills get-advisory <ghsa_id>
   ```

   Re-confirm against the current checkout that the vulnerability is still present
   and unfixed (e.g. `git log -S'<guard snippet>' -- <path>`, or grep for an
   existing guard/validator). If a fix or guard already exists, stop and report
   "already fixed" - do NOT patch. If the claim does not hold against the current
   tree, stop and defer back to `validate-security-advisory`.

2. **Locate the sink.** Use Grep/Glob/Read to pin the exact vulnerable code path
   (route handler, function, validator, or sink) with concrete `file:line`
   references. Read enough surrounding code to understand real behavior and avoid
   regressions.

3. **Design a minimal fix.** Choose the smallest targeted change that closes the
   root cause without changing unrelated behavior. Follow the repo code style:
   - use top-level imports (only use lazy imports when necessary);
   - add comments only for non-obvious intent, not to narrate the code;
   - when touching the SQLAlchemy tracking store, keep ALL workspace-aware paths
     and validations intact - never drop workspace plumbing even if the fix
     focuses on single-tenant behavior.

4. **Apply the fix locally** with Edit/Write. Keep the change scoped to what the
   vulnerability requires.

5. **Add a regression test** that fails before the fix and passes after it,
   asserting the vulnerable behavior is now blocked. If the change is in the
   tracking layer, also add the workspace-aware variant in
   `tests/store/tracking/test_sqlalchemy_store_workspace.py`. Only add a docstring
   when it provides context beyond the test name.

6. **Verify.** Run the targeted test plus linters on the changed files, and fix
   anything they flag:

   ```bash
   uv run pytest <path to new/affected test>
   uv run ruff format <changed files> && uv run ruff check <changed files>
   uv run clint <changed files>
   ```

7. **Report, do not ship.** Summarize for the maintainer:
   - the root cause and the patch, with `file:line` citations;
   - the regression test added and how to run it;
   - the local change surface via `git diff --stat` (and `git status`).

   Render the advisory reference as a Markdown link whose visible text is the GHSA
   id, e.g.
   `[GHSA-mf9x-42mc-wh54](https://github.com/mlflow/mlflow/security/advisories/GHSA-mf9x-42mc-wh54)`.
   Explicitly state that nothing was committed, pushed, branched, or PR'd and that
   the advisory was untouched - the maintainer reviews the working tree and
   handles commit, PR, and disclosure privately.

## Non-goals

- No proof-of-concept execution (consistent with `validate-security-advisory`):
  correctness is argued via code reasoning plus the regression test, not by
  detonating the exploit.
- No commit, push, branch, or PR - see the hard constraint above.

## Invocation example

```bash
/draft-security-fix GHSA-mf9x-42mc-wh54
```

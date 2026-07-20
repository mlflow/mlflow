---
name: validate-security-advisory
description: Deep-dive one MLflow security advisory, assess its validity against the codebase, and (when already fixed) draft a maintainer-approved reply for manual posting.
argument-hint: "a GHSA id (optionally --repo owner/repo)"
allowed-tools:
  - Bash(uv run --package skills skills get-advisory:*)
  - Bash(uv run --package skills skills list-advisories:*)
  - Bash(git log:*)
  - Bash(git show:*)
  - Bash(git blame:*)
  - Grep
  - Glob
  - Read
---

# Validate Security Advisory

Deep-dive a single security advisory and assess whether the reported
vulnerability is plausible, by correlating the report with the actual MLflow
source in the current checkout. When a fix is already present, link the pull
request that introduced it, and cross-check whether a previously closed advisory
already addressed the same issue.

## Read-only / notify-only (hard constraint)

This skill only reads. It MUST NOT modify any code or any advisory. It MAY draft
a reply for the maintainer to review (see step 8), but it MUST NOT post that reply
and MUST NOT accept, close, comment on, publish, or edit the advisory (nor do so
on the maintainer's behalf). It fetches advisory detail with a read-only GET and
inspects the local repository. All output - including any drafted reply - is for
the maintainer only.

Posting a reply is not even possible through the API: GitHub exposes no endpoint
to comment on a repository security advisory, so the advisory conversation is
web-UI only. Any reply is therefore posted manually by the maintainer, and only
after they explicitly approve the wording.

## Prerequisites

- **GitHub token**: Auto-detected via `gh auth token`, or set `GH_TOKEN`.
- **Advisory access**: Unpublished advisories require the token to belong to a
  security manager/admin of the repo or a collaborator on the advisory, with the
  `repo` or `repository_advisories:read` scope.

## Steps

1. **Fetch the advisory detail.** Run:

   ```bash
   uv run --package skills skills get-advisory <ghsa_id>
   ```

   This prints the summary, severity, CWE(s), reporter, affected/patched
   versions, and the full description (which usually contains the PoC / repro).

2. **Extract the claim.** From the description, identify the claimed affected
   component, endpoint/route, function or file, the attack vector, and any
   preconditions (auth required, config flags, network position).

3. **Locate the code.** Use Grep/Glob/Read to find the claimed code paths in the
   local MLflow tree (e.g. the route handler, the function named in the PoC, the
   validator or sink). Read enough surrounding code to understand the real
   behavior.

4. **Assess validity.** Determine whether:
   - the referenced code path actually exists and is reachable as described;
   - the vulnerability is plausible given the real code (not just the report's
     wording);
   - a fix or guard is already present, and whether a regression test covers it;
   - the reported affected version differs from the current checkout - if so,
     say the assessment is against the current tree and the reported version may
     behave differently.

5. **Identify the fixing PR (if fixed).** If a fix or guard is already present,
   trace it to the pull request that introduced it. Use `git log`/`git blame`/
   `git show` on the relevant file or code (e.g. `git log --oneline -S'<snippet>'
   -- <path>`, or blame the guard line) and extract the PR number from the
   squash-merge commit subject (MLflow uses `... (#12345)`). Report it as a
   Markdown link with the PR number as the visible text, e.g.
   `[#12345](https://github.com/mlflow/mlflow/pull/12345)` (URL is always
   `https://github.com/<owner>/<repo>/pull/<number>`). If you cannot confidently
   pin the PR, say so rather than guessing.

6. **Cross-check closed advisories.** Check whether a previously closed advisory
   already covers this same vulnerability (e.g. an earlier report of the same
   bug that was handled/fixed). List closed advisories with
   `uv run --package skills skills list-advisories --state closed` and, for
   plausible matches, fetch detail with `get-advisory <closed_ghsa>` to confirm
   it is the same root cause/component. If one matches, report it as a related
   closed advisory (Markdown-linked GHSA) and note whether the current report is
   already addressed by that fix.

7. **Report.** Give a verdict (valid / already-fixed / likely-invalid /
   needs-more-info) with a confidence level, backed by specific `file:line`
   citations. Render the GHSA reference (and any related closed GHSA) as a
   Markdown link with the GHSA id as the visible text, and link the fixing PR
   when identified. Do not take or recommend taking any action on the advisory
   itself.

8. **Draft a reply when already fixed (maintainer-approved, manual posting).** If,
   and only if, the verdict is `already-fixed`, draft a short, courteous reply to
   the reporter suitable for the advisory conversation. It should:
   - thank the reporter for the report;
   - state that the reported issue is already addressed in the current code,
     citing the fixing PR (Markdown-linked) and the guard/regression test with
     `file:line`;
   - note the fixed/affected versions when known;
   - avoid disclosing any unrelated sensitive detail.

   Then present the draft to the maintainer and ask for explicit approval. Do NOT
   post it, and do not imply it has been or will be posted automatically. Posting
   is not possible via the GitHub API (there is no endpoint to comment on a
   repository security advisory) and this skill is read-only regardless, so the
   maintainer posts the approved wording manually in the advisory web UI
   (`https://github.com/<owner>/<repo>/security/advisories/<ghsa_id>`). Always
   require the maintainer's permission before any reply is posted; never post or
   claim to post on their behalf.

## Invocation example

```bash
/validate-security-advisory GHSA-mf9x-42mc-wh54
```
